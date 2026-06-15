"""Concrete learner for the Graph→Schedule policy.

Implements both the legacy REINFORCE step and the recommended PPO loop with
paired baselines, K rollouts per graph, advantage normalization, optional
replay buffer, behavioral cloning pretraining, residual-α curriculum, and
probe diagnostics.

The ``algorithm`` field of :class:`TrainingConfig` selects between them.
``ppo`` is the default and recommended choice.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import networkx as nx
import torch

from config import ProjectConfig, UDGConfig
from schedules import GlobalSchedule
from graphs.unit_disk import generate_square_lattice_udg
from module1.policy import SchedulePolicy
from module1.base import FixedScheduleBaseline
from module3.interfaces import Learner, TrainingConfig
from module3.reinforce import reinforce_step
from module3.ppo import ppo_step
from module3.replay import ReplayBuffer


def _build_graph_pool(
    udg_cfg: UDGConfig,
    pool_size: int,
    base_seed: int,
    *,
    extra_factor: int = 2,
) -> list[nx.Graph]:
    """Generate a pool of UDGs with varying random seeds.

    Generates ``pool_size * extra_factor`` candidate graphs and keeps the
    first ``pool_size`` with at least one edge.  Positions are attached as
    ``G.graph['positions']`` so the backend adapter can find them.
    """
    pool: list[nx.Graph] = []
    for i in range(pool_size * max(extra_factor, 1)):
        if len(pool) >= pool_size:
            break
        cfg = UDGConfig(
            nx=udg_cfg.nx,
            ny=udg_cfg.ny,
            spacing=udg_cfg.spacing,
            radius=udg_cfg.radius,
            dropout_rate=udg_cfg.dropout_rate,
            seed=base_seed + i,
        )
        G, pos = generate_square_lattice_udg(cfg)
        if G.number_of_edges() == 0:
            continue
        G.graph["positions"] = pos
        G.graph["seed"] = base_seed + i
        pool.append(G)
    return pool


def _curate_pool(
    graphs: list[nx.Graph],
    baseline_rewards: dict[object, float],
    lo: float,
    hi: float,
) -> list[nx.Graph]:
    """Drop graphs whose baseline reward is outside ``(lo, hi)``.

    Graphs whose baseline already attains the maximum reward carry no
    learning signal (nothing to improve), and graphs where the baseline
    achieves nothing usually correspond to instances the AHS protocol
    cannot solve at all — both extremes only add noise.
    """
    kept: list[nx.Graph] = []
    for G in graphs:
        key = G.graph.get("seed", id(G))
        if key not in baseline_rewards:
            kept.append(G)
            continue
        r = float(baseline_rewards[key])
        if lo < r < hi:
            kept.append(G)
    return kept


class ReinforceLearner(Learner):
    """Policy-gradient learner with REINFORCE or PPO backends.

    Despite the historical name, ``algorithm='ppo'`` (default) uses the
    paired-baseline PPO step from :mod:`module3.ppo`.

    Parameters
    ----------
    config : TrainingConfig
        Training hyperparameters.
    project : ProjectConfig
        Physics / hardware configuration.
    backend_fn : callable
        ``(nx.Graph, GlobalSchedule) -> float`` reward function used for
        learned-schedule rewards.  May be the normalized
        ``is_cost_vs_baseline`` variant.
    raw_backend_fn : callable | None
        ``(nx.Graph, GlobalSchedule) -> float`` reward function returning
        un-normalized rewards.  Required when paired baselines are enabled
        and ``backend_fn`` is normalized.  Defaults to ``backend_fn``.
    device : str
        Torch device for the policy network.
    hidden_dim : int
        Hidden dimension for the policy GIN + decoder MLPs.
    """

    def __init__(
        self,
        config: TrainingConfig,
        project: ProjectConfig,
        backend_fn: Callable[[nx.Graph, GlobalSchedule], float],
        *,
        raw_backend_fn: Callable[[nx.Graph, GlobalSchedule], float] | None = None,
        device: str = "cpu",
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(config, project)
        self.backend_fn = backend_fn
        self.raw_backend_fn = raw_backend_fn or backend_fn
        self.device = torch.device(device)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)
            np.random.seed(config.seed)

        self.policy = SchedulePolicy(
            project,
            hidden_dim=hidden_dim,
            init_log_std=config.init_log_std,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.baseline_ema: dict[int, float] = {}
        self.step_count = 0
        self.best_eval_reward = -float("inf")

        self.baseline_model = FixedScheduleBaseline(project)
        self.replay = ReplayBuffer(config.replay_buffer_size)

        base_seed = config.seed if config.seed is not None else 0
        raw_train_pool = _build_graph_pool(
            project.udg, config.graph_pool_size, base_seed
        )
        self.eval_pool = _build_graph_pool(
            project.udg, config.eval_graphs, base_seed + 10_000
        )

        self.baseline_reward_cache: dict[object, float] = {}
        if config.pool_curation or config.algorithm == "ppo":
            for G in raw_train_pool + self.eval_pool:
                key = G.graph.get("seed", id(G))
                if key in self.baseline_reward_cache:
                    continue
                sched_b = self.baseline_model.make_schedule(G)
                self.baseline_reward_cache[key] = float(
                    self.raw_backend_fn(G, sched_b)
                )

        if config.pool_curation:
            self.train_pool = _curate_pool(
                raw_train_pool, self.baseline_reward_cache,
                config.curation_lo, config.curation_hi,
            )
            if len(self.train_pool) < max(config.batch_size, 1):
                # Fall back to uncurated pool if curation was too aggressive.
                self.train_pool = raw_train_pool
        else:
            self.train_pool = raw_train_pool

    # ── core RL step ────────────────────────────────────────────────────

    def _update_residual_alpha(self) -> float:
        alpha = self.policy.current_alpha(self.step_count)
        self.policy.set_residual_alpha(alpha)
        return alpha

    def train_step(self, graphs: list[nx.Graph]) -> dict[str, Any]:
        alpha = self._update_residual_alpha()

        if self.config.algorithm == "reinforce":
            metrics = reinforce_step(
                policy=self.policy,
                graphs=graphs,
                backend_fn=self.backend_fn,
                optimizer=self.optimizer,
                baseline_ema=self.baseline_ema,
                ema_alpha=self.config.ema_alpha,
                entropy_coef=self.config.entropy_coef,
                value_loss_coef=self.config.value_loss_coef,
                grad_clip=self.config.grad_clip,
            )
            metrics["residual_alpha"] = alpha
        elif self.config.algorithm == "ppo":
            extra = None
            if len(self.replay) > 0 and self.config.replay_mix_ratio > 0:
                k = max(
                    1,
                    int(round(
                        self.config.replay_mix_ratio
                        * len(graphs)
                        * self.config.rollouts_per_graph
                    )),
                )
                extra = self.replay.sample(k)

            metrics = ppo_step(
                policy=self.policy,
                graphs=graphs,
                backend_fn=self.backend_fn,
                optimizer=self.optimizer,
                baseline_model=self.baseline_model
                if self.config.use_paired_baseline else None,
                raw_backend_fn=self.raw_backend_fn,
                rollouts_per_graph=self.config.rollouts_per_graph,
                ppo_epochs=self.config.ppo_epochs,
                ppo_minibatch_size=self.config.ppo_minibatch_size,
                ppo_clip=self.config.ppo_clip,
                entropy_coef=self.config.entropy_coef,
                value_loss_coef=self.config.value_loss_coef,
                grad_clip=self.config.grad_clip,
                use_paired_baseline=self.config.use_paired_baseline,
                advantage_normalization=self.config.advantage_normalization,
                extra_rollouts=extra,
                return_rollouts=self.config.replay_buffer_size > 0,
            )

            if self.config.replay_buffer_size > 0 and "_fresh_rollouts" in metrics:
                self.replay.add_rollouts(metrics.pop("_fresh_rollouts"))
        else:
            raise ValueError(
                f"Unknown algorithm {self.config.algorithm!r}; "
                "expected 'reinforce' or 'ppo'."
            )

        self.step_count += 1
        metrics["step"] = self.step_count
        return metrics

    # ── evaluation ──────────────────────────────────────────────────────

    def evaluate(self, graphs: list[nx.Graph]) -> dict[str, Any]:
        """Run deterministic inference on held-out graphs, compare to baseline."""
        self.policy.eval()
        learned_rewards: list[float] = []
        baseline_rewards: list[float] = []

        for G in graphs:
            sched_learned = self.policy.make_schedule(G)
            r_learned = self.raw_backend_fn(G, sched_learned)
            learned_rewards.append(r_learned)

            key = G.graph.get("seed", id(G))
            if key in self.baseline_reward_cache:
                r_baseline = self.baseline_reward_cache[key]
            else:
                sched_baseline = self.baseline_model.make_schedule(G)
                r_baseline = self.raw_backend_fn(G, sched_baseline)
                self.baseline_reward_cache[key] = r_baseline
            baseline_rewards.append(r_baseline)

        self.policy.train()

        learned_arr = np.array(learned_rewards)
        baseline_arr = np.array(baseline_rewards)
        return {
            "eval_mean_reward": float(learned_arr.mean()),
            "eval_std_reward": float(learned_arr.std()),
            "eval_max_reward": float(learned_arr.max()),
            "baseline_mean_reward": float(baseline_arr.mean()),
            "improvement": float((learned_arr - baseline_arr).mean()),
            "n_better": int((learned_arr > baseline_arr).sum()),
            "n_graphs": len(graphs),
        }

    def select_batch(self, pool: list[nx.Graph]) -> list[nx.Graph]:
        """Uniform random sampling from pool."""
        k = min(self.config.batch_size, len(pool))
        return random.sample(pool, k)

    def refresh_pool(self) -> None:
        """Regenerate the training graph pool with new random seeds."""
        base_seed = (self.config.seed or 0) + self.step_count * 1000
        raw = _build_graph_pool(
            self.project.udg, self.config.graph_pool_size, base_seed
        )
        # Re-fill baseline cache for the new graphs.
        for G in raw:
            key = G.graph.get("seed", id(G))
            if key in self.baseline_reward_cache:
                continue
            sched_b = self.baseline_model.make_schedule(G)
            self.baseline_reward_cache[key] = float(
                self.raw_backend_fn(G, sched_b)
            )
        if self.config.pool_curation:
            self.train_pool = _curate_pool(
                raw, self.baseline_reward_cache,
                self.config.curation_lo, self.config.curation_hi,
            )
            if len(self.train_pool) < max(self.config.batch_size, 1):
                self.train_pool = raw
        else:
            self.train_pool = raw

    # ── checkpointing ───────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "baseline_ema": dict(self.baseline_ema),
                "baseline_reward_cache": dict(self.baseline_reward_cache),
                "step_count": self.step_count,
                "best_eval_reward": self.best_eval_reward,
            },
            ckpt_path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.baseline_ema = ckpt.get("baseline_ema", {})
        self.baseline_reward_cache = ckpt.get("baseline_reward_cache", {})
        self.step_count = ckpt.get("step_count", 0)
        self.best_eval_reward = ckpt.get("best_eval_reward", -float("inf"))
