"""Concrete REINFORCE learner for the Graph→Schedule policy."""
from __future__ import annotations

import json
import random
from copy import deepcopy
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


def _build_graph_pool(
    udg_cfg: UDGConfig,
    pool_size: int,
    base_seed: int,
) -> list[nx.Graph]:
    """Generate a pool of UDGs with varying random seeds.

    Positions are attached as ``G.graph["positions"]`` so the backend
    adapter can find them.
    """
    pool: list[nx.Graph] = []
    for i in range(pool_size):
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


class ReinforceLearner(Learner):
    """REINFORCE learner with EMA baseline and optional value-function critic.

    Parameters
    ----------
    config : TrainingConfig
        Training hyperparameters.
    project : ProjectConfig
        Physics / hardware configuration.
    backend_fn : callable
        ``(nx.Graph, GlobalSchedule) -> float`` reward function.
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
        device: str = "cpu",
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(config, project)
        self.backend_fn = backend_fn
        self.device = torch.device(device)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            random.seed(config.seed)
            np.random.seed(config.seed)

        self.policy = SchedulePolicy(
            project, hidden_dim=hidden_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.baseline_ema: dict[int, float] = {}
        self.step_count = 0
        self.best_eval_reward = -float("inf")

        self.baseline_model = FixedScheduleBaseline(project)

        base_seed = config.seed if config.seed is not None else 0
        self.train_pool = _build_graph_pool(
            project.udg, config.graph_pool_size, base_seed
        )
        self.eval_pool = _build_graph_pool(
            project.udg, config.eval_graphs, base_seed + 10_000
        )

    def train_step(self, graphs: list[nx.Graph]) -> dict[str, Any]:
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
        self.step_count += 1
        metrics["step"] = self.step_count
        return metrics

    def evaluate(self, graphs: list[nx.Graph]) -> dict[str, Any]:
        """Run deterministic inference on held-out graphs, compare to baseline."""
        self.policy.eval()
        learned_rewards: list[float] = []
        baseline_rewards: list[float] = []

        for G in graphs:
            sched_learned = self.policy.make_schedule(G)
            r_learned = self.backend_fn(G, sched_learned)
            learned_rewards.append(r_learned)

            sched_baseline = self.baseline_model.make_schedule(G)
            r_baseline = self.backend_fn(G, sched_baseline)
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
        self.train_pool = _build_graph_pool(
            self.project.udg, self.config.graph_pool_size, base_seed
        )

    def save_checkpoint(self, path: str) -> None:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "baseline_ema": dict(self.baseline_ema),
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
        self.step_count = ckpt.get("step_count", 0)
        self.best_eval_reward = ckpt.get("best_eval_reward", -float("inf"))
