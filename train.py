#!/usr/bin/env python3
"""Train the Graph→Schedule policy with PPO (or legacy REINFORCE).

Usage
-----
    python train.py                              # defaults from config.json
    python train.py --steps 500 --shots 20       # quick test run
    python train.py --config my_config.json      # custom config
    python train.py --algorithm reinforce        # legacy REINFORCE

The script loads ``config.json`` and ``hardware_specs.json`` for physics
settings, then runs the PPO training loop using the Braket local simulator
as the reward backend.  PPO is the default and recommended choice.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import load_project_config_json, ProjectConfig, RewardConfig
from module3.interfaces import TrainingConfig
from module3.learner import ReinforceLearner
from module3.orchestrator import TrainingOrchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Graph→Schedule policy")
    p.add_argument("--config", default="config.json",
                   help="Path to project config JSON")
    p.add_argument("--hardware", default="hardware_specs.json",
                   help="Path to hardware specs JSON")

    p.add_argument("--steps", type=int, default=None,
                   help="Override total training steps")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--shots", type=int, default=None,
                   help="Override n_shots per p_MIS evaluation")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--seed", type=int, default=None,
                   help="Override random seed")
    p.add_argument("--pool-size", type=int, default=None,
                   help="Override training graph pool size")
    p.add_argument("--eval-every", type=int, default=None,
                   help="Override evaluation frequency")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Override checkpoint directory")
    p.add_argument("--device", default="cpu",
                   help="Torch device (cpu or cuda)")

    p.add_argument("--algorithm", choices=["ppo", "reinforce"], default=None,
                   help="Policy-gradient algorithm (default from config or 'ppo')")
    p.add_argument("--rollouts-per-graph", type=int, default=None,
                   help="K rollouts per graph per PPO step")
    p.add_argument("--no-paired-baseline", action="store_true",
                   help="Disable paired-baseline advantages")
    p.add_argument("--no-bc-pretrain", action="store_true",
                   help="Disable behavioral-cloning pretraining")

    p.add_argument("--backend", choices=["simulator", "mock"],
                   default="simulator",
                   help="Reward backend: 'simulator' (Braket AHS) or "
                        "'mock' (random rewards for testing)")
    p.add_argument("--reward",
                   choices=["is_cost", "is_cost_vs_baseline",
                            "p_mis", "composite"],
                   default=None,
                   help="Override reward function kind")
    p.add_argument("--penalty-U", type=float, default=None,
                   help="Override edge-violation penalty U for is_cost reward")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _make_reward_fns(args, project_config):
    """Build (reward_fn, raw_reward_fn) used by the learner.

    * ``reward_fn`` returns rewards in the form the optimizer expects
      (may be normalized vs. baseline).
    * ``raw_reward_fn`` always returns un-normalized ``is_cost``-style
      values; used for paired-baseline and evaluation.
    """
    if args.backend == "mock":
        import random as _rng

        def mock_fn(G, schedule):
            return _rng.random() * 0.3
        return mock_fn, mock_fn

    from module2 import BraketBackend
    from module3.backend_adapter import (
        BaselineRewardCache, make_reward_fn, make_raw_reward_fn,
    )
    from module1.base import FixedScheduleBaseline

    n_shots = args.shots or 50
    backend = BraketBackend(
        project_config, n_shots=n_shots,
        backend_type="simulator", validate=False,
    )

    raw_fn = make_raw_reward_fn(backend, reward_cfg=project_config.reward)

    if project_config.reward.kind == "is_cost_vs_baseline":
        baseline_model = FixedScheduleBaseline(project_config)

        def _baseline_eval(G):
            return raw_fn(G, baseline_model.make_schedule(G))

        cache = BaselineRewardCache(_baseline_eval)
        reward_fn = make_reward_fn(
            backend, reward_cfg=project_config.reward, baseline_cache=cache,
        )
    else:
        reward_fn = make_reward_fn(backend, reward_cfg=project_config.reward)

    return reward_fn, raw_fn


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    project = load_project_config_json(
        Path(args.config), Path(args.hardware)
    )

    if args.reward is not None or args.penalty_U is not None:
        rc = project.reward
        project = ProjectConfig(
            backend=project.backend,
            controls=project.controls,
            udg=project.udg,
            hardware=project.hardware,
            reward=RewardConfig(
                kind=args.reward or rc.kind,
                penalty_U=args.penalty_U if args.penalty_U is not None else rc.penalty_U,
                mis_bonus=rc.mis_bonus,
                normalize_by_nodes=rc.normalize_by_nodes,
                baseline_norm_eps=rc.baseline_norm_eps,
            ),
        )

    train_cfg = TrainingConfig()
    if args.steps is not None:
        train_cfg.total_steps = args.steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.shots is not None:
        train_cfg.n_shots = args.shots
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.pool_size is not None:
        train_cfg.graph_pool_size = args.pool_size
    if args.eval_every is not None:
        train_cfg.eval_every = args.eval_every
    if args.checkpoint_dir is not None:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.algorithm is not None:
        train_cfg.algorithm = args.algorithm
    if args.rollouts_per_graph is not None:
        train_cfg.rollouts_per_graph = args.rollouts_per_graph
    if args.no_paired_baseline:
        train_cfg.use_paired_baseline = False
    if args.no_bc_pretrain:
        train_cfg.bc_pretrain_steps = 0
        train_cfg.bc_critic_steps = 0

    reward_fn, raw_reward_fn = _make_reward_fns(args, project)

    learner = ReinforceLearner(
        config=train_cfg,
        project=project,
        backend_fn=reward_fn,
        raw_backend_fn=raw_reward_fn,
        device=args.device,
    )

    orchestrator = TrainingOrchestrator(learner)
    orchestrator.run()


if __name__ == "__main__":
    main()
