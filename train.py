#!/usr/bin/env python3
"""Train the Graph→Schedule policy with REINFORCE.

Usage
-----
    python train.py                           # defaults from config.json
    python train.py --steps 500 --shots 20    # quick test run
    python train.py --config my_config.json   # custom config

The script loads ``config.json`` and ``hardware_specs.json`` for physics
settings, then runs the REINFORCE training loop using the Braket local
simulator as the reward backend.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import load_project_config_json
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
    p.add_argument("--backend", choices=["simulator", "mock"],
                   default="simulator",
                   help="Reward backend: 'simulator' (Braket AHS) or "
                        "'mock' (random rewards for testing)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _make_backend_fn(args, project_config):
    """Construct the reward function based on --backend flag."""
    if args.backend == "mock":
        import random as _rng

        def mock_fn(G, schedule):
            return _rng.random() * 0.3
        return mock_fn

    from module2 import BraketBackend
    from module3.backend_adapter import make_reward_fn

    n_shots = args.shots or 50
    backend = BraketBackend(
        project_config, n_shots=n_shots,
        backend_type="simulator", validate=False,
    )
    return make_reward_fn(backend)


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

    reward_fn = _make_backend_fn(args, project)

    learner = ReinforceLearner(
        config=train_cfg,
        project=project,
        backend_fn=reward_fn,
        device=args.device,
    )

    orchestrator = TrainingOrchestrator(learner)
    orchestrator.run()


if __name__ == "__main__":
    main()
