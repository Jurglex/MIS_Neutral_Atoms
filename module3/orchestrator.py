"""Training orchestrator — main loop with evaluation, logging, and checkpointing."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from module3.interfaces import Orchestrator, TrainingConfig
from module3.learner import ReinforceLearner

logger = logging.getLogger(__name__)


class TrainingOrchestrator(Orchestrator):
    """Runs the full training loop: train → log → eval → checkpoint.

    Parameters
    ----------
    learner : ReinforceLearner
        Concrete learner that owns the policy and optimizer.
    """

    def __init__(self, learner: ReinforceLearner) -> None:
        super().__init__(learner)
        self._learner: ReinforceLearner = learner
        self.history: list[dict[str, Any]] = []

    def run(self) -> None:
        cfg = self._learner.config
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting training: %d steps, batch_size=%d, lr=%.1e",
            cfg.total_steps, cfg.batch_size, cfg.learning_rate,
        )

        for step in range(1, cfg.total_steps + 1):
            t0 = time.time()

            batch = self._learner.select_batch(self._learner.train_pool)
            metrics = self._learner.train_step(batch)
            metrics["wall_time"] = time.time() - t0
            self.history.append(metrics)

            if step % 10 == 0:
                logger.info(
                    "step %4d | reward %.4f | loss %.4f | entropy %.3f | "
                    "grad_norm %.3f | %.2fs",
                    step,
                    metrics["mean_reward"],
                    metrics["loss"],
                    metrics["mean_entropy"],
                    metrics["grad_norm"],
                    metrics["wall_time"],
                )

            if step % cfg.eval_every == 0:
                eval_metrics = self._learner.evaluate(self._learner.eval_pool)
                eval_metrics["step"] = step
                logger.info(
                    "  EVAL step %4d | learned %.4f | baseline %.4f | "
                    "improvement %+.4f | better %d/%d",
                    step,
                    eval_metrics["eval_mean_reward"],
                    eval_metrics["baseline_mean_reward"],
                    eval_metrics["improvement"],
                    eval_metrics["n_better"],
                    eval_metrics["n_graphs"],
                )
                self.history.append(eval_metrics)

                if eval_metrics["eval_mean_reward"] > self._learner.best_eval_reward:
                    self._learner.best_eval_reward = eval_metrics["eval_mean_reward"]
                    self._learner.save_checkpoint(
                        str(ckpt_dir / "best_model.pt")
                    )
                    logger.info("  New best model saved (reward=%.4f)",
                                self._learner.best_eval_reward)

            if cfg.graph_pool_refresh > 0 and step % cfg.graph_pool_refresh == 0:
                self._learner.refresh_pool()
                logger.info("  Refreshed training graph pool at step %d", step)

        self._learner.save_checkpoint(str(ckpt_dir / "final_model.pt"))

        history_path = log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(
            "Training complete. %d steps, final model saved. "
            "History at %s", cfg.total_steps, history_path,
        )
