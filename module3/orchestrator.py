"""Training orchestrator — main loop with BC pretraining, evaluation,
diagnostics, logging, and checkpointing.

The orchestrator now:

* Runs **behavioral-cloning pretraining** on the policy mean (matches
  baseline schedule) and on the critic (predicts baseline reward) before
  any RL.  Configurable via :attr:`TrainingConfig.bc_pretrain_steps` and
  :attr:`TrainingConfig.bc_critic_steps`.
* Updates the policy's **residual-α** every training step via the
  learner's curriculum schedule.
* Periodically runs **probe diagnostics** (schedule deviation vs. graph
  features, graph-conditioning index) to verify the learned policy is
  meaningfully graph-conditional.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from module3.interfaces import Orchestrator, TrainingConfig
from module3.learner import ReinforceLearner
from module3.pretrain import behavioral_clone_policy, behavioral_clone_critic
from module3.diagnostics import (
    schedule_deviation_probe,
    graph_conditioning_index,
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator(Orchestrator):
    """Runs the full pipeline: BC pretrain → RL → eval → checkpoint."""

    def __init__(self, learner: ReinforceLearner) -> None:
        super().__init__(learner)
        self._learner: ReinforceLearner = learner
        self.history: list[dict[str, Any]] = []

    # ── pretraining phase ───────────────────────────────────────────────

    def _run_bc_pretrain(self) -> None:
        cfg = self._learner.config
        if cfg.bc_pretrain_steps > 0:
            logger.info("BC pretraining policy mean → baseline (%d steps)",
                        cfg.bc_pretrain_steps)
            losses = behavioral_clone_policy(
                policy=self._learner.policy,
                graphs=self._learner.train_pool,
                baseline_model=self._learner.baseline_model,
                n_steps=cfg.bc_pretrain_steps,
                lr=cfg.bc_pretrain_lr,
                log_fn=logger.info,
            )
            self.history.append({
                "phase": "bc_policy",
                "n_steps": cfg.bc_pretrain_steps,
                "final_loss": losses[-1] if losses else None,
            })

        if cfg.bc_critic_steps > 0:
            if not self._learner.baseline_reward_cache:
                logger.info("Skipping critic BC: baseline reward cache empty")
            else:
                logger.info("BC pretraining critic → baseline reward (%d steps)",
                            cfg.bc_critic_steps)
                losses = behavioral_clone_critic(
                    policy=self._learner.policy,
                    graphs=self._learner.train_pool,
                    baseline_rewards=self._learner.baseline_reward_cache,
                    n_steps=cfg.bc_critic_steps,
                    lr=cfg.bc_pretrain_lr,
                    log_fn=logger.info,
                )
                self.history.append({
                    "phase": "bc_critic",
                    "n_steps": cfg.bc_critic_steps,
                    "final_loss": losses[-1] if losses else None,
                })

    # ── diagnostics ─────────────────────────────────────────────────────

    def _run_diagnostics(self, step: int) -> None:
        try:
            probe = schedule_deviation_probe(
                self._learner.policy,
                self._learner.eval_pool,
                baseline_model=self._learner.baseline_model,
            )
            ci = graph_conditioning_index(
                self._learner.policy, self._learner.eval_pool, n_rollouts=4,
            )
        except Exception as e:
            logger.warning("Diagnostics failed at step %d: %s", step, e)
            return

        logger.info(
            "  DIAG step %4d | mean dev %.3f ± %.3f | "
            "cond_index %.3f | corr(λ₂)=%+.2f corr(density)=%+.2f",
            step,
            probe["mean_deviation"], probe["std_deviation"],
            ci["conditioning_index"],
            probe["feature_correlations"].get("lambda_2", 0.0),
            probe["feature_correlations"].get("density", 0.0),
        )
        self.history.append({
            "phase": "diagnostics",
            "step": step,
            "mean_deviation": probe["mean_deviation"],
            "std_deviation": probe["std_deviation"],
            "feature_correlations": probe["feature_correlations"],
            "conditioning_index": ci["conditioning_index"],
        })

    # ── main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self._learner.config
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting training: algorithm=%s, %d steps, batch_size=%d, lr=%.1e",
            cfg.algorithm, cfg.total_steps, cfg.batch_size, cfg.learning_rate,
        )

        self._run_bc_pretrain()

        for step in range(1, cfg.total_steps + 1):
            t0 = time.time()

            batch = self._learner.select_batch(self._learner.train_pool)
            metrics = self._learner.train_step(batch)
            metrics["wall_time"] = time.time() - t0
            self.history.append(metrics)

            if step % 10 == 0:
                logger.info(
                    "step %4d | reward %.4f | loss %.4f | entropy %.3f | "
                    "alpha %.3f | grad %.3f | %.2fs",
                    step,
                    metrics["mean_reward"],
                    metrics["loss"],
                    metrics["mean_entropy"],
                    metrics.get("residual_alpha", 0.0),
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

            if cfg.diagnostics_every > 0 and step % cfg.diagnostics_every == 0:
                self._run_diagnostics(step)

            if cfg.graph_pool_refresh > 0 and step % cfg.graph_pool_refresh == 0:
                self._learner.refresh_pool()
                logger.info("  Refreshed training graph pool at step %d", step)

        self._learner.save_checkpoint(str(ckpt_dir / "final_model.pt"))

        history_path = log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(
            "Training complete. %d steps, final model saved. History at %s",
            cfg.total_steps, history_path,
        )
