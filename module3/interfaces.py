from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol

import networkx as nx

from schedules import GlobalSchedule
from config import ProjectConfig


@dataclass
class TrainingConfig:
    """Configuration for the policy-gradient training loop.

    Supports either REINFORCE (``algorithm='reinforce'``) or PPO with paired
    baselines (``algorithm='ppo'``, default).  PPO additionally uses K
    rollouts per graph, advantage normalization, multiple inner epochs, and
    an optional replay buffer for off-policy reuse.
    """

    # ── core RL loop ───────────────────────────────────────────────
    total_steps: int = 2000
    batch_size: int = 8
    eval_every: int = 50
    eval_graphs: int = 16
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    ema_alpha: float = 0.9
    grad_clip: float = 1.0
    n_shots: int = 50
    graph_pool_size: int = 64
    graph_pool_refresh: int = 500
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int | None = 42

    # ── algorithm choice ─────────────────────────────────────────────
    algorithm: str = "ppo"
    """One of 'reinforce' (legacy) or 'ppo' (default, recommended)."""

    # ── PPO / paired-baseline knobs ──────────────────────────────────
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 16
    rollouts_per_graph: int = 4
    """Number of stochastic schedule rollouts evaluated per graph per step.
    Higher = lower-variance gradient at higher simulator cost."""
    use_paired_baseline: bool = True
    """If True, each graph's analytic baseline is also evaluated and used
    as the advantage baseline (replacing the EMA).  Strongly recommended."""
    advantage_normalization: bool = True
    """If True, normalize advantages to zero-mean, unit-variance within
    each gradient step before computing the PPO loss."""

    # ── replay buffer (off-policy reuse) ─────────────────────────────
    replay_buffer_size: int = 0
    """If >0, keep this many recent (graph, action, reward) tuples and mix
    them into PPO updates with importance-sampling correction.  0 disables."""
    replay_mix_ratio: float = 0.5
    """Fraction of each PPO minibatch drawn from the replay buffer
    (rest is on-policy).  Ignored when buffer is empty."""

    # ── behavioral cloning pretraining ───────────────────────────────
    bc_pretrain_steps: int = 200
    """Number of supervised steps that pretrain the policy mean toward
    the analytic baseline before any RL.  0 disables BC."""
    bc_pretrain_lr: float = 1e-3
    bc_critic_steps: int = 100
    """Number of simulator-based supervised steps that pretrain the value
    head against the baseline reward on each pool graph."""

    # ── pool curation ────────────────────────────────────────────────
    pool_curation: bool = True
    """If True, filter the training pool to graphs whose baseline reward
    falls in (curation_lo, curation_hi) — discards trivial / unsolvable
    instances that carry no learning signal."""
    curation_lo: float = 0.0
    """Lower bound on baseline reward for kept graphs."""
    curation_hi: float = 1.0
    """Upper bound (in normalized units).  For ``is_cost`` this is roughly
    the max-possible-IS-size / n_nodes, so 1.0 is permissive by default."""

    # ── exploration / diagnostics ────────────────────────────────────
    init_log_std: float = -0.5
    """Initial log-std of the Gaussian policy.  Higher = more exploration
    around the baseline."""
    diagnostics_every: int = 100
    """Steps between probe diagnostics (schedule-deviation vs. graph
    features).  Set to 0 to disable."""


class ScheduleNetwork(Protocol):
    """Protocol for a Graph→Schedule model."""

    def __call__(self, g: nx.Graph) -> GlobalSchedule: ...


class Learner(abc.ABC):
    """Abstract learner interface for Module 3."""

    def __init__(self, config: TrainingConfig, project: ProjectConfig) -> None:
        self.config = config
        self.project = project

    @abc.abstractmethod
    def train_step(
        self, graphs: list[nx.Graph]
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self, graphs: list[nx.Graph]
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def select_batch(self, pool: list[nx.Graph]) -> list[nx.Graph]:
        raise NotImplementedError

    @abc.abstractmethod
    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError


class Orchestrator(abc.ABC):
    """Abstract orchestration interface to coordinate training/evaluation."""

    def __init__(self, learner: Learner) -> None:
        self.learner = learner

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError
