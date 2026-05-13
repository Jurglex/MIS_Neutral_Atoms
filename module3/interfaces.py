from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol

import networkx as nx

from schedules import GlobalSchedule
from config import ProjectConfig


@dataclass
class TrainingConfig:
    """Configuration for the REINFORCE / actor-critic training loop."""

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
