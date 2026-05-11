from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx

from schedules import GlobalSchedule
from config import ProjectConfig


@dataclass
class TrainingConfig:
    """Top-level configuration for learning/orchestration (skeleton)."""

    total_steps: int = 1000
    eval_every: int = 100
    learning_rate: float = 1e-3
    shot_budget_total: int = 100_000
    seed: int | None = None


class ScheduleNetwork(Protocol):
    """Protocol for a Graph→Schedule model."""

    def __call__(self, g: nx.Graph) -> GlobalSchedule: ...


class Learner(abc.ABC):
    """Abstract learner interface for Module 3 (skeleton)."""

    def __init__(self, config: TrainingConfig, project: ProjectConfig) -> None:
        self.config = config
        self.project = project

    @abc.abstractmethod
    def train_step(self, model: ScheduleNetwork, graphs: list[nx.Graph]) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def select_next_graph(self, pool: list[nx.Graph]) -> nx.Graph:
        raise NotImplementedError


class Orchestrator(abc.ABC):
    """Abstract orchestration interface to coordinate training/evaluation (skeleton)."""

    def __init__(self, learner: Learner) -> None:
        self.learner = learner

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError
