from __future__ import annotations

import abc
import networkx as nx

from config import ProjectConfig
from schedules import GlobalSchedule


class ScheduleModel(abc.ABC):
    """Abstract base class for Graph→Schedule models.

    Implementations must map a NetworkX graph to a `GlobalSchedule` using the
    shared `ProjectConfig`.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def __call__(self, g: nx.Graph) -> GlobalSchedule:
        return self.make_schedule(g)

    @abc.abstractmethod
    def make_schedule(self, g: nx.Graph) -> GlobalSchedule:
        """Produce a schedule for the given graph."""
        raise NotImplementedError
