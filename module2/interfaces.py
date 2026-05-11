from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from schedules import GlobalSchedule


@dataclass
class DeviceMetadata:
    """Device/backend configuration for quantum runs (interface only)."""

    backend: Literal["bloqade", "aquila"]
    n_sites: int
    shot_budget: int
    lattice_spacing_um: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResult:
    """Result from a backend p_MIS estimation run (interface only)."""

    p_mis: float
    shots: int
    std_err: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QuantumBackend(abc.ABC):
    """Abstract interface for quantum backends (Module 2 skeleton)."""

    def __init__(self, device: DeviceMetadata) -> None:
        self.device = device

    @abc.abstractmethod
    def estimate_p_mis(
        self,
        schedule: GlobalSchedule,
        graph: nx.Graph,
        seed: int | None = None,
    ) -> BackendResult:
        raise NotImplementedError
