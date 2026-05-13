from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

from schedules import GlobalSchedule


@dataclass
class DeviceMetadata:
    """Device/backend configuration for quantum runs."""

    backend: Literal["bloqade", "aquila"]
    n_sites: int
    shot_budget: int
    lattice_spacing_um: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResult:
    """Result from a backend p_MIS estimation run.

    Attributes
    ----------
    p_mis : float
        Estimated probability of measuring a maximum independent set.
    shots : int
        Number of measurement shots used.
    std_err : float | None
        Binomial standard error sqrt(p(1-p)/n), if available.
    counts : dict[str, int] | None
        Raw bitstring counts from the device (``'r'``/``'g'`` alphabet).
    metadata : dict[str, Any]
        Arbitrary extra information from the backend.
    """

    p_mis: float
    shots: int
    std_err: float | None = None
    counts: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def binomial_std_err(p: float, n: int) -> float:
        return math.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))


Positions = dict[int, tuple[float, float]]
"""Mapping from node ID → (x, y) coordinates in μm."""


class QuantumBackend(abc.ABC):
    """Abstract interface for quantum backends (Module 2).

    Implementations convert a :class:`GlobalSchedule` (Ω/Δ in rad/μs,
    ``dt`` in seconds) and atom ``positions`` (μm) into a device-specific
    program, execute it, and return a :class:`BackendResult`.
    """

    def __init__(self, device: DeviceMetadata) -> None:
        self.device = device

    @abc.abstractmethod
    def estimate_p_mis(
        self,
        schedule: GlobalSchedule,
        graph: nx.Graph,
        positions: Positions,
        *,
        seed: int | None = None,
    ) -> BackendResult:
        """Run the schedule on the backend and estimate p_MIS.

        Parameters
        ----------
        schedule : GlobalSchedule
            Time-discretized Ω(t) and Δ(t) arrays (rad/μs) with metadata.
        graph : nx.Graph
            The interaction graph whose MIS we are targeting.
        positions : Positions
            ``{node_id: (x_μm, y_μm)}`` — physical atom coordinates in μm.
        seed : int | None
            Optional RNG seed for reproducibility (simulator only).
        """
        raise NotImplementedError
