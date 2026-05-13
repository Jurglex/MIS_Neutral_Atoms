from __future__ import annotations

import abc

import numpy as np
import networkx as nx

from config import ProjectConfig, compute_blockade_omega
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


class FixedScheduleBaseline(ScheduleModel):
    """Graph-agnostic baseline: trapezoidal Omega + linear Delta sweep.

    Produces the same schedule for every graph — the standard Ebadi-style
    adiabatic protocol.  Serves as the comparison target: any learned policy
    must outperform this to demonstrate graph-conditioned value.
    """

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__(config)
        ctrl = config.controls
        hw = config.hardware
        N_t = ctrl.N_t

        R_b_um = config.udg.radius * config.udg.spacing
        omega_peak = compute_blockade_omega(hw.C6, R_b_um, hw.omega_max)

        T_us = ctrl.T * 1e6
        t = np.linspace(0.0, 1.0, N_t)

        r = max(hw.t_ramp / T_us, 1e-9)
        o = hw.t_onset / T_us
        rise = np.clip((t - o) / r, 0.0, 1.0)
        fall = np.clip((1.0 - o - t) / r, 0.0, 1.0)
        self._omega = (omega_peak * np.minimum(rise, fall)).astype(np.float64)

        self._delta = np.linspace(
            ctrl.delta_min, ctrl.delta_max, N_t, dtype=np.float64
        )
        self._dt = ctrl.dt
        self._param_kind = ctrl.param_kind

    def make_schedule(self, g: nx.Graph) -> GlobalSchedule:
        return GlobalSchedule(
            omega=self._omega.copy(),
            delta=self._delta.copy(),
            dt=self._dt,
            param_kind=self._param_kind,
        )
