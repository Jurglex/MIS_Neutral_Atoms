from __future__ import annotations

import networkx as nx
import numpy as np

from config import ProjectConfig, derive_omega_schedule
from schedules import GlobalSchedule
from module1.base import ScheduleModel


class AdjacencyMLP(ScheduleModel):
    """Skeleton: Graph→Schedule model that consumes the adjacency matrix.

    Notes
    -----
    - This is a skeleton for comparing model families.
    - The intended implementation will flatten or otherwise embed the NxN
      adjacency matrix and map it to a Δ(t) schedule.
    - Ω(t) is derived from the unit-disk radius using `derive_omega_schedule`.
    """

    def make_schedule(self, g: nx.Graph) -> GlobalSchedule:
        _A = nx.to_numpy_array(g, dtype=np.float32, weight=None)

        omega = derive_omega_schedule(self.config.controls, self.config.udg)

        N_t = self.config.controls.N_t
        delta = np.zeros((N_t,), dtype=np.float64)

        sched = GlobalSchedule(
            omega=omega,
            delta=delta,
            dt=self.config.controls.dt,
            param_kind=self.config.controls.param_kind,
        )
        sched.validate_shapes()
        return sched
