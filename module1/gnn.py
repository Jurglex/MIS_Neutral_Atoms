from __future__ import annotations

import networkx as nx
import numpy as np

from config import ProjectConfig, derive_omega_schedule
from schedules import GlobalSchedule
from module1.base import ScheduleModel


class GNNModel(ScheduleModel):
    """Skeleton: Graph→Schedule model that uses a GNN front-end.

    Notes
    -----
    - This is a placeholder; no GNN library is wired yet.
    - Intended path: encode node/edge features with a GNN, pool to a graph
      embedding, then map to Δ(t).
    - Ω(t) is derived from the unit-disk radius using `derive_omega_schedule`.
    """

    def make_schedule(self, g: nx.Graph) -> GlobalSchedule:
        N_t = self.config.controls.N_t
        delta = np.zeros((N_t,), dtype=np.float64)
        omega = derive_omega_schedule(self.config.controls, self.config.udg)

        sched = GlobalSchedule(
            omega=omega,
            delta=delta,
            dt=self.config.controls.dt,
            param_kind=self.config.controls.param_kind,
        )
        sched.validate_shapes()
        return sched
