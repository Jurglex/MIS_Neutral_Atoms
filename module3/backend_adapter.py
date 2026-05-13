"""Adapter that bridges a QuantumBackend into the reward-function signature
expected by the REINFORCE training step.

The key concern is that ``QuantumBackend.estimate_p_mis`` requires atom
*positions* in addition to the schedule and graph, while the training loop
only has (graph, schedule) pairs.  This module resolves that by attaching
positions as a graph attribute during graph-pool construction.
"""
from __future__ import annotations

from typing import Callable

import networkx as nx

from schedules import GlobalSchedule
from module2.interfaces import Positions, QuantumBackend


def get_positions(G: nx.Graph) -> Positions:
    """Retrieve positions stored as ``G.graph["positions"]``.

    Graphs produced by ``generate_square_lattice_udg`` carry positions;
    the learner stores them via this convention.
    """
    pos = G.graph.get("positions")
    if pos is None:
        raise ValueError(
            "Graph has no 'positions' attribute.  Use graphs from "
            "generate_square_lattice_udg or attach positions manually."
        )
    return pos


def make_reward_fn(
    backend: QuantumBackend,
    *,
    seed: int | None = None,
) -> Callable[[nx.Graph, GlobalSchedule], float]:
    """Wrap a QuantumBackend into the ``(Graph, Schedule) -> reward`` signature.

    Positions are read from ``G.graph["positions"]`` (set during pool
    construction in the learner).
    """

    def reward_fn(G: nx.Graph, schedule: GlobalSchedule) -> float:
        positions = get_positions(G)
        result = backend.estimate_p_mis(schedule, G, positions, seed=seed)
        return result.p_mis

    return reward_fn
