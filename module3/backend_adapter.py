"""Adapter that bridges a QuantumBackend into the reward-function signature
expected by the REINFORCE / PPO training step.

The key concerns are:

1. ``QuantumBackend.estimate_p_mis`` requires atom *positions* in addition to
   the schedule and graph, while the training loop only has (graph, schedule)
   pairs.  This module attaches positions as ``G.graph['positions']`` during
   pool construction and retrieves them here.

2. Several reward formulations are supported and selected via
   ``RewardConfig.kind``:

   ``"is_cost"``
       Weighted independent-set cost from raw measurement bitstrings.
       Dense gradient signal; does not require knowing |MIS|.

   ``"is_cost_vs_baseline"`` (recommended)
       Per-graph normalized improvement of ``is_cost`` over the analytic
       adiabatic baseline:
           r_norm(G) = (r_learned(G) − r_baseline(G)) / (|r_baseline(G)| + ε)
       Removes per-graph reward heterogeneity so the gradient signal is
       "fractional improvement over baseline".  Requires the caller to
       provide a ``BaselineRewardCache``; ``make_reward_fn`` will lazily
       evaluate (and cache) the baseline reward per graph.

   ``"p_mis"``
       Binary MIS hit rate (legacy).  Very sparse.

   ``"composite"``
       (1-λ)·is_cost + λ·p_mis.
"""
from __future__ import annotations

from typing import Callable, Protocol

import networkx as nx

from config import RewardConfig
from schedules import GlobalSchedule
from module2.interfaces import Positions, QuantumBackend, BackendResult


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


# ── Reward computation from BackendResult ────────────────────────────────

def _is_cost_from_counts(
    counts: dict[str, int],
    graph: nx.Graph,
    penalty_U: float,
    normalize_by_nodes: bool,
) -> float:
    """Weighted IS cost: Σ s_i  −  U · Σ_{(i,j)∈E} s_i·s_j, averaged over shots.

    Each shot's bitstring contributes its cost to the average.
    ``'r'`` = Rydberg (selected / excited), ``'g'`` = ground (not selected).
    """
    edges = list(graph.edges())
    n_nodes = graph.number_of_nodes()
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    total_cost = 0.0
    for bitstring, count in counts.items():
        selected = [i for i, c in enumerate(bitstring) if c == 'r']
        n_selected = len(selected)

        violations = 0
        selected_set = set(selected)
        for u, v in edges:
            if u in selected_set and v in selected_set:
                violations += 1

        shot_cost = n_selected - penalty_U * violations
        total_cost += shot_cost * count

    avg_cost = total_cost / total_shots
    if normalize_by_nodes and n_nodes > 0:
        avg_cost /= n_nodes
    return avg_cost


def _raw_reward(
    result: BackendResult,
    graph: nx.Graph,
    reward_cfg: RewardConfig,
) -> float:
    """Compute the un-normalized scalar reward (before any vs-baseline scaling)."""
    kind = reward_cfg.kind

    if kind == "p_mis":
        return result.p_mis

    if kind in ("is_cost", "is_cost_vs_baseline"):
        if result.counts is None:
            return result.p_mis
        return _is_cost_from_counts(
            result.counts, graph, reward_cfg.penalty_U,
            reward_cfg.normalize_by_nodes,
        )

    if kind == "composite":
        if result.counts is None:
            return result.p_mis
        r_cost = _is_cost_from_counts(
            result.counts, graph, reward_cfg.penalty_U,
            reward_cfg.normalize_by_nodes,
        )
        lam = reward_cfg.mis_bonus
        return (1.0 - lam) * r_cost + lam * result.p_mis

    raise ValueError(
        f"Unknown reward kind: {kind!r}. "
        "Choose from 'is_cost', 'is_cost_vs_baseline', 'p_mis', 'composite'."
    )


# ── Baseline reward cache (for is_cost_vs_baseline) ───────────────────────

class BaselineEvaluator(Protocol):
    """Callable that returns the analytic-baseline reward for a graph.

    Used by ``BaselineRewardCache`` to lazily fill its table on first access.
    """

    def __call__(self, G: nx.Graph) -> float: ...


class BaselineRewardCache:
    """Per-graph cache of the analytic-baseline reward.

    Keyed by ``G.graph.get('seed', id(G))`` so repeated evaluations on the
    same graph reuse the simulator result.

    Parameters
    ----------
    evaluator : BaselineEvaluator
        Function that takes a graph and returns its baseline reward.
        Typically ``lambda G: raw_reward(backend.estimate_p_mis(
        baseline_model.make_schedule(G), G, positions))``.
    """

    def __init__(self, evaluator: BaselineEvaluator) -> None:
        self._cache: dict[object, float] = {}
        self._evaluator = evaluator

    @staticmethod
    def _key(G: nx.Graph) -> object:
        return G.graph.get("seed", id(G))

    def get(self, G: nx.Graph) -> float:
        k = self._key(G)
        if k not in self._cache:
            self._cache[k] = float(self._evaluator(G))
        return self._cache[k]

    def set(self, G: nx.Graph, value: float) -> None:
        self._cache[self._key(G)] = float(value)

    def clear(self) -> None:
        self._cache.clear()

    def __contains__(self, G: nx.Graph) -> bool:
        return self._key(G) in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ── Public factory ───────────────────────────────────────────────────────

def make_reward_fn(
    backend: QuantumBackend,
    reward_cfg: RewardConfig | None = None,
    *,
    baseline_cache: BaselineRewardCache | None = None,
    seed: int | None = None,
) -> Callable[[nx.Graph, GlobalSchedule], float]:
    """Wrap a QuantumBackend into the ``(Graph, Schedule) -> reward`` signature.

    Parameters
    ----------
    backend : QuantumBackend
        Quantum backend that runs schedules and returns measurement counts.
    reward_cfg : RewardConfig | None
        Reward function configuration.  Defaults to ``RewardConfig()``
        (``is_cost_vs_baseline`` with U=3, normalized by nodes).
    baseline_cache : BaselineRewardCache | None
        Required for ``is_cost_vs_baseline``.  Caches the analytic-baseline
        reward per graph so the simulator only runs it once per graph.
        If None and the kind is ``is_cost_vs_baseline``, falls back to
        plain ``is_cost``.
    seed : int | None
        Optional RNG seed forwarded to the backend.
    """
    if reward_cfg is None:
        reward_cfg = RewardConfig()

    def reward_fn(G: nx.Graph, schedule: GlobalSchedule) -> float:
        positions = get_positions(G)
        result = backend.estimate_p_mis(schedule, G, positions, seed=seed)
        r_raw = _raw_reward(result, G, reward_cfg)

        if reward_cfg.kind == "is_cost_vs_baseline" and baseline_cache is not None:
            r_base = baseline_cache.get(G)
            return (r_raw - r_base) / (abs(r_base) + reward_cfg.baseline_norm_eps)
        return r_raw

    return reward_fn


def make_raw_reward_fn(
    backend: QuantumBackend,
    reward_cfg: RewardConfig | None = None,
    *,
    seed: int | None = None,
) -> Callable[[nx.Graph, GlobalSchedule], float]:
    """Variant of ``make_reward_fn`` that always returns the *un-normalized*
    reward (``is_cost`` semantics) regardless of ``reward_cfg.kind``.

    Used by:
    * the baseline cache to fill itself (must not recurse on
      ``is_cost_vs_baseline``);
    * evaluation diagnostics where absolute rewards are reported.
    """
    if reward_cfg is None:
        reward_cfg = RewardConfig()

    def reward_fn(G: nx.Graph, schedule: GlobalSchedule) -> float:
        positions = get_positions(G)
        result = backend.estimate_p_mis(schedule, G, positions, seed=seed)
        return _raw_reward(result, G, reward_cfg)

    return reward_fn
