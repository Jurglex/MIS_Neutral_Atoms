"""Probe diagnostics: did the policy actually learn something graph-conditional?

The central scientific claim of this project is that the learned policy
produces *meaningfully graph-dependent* schedules — i.e. it has internalized
some piece of physics rather than overfitting to a fixed shape.

These probes quantify that claim by:

1. Measuring per-graph schedule deviation from the analytic baseline.
2. Correlating that deviation with graph features (size, density,
   algebraic connectivity λ₂, mean degree) that the underlying physics
   *should* care about.
3. Reporting a graph-conditioning index (variance across graphs / variance
   within rollouts) — high values mean the policy varies more between
   graphs than between rollouts on the same graph.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import networkx as nx

from module1.policy import SchedulePolicy
from module1.base import FixedScheduleBaseline


def _schedule_deviation(omega_a, delta_a, omega_b, delta_b) -> dict[str, float]:
    """L2 distance between two schedules, broken down by component."""
    dom = float(np.sqrt(((omega_a - omega_b) ** 2).mean()))
    ddl = float(np.sqrt(((delta_a - delta_b) ** 2).mean()))
    return {"omega_l2": dom, "delta_l2": ddl, "total_l2": math.hypot(dom, ddl)}


def _graph_features(G: nx.Graph) -> dict[str, float]:
    """Graph statistics that the policy should plausibly condition on."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    mean_deg = (2 * m / n) if n > 0 else 0.0
    density = nx.density(G) if n > 1 else 0.0
    if n >= 2:
        try:
            lam2 = float(nx.algebraic_connectivity(G))
        except Exception:
            lam2 = 0.0
    else:
        lam2 = 0.0
    clustering = float(nx.average_clustering(G)) if n > 0 else 0.0
    return {
        "n_nodes": float(n), "n_edges": float(m),
        "density": float(density),
        "mean_degree": float(mean_deg),
        "lambda_2": lam2,
        "clustering": clustering,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns 0 when undefined (zero variance)."""
    if len(x) < 2:
        return 0.0
    sx = x.std()
    sy = y.std()
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def schedule_deviation_probe(
    policy: SchedulePolicy,
    graphs: list[nx.Graph],
    baseline_model: FixedScheduleBaseline | None = None,
) -> dict[str, Any]:
    """Compute per-graph schedule deviations and feature correlations.

    Returns
    -------
    dict with keys::

        per_graph : list of dicts (one per graph) with deviation + features.
        feature_correlations : Pearson r between total_l2 deviation and
                                each graph feature.
        mean_deviation : mean total_l2 across graphs.
        std_deviation : std of total_l2 across graphs.
    """
    if baseline_model is None:
        baseline_model = FixedScheduleBaseline(policy.config)

    rows: list[dict[str, float]] = []
    for G in graphs:
        sched_learned = policy.make_schedule(G)
        sched_baseline = baseline_model.make_schedule(G)
        dev = _schedule_deviation(
            sched_learned.omega, sched_learned.delta,
            sched_baseline.omega, sched_baseline.delta,
        )
        feats = _graph_features(G)
        rows.append({**dev, **feats})

    if not rows:
        return {
            "per_graph": [],
            "feature_correlations": {},
            "mean_deviation": 0.0, "std_deviation": 0.0,
        }

    total = np.array([r["total_l2"] for r in rows])
    feature_keys = [
        "n_nodes", "n_edges", "density",
        "mean_degree", "lambda_2", "clustering",
    ]
    corrs = {
        k: _pearson(total, np.array([r[k] for r in rows]))
        for k in feature_keys
    }

    return {
        "per_graph": rows,
        "feature_correlations": corrs,
        "mean_deviation": float(total.mean()),
        "std_deviation": float(total.std()),
        "max_deviation": float(total.max()),
        "min_deviation": float(total.min()),
    }


def graph_conditioning_index(
    policy: SchedulePolicy,
    graphs: list[nx.Graph],
    n_rollouts: int = 8,
) -> dict[str, float]:
    """How much does the policy mean vary between graphs vs. within rollouts?

    Computes::

        between_graph_var = Var_G[ E_k[ sched_k(G) ] ]
        within_graph_var  = E_G[ Var_k[ sched_k(G) ] ]
        conditioning_index = between_graph_var / (within_graph_var + ε)

    A high conditioning index means the policy produces materially
    different schedules across graphs (graph-conditional behavior),
    whereas a low value means it produces near-identical schedules
    regardless of graph (i.e. it collapsed to the baseline / a single
    mode).
    """
    means_per_graph: list[np.ndarray] = []
    within_vars: list[float] = []

    for G in graphs:
        rollouts = []
        for _ in range(n_rollouts):
            sched = policy.make_schedule(G)
            rollouts.append(np.concatenate([sched.omega, sched.delta]))
        arr = np.stack(rollouts)
        means_per_graph.append(arr.mean(axis=0))
        within_vars.append(float(arr.var(axis=0).mean()))

    if not means_per_graph:
        return {
            "between_graph_var": 0.0,
            "within_graph_var": 0.0,
            "conditioning_index": 0.0,
        }

    M = np.stack(means_per_graph)
    between_var = float(M.var(axis=0).mean())
    within_var = float(np.mean(within_vars))
    cond_idx = between_var / (within_var + 1e-9)
    return {
        "between_graph_var": between_var,
        "within_graph_var": within_var,
        "conditioning_index": cond_idx,
    }
