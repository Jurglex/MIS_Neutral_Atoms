"""Behavioral cloning pretraining for the Graph→Schedule policy.

Two short supervised loops run *before* any RL:

1. **Policy BC.**  Force the policy's deterministic forward pass to match
   the analytic adiabatic baseline schedule on every pool graph.  No
   simulator calls — entirely a torch MSE loss against the closed-form
   baseline.

2. **Critic BC.**  Force the value head to predict each pool graph's
   measured baseline reward.  Requires one simulator call per pool graph
   (these results are then reused by the ``BaselineRewardCache`` during RL).

Why bother
----------
A randomly-initialized policy starts (after Arch 3 / warm-start) producing
*approximately* the baseline schedule, but its critic is uniformly zero
and the action distribution is centered on the network's untrained mean.
Pretraining gives RL a clean starting point: policy mean ≡ baseline,
value head ≈ true baseline reward, so the first PPO advantage already
reflects "does this perturbation help?" rather than "is the network even
near the baseline?".
"""
from __future__ import annotations

from typing import Callable

import networkx as nx
import torch
from torch_geometric.data import Batch

from module1.featurize import graph_to_pyg
from module1.policy import SchedulePolicy
from module1.base import FixedScheduleBaseline
from schedules import GlobalSchedule


# ── Policy BC ────────────────────────────────────────────────────────────

def behavioral_clone_policy(
    policy: SchedulePolicy,
    graphs: list[nx.Graph],
    baseline_model: FixedScheduleBaseline,
    *,
    n_steps: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
    log_every: int = 50,
    log_fn: Callable[[str], None] | None = None,
) -> list[float]:
    """Supervised pretraining: policy mean output ≈ baseline schedule.

    The schedule loss is computed in the *output* (Ω, Δ) space rather than
    on the latent action parameters, so the policy is free to discover any
    parameterization that reproduces the baseline shape.

    Returns
    -------
    list of training losses, one per step.
    """
    device = next(policy.parameters()).device
    optimizer = torch.optim.Adam(
        [p for n, p in policy.named_parameters() if "value_head" not in n],
        lr=lr,
    )
    losses: list[float] = []

    pool_data = [graph_to_pyg(G) for G in graphs]
    pool_baseline = []
    for G in graphs:
        sched: GlobalSchedule = baseline_model.make_schedule(G)
        pool_baseline.append((
            torch.from_numpy(sched.omega).float(),
            torch.from_numpy(sched.delta).float(),
        ))

    omega_max = policy.config.controls.omega_max
    delta_span = policy.config.controls.delta_max - policy.config.controls.delta_min

    policy.train()
    for step in range(n_steps):
        idx = torch.randperm(len(graphs))[:batch_size].tolist()
        batch_data = [pool_data[i] for i in idx]
        batch = Batch.from_data_list(batch_data).to(device)

        out = policy.forward(batch)
        target_omega = torch.stack(
            [pool_baseline[i][0] for i in idx]
        ).to(device)
        target_delta = torch.stack(
            [pool_baseline[i][1] for i in idx]
        ).to(device)

        loss_omega = ((out["omega"] - target_omega) / max(omega_max, 1e-6)).pow(2).mean()
        loss_delta = ((out["delta"] - target_delta) / max(delta_span, 1e-6)).pow(2).mean()
        loss = loss_omega + loss_delta

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if log_fn is not None and (step + 1) % log_every == 0:
            log_fn(
                f"  BC policy | step {step+1:4d}/{n_steps} | "
                f"loss {loss.item():.4f} (omega={loss_omega.item():.4f}, "
                f"delta={loss_delta.item():.4f})"
            )

    return losses


# ── Critic BC ────────────────────────────────────────────────────────────

def behavioral_clone_critic(
    policy: SchedulePolicy,
    graphs: list[nx.Graph],
    baseline_rewards: dict[object, float],
    *,
    n_steps: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    log_every: int = 50,
    log_fn: Callable[[str], None] | None = None,
) -> list[float]:
    """Supervised pretraining: value head ≈ measured baseline reward.

    Parameters
    ----------
    policy : SchedulePolicy
        Policy with a value head.
    graphs : list of networkx.Graph
    baseline_rewards : dict
        Mapping ``{G.graph['seed'] or id(G): r_baseline}`` of pre-computed
        baseline rewards (one simulator call per graph, performed once).
    """
    device = next(policy.parameters()).device
    optimizer = torch.optim.Adam(policy.value_head.parameters(), lr=lr)
    losses: list[float] = []

    pool_data = [graph_to_pyg(G) for G in graphs]
    pool_targets: list[float] = []
    for G in graphs:
        key = G.graph.get("seed", id(G))
        if key not in baseline_rewards:
            raise KeyError(
                f"baseline_rewards is missing graph key {key!r}.  "
                "Populate the cache by running the baseline on every pool "
                "graph before calling behavioral_clone_critic."
            )
        pool_targets.append(float(baseline_rewards[key]))

    targets_t = torch.tensor(pool_targets, dtype=torch.float32, device=device)

    policy.train()
    for step in range(n_steps):
        idx = torch.randperm(len(graphs))[:batch_size].tolist()
        batch_data = [pool_data[i] for i in idx]
        batch = Batch.from_data_list(batch_data).to(device)

        out = policy.forward(batch)
        target = targets_t[idx]
        loss = ((out["value"] - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if log_fn is not None and (step + 1) % log_every == 0:
            log_fn(
                f"  BC critic | step {step+1:4d}/{n_steps} | "
                f"loss {loss.item():.4f}"
            )

    return losses
