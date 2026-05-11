"""REINFORCE training step with per-graph EMA baseline.

This is a scaffold — it requires a working backend function that maps
(nx.Graph, GlobalSchedule) -> float reward.  Wire to Bloqade / Aquila
before running.
"""
from __future__ import annotations

from typing import Callable

import networkx as nx
import torch
from torch_geometric.data import Batch

from schedules import GlobalSchedule
from module1.featurize import graph_to_pyg
from module1.policy import SchedulePolicy


def reinforce_step(
    policy: SchedulePolicy,
    graphs: list[nx.Graph],
    backend_fn: Callable[[nx.Graph, GlobalSchedule], float],
    optimizer: torch.optim.Optimizer,
    baseline_ema: dict[int, float],
    ema_alpha: float = 0.9,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    """One REINFORCE update with per-graph EMA baseline as a control variate.

    Parameters
    ----------
    policy : SchedulePolicy
        The policy network to update.
    graphs : list[nx.Graph]
        Batch of graphs to train on.
    backend_fn : callable
        ``backend_fn(G, schedule) -> reward``.  Must be wired to a real
        backend (Bloqade simulator or Aquila hardware) before use.
    optimizer : torch.optim.Optimizer
        Optimizer for ``policy.parameters()``.
    baseline_ema : dict[int, float]
        Mutable dict mapping graph hash -> running reward baseline.
        Passed in so state persists across calls.
    ema_alpha : float
        Smoothing factor for the EMA baseline (higher = more smoothing).
    entropy_coef : float
        Weight for the entropy bonus (encourages exploration).

    Returns
    -------
    dict with keys: loss, policy_loss, mean_reward, mean_entropy.
    """
    data_list = [graph_to_pyg(G) for G in graphs]
    batch = Batch.from_data_list(data_list)
    device = next(policy.parameters()).device
    batch = batch.to(device)

    out = policy.sample_schedule(batch, deterministic=False)

    rewards: list[float] = []
    for i, G in enumerate(graphs):
        sched = GlobalSchedule(
            omega=out["omega"][i].detach().cpu().numpy(),
            delta=out["delta"][i].detach().cpu().numpy(),
            dt=policy.config.controls.dt,
            param_kind=policy.config.controls.param_kind,
        )
        rewards.append(backend_fn(G, sched))

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    advantages: list[float] = []
    for i, G in enumerate(graphs):
        key = hash(tuple(sorted(G.edges())))
        b = baseline_ema.get(key, rewards[i])
        advantages.append(rewards[i] - b)
        baseline_ema[key] = ema_alpha * b + (1 - ema_alpha) * rewards[i]
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)

    policy_loss = -(out["logprob"] * adv.detach()).mean()
    entropy_bonus = -entropy_coef * out["entropy"].mean()
    loss = policy_loss + entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "mean_reward": rewards_t.mean().item(),
        "mean_entropy": out["entropy"].mean().item(),
    }
