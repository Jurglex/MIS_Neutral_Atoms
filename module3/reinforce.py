"""REINFORCE training step with per-graph EMA baseline and optional critic.

Requires a backend function mapping (nx.Graph, GlobalSchedule) -> float reward.
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
    value_loss_coef: float = 0.5,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One REINFORCE update with EMA baseline and optional value-function loss.

    Parameters
    ----------
    policy : SchedulePolicy
        The policy network to update.
    graphs : list[nx.Graph]
        Batch of graphs to train on.
    backend_fn : callable
        ``backend_fn(G, schedule) -> reward``.
    optimizer : torch.optim.Optimizer
        Optimizer for ``policy.parameters()``.
    baseline_ema : dict[int, float]
        Mutable dict mapping graph hash -> running reward baseline.
    ema_alpha : float
        Smoothing factor for the EMA baseline.
    entropy_coef : float
        Weight for the entropy bonus.
    value_loss_coef : float
        Weight for the critic (value head) MSE loss.  Set to 0 to disable.
    grad_clip : float
        Max gradient norm for clipping.

    Returns
    -------
    dict with keys: loss, policy_loss, value_loss, mean_reward,
    mean_entropy, grad_norm, max_reward, min_reward.
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

    value_loss = torch.tensor(0.0, device=device)
    if value_loss_coef > 0:
        value_loss = value_loss_coef * (
            (out["value"] - rewards_t.detach()) ** 2
        ).mean()

    loss = policy_loss + entropy_bonus + value_loss

    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), max_norm=grad_clip
    )

    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_reward": rewards_t.mean().item(),
        "max_reward": rewards_t.max().item(),
        "min_reward": rewards_t.min().item(),
        "mean_entropy": out["entropy"].mean().item(),
        "grad_norm": float(grad_norm),
    }
