"""PPO trainer for the Graph→Schedule policy.

Major design differences from the legacy ``reinforce_step``:

* **Paired baseline advantages.**  Every graph in the batch is evaluated
  twice — once with a sampled policy schedule, once with the analytic
  adiabatic baseline schedule.  The advantage is
  ``A(G) = r_learned(G) − r_baseline(G)``.  This eliminates per-graph
  reward heterogeneity that EMA baselines cannot track quickly.

* **K rollouts per graph.**  For each graph we sample K independent
  schedules from the current policy and evaluate all of them.  The
  advantage averaging across K rollouts dramatically reduces variance
  per simulator call.

* **Advantage normalization.**  Advantages are scaled to zero-mean,
  unit-variance within each gradient step.

* **PPO clipping + multiple inner epochs.**  Rollouts are split into
  minibatches and the clipped policy ratio loss is applied across
  multiple epochs over the same collected data.

* **Replay buffer integration.**  Optional off-policy reuse via a
  ``ReplayBuffer`` (see ``module3.replay``).
"""
from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch

from schedules import GlobalSchedule
from module1.featurize import graph_to_pyg
from module1.policy import SchedulePolicy
from module1.base import FixedScheduleBaseline


# ── Rollout collection ───────────────────────────────────────────────────

def collect_rollouts(
    policy: SchedulePolicy,
    graphs: list[nx.Graph],
    backend_fn: Callable[[nx.Graph, GlobalSchedule], float],
    baseline_model: FixedScheduleBaseline | None,
    rollouts_per_graph: int,
    raw_backend_fn: Callable[[nx.Graph, GlobalSchedule], float] | None = None,
) -> dict[str, torch.Tensor | list]:
    """Generate K rollouts per graph and (optionally) one baseline evaluation.

    Returns
    -------
    dict with keys::

        graph_idx : (N,) long  — index into ``graphs``
        sampled_params : (N, A) — the action parameters that were rolled out
        old_logprob : (N,) — log-prob under the *current* policy
        old_value : (N,) — value estimate at the time of rollout
        reward : (N,) — backend reward (may be normalized-vs-baseline)
        baseline_reward : (N,) — analytic-baseline reward for the same graph
                                   (uses ``raw_backend_fn`` for raw rewards
                                    when normalization is desired)
        omega, delta : (N, N_t) — schedules produced (kept for diagnostics)

    Where ``N = len(graphs) * rollouts_per_graph``.
    """
    device = next(policy.parameters()).device
    data_list = [graph_to_pyg(G) for G in graphs]
    base_batch = Batch.from_data_list(data_list).to(device)

    all_sampled, all_logp, all_value = [], [], []
    all_omega, all_delta, all_graph_idx, all_rewards = [], [], [], []

    policy_was_training = policy.training
    policy.eval()
    for k in range(rollouts_per_graph):
        with torch.no_grad():
            out = policy.sample_schedule(base_batch, deterministic=False)
        for i, G in enumerate(graphs):
            sched = GlobalSchedule(
                omega=out["omega"][i].cpu().numpy(),
                delta=out["delta"][i].cpu().numpy(),
                dt=policy.config.controls.dt,
                param_kind=policy.config.controls.param_kind,
            )
            r = float(backend_fn(G, sched))
            all_rewards.append(r)
            all_graph_idx.append(i)
            all_sampled.append(out["sampled_params"][i].detach().cpu())
            all_logp.append(out["logprob"][i].detach().cpu())
            all_value.append(out["value"][i].detach().cpu())
            all_omega.append(out["omega"][i].detach().cpu())
            all_delta.append(out["delta"][i].detach().cpu())
    if policy_was_training:
        policy.train()

    baseline_rewards = torch.zeros(len(all_graph_idx))
    if baseline_model is not None and raw_backend_fn is not None:
        per_graph_base: dict[int, float] = {}
        for i, G in enumerate(graphs):
            sched_b = baseline_model.make_schedule(G)
            per_graph_base[i] = float(raw_backend_fn(G, sched_b))
        for j, gi in enumerate(all_graph_idx):
            baseline_rewards[j] = per_graph_base[gi]

    return {
        "graph_idx": torch.tensor(all_graph_idx, dtype=torch.long),
        "sampled_params": torch.stack(all_sampled),
        "old_logprob": torch.stack(all_logp),
        "old_value": torch.stack(all_value),
        "reward": torch.tensor(all_rewards, dtype=torch.float32),
        "baseline_reward": baseline_rewards,
        "omega": torch.stack(all_omega),
        "delta": torch.stack(all_delta),
        "data_list": data_list,
    }


# ── Advantage computation ────────────────────────────────────────────────

def compute_advantages(
    rewards: torch.Tensor,
    baseline_rewards: torch.Tensor,
    use_paired_baseline: bool,
    advantage_normalization: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build the advantage tensor from rollout rewards.

    With ``use_paired_baseline=True``::
        A = r_learned − r_baseline
    Otherwise the rewards are used directly (the EMA-style baseline must
    have been folded in by the caller; PPO's clipping then absorbs the
    remaining bias).
    """
    if use_paired_baseline:
        adv = rewards - baseline_rewards
    else:
        adv = rewards - rewards.mean()
    if advantage_normalization:
        adv = (adv - adv.mean()) / (adv.std() + eps)
    return adv


# ── PPO inner loss ───────────────────────────────────────────────────────

def ppo_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    new_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip: float,
    entropy_coef: float,
    value_loss_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard PPO-clip policy loss + value loss + entropy bonus."""
    ratio = (new_logprob - old_logprob).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = ((new_value - returns) ** 2).mean()
    entropy_bonus = -entropy.mean()

    loss = (
        policy_loss
        + value_loss_coef * value_loss
        + entropy_coef * entropy_bonus
    )

    with torch.no_grad():
        approx_kl = ((old_logprob - new_logprob)).mean().item()
        clip_frac = ((ratio - 1).abs() > clip).float().mean().item()

    return loss, {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float((-entropy_bonus).item()),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }


# ── Full PPO step ────────────────────────────────────────────────────────

def ppo_step(
    policy: SchedulePolicy,
    graphs: list[nx.Graph],
    backend_fn: Callable[[nx.Graph, GlobalSchedule], float],
    optimizer: torch.optim.Optimizer,
    baseline_model: FixedScheduleBaseline | None = None,
    raw_backend_fn: Callable[[nx.Graph, GlobalSchedule], float] | None = None,
    *,
    rollouts_per_graph: int = 4,
    ppo_epochs: int = 4,
    ppo_minibatch_size: int = 16,
    ppo_clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    grad_clip: float = 1.0,
    use_paired_baseline: bool = True,
    advantage_normalization: bool = True,
    extra_rollouts: dict | None = None,
    return_rollouts: bool = False,
) -> dict[str, float]:
    """One PPO update over K-rollout paired-baseline data.

    Parameters
    ----------
    policy, graphs, backend_fn, optimizer
        As in REINFORCE.
    baseline_model : FixedScheduleBaseline | None
        Required for ``use_paired_baseline=True``.
    raw_backend_fn : callable | None
        Reward function that returns *raw* (un-normalized) rewards.  Used to
        evaluate the baseline schedule when ``backend_fn`` itself returns
        ``is_cost_vs_baseline``-style normalized values.  Defaults to
        ``backend_fn`` when omitted.
    extra_rollouts : dict | None
        Additional pre-collected rollouts to mix in (e.g. from a replay
        buffer).  Must follow the schema of ``collect_rollouts``.
    """
    if raw_backend_fn is None:
        raw_backend_fn = backend_fn

    fresh_rollouts = collect_rollouts(
        policy=policy,
        graphs=graphs,
        backend_fn=backend_fn,
        baseline_model=baseline_model if use_paired_baseline else None,
        rollouts_per_graph=rollouts_per_graph,
        raw_backend_fn=raw_backend_fn if use_paired_baseline else None,
    )

    if extra_rollouts is not None:
        rollouts = _concat_rollouts(fresh_rollouts, extra_rollouts)
    else:
        rollouts = fresh_rollouts

    device = next(policy.parameters()).device

    rewards = rollouts["reward"].to(device)
    baseline_rewards = rollouts["baseline_reward"].to(device)
    advantages = compute_advantages(
        rewards, baseline_rewards,
        use_paired_baseline=use_paired_baseline,
        advantage_normalization=advantage_normalization,
    )

    if use_paired_baseline:
        returns = rewards - baseline_rewards
    else:
        returns = rewards.clone()

    sampled_params = rollouts["sampled_params"].to(device)
    old_logprob = rollouts["old_logprob"].to(device)
    graph_idx = rollouts["graph_idx"].numpy()
    data_list = rollouts["data_list"]

    N = sampled_params.shape[0]
    indices = np.arange(N)

    metrics_accum: dict[str, float] = {
        "policy_loss": 0.0, "value_loss": 0.0,
        "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0,
    }
    n_minibatches = 0
    grad_norm_last = 0.0

    for _epoch in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, N, ppo_minibatch_size):
            mb = indices[start:start + ppo_minibatch_size]
            mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)

            mb_graph_idx = graph_idx[mb]
            mb_data = [data_list[gi] for gi in mb_graph_idx]
            mb_batch = Batch.from_data_list(mb_data).to(device)

            new_lp, ent, new_val = policy.log_prob_for(
                mb_batch, sampled_params[mb_t]
            )

            loss, batch_metrics = ppo_loss(
                new_logprob=new_lp,
                old_logprob=old_logprob[mb_t],
                advantages=advantages[mb_t],
                new_value=new_val,
                returns=returns[mb_t],
                entropy=ent,
                clip=ppo_clip,
                entropy_coef=entropy_coef,
                value_loss_coef=value_loss_coef,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_norm=grad_clip
            )
            grad_norm_last = float(grad_norm)
            optimizer.step()

            for k, v in batch_metrics.items():
                metrics_accum[k] += v
            n_minibatches += 1

    if n_minibatches > 0:
        for k in metrics_accum:
            metrics_accum[k] /= n_minibatches

    result = {
        **metrics_accum,
        "loss": metrics_accum["policy_loss"]
                + value_loss_coef * metrics_accum["value_loss"]
                - entropy_coef * metrics_accum["entropy"],
        "mean_reward": float(rewards.mean().item()),
        "max_reward": float(rewards.max().item()),
        "min_reward": float(rewards.min().item()),
        "mean_baseline_reward": float(baseline_rewards.mean().item()),
        "mean_advantage": float(advantages.mean().item()),
        "mean_entropy": metrics_accum["entropy"],
        "grad_norm": grad_norm_last,
        "n_rollouts": int(N),
        "residual_alpha": float(policy.residual_alpha.item()),
    }
    if return_rollouts:
        result["_fresh_rollouts"] = fresh_rollouts
    return result


def _concat_rollouts(a: dict, b: dict) -> dict:
    """Concatenate two rollout dicts (used for replay-buffer mixing)."""
    out = {}
    for k in (
        "graph_idx", "sampled_params", "old_logprob",
        "old_value", "reward", "baseline_reward", "omega", "delta",
    ):
        out[k] = torch.cat([a[k], b[k]], dim=0)
    out["data_list"] = a["data_list"] + b["data_list"]
    return out
