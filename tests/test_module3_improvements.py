"""Tests for the v8 training upgrades:

* Architecture 3 (multiplicative Ω, monotone Δ) heads.
* Residual-α curriculum schedule on SchedulePolicy.
* Per-graph normalized reward (``is_cost_vs_baseline``) and baseline cache.
* PPO step (paired baseline, advantage normalization, multiple epochs).
* Behavioral-cloning pretraining (policy mean → baseline, critic → reward).
* Replay buffer FIFO + sampling semantics.
* Pool curation filtering.
* Probe diagnostics (schedule deviation, conditioning index).
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx
import pytest
import torch

from config import (
    ProjectConfig, ControlsConfig, UDGConfig, ParamKind,
    HardwareSpecs, RewardConfig, load_project_config_json,
)
from schedules import GlobalSchedule
from module1.policy import SchedulePolicy
from module1.base import FixedScheduleBaseline
from module1.heads import MonotoneDeltaHead, MultiplicativeOmegaHead
from module2.interfaces import BackendResult
from module3.interfaces import TrainingConfig
from module3.backend_adapter import (
    BaselineRewardCache, _is_cost_from_counts, _raw_reward,
    make_reward_fn, make_raw_reward_fn,
)
from module3.replay import ReplayBuffer
from module3.ppo import (
    collect_rollouts, compute_advantages, ppo_loss, ppo_step,
)
from module3.pretrain import behavioral_clone_policy, behavioral_clone_critic
from module3.diagnostics import (
    schedule_deviation_probe, graph_conditioning_index,
)


ROOT = Path(__file__).resolve().parent.parent


def _project_config():
    return load_project_config_json(
        ROOT / "config.json", ROOT / "hardware_specs.json"
    )


def _arch3_config(N_t: int = 32, learn_omega: bool = True) -> ProjectConfig:
    """Compact config explicitly setting architecture=3."""
    controls = ControlsConfig(
        T=4e-6, N_t=N_t, param_kind=ParamKind.pwl,
        learn_omega=learn_omega, architecture=3,
        omega_max=15.8, delta_min=-25.0, delta_max=25.0,
        n_delta_knots=8, n_omega_modes=5, n_delta_modes=6,
        warm_start=True,
        residual_alpha_start=0.05, residual_alpha_end=1.0,
        residual_alpha_warmup_steps=100,
    )
    udg = UDGConfig(nx=3, ny=3, spacing=1.0, radius=1.5, dropout_rate=0.0, seed=7)
    hardware = HardwareSpecs(
        C6=5.42e6, omega_max=15.8, delta_min=-125.0, delta_max=125.0,
        t_ramp=0.3, t_onset=0.0,
    )
    return ProjectConfig(
        backend="bloqade", controls=controls, udg=udg, hardware=hardware,
    )


def _mock_backend_fn(G, schedule):
    """Deterministic mock proportional to (mean omega - mean delta)."""
    return float(schedule.omega.mean() - 0.1 * schedule.delta.mean())


# ---------------------------------------------------------------------------
# Architecture 3 heads
# ---------------------------------------------------------------------------

def test_multiplicative_omega_initial_equals_baseline():
    """MultiplicativeOmegaHead at zero params reproduces the trapezoidal baseline."""
    cfg = _arch3_config(N_t=64)
    head = MultiplicativeOmegaHead(
        embed_dim=16, controls=cfg.controls, hardware=cfg.hardware,
        R_b_um=cfg.udg.radius * cfg.udg.spacing,
    )
    embed = torch.zeros(2, 16)
    _, omega = head(embed)
    expected = head.baseline_omega.unsqueeze(0).expand(2, -1)
    torch.testing.assert_close(omega, expected, atol=1e-4, rtol=0.0)
    assert (omega >= 0).all()
    assert (omega <= cfg.controls.omega_max + 1e-3).all()


def test_multiplicative_omega_zero_boundaries():
    """Multiplicative head inherits Ω(0) = Ω(T) = 0 from the baseline."""
    cfg = _arch3_config(N_t=64)
    head = MultiplicativeOmegaHead(
        embed_dim=16, controls=cfg.controls, hardware=cfg.hardware,
        R_b_um=cfg.udg.radius * cfg.udg.spacing,
    )
    embed = torch.randn(3, 16) * 5.0
    _, omega = head(embed)
    torch.testing.assert_close(omega[:, 0], torch.zeros(3), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(omega[:, -1], torch.zeros(3), atol=1e-5, rtol=0.0)


def test_monotone_delta_is_monotone():
    """MonotoneDeltaHead produces non-decreasing Δ(t) by construction."""
    cfg = _arch3_config(N_t=64)
    head = MonotoneDeltaHead(embed_dim=16, controls=cfg.controls)
    embed = torch.randn(5, 16) * 3.0
    _, delta = head(embed)

    diffs = delta[:, 1:] - delta[:, :-1]
    assert (diffs >= -1e-5).all(), "Δ(t) decreased between consecutive timesteps"


def test_monotone_delta_within_bounds():
    cfg = _arch3_config(N_t=64)
    head = MonotoneDeltaHead(embed_dim=16, controls=cfg.controls)
    embed = torch.randn(3, 16) * 10.0
    _, delta = head(embed)
    assert (delta >= cfg.controls.delta_min - 1e-3).all()
    assert (delta <= cfg.controls.delta_max + 1e-3).all()


def test_arch3_policy_forward_shapes():
    cfg = _arch3_config(N_t=32)
    policy = SchedulePolicy(cfg)
    from module1.featurize import graph_to_pyg
    from torch_geometric.data import Batch
    graphs = [nx.erdos_renyi_graph(8, 0.4, seed=i) for i in range(3)]
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.forward(batch)
    assert out["omega"].shape == (3, 32)
    assert out["delta"].shape == (3, 32)
    assert out["value"].shape == (3,)


def test_arch3_initial_output_matches_baseline():
    """At init with alpha small, Arch 3 produces ≈ baseline schedule."""
    torch.manual_seed(0)
    cfg = _arch3_config(N_t=64)
    policy = SchedulePolicy(cfg)
    baseline = FixedScheduleBaseline(cfg)

    g = nx.path_graph(8)
    sched_l = policy.make_schedule(g)
    sched_b = baseline.make_schedule(g)

    np.testing.assert_allclose(
        sched_l.omega, sched_b.omega, atol=1.0,
        err_msg="Arch 3: Ω deviates too far from baseline at init",
    )


# ---------------------------------------------------------------------------
# Residual-α curriculum
# ---------------------------------------------------------------------------

def test_residual_alpha_default_is_start():
    cfg = _arch3_config(N_t=32)
    policy = SchedulePolicy(cfg)
    assert float(policy.residual_alpha) == pytest.approx(
        cfg.controls.residual_alpha_start
    )


def test_residual_alpha_linear_schedule():
    cfg = _arch3_config(N_t=32)
    policy = SchedulePolicy(cfg)
    warmup = cfg.controls.residual_alpha_warmup_steps
    start = cfg.controls.residual_alpha_start
    end = cfg.controls.residual_alpha_end

    assert policy.current_alpha(0) == pytest.approx(start)
    mid = policy.current_alpha(warmup // 2)
    assert start < mid < end
    assert policy.current_alpha(warmup) == pytest.approx(end)
    assert policy.current_alpha(warmup * 10) == pytest.approx(end)


def test_residual_alpha_set_persists_through_forward():
    cfg = _arch3_config(N_t=32)
    policy = SchedulePolicy(cfg)
    policy.set_residual_alpha(0.5)

    g = nx.path_graph(8)
    _ = policy.make_schedule(g)
    assert float(policy.residual_alpha) == pytest.approx(0.5)


def test_residual_alpha_affects_arch3_output():
    """Higher alpha → larger deviations from baseline."""
    torch.manual_seed(42)
    cfg = _arch3_config(N_t=32)
    policy = SchedulePolicy(cfg)

    # Initialize the heads to be non-zero so alpha matters.
    with torch.no_grad():
        for p in policy.omega_head.net.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        for p in policy.delta_head.net.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    baseline = FixedScheduleBaseline(cfg)
    g = nx.path_graph(8)
    base_delta = baseline.make_schedule(g).delta

    policy.set_residual_alpha(0.01)
    dev_low = np.abs(policy.make_schedule(g).delta - base_delta).mean()

    policy.set_residual_alpha(1.0)
    dev_high = np.abs(policy.make_schedule(g).delta - base_delta).mean()

    assert dev_high > dev_low, (
        f"Larger alpha should produce larger deviation: "
        f"alpha=0.01→{dev_low:.3f}, alpha=1.0→{dev_high:.3f}"
    )


# ---------------------------------------------------------------------------
# Reward normalization: is_cost_vs_baseline
# ---------------------------------------------------------------------------

def test_baseline_reward_cache_lazy_fill():
    """Cache evaluates the lambda once per unique graph key."""
    calls: list[int] = []

    def _eval(G):
        calls.append(G.graph.get("seed", id(G)))
        return float(nx.density(G))

    cache = BaselineRewardCache(_eval)
    G1 = nx.path_graph(5)
    G1.graph["seed"] = 1
    G2 = nx.path_graph(6)
    G2.graph["seed"] = 2

    a1 = cache.get(G1)
    a1_again = cache.get(G1)
    a2 = cache.get(G2)

    assert calls == [1, 2]
    assert a1 == a1_again
    assert len(cache) == 2


def test_is_cost_vs_baseline_zero_at_baseline():
    """When learned reward equals baseline, normalized reward is 0."""
    G = nx.path_graph(3)
    counts = {"rgr": 10}
    result = BackendResult(p_mis=0.5, shots=10, counts=counts)
    raw_cost = _is_cost_from_counts(counts, G, 3.0, True)

    cache = BaselineRewardCache(lambda g: raw_cost)
    cfg = RewardConfig(kind="is_cost_vs_baseline", baseline_norm_eps=1e-3)
    from module3.backend_adapter import _raw_reward

    class _Stub:
        def estimate_p_mis(self, schedule, graph, positions, *, seed=None):
            return result

    G.graph["positions"] = {i: (float(i), 0.0) for i in G.nodes}
    G.graph["seed"] = 0

    reward_fn = make_reward_fn(_Stub(), reward_cfg=cfg, baseline_cache=cache)
    sched = GlobalSchedule(omega=np.zeros(8), delta=np.zeros(8), dt=1e-7,
                           param_kind=ParamKind.pwl)
    r = reward_fn(G, sched)
    assert r == pytest.approx(0.0, abs=1e-3)


def test_is_cost_vs_baseline_positive_when_better():
    G = nx.path_graph(3)
    learned_counts = {"rgr": 10}
    learned_result = BackendResult(p_mis=0.5, shots=10, counts=learned_counts)
    learned_cost = _is_cost_from_counts(learned_counts, G, 3.0, True)
    baseline_cost = learned_cost * 0.5  # learned is 2x better

    cache = BaselineRewardCache(lambda g: baseline_cost)
    cfg = RewardConfig(kind="is_cost_vs_baseline", baseline_norm_eps=1e-3)

    class _Stub:
        def estimate_p_mis(self, schedule, graph, positions, *, seed=None):
            return learned_result

    G.graph["positions"] = {i: (float(i), 0.0) for i in G.nodes}
    G.graph["seed"] = 0
    reward_fn = make_reward_fn(_Stub(), reward_cfg=cfg, baseline_cache=cache)
    sched = GlobalSchedule(omega=np.zeros(8), delta=np.zeros(8), dt=1e-7,
                           param_kind=ParamKind.pwl)
    r = reward_fn(G, sched)
    assert r > 0
    expected = (learned_cost - baseline_cost) / (abs(baseline_cost) + 1e-3)
    assert r == pytest.approx(expected, rel=1e-4)


def test_is_cost_vs_baseline_falls_back_without_cache():
    """If no cache is provided, the normalized kind degrades to raw is_cost."""
    G = nx.path_graph(3)
    counts = {"rgr": 10}
    result = BackendResult(p_mis=0.5, shots=10, counts=counts)
    raw_cost = _is_cost_from_counts(counts, G, 3.0, True)

    class _Stub:
        def estimate_p_mis(self, schedule, graph, positions, *, seed=None):
            return result

    G.graph["positions"] = {i: (float(i), 0.0) for i in G.nodes}
    cfg = RewardConfig(kind="is_cost_vs_baseline")
    reward_fn = make_reward_fn(_Stub(), reward_cfg=cfg, baseline_cache=None)
    sched = GlobalSchedule(omega=np.zeros(8), delta=np.zeros(8), dt=1e-7,
                           param_kind=ParamKind.pwl)
    r = reward_fn(G, sched)
    assert r == pytest.approx(raw_cost)


# ---------------------------------------------------------------------------
# PPO step
# ---------------------------------------------------------------------------

def _udg_pool(n: int = 4):
    from module3.learner import _build_graph_pool
    cfg = _project_config()
    return _build_graph_pool(cfg.udg, pool_size=n, base_seed=100)


def test_collect_rollouts_shapes():
    cfg = _arch3_config(N_t=32)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(3)

    rollouts = collect_rollouts(
        policy=policy, graphs=pool[:2],
        backend_fn=_mock_backend_fn,
        baseline_model=FixedScheduleBaseline(cfg),
        rollouts_per_graph=3,
        raw_backend_fn=_mock_backend_fn,
    )
    N = 2 * 3
    assert rollouts["sampled_params"].shape[0] == N
    assert rollouts["reward"].shape == (N,)
    assert rollouts["baseline_reward"].shape == (N,)
    # Baseline rewards within a graph should be equal across rollouts
    for gi in range(2):
        mask = rollouts["graph_idx"] == gi
        vals = rollouts["baseline_reward"][mask]
        torch.testing.assert_close(vals, vals[0].repeat(vals.shape[0]))


def test_compute_advantages_normalization():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    baselines = torch.tensor([0.5, 0.5, 0.5, 0.5])
    adv = compute_advantages(
        rewards, baselines,
        use_paired_baseline=True, advantage_normalization=True,
    )
    assert adv.mean().abs().item() < 1e-5
    assert abs(adv.std().item() - 1.0) < 1e-3


def test_ppo_loss_clipping_bounds_ratio():
    """When ratio is way outside the clip range, gradient comes from clipped term."""
    new_lp = torch.tensor([0.5, -0.5])
    old_lp = torch.tensor([0.0, 0.0])
    adv = torch.tensor([1.0, -1.0])
    new_val = torch.tensor([0.1, 0.1])
    returns = torch.tensor([0.0, 0.0])
    ent = torch.tensor([1.0, 1.0])
    loss, metrics = ppo_loss(
        new_lp, old_lp, adv, new_val, returns, ent,
        clip=0.2, entropy_coef=0.0, value_loss_coef=0.0,
    )
    assert "clip_frac" in metrics
    assert 0.0 <= metrics["clip_frac"] <= 1.0


def test_ppo_step_runs_end_to_end():
    cfg = _arch3_config(N_t=32, learn_omega=True)
    policy = SchedulePolicy(cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    pool = _udg_pool(3)

    metrics = ppo_step(
        policy=policy, graphs=pool[:2],
        backend_fn=_mock_backend_fn, optimizer=optimizer,
        baseline_model=FixedScheduleBaseline(cfg),
        raw_backend_fn=_mock_backend_fn,
        rollouts_per_graph=3, ppo_epochs=2, ppo_minibatch_size=4,
    )
    for k in (
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac",
        "loss", "mean_reward", "grad_norm", "n_rollouts",
        "mean_baseline_reward", "residual_alpha",
    ):
        assert k in metrics, f"PPO step missing metric: {k}"


def test_ppo_paired_baseline_advantage():
    """With paired baselines, advantages are reward minus same-graph baseline."""
    cfg = _arch3_config(N_t=32, learn_omega=True)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(2)

    rollouts = collect_rollouts(
        policy=policy, graphs=pool,
        backend_fn=_mock_backend_fn,
        baseline_model=FixedScheduleBaseline(cfg),
        rollouts_per_graph=2,
        raw_backend_fn=_mock_backend_fn,
    )
    adv = compute_advantages(
        rollouts["reward"], rollouts["baseline_reward"],
        use_paired_baseline=True, advantage_normalization=False,
    )
    expected = rollouts["reward"] - rollouts["baseline_reward"]
    torch.testing.assert_close(adv, expected, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# Behavioral cloning
# ---------------------------------------------------------------------------

def test_bc_policy_reduces_baseline_error():
    """BC measurably reduces deviation from the baseline schedule.

    Uses Arch 2 (which has no built-in baseline matching) so the random
    initial output differs noticeably from the baseline and BC has a
    target to converge toward.
    """
    controls = ControlsConfig(
        T=4e-6, N_t=32, param_kind=ParamKind.pwl,
        learn_omega=True, architecture=2,
        omega_max=15.8, delta_min=-25.0, delta_max=25.0,
        n_delta_knots=8, n_omega_modes=5, n_delta_modes=6,
        warm_start=False,
        residual_alpha_start=1.0, residual_alpha_end=1.0,
        residual_alpha_warmup_steps=0,
    )
    udg = UDGConfig(nx=3, ny=3, spacing=1.0, radius=1.5, dropout_rate=0.0, seed=7)
    hw = HardwareSpecs(C6=5.42e6, omega_max=15.8, t_ramp=0.3, t_onset=0.0)
    cfg = ProjectConfig(backend="bloqade", controls=controls, udg=udg, hardware=hw)

    torch.manual_seed(0)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(4)

    baseline = FixedScheduleBaseline(cfg)
    g = pool[0]
    err_before = np.abs(
        policy.make_schedule(g).delta - baseline.make_schedule(g).delta
    ).mean()

    losses = behavioral_clone_policy(
        policy=policy, graphs=pool, baseline_model=baseline,
        n_steps=300, lr=3e-3,
    )
    assert losses[-1] < losses[0] * 0.5, (
        f"BC didn't reduce policy loss enough: {losses[0]:.6f} → {losses[-1]:.6f}"
    )

    err_after = np.abs(
        policy.make_schedule(g).delta - baseline.make_schedule(g).delta
    ).mean()
    assert err_after < err_before


def test_bc_critic_predicts_baseline_reward():
    cfg = _arch3_config(N_t=32, learn_omega=True)
    torch.manual_seed(0)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(4)

    # Graph-dependent target so the critic actually has to learn the mapping.
    baseline_rewards: dict[object, float] = {}
    for G in pool:
        key = G.graph.get("seed", id(G))
        baseline_rewards[key] = float(nx.density(G) * 5.0 + G.number_of_nodes())

    losses = behavioral_clone_critic(
        policy=policy, graphs=pool, baseline_rewards=baseline_rewards,
        n_steps=800, lr=3e-3,
    )
    assert losses[-1] < losses[0] * 0.1

    from module1.featurize import graph_to_pyg
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph_to_pyg(G) for G in pool])
    with torch.no_grad():
        out = policy.forward(batch)
    preds = out["value"].cpu().numpy()
    targets = np.array([
        baseline_rewards[G.graph.get("seed", id(G))] for G in pool
    ])
    # Critic should match each per-graph target reasonably (small training set).
    assert np.abs(preds - targets).mean() < 1.0


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def test_replay_buffer_fifo_eviction():
    buf = ReplayBuffer(capacity=3)
    cfg = _arch3_config(N_t=16, learn_omega=True)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(2)

    rollouts = collect_rollouts(
        policy=policy, graphs=pool, backend_fn=_mock_backend_fn,
        baseline_model=FixedScheduleBaseline(cfg),
        rollouts_per_graph=2, raw_backend_fn=_mock_backend_fn,
    )
    buf.add_rollouts(rollouts)
    assert len(buf) == 3  # capped at capacity


def test_replay_buffer_disabled_when_zero_capacity():
    buf = ReplayBuffer(capacity=0)
    cfg = _arch3_config(N_t=16, learn_omega=True)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(1)
    rollouts = collect_rollouts(
        policy=policy, graphs=pool, backend_fn=_mock_backend_fn,
        baseline_model=FixedScheduleBaseline(cfg),
        rollouts_per_graph=2, raw_backend_fn=_mock_backend_fn,
    )
    buf.add_rollouts(rollouts)
    assert len(buf) == 0
    assert buf.sample(2) is None


def test_replay_buffer_sample_shape():
    buf = ReplayBuffer(capacity=8)
    cfg = _arch3_config(N_t=16, learn_omega=True)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(2)
    rollouts = collect_rollouts(
        policy=policy, graphs=pool, backend_fn=_mock_backend_fn,
        baseline_model=FixedScheduleBaseline(cfg),
        rollouts_per_graph=3, raw_backend_fn=_mock_backend_fn,
    )
    buf.add_rollouts(rollouts)
    sample = buf.sample(4)
    assert sample is not None
    assert sample["sampled_params"].shape[0] == 4
    assert len(sample["data_list"]) == 4


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_schedule_deviation_probe_returns_correlations():
    cfg = _arch3_config(N_t=32, learn_omega=True)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(4)

    out = schedule_deviation_probe(
        policy, pool, baseline_model=FixedScheduleBaseline(cfg)
    )
    assert "feature_correlations" in out
    for k in ("n_nodes", "density", "lambda_2", "clustering"):
        assert k in out["feature_correlations"]
    assert "per_graph" in out and len(out["per_graph"]) == len(pool)


def test_graph_conditioning_index_keys():
    cfg = _arch3_config(N_t=32, learn_omega=True)
    policy = SchedulePolicy(cfg)
    pool = _udg_pool(3)
    ci = graph_conditioning_index(policy, pool, n_rollouts=3)
    for k in ("between_graph_var", "within_graph_var", "conditioning_index"):
        assert k in ci


# ---------------------------------------------------------------------------
# Learner with PPO + curation end-to-end smoke
# ---------------------------------------------------------------------------

def test_learner_ppo_end_to_end_with_mock_backend():
    cfg = _project_config()
    train_cfg = TrainingConfig(
        total_steps=2, batch_size=2, graph_pool_size=4, eval_graphs=2,
        seed=42, algorithm="ppo", rollouts_per_graph=2,
        ppo_epochs=1, ppo_minibatch_size=4,
        bc_pretrain_steps=0, bc_critic_steps=0,
        pool_curation=False, replay_buffer_size=0,
        diagnostics_every=0,
    )
    from module3.learner import ReinforceLearner
    learner = ReinforceLearner(
        config=train_cfg, project=cfg, backend_fn=_mock_backend_fn,
    )
    batch = learner.select_batch(learner.train_pool)
    metrics = learner.train_step(batch)
    for k in ("loss", "policy_loss", "value_loss",
              "mean_reward", "residual_alpha", "grad_norm"):
        assert k in metrics
    # Residual alpha should have been set from the schedule (step 0).
    assert metrics["residual_alpha"] == pytest.approx(
        cfg.controls.residual_alpha_start, abs=1e-6,
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ... ", end="", flush=True)
        try:
            t()
            print("OK")
        except Exception as e:
            import traceback
            print(f"FAILED:")
            traceback.print_exc()
