"""Tests for Module 1 schedule models (baselines and SchedulePolicy)."""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch
import pytest

from config import (
    ControlsConfig, UDGConfig, ProjectConfig, ParamKind,
    HardwareSpecs, compute_blockade_omega,
)
from schedules import GlobalSchedule
from module1 import ScheduleModel, GNNModel, AdjacencyMLP, SchedulePolicy
from module1.featurize import graph_to_pyg, laplacian_positional_encoding, algebraic_connectivity
from module1.heads import (
    OmegaHead, AnalyticOmega, DeltaHead,
    FourierOmegaHead, FourierDeltaHead,
)
from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    N_t: int = 16,
    learn_omega: bool = False,
    architecture: int = 1,
    warm_start: bool = False,
) -> ProjectConfig:
    controls = ControlsConfig(
        T=4e-6, N_t=N_t, param_kind=ParamKind.pwc,
        learn_omega=learn_omega, architecture=architecture,
        omega_max=15.8, delta_min=-25.0, delta_max=25.0,
        n_delta_knots=8, n_omega_modes=5, n_delta_modes=8,
        warm_start=warm_start, omega_scale=2.0, omega_cap=None,
    )
    udg = UDGConfig(nx=4, ny=4, spacing=1.0, radius=2.0, dropout_rate=0.0, seed=0)
    hardware = HardwareSpecs(C6=5.42e6, omega_max=15.8, t_ramp=0.3, t_onset=0.0)
    return ProjectConfig(backend="bloqade", controls=controls, udg=udg, hardware=hardware)


def _test_graphs() -> list[nx.Graph]:
    return [nx.erdos_renyi_graph(n, 0.3, seed=i) for i, n in enumerate([20, 25, 30])]


# ---------------------------------------------------------------------------
# Hardware specs and blockade omega tests
# ---------------------------------------------------------------------------

def test_compute_blockade_omega_basic():
    C6 = 5.42e6
    R_b = 2.0  # μm
    omega = compute_blockade_omega(C6, R_b, omega_max_hw=15.8)
    expected = C6 / 2.0**6  # 84687.5 rad/μs
    assert omega == 15.8, "Should be capped at hardware max"


def test_compute_blockade_omega_large_radius():
    C6 = 5.42e6
    R_b = 30.0  # very large radius → small omega
    omega = compute_blockade_omega(C6, R_b, omega_max_hw=15.8)
    expected = C6 / 30.0**6
    assert abs(omega - expected) < 1e-6, "Should use computed value (below hw max)"
    assert omega < 15.8


# ---------------------------------------------------------------------------
# Baseline model tests (GNNModel, AdjacencyMLP)
# ---------------------------------------------------------------------------

def test_baselines_are_schedule_models():
    cfg = _make_config()
    assert isinstance(GNNModel(cfg), ScheduleModel)
    assert isinstance(AdjacencyMLP(cfg), ScheduleModel)


def test_baselines_schedule_shapes():
    N_t = 16
    cfg = _make_config(N_t=N_t)
    g = nx.path_graph(5)
    for ModelCls in (GNNModel, AdjacencyMLP):
        sched = ModelCls(cfg)(g)
        assert isinstance(sched, GlobalSchedule)
        assert sched.omega.shape == (N_t,)
        assert sched.delta.shape == (N_t,)
        assert sched.n_steps == N_t


# ---------------------------------------------------------------------------
# Featurization tests
# ---------------------------------------------------------------------------

def test_graph_to_pyg_shapes():
    g = nx.path_graph(10)
    data = graph_to_pyg(g, k_pe=4)
    assert data.x.shape == (10, 7)  # 3 + 4 PE dims (degree, clustering, triangles + PE)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 2 * g.number_of_edges()
    assert data.graph_feats.shape == (1, 4)


def test_laplacian_pe_padding_small_graph():
    g = nx.path_graph(3)
    pe = laplacian_positional_encoding(g, k=8)
    assert pe.shape == (3, 8)


def test_algebraic_connectivity_disconnected():
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    assert algebraic_connectivity(g) == 0.0


# ---------------------------------------------------------------------------
# Arch 1 head tests
# ---------------------------------------------------------------------------

def test_omega_head_shapes_and_constraints():
    torch.manual_seed(0)
    cfg = _make_config(N_t=64)
    head = OmegaHead(embed_dim=32, controls=cfg.controls)
    embed = torch.randn(3, 32)
    params, omega = head(embed)
    assert params.shape == (3, 3)
    assert omega.shape == (3, 64)
    assert (omega >= 0).all()
    assert torch.allclose(omega[:, 0], torch.zeros(3), atol=1e-5)
    assert torch.allclose(omega[:, -1], torch.zeros(3), atol=1e-5)
    assert (omega <= cfg.controls.omega_max + 1e-5).all()


def test_analytic_omega_trapezoidal_shape():
    """Trapezoidal envelope: zero at edges, flat plateau in the middle."""
    N_t = 64
    omega_peak = 10.0
    t_ramp_frac = 0.1  # 10% ramp on each side → 80% hold
    head = AnalyticOmega(embed_dim=32, N_t=N_t, omega_peak=omega_peak,
                         t_ramp_frac=t_ramp_frac, t_onset_frac=0.0)
    embed = torch.randn(2, 32)
    params, omega = head(embed)

    assert params.shape == (2, 0)
    assert omega.shape == (2, N_t)
    assert (omega >= -1e-6).all(), "Omega went negative"
    assert (omega <= omega_peak + 1e-5).all(), "Omega exceeded peak"
    # Endpoints should be zero
    assert torch.allclose(omega[:, 0], torch.zeros(2), atol=1e-5)
    assert torch.allclose(omega[:, -1], torch.zeros(2), atol=1e-5)
    # Mid-point should be at peak (well within the hold region)
    mid = N_t // 2
    assert torch.allclose(omega[:, mid], torch.full((2,), omega_peak), atol=0.5)


def test_analytic_omega_with_onset():
    """Non-zero onset delay shifts the ramp start."""
    N_t = 100
    omega_peak = 12.0
    head = AnalyticOmega(embed_dim=32, N_t=N_t, omega_peak=omega_peak,
                         t_ramp_frac=0.1, t_onset_frac=0.1)
    embed = torch.randn(1, 32)
    _, omega = head(embed)
    # During onset period (first 10%), omega should be ~0
    onset_end = int(0.1 * N_t)
    assert omega[0, :onset_end].abs().max() < 1e-5


def test_delta_head_shapes_and_bounds():
    torch.manual_seed(0)
    cfg = _make_config(N_t=64)
    head = DeltaHead(embed_dim=32, controls=cfg.controls)
    embed = torch.randn(3, 32)
    params, delta = head(embed)
    assert params.shape == (3, 8)
    assert delta.shape == (3, 64)
    assert (delta >= cfg.controls.delta_min - 1e-3).all()
    assert (delta <= cfg.controls.delta_max + 1e-3).all()


# ---------------------------------------------------------------------------
# Arch 2 head tests
# ---------------------------------------------------------------------------

def test_fourier_omega_head_shapes_and_constraints():
    torch.manual_seed(0)
    cfg = _make_config(N_t=64)
    head = FourierOmegaHead(embed_dim=32, controls=cfg.controls)
    embed = torch.randn(3, 32)
    params, omega = head(embed)
    assert params.shape == (3, 5)
    assert omega.shape == (3, 64)
    assert (omega >= 0).all()
    assert (omega <= cfg.controls.omega_max + 1e-5).all()
    expected_boundary = cfg.controls.omega_max * 0.5
    assert torch.allclose(omega[:, 0], torch.full((3,), expected_boundary), atol=1e-4)
    assert torch.allclose(omega[:, -1], torch.full((3,), expected_boundary), atol=1e-4)


def test_fourier_delta_head_shapes_and_bounds():
    torch.manual_seed(0)
    cfg = _make_config(N_t=64)
    head = FourierDeltaHead(embed_dim=32, controls=cfg.controls)
    embed = torch.randn(3, 32)
    params, delta = head(embed)
    assert params.shape == (3, 9)  # 1 DC + 8 cosine modes
    assert delta.shape == (3, 64)
    assert (delta >= cfg.controls.delta_min - 1e-3).all()
    assert (delta <= cfg.controls.delta_max + 1e-3).all()


# ---------------------------------------------------------------------------
# SchedulePolicy tests — parametrized over architectures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch", [1, 2])
def test_policy_forward_shapes(arch):
    torch.manual_seed(0)
    cfg = _make_config(N_t=64, learn_omega=True, architecture=arch)
    policy = SchedulePolicy(cfg)
    graphs = _test_graphs()
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.forward(batch)
    B = len(graphs)
    assert out["omega"].shape == (B, 64)
    assert out["delta"].shape == (B, 64)
    assert out["value"].shape == (B,)
    assert out["mean_params"].shape[0] == B


@pytest.mark.parametrize("arch", [1, 2])
def test_policy_sample_schedule(arch):
    torch.manual_seed(0)
    cfg = _make_config(N_t=64, learn_omega=False, architecture=arch)
    policy = SchedulePolicy(cfg)
    graphs = _test_graphs()
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.sample_schedule(batch, deterministic=False)
    B = len(graphs)
    assert out["omega"].shape == (B, 64)
    assert out["delta"].shape == (B, 64)
    assert out["logprob"].shape == (B,)
    assert out["entropy"].shape == (B,)
    assert out["value"].shape == (B,)


@pytest.mark.parametrize("arch", [1, 2])
def test_policy_make_schedule_interface(arch):
    torch.manual_seed(0)
    cfg = _make_config(N_t=32, learn_omega=False, architecture=arch)
    policy = SchedulePolicy(cfg)
    g = nx.path_graph(10)

    sched = policy.make_schedule(g)

    assert isinstance(sched, GlobalSchedule)
    assert sched.omega.shape == (32,)
    assert sched.delta.shape == (32,)
    assert sched.omega.dtype == np.float64
    assert sched.delta.dtype == np.float64
    assert sched.dt == cfg.controls.dt
    assert sched.param_kind == ParamKind.pwc


@pytest.mark.parametrize("arch", [1, 2])
def test_policy_gradient_flow(arch):
    """Gradients flow through the reparameterized schedule values (omega, delta)."""
    torch.manual_seed(0)
    cfg = _make_config(N_t=32, learn_omega=True, architecture=arch)
    policy = SchedulePolicy(cfg)
    graphs = _test_graphs()
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.sample_schedule(batch)

    loss = out["omega"].sum() + out["delta"].sum()
    loss.backward()

    params_with_grad = {
        name for name, p in policy.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert any("encoder" in n for n in params_with_grad), "Encoder got no gradients"
    assert any("omega_head" in n for n in params_with_grad), "Omega head got no gradients"
    assert any("delta_head" in n for n in params_with_grad), "Delta head got no gradients"


def test_policy_omega_constraints_arch1():
    """Arch 1 Omega: boundaries at 0, non-negative, bounded by omega_max."""
    torch.manual_seed(0)
    cfg = _make_config(N_t=64, learn_omega=True, architecture=1)
    policy = SchedulePolicy(cfg)
    graphs = _test_graphs()
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.sample_schedule(batch, deterministic=True)
    omega = out["omega"]
    assert (omega >= 0).all(), "Omega went negative"
    assert torch.allclose(omega[:, 0], torch.zeros(len(graphs)), atol=1e-5), "Omega(0) != 0"
    assert torch.allclose(omega[:, -1], torch.zeros(len(graphs)), atol=1e-5), "Omega(T) != 0"


def test_policy_omega_constraints_arch2():
    """Arch 2 Omega: non-negative, bounded by omega_max, sigmoid(0)=0.5 at boundaries."""
    torch.manual_seed(0)
    cfg = _make_config(N_t=64, learn_omega=True, architecture=2)
    policy = SchedulePolicy(cfg)
    graphs = _test_graphs()
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.sample_schedule(batch, deterministic=True)
    omega = out["omega"]
    assert (omega >= 0).all(), "Omega went negative"
    assert (omega <= cfg.controls.omega_max + 1e-3).all(), "Omega exceeded omega_max"
    boundary_val = cfg.controls.omega_max * 0.5
    assert torch.allclose(omega[:, 0], torch.full((len(graphs),), boundary_val), atol=1e-4)
    assert torch.allclose(omega[:, -1], torch.full((len(graphs),), boundary_val), atol=1e-4)


def test_policy_analytic_omega_is_trapezoidal():
    """When learn_omega=False, the policy produces a trapezoidal Omega schedule."""
    torch.manual_seed(0)
    cfg = _make_config(N_t=64, learn_omega=False, architecture=1)
    policy = SchedulePolicy(cfg)
    g = nx.path_graph(10)
    sched = policy.make_schedule(g)
    omega = sched.omega
    assert omega[0] == pytest.approx(0.0, abs=1e-5)
    assert omega[-1] == pytest.approx(0.0, abs=1e-5)
    assert (omega >= -1e-6).all()
    # Plateau region should be constant (within the hold section)
    mid = len(omega) // 2
    assert omega[mid] > 0, "Omega plateau should be positive"
    assert omega[mid] == pytest.approx(omega[mid + 1], abs=1e-3)


@pytest.mark.parametrize("arch", [1, 2])
def test_policy_delta_bounds(arch):
    torch.manual_seed(0)
    cfg = _make_config(N_t=64, learn_omega=False, architecture=arch)
    policy = SchedulePolicy(cfg)
    graphs = _test_graphs()
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.sample_schedule(batch, deterministic=True)
    delta = out["delta"]
    assert (delta >= cfg.controls.delta_min - 1e-3).all()
    assert (delta <= cfg.controls.delta_max + 1e-3).all()


# ---------------------------------------------------------------------------
# Warm-start / residual parameterization tests
# ---------------------------------------------------------------------------

from module1.base import FixedScheduleBaseline


@pytest.mark.parametrize("arch", [1, 2])
def test_warm_start_omega_near_baseline_at_init(arch):
    """With warm_start, learned omega at init should closely match the baseline."""
    torch.manual_seed(0)
    cfg = _make_config(N_t=64, learn_omega=True, architecture=arch, warm_start=True)
    policy = SchedulePolicy(cfg)
    baseline = FixedScheduleBaseline(cfg)

    g = nx.path_graph(10)
    sched_learned = policy.make_schedule(g)
    sched_baseline = baseline.make_schedule(g)

    np.testing.assert_allclose(
        sched_learned.omega, sched_baseline.omega, atol=3.0,
        err_msg=f"Arch {arch}: warm-start omega deviates too far from baseline at init",
    )


@pytest.mark.parametrize("arch", [1, 2])
def test_warm_start_delta_closer_to_baseline_than_without(arch):
    """With warm_start, delta at init should be closer to the baseline than without."""
    torch.manual_seed(0)
    cfg_ws = _make_config(N_t=64, learn_omega=False, architecture=arch, warm_start=True)
    cfg_no = _make_config(N_t=64, learn_omega=False, architecture=arch, warm_start=False)
    policy_ws = SchedulePolicy(cfg_ws)
    # Use the same weights for a fair comparison
    policy_no = SchedulePolicy(cfg_no)
    policy_no.load_state_dict(policy_ws.state_dict(), strict=False)

    baseline = FixedScheduleBaseline(cfg_ws)

    g = nx.path_graph(10)
    delta_ws = policy_ws.make_schedule(g).delta
    delta_no = policy_no.make_schedule(g).delta
    delta_bl = baseline.make_schedule(g).delta

    err_ws = np.abs(delta_ws - delta_bl).mean()
    err_no = np.abs(delta_no - delta_bl).mean()
    assert err_ws < err_no, (
        f"Arch {arch}: warm-start delta (mean err={err_ws:.2f}) should be "
        f"closer to baseline than without (mean err={err_no:.2f})"
    )


@pytest.mark.parametrize("arch", [1, 2])
def test_warm_start_gradient_flow(arch):
    """Gradients still flow through the residual correction path."""
    torch.manual_seed(0)
    cfg = _make_config(N_t=32, learn_omega=True, architecture=arch, warm_start=True)
    policy = SchedulePolicy(cfg)
    graphs = _test_graphs()
    batch = Batch.from_data_list([graph_to_pyg(g) for g in graphs])
    out = policy.sample_schedule(batch)

    loss = out["omega"].sum() + out["delta"].sum()
    loss.backward()

    params_with_grad = {
        name for name, p in policy.named_parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    }
    assert any("encoder" in n for n in params_with_grad)
    assert any("omega_head" in n for n in params_with_grad)
    assert any("delta_head" in n for n in params_with_grad)


def test_warm_start_omega_unaffected_when_not_learned():
    """When learn_omega=False, warm_start should not alter the analytic omega."""
    torch.manual_seed(0)
    cfg_ws = _make_config(N_t=64, learn_omega=False, architecture=1, warm_start=True)
    cfg_no = _make_config(N_t=64, learn_omega=False, architecture=1, warm_start=False)
    policy_ws = SchedulePolicy(cfg_ws)
    policy_no = SchedulePolicy(cfg_no)

    g = nx.path_graph(10)
    omega_ws = policy_ws.make_schedule(g).omega
    omega_no = policy_no.make_schedule(g).omega
    np.testing.assert_allclose(omega_ws, omega_no, atol=1e-5,
                               err_msg="Warm-start should not change analytic omega")


def test_warm_start_physical_bounds_respected():
    """Warm-start outputs stay within physical bounds even with large corrections."""
    torch.manual_seed(42)
    cfg = _make_config(N_t=32, learn_omega=True, architecture=1, warm_start=True)
    policy = SchedulePolicy(cfg)

    # Force large network outputs by scaling weights
    with torch.no_grad():
        for p in policy.omega_head.parameters():
            p.mul_(10.0)
        for p in policy.delta_head.parameters():
            p.mul_(10.0)

    g = nx.path_graph(10)
    sched = policy.make_schedule(g)
    assert (sched.omega >= -1e-6).all(), "Omega went negative"
    assert (sched.omega <= cfg.controls.omega_max + 1e-3).all()
    assert (sched.delta >= cfg.controls.delta_min - 1e-3).all()
    assert (sched.delta <= cfg.controls.delta_max + 1e-3).all()
