"""Minimal integration tests: Graph → Module 1 → Schedule → Module 2 → Report.

Designed to be cheap: uses the smallest possible graph (3-4 atoms) and
very few shots (5-10) so each test completes in seconds.
"""
from __future__ import annotations

import sys
import numpy as np
import networkx as nx

try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, ".")

from config import (
    ControlsConfig,
    HardwareSpecs,
    ParamKind,
    ProjectConfig,
    UDGConfig,
    compute_blockade_omega,
    load_project_config_json,
)
from graphs.unit_disk import generate_square_lattice_udg
from module1.policy import SchedulePolicy
from schedules import GlobalSchedule
from module2.interfaces import BackendResult, Positions


# ── helpers ─────────────────────────────────────────────────────────────

def _tiny_config(arch: int = 1, learn_omega: bool = False) -> ProjectConfig:
    """3x3 lattice, small N_t, minimal settings for fast tests."""
    return ProjectConfig(
        backend="bloqade",
        controls=ControlsConfig(
            T=4.0e-6,
            N_t=10,
            param_kind=ParamKind.pwl,
            learn_omega=learn_omega,
            architecture=arch,
            omega_max=15.8,
            delta_min=-25.0,
            delta_max=25.0,
            n_delta_knots=4,
            n_omega_modes=3,
            n_delta_modes=4,
        ),
        udg=UDGConfig(nx=3, ny=3, spacing=4.0, radius=1.5, dropout_rate=0.2, seed=42),
        hardware=HardwareSpecs(),
    )


def _make_graph_and_positions(cfg: ProjectConfig):
    """Generate a small graph + positions from config."""
    G, pos = generate_square_lattice_udg(cfg.udg)
    assert G.number_of_nodes() >= 2, "Need at least 2 atoms"
    return G, pos


# ── Test 1: Module 1 produces a valid GlobalSchedule ────────────────────

def test_module1_produces_valid_schedule():
    cfg = _tiny_config()
    G, pos = _make_graph_and_positions(cfg)
    policy = SchedulePolicy(cfg)
    sched = policy.make_schedule(G)

    assert isinstance(sched, GlobalSchedule)
    sched.validate_shapes()
    assert sched.omega.shape == (cfg.controls.N_t,)
    assert sched.delta.shape == (cfg.controls.N_t,)
    assert sched.dt > 0
    assert np.all(sched.omega >= -1e-6), "Omega must be non-negative"
    print(f"[PASS] Module 1 schedule: omega [{sched.omega.min():.2f}, {sched.omega.max():.2f}], "
          f"delta [{sched.delta.min():.2f}, {sched.delta.max():.2f}] rad/μs")


# ── Test 2: Schedule → Braket TimeSeries conversion ─────────────────────

def test_schedule_to_timeseries_conversion():
    from module2.braket_backend import _schedule_to_braket_timeseries

    omega_rad_us = np.array([0.0, 5.0, 15.8, 15.8, 5.0, 0.0])
    dt_s = 1e-6
    ts = _schedule_to_braket_timeseries(omega_rad_us, dt_s)

    times = ts.times()
    values = ts.values()
    assert len(times) == 6
    assert abs(times[0]) < 1e-12
    assert abs(times[-1] - 5e-6) < 1e-12
    assert abs(values[2] - 15.8e6) < 1.0, "Should be 15.8 Mrad/s"
    print(f"[PASS] TimeSeries conversion: 6 knots, t=[0, {times[-1]*1e6:.1f}] μs")


# ── Test 3: Hardware validation catches violations ──────────────────────

def test_validation_catches_violations():
    from module2.braket_backend import _validate_program

    positions_m = [(0, 0), (1e-6, 0)]  # 1 μm apart — below 4 μm minimum
    omega = np.array([0.0, 15.8e6, 0.0])
    delta = np.array([0.0, 0.0, 0.0])
    times = np.array([0.0, 2e-6, 4e-6])

    try:
        _validate_program(positions_m, omega, delta, times, 4e-6)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "spacing" in str(e).lower(), f"Unexpected error message: {e}"
    print("[PASS] Validation correctly rejects too-close atoms")


# ── Test 4: Full pipeline — Graph → Module 1 → Module 2 → BackendResult

def test_full_pipeline_end_to_end():
    """The main integration test: graph → schedule → Braket sim → p_MIS."""
    from module2.braket_backend import BraketBackend

    cfg = _tiny_config()
    G, pos = _make_graph_and_positions(cfg)

    policy = SchedulePolicy(cfg)
    sched = policy.make_schedule(G)

    backend = BraketBackend(cfg, n_shots=5, backend_type="simulator")
    result = backend.estimate_p_mis(sched, G, pos)

    assert isinstance(result, BackendResult)
    assert 0.0 <= result.p_mis <= 1.0
    assert result.shots == 5
    assert result.std_err is not None
    assert result.counts is not None
    assert len(result.counts) > 0

    print(f"[PASS] Full pipeline: {G.number_of_nodes()} atoms, 5 shots → "
          f"p_MIS={result.p_mis:.2f} ± {result.std_err:.2f}")
    print(f"       Counts: {result.counts}")


# ── Test 5: Both architectures work through the pipeline ────────────────

def test_pipeline_both_architectures(arch):
    from module2.braket_backend import BraketBackend

    cfg = _tiny_config(arch=arch)
    G, pos = _make_graph_and_positions(cfg)

    policy = SchedulePolicy(cfg)
    sched = policy.make_schedule(G)
    sched.validate_shapes()

    backend = BraketBackend(cfg, n_shots=5, backend_type="simulator")
    result = backend.estimate_p_mis(sched, G, pos)

    assert 0.0 <= result.p_mis <= 1.0
    print(f"[PASS] Arch {arch}: p_MIS={result.p_mis:.2f}")


# ── Run directly ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Integration tests: Graph → Module 1 → Module 2 → Report")
    print("=" * 60)
    print()

    test_module1_produces_valid_schedule()
    print()

    test_schedule_to_timeseries_conversion()
    print()

    test_validation_catches_violations()
    print()

    print("Running full pipeline (Braket LocalSimulator, 5 shots)…")
    test_full_pipeline_end_to_end()
    print()

    for a in [1, 2]:
        print(f"Running Arch {a} pipeline…")
        test_pipeline_both_architectures(a)
    print()

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
