"""Tests for Module 3: training loop, learner, checkpoint, evaluation."""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx
import pytest
import torch

from config import load_project_config_json, RewardConfig
from schedules import GlobalSchedule
from module1.policy import SchedulePolicy
from module1.base import FixedScheduleBaseline
from module2.interfaces import BackendResult
from module3.interfaces import TrainingConfig
from module3.learner import ReinforceLearner, _build_graph_pool
from module3.reinforce import reinforce_step
from module3.backend_adapter import _is_cost_from_counts, _raw_reward as _compute_reward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

def _project_config():
    return load_project_config_json(
        ROOT / "config.json", ROOT / "hardware_specs.json"
    )

def _mock_backend_fn(G: nx.Graph, schedule: GlobalSchedule) -> float:
    """Deterministic-ish mock: reward proportional to graph density."""
    return nx.density(G) * 0.5 + 0.1


def _random_backend_fn(G: nx.Graph, schedule: GlobalSchedule) -> float:
    return random.random() * 0.3


# ---------------------------------------------------------------------------
# Graph pool
# ---------------------------------------------------------------------------

def test_build_graph_pool():
    cfg = _project_config()
    pool = _build_graph_pool(cfg.udg, pool_size=8, base_seed=0)
    assert len(pool) >= 1
    for G in pool:
        assert G.number_of_edges() > 0
        assert "positions" in G.graph
        pos = G.graph["positions"]
        assert len(pos) == G.number_of_nodes()


# ---------------------------------------------------------------------------
# FixedScheduleBaseline
# ---------------------------------------------------------------------------

def test_fixed_baseline_produces_schedule():
    cfg = _project_config()
    baseline = FixedScheduleBaseline(cfg)
    g1 = nx.path_graph(10)
    g2 = nx.cycle_graph(8)
    s1 = baseline.make_schedule(g1)
    s2 = baseline.make_schedule(g2)
    assert s1.omega.shape == (cfg.controls.N_t,)
    np.testing.assert_array_equal(s1.omega, s2.omega)
    np.testing.assert_array_equal(s1.delta, s2.delta)


def test_fixed_baseline_delta_is_linear_sweep():
    cfg = _project_config()
    baseline = FixedScheduleBaseline(cfg)
    s = baseline.make_schedule(nx.path_graph(5))
    assert s.delta[0] < 0, "Delta should start negative"
    assert s.delta[-1] > 0, "Delta should end positive"
    diffs = np.diff(s.delta)
    assert np.all(diffs > 0), "Delta should be monotonically increasing"


# ---------------------------------------------------------------------------
# ReinforceLearner instantiation
# ---------------------------------------------------------------------------

def test_learner_instantiation():
    cfg = _project_config()
    train_cfg = TrainingConfig(
        total_steps=10, batch_size=2, graph_pool_size=4,
        eval_graphs=2, seed=42,
    )
    learner = ReinforceLearner(
        config=train_cfg, project=cfg,
        backend_fn=_mock_backend_fn,
    )
    assert learner.policy is not None
    assert len(learner.train_pool) >= 1
    assert len(learner.eval_pool) >= 1
    assert isinstance(learner.policy, SchedulePolicy)


# ---------------------------------------------------------------------------
# Single train step
# ---------------------------------------------------------------------------

def test_single_train_step():
    cfg = _project_config()
    train_cfg = TrainingConfig(
        total_steps=5, batch_size=2, graph_pool_size=4,
        eval_graphs=2, seed=42,
    )
    learner = ReinforceLearner(
        config=train_cfg, project=cfg,
        backend_fn=_mock_backend_fn,
    )
    batch = learner.select_batch(learner.train_pool)
    assert len(batch) == 2

    metrics = learner.train_step(batch)
    assert "loss" in metrics
    assert "mean_reward" in metrics
    assert "grad_norm" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert metrics["step"] == 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluation():
    cfg = _project_config()
    train_cfg = TrainingConfig(
        total_steps=5, batch_size=2, graph_pool_size=4,
        eval_graphs=2, seed=42,
    )
    learner = ReinforceLearner(
        config=train_cfg, project=cfg,
        backend_fn=_mock_backend_fn,
    )
    eval_metrics = learner.evaluate(learner.eval_pool)
    assert "eval_mean_reward" in eval_metrics
    assert "baseline_mean_reward" in eval_metrics
    assert "improvement" in eval_metrics
    assert "n_better" in eval_metrics
    assert eval_metrics["n_graphs"] == len(learner.eval_pool)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_save_load():
    cfg = _project_config()
    train_cfg = TrainingConfig(
        total_steps=5, batch_size=2, graph_pool_size=4,
        eval_graphs=2, seed=42,
    )
    learner = ReinforceLearner(
        config=train_cfg, project=cfg,
        backend_fn=_mock_backend_fn,
    )

    batch = learner.select_batch(learner.train_pool)
    learner.train_step(batch)
    learner.train_step(batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "test_ckpt.pt")
        learner.save_checkpoint(ckpt_path)

        learner2 = ReinforceLearner(
            config=train_cfg, project=cfg,
            backend_fn=_mock_backend_fn,
        )
        learner2.load_checkpoint(ckpt_path)

        assert learner2.step_count == 2
        for p1, p2 in zip(
            learner.policy.parameters(), learner2.policy.parameters()
        ):
            torch.testing.assert_close(p1, p2)


# ---------------------------------------------------------------------------
# reinforce_step directly
# ---------------------------------------------------------------------------

def test_reinforce_step_standalone():
    cfg = _project_config()
    policy = SchedulePolicy(cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    baseline = {}

    pool = _build_graph_pool(cfg.udg, pool_size=4, base_seed=99)
    graphs = pool[:2]

    metrics = reinforce_step(
        policy=policy,
        graphs=graphs,
        backend_fn=_random_backend_fn,
        optimizer=optimizer,
        baseline_ema=baseline,
    )
    assert metrics["mean_reward"] >= 0
    assert "grad_norm" in metrics
    assert metrics["grad_norm"] >= 0


# ---------------------------------------------------------------------------
# SchedulePolicy is also a ScheduleModel
# ---------------------------------------------------------------------------

def test_policy_is_schedule_model():
    from module1.base import ScheduleModel
    cfg = _project_config()
    policy = SchedulePolicy(cfg)
    assert isinstance(policy, ScheduleModel)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _triangle_graph():
    """3-node triangle: MIS size = 1, any pair violates."""
    G = nx.cycle_graph(3)
    return G


def test_is_cost_no_violations():
    """Single selected node, no edge violations → cost = 1 / n_nodes."""
    G = _triangle_graph()
    counts = {"rgg": 5, "grg": 3, "ggr": 2}
    cost = _is_cost_from_counts(counts, G, penalty_U=3.0, normalize_by_nodes=True)
    assert cost == pytest.approx(1.0 / 3.0)


def test_is_cost_with_violations():
    """Two adjacent selected → cost = (2 - U) per violating shot."""
    G = _triangle_graph()
    counts = {"rrg": 10}
    cost = _is_cost_from_counts(counts, G, penalty_U=3.0, normalize_by_nodes=False)
    assert cost == pytest.approx(2.0 - 3.0)


def test_is_cost_all_ground():
    """All ground state → cost = 0."""
    G = _triangle_graph()
    counts = {"ggg": 100}
    cost = _is_cost_from_counts(counts, G, penalty_U=3.0, normalize_by_nodes=False)
    assert cost == 0.0


def test_is_cost_mixed_shots():
    """Mix of valid and violating shots."""
    G = nx.path_graph(3)  # edges: 0-1, 1-2
    counts = {
        "rgr": 6,  # nodes 0,2 selected, no edge between them → cost = 2
        "rrg": 4,  # nodes 0,1 selected, edge 0-1 violated → cost = 2 - U
    }
    U = 3.0
    expected = (6 * 2.0 + 4 * (2.0 - U)) / 10.0 / 3.0  # normalized by 3 nodes
    cost = _is_cost_from_counts(counts, G, penalty_U=U, normalize_by_nodes=True)
    assert cost == pytest.approx(expected)


def test_compute_reward_p_mis():
    """p_mis reward kind returns result.p_mis directly."""
    result = BackendResult(p_mis=0.42, shots=100, counts={"rgr": 42, "ggg": 58})
    G = nx.path_graph(3)
    cfg = RewardConfig(kind="p_mis")
    assert _compute_reward(result, G, cfg) == pytest.approx(0.42)


def test_compute_reward_is_cost():
    """is_cost reward kind computes from counts."""
    G = nx.path_graph(3)  # edges: 0-1, 1-2
    counts = {"rgr": 10}  # IS of size 2, no violations
    result = BackendResult(p_mis=0.5, shots=10, counts=counts)
    cfg = RewardConfig(kind="is_cost", penalty_U=3.0, normalize_by_nodes=True)
    reward = _compute_reward(result, G, cfg)
    assert reward == pytest.approx(2.0 / 3.0)


def test_compute_reward_composite():
    """composite blends is_cost and p_mis."""
    G = nx.path_graph(3)
    counts = {"rgr": 10}
    result = BackendResult(p_mis=0.5, shots=10, counts=counts)
    cfg = RewardConfig(kind="composite", penalty_U=3.0, mis_bonus=0.3,
                       normalize_by_nodes=True)
    r_cost = 2.0 / 3.0
    r_mis = 0.5
    expected = 0.7 * r_cost + 0.3 * r_mis
    reward = _compute_reward(result, G, cfg)
    assert reward == pytest.approx(expected)


def test_compute_reward_unknown_kind_raises():
    result = BackendResult(p_mis=0.1, shots=10)
    G = nx.path_graph(3)
    cfg = RewardConfig(kind="invalid")
    with pytest.raises(ValueError, match="Unknown reward kind"):
        _compute_reward(result, G, cfg)


def test_reward_config_defaults():
    """RewardConfig defaults are is_cost_vs_baseline, U=3, normalized."""
    cfg = RewardConfig()
    assert cfg.kind == "is_cost_vs_baseline"
    assert cfg.penalty_U == 3.0
    assert cfg.normalize_by_nodes is True
    assert cfg.mis_bonus == 0.0
    assert cfg.baseline_norm_eps > 0.0


# ---------------------------------------------------------------------------
# Run as standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"Running {t.__name__} ... ", end="", flush=True)
        try:
            t()
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
