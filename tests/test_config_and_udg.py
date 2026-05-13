import numpy as np
import networkx as nx

from config import ControlsConfig, UDGConfig, ProjectConfig, derive_omega_schedule, ParamKind
from graphs.unit_disk import generate_square_lattice_udg


def test_derive_omega_schedule_constant_and_scaled():
    # Use new (T, N_t) API; legacy dt implied via T / (N_t-1)
    controls = ControlsConfig(T=7e-6, N_t=8, param_kind=ParamKind.pwc, omega_scale=2.0, omega_cap=None)
    udg = UDGConfig(nx=4, ny=4, spacing=1.0, radius=2.0, dropout_rate=0.0, seed=123)
    omega = derive_omega_schedule(controls, udg)
    assert omega.shape == (controls.N_t,)
    # expected base = omega_scale / radius^6
    expected = controls.omega_scale / (udg.radius ** 6)
    assert np.allclose(omega, expected)


def test_generate_square_lattice_udg_deterministic_and_positions():
    cfg = UDGConfig(nx=5, ny=3, spacing=1.2, radius=1.5, dropout_rate=0.35, seed=999)
    G1, pos1 = generate_square_lattice_udg(cfg)
    G2, pos2 = generate_square_lattice_udg(cfg)

    # Deterministic with same seed
    assert nx.is_isomorphic(G1, G2)
    assert pos1 == pos2

    # Nodes are 0..N-1
    if G1.number_of_nodes() > 0:
        assert set(G1.nodes()) == set(range(G1.number_of_nodes()))

    # Edge lengths are within radius (physical distance = radius * spacing)
    r_phys = cfg.radius * cfg.spacing
    for u, v in G1.edges():
        x1, y1 = pos1[u]
        x2, y2 = pos1[v]
        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        assert d2 <= r_phys * r_phys + 1e-9


def test_dropout_fixed_count_across_seeds():
    # With fixed dropout rate, the number of dropped nodes is fixed (round(p*N))
    base = UDGConfig(nx=5, ny=5, spacing=1.0, radius=1.2, dropout_rate=0.2, seed=0)
    N_total = base.nx * base.ny
    k_drop = int(round(base.dropout_rate * N_total))
    expected_nodes = N_total - k_drop

    seeds = [0, 1, 2, 42, 99]
    counts = []
    for s in seeds:
        cfg = UDGConfig(
            nx=base.nx,
            ny=base.ny,
            spacing=base.spacing,
            radius=base.radius,
            dropout_rate=base.dropout_rate,
            seed=s,
        )
        G, pos = generate_square_lattice_udg(cfg)
        counts.append(G.number_of_nodes())

    assert all(c == expected_nodes for c in counts)
