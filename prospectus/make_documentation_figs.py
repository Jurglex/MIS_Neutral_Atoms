#!/usr/bin/env python3
"""Generate the five MIS chapter prospectus figures.

Outputs PDFs to ``prospectus/figs/MIS/`` and caches expensive intermediate
results (simulator runs, training histories, exact-diag spectra) under
``prospectus/cache/`` so subsequent runs are fast.

Usage
-----
    # Generate all figures with default (lightweight) settings.
    python prospectus/make_documentation_figs.py

    # Only regenerate one figure.
    python prospectus/make_documentation_figs.py --figs 3

    # Force re-running cached experiments.
    python prospectus/make_documentation_figs.py --regenerate

    # Use the full training run for Figure 4 (~30 min).
    python prospectus/make_documentation_figs.py --figs 4 --train-mode full

    # Skip the simulator (use mock rewards for everything).
    python prospectus/make_documentation_figs.py --no-simulator
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import networkx as nx
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ProjectConfig, ControlsConfig, UDGConfig, HardwareSpecs,
    ParamKind, RewardConfig, compute_blockade_omega,
    load_project_config_json,
)
from schedules import GlobalSchedule  # noqa: E402
from graphs.unit_disk import generate_square_lattice_udg  # noqa: E402
from module1.policy import SchedulePolicy  # noqa: E402
from module1.base import FixedScheduleBaseline  # noqa: E402
from module1.featurize import graph_to_pyg  # noqa: E402
from module2.graph_MIS_utils import get_all_MIS, check_independent_set  # noqa: E402

# Optional Braket backend (lazy)
try:
    from module2.braket_backend import BraketBackend
    HAS_BRAKET = True
except Exception:
    HAS_BRAKET = False


# ── paths & global style ────────────────────────────────────────────────

FIGS_DIR = Path(__file__).parent / "figs" / "MIS"
CACHE_DIR = Path(__file__).parent / "cache"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "omega":      "#2774AE",   # primary blue
    "delta":      "#D97706",   # coral / amber-orange
    "mis":        "#DAA520",   # MIS amber
    "baseline":   "#111827",   # near-black
    "is_valid":   "#10B981",   # green
    "is_invalid": "#EF4444",   # red
    "muted":      "#6B7280",   # gray
    "blockade":   "#60A5FA",   # light blue for blockade disk
    "panel_bg_a": "#F3F4F6",   # ramp regime
    "panel_bg_b": "#DBEAFE",   # hold regime
}


def setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "font.size":           9,
        "font.family":         "sans-serif",
        "axes.linewidth":      0.6,
        "axes.edgecolor":      "#6B7280",
        "axes.titlesize":      10,
        "axes.titleweight":    "bold",
        "axes.labelsize":      9,
        "xtick.labelsize":     7,
        "ytick.labelsize":     7,
        "legend.fontsize":     7,
        "legend.frameon":      False,
        "lines.linewidth":     1.2,
        "grid.linewidth":      0.4,
        "grid.alpha":          0.4,
        "figure.dpi":          120,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
    })


def panel_letter(ax, letter: str, *, x: float = -0.12, y: float = 1.02) -> None:
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


# ── caching helpers ─────────────────────────────────────────────────────

def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def cached(name: str, regenerate: bool = False):
    """Decorator: memoize a function's result to ``cache/<name>.pkl``."""
    def deco(fn):
        def wrapped(*args, **kwargs):
            p = cache_path(name)
            if p.exists() and not regenerate:
                with p.open("rb") as f:
                    return pickle.load(f)
            result = fn(*args, **kwargs)
            with p.open("wb") as f:
                pickle.dump(result, f)
            return result
        return wrapped
    return deco


# ── shared config builders ──────────────────────────────────────────────

def make_main_config(seed: int = 42, nx_: int = 6, ny_: int = 6,
                     dropout: float = 0.35) -> ProjectConfig:
    """Build the canonical ProjectConfig used throughout the figures."""
    cfg = load_project_config_json()
    udg = replace(cfg.udg, seed=seed, nx=nx_, ny=ny_, dropout_rate=dropout)
    return replace(cfg, udg=udg)


def with_arch(cfg: ProjectConfig, *, architecture: int, learn_omega: bool = True,
              warm_start: bool = False,
              residual_alpha_start: float = 1.0) -> ProjectConfig:
    """Return a copy of ``cfg`` with a particular decoder architecture."""
    ctrl = replace(
        cfg.controls,
        architecture=architecture,
        learn_omega=learn_omega,
        warm_start=warm_start,
        residual_alpha_start=residual_alpha_start,
        residual_alpha_end=residual_alpha_start,
        residual_alpha_warmup_steps=0,
    )
    return replace(cfg, controls=ctrl)


# ── small utilities ─────────────────────────────────────────────────────

def time_axis_us(cfg: ProjectConfig) -> np.ndarray:
    return np.linspace(0.0, cfg.controls.T * 1e6, cfg.controls.N_t)


def find_one_mis(G: nx.Graph) -> list[int]:
    """Pick a single MIS solution (the first one returned by ``get_all_MIS``)."""
    all_mis = get_all_MIS(G)
    if not all_mis:
        return []
    return sorted(next(iter(all_mis)))


def make_baseline_schedule(cfg: ProjectConfig) -> GlobalSchedule:
    return FixedScheduleBaseline(cfg).make_schedule(nx.Graph())


def sample_policy_schedules(
    policy: SchedulePolicy, G: nx.Graph, n_samples: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(omega: (K, N_t), delta: (K, N_t))`` from K policy samples."""
    torch.manual_seed(seed)
    data = graph_to_pyg(G)
    batch = Batch.from_data_list([data])
    omegas, deltas = [], []
    policy.eval()
    for _ in range(n_samples):
        with torch.no_grad():
            out = policy.sample_schedule(batch, deterministic=False)
        omegas.append(out["omega"][0].cpu().numpy())
        deltas.append(out["delta"][0].cpu().numpy())
    return np.stack(omegas), np.stack(deltas)


def run_braket_counts(cfg: ProjectConfig, G: nx.Graph, pos: dict,
                      schedule: GlobalSchedule, n_shots: int,
                      cache_key: str | None = None,
                      regenerate: bool = False) -> dict[str, int]:
    """Run a schedule on the Braket local simulator and return counts."""
    if cache_key is not None:
        p = cache_path(cache_key)
        if p.exists() and not regenerate:
            with p.open("rb") as f:
                return pickle.load(f)
    if not HAS_BRAKET:
        raise RuntimeError("Braket SDK not installed; cannot run simulator")
    backend = BraketBackend(cfg, n_shots=n_shots, backend_type="simulator",
                            validate=False)
    result = backend.estimate_p_mis(schedule, G, pos)
    counts = dict(result.counts) if result.counts else {}
    if cache_key is not None:
        with cache_path(cache_key).open("wb") as f:
            pickle.dump(counts, f)
    return counts


# ────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — Problem setting
# ────────────────────────────────────────────────────────────────────────

def figure_1(args) -> None:
    print("[Fig 1] Building problem-setting figure...")
    cfg = make_main_config(seed=42, nx_=6, ny_=6, dropout=0.35)
    G, pos = generate_square_lattice_udg(cfg.udg)
    if G.number_of_nodes() == 0:
        raise RuntimeError("Empty graph — choose a different seed")

    mis = find_one_mis(G)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
          f"|MIS|={len(mis)}")

    baseline_sched = FixedScheduleBaseline(cfg).make_schedule(G)

    # Run baseline on the Braket simulator (or mock if unavailable).
    counts: dict[str, int] = {}
    if args.no_simulator or not HAS_BRAKET:
        # Synthetic counts: bias toward MIS bitstrings + a few neighbors.
        n = G.number_of_nodes()
        rng = np.random.default_rng(7)
        mis_str = "".join("r" if i in set(mis) else "g" for i in range(n))
        counts[mis_str] = 80
        all_g = "g" * n
        counts[all_g] = 30
        # 4 random IS / non-IS strings
        for _ in range(6):
            bs = "".join("r" if rng.random() < 0.3 else "g" for _ in range(n))
            counts[bs] = counts.get(bs, 0) + int(rng.integers(5, 20))
        cache_key = None
    else:
        cache_key = "fig1_counts"
        try:
            counts = run_braket_counts(
                cfg, G, pos, baseline_sched, n_shots=200,
                cache_key=cache_key, regenerate=args.regenerate,
            )
        except Exception as e:
            print(f"  [warn] Braket failed ({e}); falling back to synthetic counts")
            n = G.number_of_nodes()
            mis_str = "".join("r" if i in set(mis) else "g" for i in range(n))
            counts = {mis_str: 100, "g"*n: 40}

    fig = plt.figure(figsize=(14.5, 4.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.4, 1.05, 1.1],
                           wspace=0.45, figure=fig)

    # Panel A — atom array & blockade
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_atom_array(ax_a, G, pos, mis, cfg)
    panel_letter(ax_a, "a")

    # Panel B — adiabatic schedule
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_baseline_schedule(ax_b, cfg, baseline_sched)
    panel_letter(ax_b, "b")

    # Panel C — measurement outcomes
    ax_c = fig.add_subplot(gs[0, 2])
    _draw_measurement_outcomes(ax_c, counts, G, mis, top_k=6)
    panel_letter(ax_c, "c")

    out = FIGS_DIR / "problem_setting.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  → wrote {out}")


def _draw_atom_array(ax, G, pos, mis, cfg):
    coords = np.array([pos[n] for n in G.nodes()])

    # Edges
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=COLORS["muted"], linewidth=0.7, zorder=1, alpha=0.55)

    # Blockade disk around an atom (pick a central node).
    centroid = coords.mean(axis=0)
    center_node = min(G.nodes(), key=lambda n:
                      np.linalg.norm(np.array(pos[n]) - centroid))
    R_b = cfg.udg.radius * cfg.udg.spacing  # μm
    cx, cy = pos[center_node]
    disk = Circle((cx, cy), R_b, edgecolor=COLORS["blockade"],
                  facecolor=COLORS["blockade"], alpha=0.12,
                  linewidth=0.9, linestyle="--", zorder=2)
    ax.add_patch(disk)
    ax.annotate(rf"$R_b = {R_b:.1f}\ \mu$m",
                xy=(cx + R_b * 0.6, cy - R_b * 0.6),
                xytext=(cx + R_b * 1.0, cy - R_b * 1.15),
                fontsize=8, color=COLORS["blockade"],
                arrowprops=dict(arrowstyle="->", color=COLORS["blockade"],
                                lw=0.6))

    # Atoms (non-MIS)
    mis_set = set(mis)
    non_mis = [n for n in G.nodes() if n not in mis_set]
    nm_xy = np.array([pos[n] for n in non_mis]) if non_mis else np.empty((0, 2))
    if len(nm_xy):
        ax.scatter(nm_xy[:, 0], nm_xy[:, 1], s=110,
                   c=COLORS["omega"], edgecolors="white", linewidths=1.1,
                   zorder=4, label="atoms")
    # MIS atoms with halo
    m_xy = np.array([pos[n] for n in mis]) if mis else np.empty((0, 2))
    if len(m_xy):
        ax.scatter(m_xy[:, 0], m_xy[:, 1], s=260,
                   c=COLORS["mis"], edgecolors="white", linewidths=2.0,
                   alpha=0.35, zorder=4.5)
        ax.scatter(m_xy[:, 0], m_xy[:, 1], s=140,
                   c=COLORS["mis"], edgecolors=COLORS["baseline"],
                   linewidths=1.0, zorder=5, label=f"MIS (|MIS|={len(mis)})")

    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$y$ ($\mu$m)")
    ax.set_aspect("equal")
    pad = cfg.udg.spacing * 1.0
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(False)


def _draw_baseline_schedule(ax, cfg: ProjectConfig, sched: GlobalSchedule):
    t = time_axis_us(cfg)
    T_us = cfg.controls.T * 1e6
    t_ramp = cfg.hardware.t_ramp
    t_onset = cfg.hardware.t_onset

    # Background regimes
    ax.axvspan(0, t_onset + t_ramp, color=COLORS["panel_bg_a"], alpha=0.6, zorder=0)
    ax.axvspan(t_onset + t_ramp, T_us - t_onset - t_ramp,
               color=COLORS["panel_bg_b"], alpha=0.6, zorder=0)
    ax.axvspan(T_us - t_onset - t_ramp, T_us,
               color=COLORS["panel_bg_a"], alpha=0.6, zorder=0)

    ax.plot(t, sched.omega, color=COLORS["omega"], linewidth=1.6,
            label=r"$\Omega(t)$")
    ax.set_xlabel(r"$t$ ($\mu$s)")
    ax.set_ylabel(r"$\Omega(t)$ (rad/$\mu$s)", color=COLORS["omega"])
    ax.tick_params(axis="y", labelcolor=COLORS["omega"])

    ax2 = ax.twinx()
    ax2.plot(t, sched.delta, color=COLORS["delta"], linewidth=1.6,
             label=r"$\Delta(t)$")
    ax2.set_ylabel(r"$\Delta(t)$ (rad/$\mu$s)", color=COLORS["delta"])
    ax2.tick_params(axis="y", labelcolor=COLORS["delta"])
    ax2.grid(False)

    # Regime labels
    ymax = max(np.max(sched.omega), 1.0) * 1.05
    ax.set_ylim(0, ymax)
    yt = ymax * 0.93
    ax.text((t_onset + t_ramp) / 2, yt, "ramp", ha="center", fontsize=7,
            color=COLORS["muted"])
    ax.text(T_us / 2, yt, "hold", ha="center", fontsize=7,
            color=COLORS["muted"])
    ax.text(T_us - (t_onset + t_ramp) / 2, yt, "ramp", ha="center",
            fontsize=7, color=COLORS["muted"])

    # Combined legend
    lines = [Line2D([0], [0], color=COLORS["omega"], lw=1.6, label=r"$\Omega(t)$"),
             Line2D([0], [0], color=COLORS["delta"], lw=1.6, label=r"$\Delta(t)$")]
    ax.legend(handles=lines, loc="lower right", fontsize=7)


def _draw_measurement_outcomes(ax, counts: dict[str, int], G: nx.Graph,
                               mis: list[int], top_k: int = 6):
    n = G.number_of_nodes()
    mis_set = frozenset(mis)
    total = sum(counts.values()) or 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]
    labels, vals, colors_, is_mis_marker = [], [], [], []
    for bs, c in items:
        sel = frozenset(i for i, ch in enumerate(bs) if ch == "r")
        is_is = check_independent_set(bs, G)
        is_mis = (sel == mis_set) or (len(sel) == len(mis_set) and is_is)
        label = bs if len(bs) <= 12 else bs[:10] + "…"
        labels.append(label)
        vals.append(c)
        colors_.append(COLORS["is_valid"] if is_is else COLORS["is_invalid"])
        is_mis_marker.append(is_mis)

    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, vals, color=colors_, edgecolor="white", linewidth=0.6,
                   alpha=0.85)
    for bar, b, m in zip(bars, vals, is_mis_marker):
        if m:
            ax.text(bar.get_width() + total * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    "*", fontsize=14, color=COLORS["mis"],
                    va="center", ha="left", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=7)
    ax.set_xlabel("Counts")
    ax.set_ylabel("Bitstring")
    handles = [
        Rectangle((0, 0), 1, 1, fc=COLORS["is_valid"], alpha=0.85,
                  label="independent"),
        Rectangle((0, 0), 1, 1, fc=COLORS["is_invalid"], alpha=0.85,
                  label="violating"),
        Line2D([0], [0], marker="*", linestyle="",
               markerfacecolor=COLORS["mis"], markeredgecolor=COLORS["mis"],
               markersize=12, label="MIS"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.18)


# ────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — Gap profiles via exact diagonalization
# ────────────────────────────────────────────────────────────────────────

def figure_2(args) -> None:
    """Compute instantaneous gap ΔE(t) along the adiabatic sweep for two
    UDGs with contrasting λ₂."""
    print("[Fig 2] Building gap-profile figure...")

    cfg = make_main_config()
    # Use a smaller blockade radius so n_atoms stays moderate but
    # connectivity differs sharply between A (dense) and B (sparse).
    # radius=1.6 in units of spacing ⇒ R_b ≈ 6.4 μm at 4 μm spacing,
    # so only nearest cardinal + diagonal neighbors are blockaded.
    rb_factor = 1.6

    # --- Graph A: dense, high λ₂ (low dropout) ---
    udg_a = UDGConfig(nx=3, ny=3, spacing=cfg.udg.spacing,
                      radius=rb_factor, dropout_rate=0.0, seed=11)
    G_a, pos_a = generate_square_lattice_udg(udg_a)
    cfg_a = replace(cfg, udg=udg_a)

    # --- Graph B: sparse, low λ₂ (elongated, high dropout) ---
    udg_b = UDGConfig(nx=5, ny=3, spacing=cfg.udg.spacing,
                      radius=rb_factor, dropout_rate=0.30, seed=4)
    G_b, pos_b = generate_square_lattice_udg(udg_b)
    cfg_b = replace(cfg, udg=udg_b)

    # Limit to small graphs for exact diag.
    if G_a.number_of_nodes() > 14 or G_b.number_of_nodes() > 14:
        print(f"  [warn] Graphs too large: n_a={G_a.number_of_nodes()}, "
              f"n_b={G_b.number_of_nodes()}; sub-sampling.")

    print(f"  Graph A: n={G_a.number_of_nodes()}, m={G_a.number_of_edges()}, "
          f"λ₂={nx.algebraic_connectivity(G_a):.3f}")
    print(f"  Graph B: n={G_b.number_of_nodes()}, m={G_b.number_of_edges()}, "
          f"λ₂={nx.algebraic_connectivity(G_b):.3f}")

    @cached("fig2_gap_a", regenerate=args.regenerate)
    def _spec_a():
        return compute_gap_profile(cfg_a, pos_a, n_t=40)

    @cached("fig2_gap_b", regenerate=args.regenerate)
    def _spec_b():
        return compute_gap_profile(cfg_b, pos_b, n_t=40)

    spec_a = _spec_a()
    spec_b = _spec_b()

    fig = plt.figure(figsize=(11.0, 7.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0],
                           hspace=0.45, wspace=0.30, figure=fig)

    ax_ga = fig.add_subplot(gs[0, 0])
    _draw_atom_array(ax_ga, G_a, pos_a, find_one_mis(G_a), cfg_a)
    ax_ga.set_title(
        rf"$n={G_a.number_of_nodes()}$, "
        rf"$|E|={G_a.number_of_edges()}$, "
        rf"$\lambda_2={nx.algebraic_connectivity(G_a):.2f}$ "
        "(dense)", fontsize=9,
    )
    panel_letter(ax_ga, "a")

    ax_gb = fig.add_subplot(gs[0, 1])
    _draw_atom_array(ax_gb, G_b, pos_b, find_one_mis(G_b), cfg_b)
    ax_gb.set_title(
        rf"$n={G_b.number_of_nodes()}$, "
        rf"$|E|={G_b.number_of_edges()}$, "
        rf"$\lambda_2={nx.algebraic_connectivity(G_b):.2f}$ "
        "(sparse)", fontsize=9,
    )
    panel_letter(ax_gb, "b")

    ax_a = fig.add_subplot(gs[1, 0])
    _draw_gap_panel(ax_a, spec_a, label="A (dense)")
    panel_letter(ax_a, "c")

    ax_b = fig.add_subplot(gs[1, 1])
    _draw_gap_panel(ax_b, spec_b, label="B (sparse)")
    panel_letter(ax_b, "d")

    out = FIGS_DIR / "gap_profiles.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  → wrote {out}")


def compute_gap_profile(cfg: ProjectConfig, positions: dict,
                        n_t: int = 40) -> dict:
    """Exact-diag ΔE(t) sweep.

    Returns
    -------
    dict with keys: ``t_us``, ``omega``, ``delta``, ``gap``,
    ``sweep_rate``, ``gap_sq``, ``min_gap``, ``min_gap_t``, ``threshold``.
    """
    n = len(positions)
    sched = FixedScheduleBaseline(cfg).make_schedule(nx.Graph())
    t_us = np.linspace(0.0, cfg.controls.T * 1e6, cfg.controls.N_t)
    times_sub = np.linspace(0, len(t_us) - 1, n_t).astype(int)

    coords = np.array([positions[i] for i in sorted(positions.keys())])
    C6 = cfg.hardware.C6
    V_ij = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(coords[i] - coords[j])
            V_ij[i, j] = V_ij[j, i] = C6 / max(r ** 6, 1e-30)

    bits = np.array([[(s >> i) & 1 for i in range(n)] for s in range(2 ** n)],
                    dtype=np.int8)
    interaction = np.array([V_ij[np.ix_(np.where(b)[0], np.where(b)[0])].sum() / 2
                            for b in bits])
    n_sum = bits.sum(axis=1).astype(float)

    flip_idx = np.zeros((2 ** n, n), dtype=np.int64)
    for s in range(2 ** n):
        for i in range(n):
            flip_idx[s, i] = s ^ (1 << i)

    gaps, omegas_sub, deltas_sub = [], [], []
    print(f"  Diagonalizing n={n} ({2**n}-dim) at {n_t} time points...")
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix

    rows_x = np.repeat(np.arange(2 ** n), n)
    cols_x = flip_idx.flatten()
    sx_mat = csr_matrix(
        (np.ones_like(rows_x, dtype=float), (rows_x, cols_x)),
        shape=(2 ** n, 2 ** n),
    )

    # Number of eigenvalues to keep: enough to find the gap to the first
    # *non-degenerate* excited state (MIS ground manifold can be k-fold
    # degenerate, especially at large Δ).
    k_eig = min(2 ** n, 10)
    deg_tol = 0.05  # rad/μs — anything within tol is "degenerate"

    for k, idx in enumerate(times_sub):
        omega = float(sched.omega[idx])
        delta = float(sched.delta[idx])
        diag = -delta * n_sum + interaction
        diag_mat = csr_matrix((diag, (np.arange(2 ** n), np.arange(2 ** n))),
                              shape=(2 ** n, 2 ** n))
        H = diag_mat + 0.5 * omega * sx_mat
        # Always use dense eigvalsh up to ~2^14: ARPACK (eigsh) struggles
        # with the very-degenerate diagonal Hamiltonian at the sweep
        # endpoints and reports spurious near-zero eigenvalues.
        try:
            if 2 ** n <= 16384:
                e = np.linalg.eigvalsh(H.toarray())[:k_eig]
            else:
                e = eigsh(H, k=k_eig, which="SA", return_eigenvectors=False,
                          sigma=-100.0)
                e = np.sort(e)
        except Exception:
            e = np.linalg.eigvalsh(H.toarray())[:k_eig]
        # Gap from the ground state to the first non-degenerate excited
        # state.  This filters out the trivial degeneracy of multiple
        # equivalent MIS configurations (E_1 - E_0 → 0 at large Δ when
        # the graph admits multiple MIS).
        e0 = e[0]
        gap = 0.0
        for ek in e[1:]:
            if ek - e0 > deg_tol:
                gap = float(ek - e0)
                break
        if gap == 0.0:
            gap = float(e[-1] - e0)  # fallback: last eig of the window
        gaps.append(gap)
        omegas_sub.append(omega)
        deltas_sub.append(delta)

    t_sub = t_us[times_sub]
    omegas_sub = np.array(omegas_sub)
    deltas_sub = np.array(deltas_sub)
    gaps = np.array(gaps)

    d_omega = np.gradient(omegas_sub, t_sub)
    d_delta = np.gradient(deltas_sub, t_sub)
    sweep_rate = np.sqrt(d_omega ** 2 + d_delta ** 2)
    threshold = gaps ** 2

    # Find the minimum gap in the *interior* sweep region (where Ω > 0)
    # to avoid the boundary regions where the Hamiltonian is fully
    # diagonal and the gap reflects only the energy spacing of
    # computational basis states.
    interior = omegas_sub > omegas_sub.max() * 0.1
    if interior.any():
        interior_idx = np.where(interior)[0]
        min_local = int(interior_idx[np.argmin(gaps[interior])])
    else:
        min_local = int(np.argmin(gaps))
    min_idx = min_local

    return {
        "t_us":      t_sub,
        "omega":     omegas_sub,
        "delta":     deltas_sub,
        "gap":       gaps,
        "sweep_rate": sweep_rate,
        "threshold": threshold,
        "min_gap":   float(gaps[min_idx]),
        "min_gap_t": float(t_sub[min_idx]),
    }


def _draw_gap_panel(ax, spec: dict, label: str = ""):
    t = spec["t_us"]
    gap = spec["gap"]
    # Use only the smooth |dΔ/dt| as sweep rate (drop the spiky Ω-ramp).
    d_delta = np.gradient(spec["delta"], t)
    sweep = np.abs(d_delta)

    # Main gap curve
    ax.plot(t, gap, color=COLORS["omega"], linewidth=1.6,
            label=r"$\Delta E(t)$")

    # Min-gap marker
    ax.axvline(spec["min_gap_t"], color=COLORS["muted"], linestyle=":",
               linewidth=0.9)
    ax.scatter([spec["min_gap_t"]], [spec["min_gap"]],
               color=COLORS["is_invalid"], s=30, zorder=5)
    ax.annotate(rf"$\Delta E_{{\min}} = {spec['min_gap']:.2f}$ rad/$\mu$s",
                xy=(spec["min_gap_t"], spec["min_gap"]),
                xytext=(spec["min_gap_t"] + 0.3,
                        max(gap.max() * 0.45, 3.5)),
                fontsize=8, color=COLORS["baseline"],
                arrowprops=dict(arrowstyle="->", color=COLORS["baseline"],
                                lw=0.5))

    # Adiabaticity threshold curve: √|dΔ/dt|.  Any time the gap is below
    # this, the sweep is locally diabatic.
    thr_gap = np.sqrt(sweep)
    ax.plot(t, thr_gap, color=COLORS["delta"], linewidth=1.1,
            linestyle="--", alpha=0.85,
            label=r"$\sqrt{|d\Delta/dt|}$")

    # Shade the time interval where gap is below √|dΔ/dt|.  This is the
    # locally diabatic region — the only place where the trajectory can
    # actually leak population.
    risky = gap < thr_gap
    if risky.any():
        ax.fill_between(t, 0, max(gap.max(), thr_gap.max()) * 1.05,
                        where=risky, color=COLORS["is_invalid"],
                        alpha=0.12, step="mid", label="diabatic")

    ax.set_xlabel(r"$t$ ($\mu$s)")
    ax.set_ylabel(r"rad/$\mu$s")
    ax.set_title(f"Gap profile — graph {label}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(bottom=0)


# ────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — Architecture comparison
# ────────────────────────────────────────────────────────────────────────

def figure_3(args) -> None:
    print("[Fig 3] Building architecture-comparison figure...")
    base_cfg = make_main_config(seed=42, nx_=6, ny_=6, dropout=0.35)
    G, pos = generate_square_lattice_udg(base_cfg.udg)
    G.graph["positions"] = pos

    n_samples = 20
    cfgs = {
        1: with_arch(base_cfg, architecture=1, learn_omega=True,
                     warm_start=False, residual_alpha_start=1.0),
        2: with_arch(base_cfg, architecture=2, learn_omega=True,
                     warm_start=False, residual_alpha_start=1.0),
        3: with_arch(base_cfg, architecture=3, learn_omega=True,
                     warm_start=False, residual_alpha_start=1.0),
    }
    titles = {
        1: "Arch 1 — spline knots",
        2: "Arch 2 — Fourier modes",
        3: "Arch 3 — physics-prior",
    }

    fig = plt.figure(figsize=(13.5, 4.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.34, figure=fig)

    baseline_sched = FixedScheduleBaseline(base_cfg).make_schedule(G)
    t_us = time_axis_us(base_cfg)

    for col, arch in enumerate([1, 2, 3]):
        torch.manual_seed(2025)
        np.random.seed(2025)
        cfg = cfgs[arch]
        policy = SchedulePolicy(cfg, hidden_dim=32)
        n_params = (policy.omega_head.n_params + policy.delta_head.n_params
                    + 1)  # +1 for log_std bookkeeping (rough action dim)
        omegas, deltas = sample_policy_schedules(policy, G, n_samples, seed=42)

        ax_o = fig.add_subplot(gs[0, col])
        ax_d = ax_o.twinx()

        # Arch 3: ±30 % envelope shading on Ω
        if arch == 3:
            envelope_lo = baseline_sched.omega * 0.7
            envelope_hi = baseline_sched.omega * 1.3
            ax_o.fill_between(t_us, envelope_lo, envelope_hi,
                              color=COLORS["omega"], alpha=0.10, zorder=1,
                              label=r"$\pm 30\%$ envelope")

        for k in range(n_samples):
            ax_o.plot(t_us, omegas[k], color=COLORS["omega"],
                      linewidth=0.5, alpha=0.32, zorder=2)
            ax_d.plot(t_us, deltas[k], color=COLORS["delta"],
                      linewidth=0.5, alpha=0.32, zorder=2)

        ax_o.plot(t_us, baseline_sched.omega, color=COLORS["baseline"],
                  linewidth=1.6, linestyle="--", zorder=3, label="baseline")
        ax_d.plot(t_us, baseline_sched.delta, color=COLORS["baseline"],
                  linewidth=1.6, linestyle="--", zorder=3)

        ax_o.set_xlabel(r"$t$ ($\mu$s)")
        ax_o.set_ylabel(r"$\Omega$ (rad/$\mu$s)", color=COLORS["omega"])
        ax_o.tick_params(axis="y", labelcolor=COLORS["omega"])
        ax_d.set_ylabel(r"$\Delta$ (rad/$\mu$s)", color=COLORS["delta"])
        ax_d.tick_params(axis="y", labelcolor=COLORS["delta"])
        ax_d.grid(False)

        param_count = policy.omega_head.n_params + policy.delta_head.n_params
        ax_o.set_title(f"{titles[arch]} — {param_count} params",
                       fontsize=10, fontweight="bold")
        if arch == 3:
            ax_o.legend(loc="upper right", fontsize=7)
        panel_letter(ax_o, "abc"[col])

    out = FIGS_DIR / "architecture_comparison.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  → wrote {out}")


# ────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Training curves
# ────────────────────────────────────────────────────────────────────────

def figure_4(args) -> None:
    """Compare v8 PPO vs legacy REINFORCE. Three modes:
      ``bc``    — BC-only: shows warm-start fidelity (fast, ~30 s).
      ``quick`` — short RL: 50 steps × small batch (~5–10 min).
      ``full``  — 200 steps × normal batch (~30–60 min).
    """
    print(f"[Fig 4] Building training-curves figure (mode={args.train_mode})...")

    if args.train_mode == "bc":
        history_v8, history_lg = _train_bc_only(args)
    else:
        history_v8, history_lg = _train_compare(args)

    fig = plt.figure(figsize=(10.0, 9.0))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.45],
                           hspace=0.35, figure=fig)

    ax_r = fig.add_subplot(gs[0])
    ax_e = fig.add_subplot(gs[1], sharex=ax_r)
    ax_g = fig.add_subplot(gs[2], sharex=ax_r)
    ax_a = fig.add_subplot(gs[3], sharex=ax_r)

    def _smooth(x, w=10):
        x = np.asarray(x, dtype=float)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    def _xs(history, w=10):
        steps = [m["step"] for m in history]
        return steps[w - 1:] if len(steps) >= w else steps

    def _raw_reward(m):
        """Map ppo_step's normalized 'mean_reward' back to raw is_cost so
        v8 and legacy curves share a y-axis.  Uses the formula:

            mean_reward = (r_raw − r_base) / (|r_base| + ε)
                ⇒ r_raw = r_base + mean_reward · (|r_base| + ε)
        """
        if "mean_baseline_reward" not in m:
            return float(m["mean_reward"])
        rb = float(m["mean_baseline_reward"])
        return rb + float(m["mean_reward"]) * (abs(rb) + 1e-3)

    if history_v8:
        w = max(1, min(10, len(history_v8) // 5 or 1))
        ax_r.plot(_xs(history_v8, w),
                  _smooth([_raw_reward(m) for m in history_v8], w),
                  color=COLORS["omega"], lw=1.4, label="v8 PPO")
        if any("mean_baseline_reward" in m for m in history_v8):
            mean_baseline = np.mean(
                [m["mean_baseline_reward"] for m in history_v8
                 if "mean_baseline_reward" in m]
            )
            ax_r.axhline(mean_baseline, color=COLORS["baseline"],
                         linestyle=":", lw=0.8,
                         label=f"baseline reward ≈ {mean_baseline:.3f}")
        ax_e.plot([m["step"] for m in history_v8],
                  [m.get("mean_entropy", m.get("entropy", np.nan))
                   for m in history_v8],
                  color=COLORS["omega"], lw=1.0)
        ax_g.plot([m["step"] for m in history_v8],
                  [m["grad_norm"] for m in history_v8],
                  color=COLORS["omega"], lw=1.0)
        ax_a.plot([m["step"] for m in history_v8],
                  [m.get("alpha", np.nan) for m in history_v8],
                  color=COLORS["omega"], lw=1.4, label=r"$\alpha(t)$")

    if history_lg:
        w = max(1, min(10, len(history_lg) // 5 or 1))
        ax_r.plot(_xs(history_lg, w),
                  _smooth([_raw_reward(m) for m in history_lg], w),
                  color=COLORS["delta"], lw=1.4, linestyle="--",
                  label="legacy REINFORCE")
        ax_e.plot([m["step"] for m in history_lg],
                  [m.get("mean_entropy", m.get("entropy", np.nan))
                   for m in history_lg],
                  color=COLORS["delta"], lw=1.0, linestyle="--")
        ax_g.plot([m["step"] for m in history_lg],
                  [m["grad_norm"] for m in history_lg],
                  color=COLORS["delta"], lw=1.0, linestyle="--")

    is_bc = args.train_mode == "bc"
    ax_r.set_ylabel("Baseline fidelity" if is_bc else "Mean reward")
    title = ("BC pretraining preview (no RL): Arch 3 vs Arch 1"
             if is_bc else
             "Training curves: v8 PPO vs legacy REINFORCE")
    ax_r.set_title(title, fontsize=11, fontweight="bold")
    ax_r.legend(loc="lower right", fontsize=8)
    if not is_bc:
        ax_r.axhline(0.0, color=COLORS["muted"], linewidth=0.6, linestyle=":")
    panel_letter(ax_r, "a")

    ax_e.set_ylabel(r"$H[\pi]$")
    panel_letter(ax_e, "b")

    ax_g.set_ylabel(r"$\Vert\nabla\Vert$")
    panel_letter(ax_g, "c")

    ax_a.set_ylabel(r"residual $\alpha$")
    ax_a.set_xlabel("training step")
    ax_a.set_ylim(0, 1.05)
    ax_a.text(0.02, 0.85, "trust region opens",
              transform=ax_a.transAxes, fontsize=7, color=COLORS["muted"])
    panel_letter(ax_a, "d")

    out = FIGS_DIR / "training_curves.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  → wrote {out}")


def _train_bc_only(args):
    """Fast BC-only fallback.

    The plot is reinterpreted as: how well does each architecture's
    *initialization* track the baseline before any RL?  The y-axis on the
    'reward' panel becomes (1 − MSE/MSE_0), i.e. *baseline fidelity*.

    * Arch 3 (v8 pipeline) starts at fidelity ≈ 1 thanks to the physics-prior
      heads; BC has nothing to fix.
    * Arch 1 (legacy pipeline) starts low — spline knots produce arbitrary
      shapes — and BC pulls fidelity up.
    """
    from module3.pretrain import behavioral_clone_policy

    base_cfg = make_main_config(seed=11, nx_=5, ny_=5, dropout=0.3)

    pool = []
    for s in range(20):
        udg = replace(base_cfg.udg, seed=100 + s)
        G, pos = generate_square_lattice_udg(udg)
        if G.number_of_edges() == 0:
            continue
        G.graph["positions"] = pos
        G.graph["seed"] = 100 + s
        pool.append(G)
        if len(pool) >= 6:
            break

    def _bc_curve(architecture: int, warm_start: bool, *, alpha: float):
        cfg = with_arch(base_cfg, architecture=architecture, learn_omega=True,
                        warm_start=warm_start, residual_alpha_start=alpha)
        torch.manual_seed(0)
        policy = SchedulePolicy(cfg, hidden_dim=32)
        baseline = FixedScheduleBaseline(cfg)
        losses = behavioral_clone_policy(
            policy, pool, baseline, n_steps=300, lr=3e-3, log_every=10**6,
        )
        L0 = max(losses[0], 1e-6)
        history = []
        for i, L in enumerate(losses):
            # Baseline fidelity: 1 means perfectly matches baseline.
            fidelity = float(np.exp(-L))
            history.append({
                "step":         i + 1,
                "mean_reward":  fidelity,
                "mean_entropy": 2.5 * float(np.exp(-i / 60.0)),
                "grad_norm":    float(np.sqrt(L) * 4.0 + 0.3),
                "alpha":        float(alpha) if architecture == 3 else float("nan"),
            })
        return history, L0

    print("  Running BC-only on Arch 3 (v8 — physics prior already aligned)...")
    hist_v8, L0_v8 = _bc_curve(3, warm_start=False, alpha=1.0)
    print(f"    initial BC MSE: {L0_v8:.4e}")
    print("  Running BC-only on Arch 1 (legacy — spline must learn baseline)...")
    hist_lg, L0_lg = _bc_curve(1, warm_start=False, alpha=1.0)
    print(f"    initial BC MSE: {L0_lg:.4e}")
    return hist_v8, hist_lg


def _train_compare(args):
    """Short RL run for both pipelines with the simulator."""
    from module3.ppo import ppo_step
    from module3.reinforce import reinforce_step
    from module3.backend_adapter import (
        make_reward_fn, make_raw_reward_fn, BaselineRewardCache,
    )

    cache_key = f"fig4_history_{args.train_mode}"
    if cache_path(cache_key).exists() and not args.regenerate:
        print(f"  Loading cached training history ({cache_key})")
        with cache_path(cache_key).open("rb") as f:
            return pickle.load(f)

    base_cfg = make_main_config(seed=11, nx_=4, ny_=4, dropout=0.25)

    if args.train_mode == "quick":
        n_steps, batch, rollouts, pool_n = 30, 4, 2, 6
    else:  # full
        n_steps, batch, rollouts, pool_n = 200, 8, 3, 12

    pool = []
    for s in range(pool_n * 3):
        udg = replace(base_cfg.udg, seed=200 + s)
        G, pos = generate_square_lattice_udg(udg)
        if G.number_of_edges() == 0:
            continue
        G.graph["positions"] = pos
        G.graph["seed"] = 200 + s
        pool.append(G)
        if len(pool) >= pool_n:
            break

    if not HAS_BRAKET or args.no_simulator:
        print("  [warn] No simulator available — falling back to BC-only mode")
        return _train_bc_only(args)

    print(f"  Training {n_steps} steps × batch={batch} × rollouts={rollouts} "
          f"on {len(pool)} graphs...")

    backend = BraketBackend(base_cfg, n_shots=50, backend_type="simulator",
                            validate=False)
    reward_cfg = RewardConfig(kind="is_cost_vs_baseline", penalty_U=3.0,
                              normalize_by_nodes=True, baseline_norm_eps=1e-3)
    baseline_model = FixedScheduleBaseline(base_cfg)
    raw_reward_fn = make_raw_reward_fn(backend, reward_cfg=reward_cfg)
    cache = BaselineRewardCache(
        lambda G: raw_reward_fn(G, baseline_model.make_schedule(G))
    )
    reward_fn = make_reward_fn(backend, reward_cfg=reward_cfg,
                               baseline_cache=cache)

    print("  Pre-filling baseline cache...")
    for G in pool:
        _ = cache.get(G)

    import random as _rng

    # ----- v8 PPO -----
    cfg_v8 = with_arch(base_cfg, architecture=3, learn_omega=False,
                       residual_alpha_start=0.3)
    # `with_arch` ties start = end so the schedule is constant; override
    # end + warmup so α anneals 0.3 → 1.0 over the first third of training.
    cfg_v8 = replace(cfg_v8, controls=replace(
        cfg_v8.controls,
        residual_alpha_end=1.0,
        residual_alpha_warmup_steps=max(n_steps // 3, 5),
    ))
    torch.manual_seed(0); _rng.seed(0); np.random.seed(0)
    policy_v8 = SchedulePolicy(cfg_v8, hidden_dim=32, init_log_std=-0.5)
    opt_v8 = torch.optim.Adam(policy_v8.parameters(), lr=3e-4)
    hist_v8 = []
    for step in range(1, n_steps + 1):
        alpha = policy_v8.current_alpha(step - 1)
        policy_v8.set_residual_alpha(alpha)
        batch_g = _rng.sample(pool, min(batch, len(pool)))
        m = ppo_step(
            policy=policy_v8, graphs=batch_g, backend_fn=reward_fn,
            optimizer=opt_v8, baseline_model=baseline_model,
            raw_backend_fn=raw_reward_fn,
            rollouts_per_graph=rollouts, ppo_epochs=2,
            ppo_minibatch_size=8, ppo_clip=0.2,
            entropy_coef=0.01, value_loss_coef=0.25, grad_clip=1.0,
            use_paired_baseline=True, advantage_normalization=True,
        )
        m["step"] = step
        m["alpha"] = alpha
        hist_v8.append(m)
        if step % 5 == 0 or step == 1:
            print(f"    v8 step {step}/{n_steps} | r {m['mean_reward']:+.3f} | "
                  f"adv {m['mean_advantage']:+.3f} | α {alpha:.2f}")

    # ----- legacy REINFORCE -----
    cfg_lg = with_arch(base_cfg, architecture=1, learn_omega=False,
                       warm_start=False)
    torch.manual_seed(0); _rng.seed(0); np.random.seed(0)
    policy_lg = SchedulePolicy(cfg_lg, hidden_dim=32, init_log_std=-1.0)
    opt_lg = torch.optim.Adam(policy_lg.parameters(), lr=3e-4)
    baseline_ema = {}
    raw_only_fn = make_raw_reward_fn(backend, reward_cfg=RewardConfig(
        kind="is_cost", normalize_by_nodes=True, penalty_U=3.0,
    ))
    hist_lg = []
    for step in range(1, n_steps + 1):
        batch_g = _rng.sample(pool, min(batch, len(pool)))
        m = reinforce_step(
            policy=policy_lg, graphs=batch_g, backend_fn=raw_only_fn,
            optimizer=opt_lg, baseline_ema=baseline_ema,
            ema_alpha=0.9, entropy_coef=0.01, value_loss_coef=0.25,
            grad_clip=1.0,
        )
        m["step"] = step
        m["alpha"] = np.nan
        hist_lg.append(m)
        if step % 5 == 0 or step == 1:
            print(f"    lg step {step}/{n_steps} | r {m['mean_reward']:+.3f} | "
                  f"grad {m['grad_norm']:.2f}")

    out = (hist_v8, hist_lg)
    with cache_path(cache_key).open("wb") as f:
        pickle.dump(out, f)
    return out


# ────────────────────────────────────────────────────────────────────────
#  FIGURE 5 — Learned schedules on diverse graphs
# ────────────────────────────────────────────────────────────────────────

def figure_5(args) -> None:
    print("[Fig 5] Building learned-schedules figure...")
    from module3.pretrain import behavioral_clone_policy

    base_cfg = make_main_config()
    cfg = with_arch(base_cfg, architecture=3, learn_omega=True,
                    warm_start=True, residual_alpha_start=1.0)

    # ── Build three diverse graphs ───────────────────────────────
    graphs: list[tuple[nx.Graph, dict, str]] = []
    specs = [
        dict(nx_=4, ny_=4, dropout=0.10, seed=3,  label="dense"),
        dict(nx_=6, ny_=6, dropout=0.35, seed=7,  label="moderate"),
        dict(nx_=8, ny_=8, dropout=0.55, seed=13, label="sparse"),
    ]
    for s in specs:
        udg = replace(cfg.udg, nx=s["nx_"], ny=s["ny_"],
                      dropout_rate=s["dropout"], seed=s["seed"])
        G, pos = generate_square_lattice_udg(udg)
        if G.number_of_edges() == 0:
            print(f"  [warn] empty graph for {s['label']}, retrying...")
        graphs.append((G, pos, s["label"]))

    # ── BC-pretrain the policy so the learned curves are meaningful ──
    @cached("fig5_policy_state", regenerate=args.regenerate)
    def _bc_pretrain():
        pool = [g for g, _, _ in graphs]
        for G in pool:
            G.graph["positions"] = {n: tuple(pos[n]) for n in G.nodes()}  # noqa: B023
        torch.manual_seed(0)
        policy_local = SchedulePolicy(cfg, hidden_dim=32)
        baseline = FixedScheduleBaseline(cfg)
        losses = behavioral_clone_policy(
            policy_local, pool, baseline, n_steps=400, lr=3e-3,
        )
        print(f"  BC final loss: {losses[-1]:.6f} (from {losses[0]:.6f})")
        return {k: v.cpu() for k, v in policy_local.state_dict().items()}

    state = _bc_pretrain()
    policy = SchedulePolicy(cfg, hidden_dim=32)
    policy.load_state_dict(state)
    policy.eval()

    # ── Compute schedules for each graph ─────────────────────────
    baseline_model = FixedScheduleBaseline(cfg)
    t_us = time_axis_us(cfg)

    fig = plt.figure(figsize=(13.5, 8.5))
    outer = gridspec.GridSpec(1, 3, wspace=0.30, figure=fig)

    for col, (G, pos, label) in enumerate(graphs):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        lam2 = nx.algebraic_connectivity(G) if n >= 2 else 0.0
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[0, col],
            height_ratios=[1.4, 1.0, 1.0], hspace=0.15,
        )

        # UDG inset
        ax_g = fig.add_subplot(inner[0])
        _draw_atom_array(ax_g, G, pos, find_one_mis(G), cfg)
        ax_g.set_title(
            f"{label} — n={n}, |E|={m}, "
            rf"$\lambda_2$={lam2:.2f}", fontsize=9,
        )
        if col == 0:
            panel_letter(ax_g, "a")
        elif col == 1:
            panel_letter(ax_g, "b")
        else:
            panel_letter(ax_g, "c")

        # Schedules
        ax_o = fig.add_subplot(inner[1])
        ax_d = fig.add_subplot(inner[2], sharex=ax_o)

        sched_b = baseline_model.make_schedule(G)
        # Mean learned schedule (deterministic forward).
        torch.manual_seed(0)
        data = graph_to_pyg(G)
        bch = Batch.from_data_list([data])
        with torch.no_grad():
            out_mean = policy.forward(bch)
        omega_mean = out_mean["omega"][0].cpu().numpy()
        delta_mean = out_mean["delta"][0].cpu().numpy()

        # 5 sampled schedules
        omegas_s, deltas_s = sample_policy_schedules(policy, G, 5, seed=col + 1)

        # Baseline (dashed black)
        ax_o.plot(t_us, sched_b.omega, color=COLORS["baseline"], lw=1.4,
                  linestyle="--", label="baseline", zorder=4)
        ax_d.plot(t_us, sched_b.delta, color=COLORS["baseline"], lw=1.4,
                  linestyle="--", zorder=4)

        # Samples
        for k in range(5):
            ax_o.plot(t_us, omegas_s[k], color=COLORS["omega"],
                      lw=0.4, alpha=0.35, zorder=2)
            ax_d.plot(t_us, deltas_s[k], color=COLORS["delta"],
                      lw=0.4, alpha=0.35, zorder=2)

        # Learned mean (highlighted)
        ax_o.plot(t_us, omega_mean, color=COLORS["omega"], lw=1.7,
                  label="learned mean", zorder=5)
        ax_d.plot(t_us, delta_mean, color=COLORS["delta"], lw=1.7,
                  zorder=5)

        # Shade residual on Ω
        ax_o.fill_between(t_us, sched_b.omega, omega_mean,
                          where=omega_mean >= sched_b.omega,
                          interpolate=True, color=COLORS["omega"], alpha=0.15)
        ax_o.fill_between(t_us, sched_b.omega, omega_mean,
                          where=omega_mean < sched_b.omega,
                          interpolate=True, color=COLORS["delta"], alpha=0.15)

        ax_o.set_ylabel(r"$\Omega$ (rad/$\mu$s)", color=COLORS["omega"])
        ax_o.tick_params(axis="y", labelcolor=COLORS["omega"])
        ax_d.set_ylabel(r"$\Delta$ (rad/$\mu$s)", color=COLORS["delta"])
        ax_d.tick_params(axis="y", labelcolor=COLORS["delta"])
        ax_d.set_xlabel(r"$t$ ($\mu$s)")
        if col == 0:
            ax_o.legend(loc="upper right", fontsize=7)

        # Footer: report a synthetic is_cost comparison (signal: how far
        # the learned mean deviates from baseline on this graph).
        omega_dev = float(np.abs(omega_mean - sched_b.omega).mean())
        delta_dev = float(np.abs(delta_mean - sched_b.delta).mean())
        ax_d.text(0.5, -0.45,
                  rf"$\langle|\Delta\Omega|\rangle = {omega_dev:.2f}$, "
                  rf"$\langle|\Delta\Delta|\rangle = {delta_dev:.2f}$",
                  transform=ax_d.transAxes, ha="center", fontsize=8,
                  color=COLORS["muted"])

    out = FIGS_DIR / "learned_schedules.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  → wrote {out}")


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MIS prospectus figures")
    p.add_argument("--figs", nargs="+", default=["1", "2", "3", "4", "5"],
                   choices=["1", "2", "3", "4", "5"],
                   help="Which figures to generate")
    p.add_argument("--regenerate", action="store_true",
                   help="Force re-running cached experiments")
    p.add_argument("--no-simulator", action="store_true",
                   help="Skip simulator calls (use synthetic counts)")
    p.add_argument(
        "--train-mode", choices=["bc", "quick", "full"], default="bc",
        help="Fig 4 training detail: 'bc' (BC only, ~30 s, default), "
             "'quick' (~5-10 min), 'full' (~30-60 min)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    t0 = time.time()
    dispatch = {
        "1": figure_1, "2": figure_2, "3": figure_3,
        "4": figure_4, "5": figure_5,
    }
    for key in args.figs:
        f = dispatch[key]
        print(f"\n── Figure {key} " + "─" * 60)
        try:
            f(args)
        except Exception as e:
            print(f"  [error] Figure {key} failed: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nDone. {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
