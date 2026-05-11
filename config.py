from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


class ParamKind(str, Enum):
    """Time parameterization kind.

    - "pwc": piecewise-constant values over T steps
    - "pwl": piecewise-linear values with knots on the grid
    """

    pwc = "pwc"
    pwl = "pwl"


@dataclass(frozen=True)
class ControlsConfig:
    """Controls grid and amplitude settings.

    Parameters
    ----------
    T : float
        Total duration in seconds.
    N_t : int
        Number of time grid points (including first and last). Must be >= 2.
    param_kind : ParamKind
        Parameterization kind: "pwc" or "pwl".
    learn_omega : bool
        If True the ML model outputs both Ω(t) and Δ(t).
        If False (default) only Δ(t) is learned; Ω(t) is a fixed analytic
        envelope.
    architecture : int
        Which decoder architecture to use.
        1 = spline-knot decoder (Arch 1), 2 = Fourier-coefficient decoder (Arch 2).
    omega_max : float
        Maximum Rabi amplitude in rad/μs. Used as the output bound for the
        learned Ω head and as the peak of the analytic envelope.
    delta_min : float
        Lower bound on detuning Δ(t) in rad/μs.
    delta_max : float
        Upper bound on detuning Δ(t) in rad/μs.
    n_delta_knots : int
        Number of spline control points for Δ(t) in Arch 1.
    n_omega_modes : int
        Number of sine modes for Ω(t) in Arch 2.
    n_delta_modes : int
        Number of cosine modes for Δ(t) in Arch 2.
    omega_scale : float
        Scale factor for the simple baseline Ω derivation (omega_scale / radius^6).
        Used only by the non-learned baseline models (GNNModel, AdjacencyMLP).
    omega_cap : float | None
        Optional cap for the baseline Ω derivation.
        Used only by the non-learned baseline models.
    """

    T: float = 4.0e-6
    N_t: int = 128
    param_kind: ParamKind = ParamKind.pwc
    learn_omega: bool = False
    architecture: int = 1
    omega_max: float = 15.8
    delta_min: float = -25.0
    delta_max: float = 25.0
    n_delta_knots: int = 8
    n_omega_modes: int = 5
    n_delta_modes: int = 8
    omega_scale: float = 1.0
    omega_cap: float | None = None

    @property
    def dt(self) -> float:
        """Derived time step size: T / (N_t - 1)."""
        return float(self.T) / max(1, (int(self.N_t) - 1))


@dataclass(frozen=True)
class UDGConfig:
    """Unit disk graph (UDG) generation on a square lattice with dropout.

    Parameters
    ----------
    nx : int
        Number of lattice sites along x.
    ny : int
        Number of lattice sites along y.
    spacing : float
        Lattice spacing in μm (default 1.0).
    radius : float
        Unit disk connection radius in units of ``spacing`` (physical distance
        = ``radius * spacing`` μm).  Also used as the blockade radius when
        computing Ω from C₆.
    dropout_rate : float
        Probability to drop each lattice site independently (deterministic with seed).
    seed : int
        RNG seed to ensure deterministic dropout and generation.
    """

    nx: int = 8
    ny: int = 8
    spacing: float = 1.0
    radius: float = 1.5
    dropout_rate: float = 0.0
    seed: int = 0


Backend = Literal["bloqade", "aquila"]


@dataclass(frozen=True)
class HardwareSpecs:
    """Physical constants and protocol timing for the neutral-atom device.

    Loaded from ``hardware_specs.json``.  Units follow the Rydberg-physics
    convention: frequencies in rad/μs, distances in μm, times in μs.

    Parameters
    ----------
    C6 : float
        Van der Waals coefficient for the Rydberg state (rad/μs · μm⁶).
        Aquila (⁸⁷Rb |70S₁/₂⟩): 5.42 × 10⁶.
    omega_max : float
        Hardware upper bound on global Rabi amplitude Ω (rad/μs).
    delta_min, delta_max : float
        Hardware bounds on global detuning Δ (rad/μs).
    t_ramp : float
        Linear ramp-up / ramp-down duration for the trapezoidal Ω envelope (μs).
    t_onset : float
        Delay before the Ω ramp begins (μs). Usually 0.
    """

    C6: float = 5.42e6
    omega_max: float = 15.8
    delta_min: float = -125.0
    delta_max: float = 125.0
    t_ramp: float = 0.3
    t_onset: float = 0.0


def compute_blockade_omega(
    C6: float, R_b_um: float, omega_max_hw: float,
) -> float:
    """Derive Ω_peak from the blockade radius, capped at the hardware limit.

    The Rydberg blockade condition sets Ω = C₆ / R_b⁶.  If that exceeds the
    hardware maximum, the hardware limit is returned (the blockade radius at
    that Ω is still larger than R_b, so the constraint is satisfied).
    """
    omega = C6 / max(R_b_um, 1e-12) ** 6
    return min(omega, omega_max_hw)


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level config shared by all modules."""

    backend: Backend
    controls: ControlsConfig
    udg: UDGConfig
    hardware: HardwareSpecs


def derive_omega_schedule(controls: ControlsConfig, udg: UDGConfig) -> NDArray[np.float64]:
    """Derive a simple Ω(t) schedule from the blockade radius.

    This is a placeholder consistent mapping: Ω ~ omega_scale / radius^6, capped by omega_cap.
    Returns a constant schedule over time for now.
    """
    eps = 1e-12
    base = controls.omega_scale / float(max(udg.radius, eps)) ** 6
    if controls.omega_cap is not None:
        base = float(min(base, controls.omega_cap))
    omega = np.full((controls.N_t,), float(base), dtype=np.float64)
    return omega


def _controls_from_dict(d: dict) -> ControlsConfig:
    pk = d.get("param_kind", ParamKind.pwc)
    if isinstance(pk, str):
        pk_enum = ParamKind(pk)
    else:
        pk_enum = pk

    # Backward compatibility:
    # - If N_t provided, assume T is total duration; ignore any legacy dt.
    # - Else, if legacy dt provided and T was legacy step count, compute total T.
    if "N_t" in d:
        N_t_val = int(d.get("N_t"))
        T_total = float(d.get("T", ControlsConfig.T))
    else:
        legacy_dt = d.get("dt", None)
        if legacy_dt is not None:
            N_t_val = int(d.get("T", ControlsConfig.N_t))
            T_total = float(legacy_dt) * max(1, (N_t_val - 1))
        else:
            # Fallback to defaults if not enough info
            N_t_val = ControlsConfig.N_t
            T_total = float(d.get("T", ControlsConfig.T))

    return ControlsConfig(
        T=T_total,
        N_t=N_t_val,
        param_kind=pk_enum,
        learn_omega=bool(d.get("learn_omega", ControlsConfig.learn_omega)),
        architecture=int(d.get("architecture", ControlsConfig.architecture)),
        omega_max=float(d.get("omega_max", ControlsConfig.omega_max)),
        delta_min=float(d.get("delta_min", ControlsConfig.delta_min)),
        delta_max=float(d.get("delta_max", ControlsConfig.delta_max)),
        n_delta_knots=int(d.get("n_delta_knots", ControlsConfig.n_delta_knots)),
        n_omega_modes=int(d.get("n_omega_modes", ControlsConfig.n_omega_modes)),
        n_delta_modes=int(d.get("n_delta_modes", ControlsConfig.n_delta_modes)),
        omega_scale=float(d.get("omega_scale", ControlsConfig.omega_scale)),
        omega_cap=d.get("omega_cap", ControlsConfig.omega_cap),
    )


def _udg_from_dict(d: dict) -> UDGConfig:
    return UDGConfig(
        nx=int(d.get("nx", UDGConfig.nx)),
        ny=int(d.get("ny", UDGConfig.ny)),
        spacing=float(d.get("spacing", UDGConfig.spacing)),
        radius=float(d.get("radius", UDGConfig.radius)),
        dropout_rate=float(d.get("dropout_rate", UDGConfig.dropout_rate)),
        seed=int(d.get("seed", UDGConfig.seed)),
    )


def _hardware_from_dict(d: dict) -> HardwareSpecs:
    return HardwareSpecs(
        C6=float(d.get("C6", HardwareSpecs.C6)),
        omega_max=float(d.get("omega_max", HardwareSpecs.omega_max)),
        delta_min=float(d.get("delta_min", HardwareSpecs.delta_min)),
        delta_max=float(d.get("delta_max", HardwareSpecs.delta_max)),
        t_ramp=float(d.get("t_ramp", HardwareSpecs.t_ramp)),
        t_onset=float(d.get("t_onset", HardwareSpecs.t_onset)),
    )


def project_config_from_dict(
    d: dict, hardware_dict: dict | None = None,
) -> ProjectConfig:
    backend: Backend = d.get("backend", "bloqade")  # type: ignore[assignment]
    controls = _controls_from_dict(d.get("controls", {}))
    udg = _udg_from_dict(d.get("udg", {}))
    hardware = _hardware_from_dict(hardware_dict or {})
    return ProjectConfig(backend=backend, controls=controls, udg=udg, hardware=hardware)


def load_project_config_json(
    path: str | Path | None = None,
    hardware_path: str | Path | None = None,
) -> ProjectConfig:
    """Load ProjectConfig from JSON files.

    Parameters
    ----------
    path : str | Path | None
        Path to ``config.json``.  Defaults to ``<repo_root>/config.json``.
    hardware_path : str | Path | None
        Path to ``hardware_specs.json``.  Defaults to
        ``<repo_root>/hardware_specs.json``.  If the file does not exist,
        built-in Aquila defaults are used.
    """
    root = Path(__file__).resolve().parent
    if path is None:
        path = root / "config.json"
    else:
        path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if hardware_path is None:
        hardware_path = root / "hardware_specs.json"
    else:
        hardware_path = Path(hardware_path)
    if hardware_path.exists():
        with hardware_path.open("r", encoding="utf-8") as f:
            hw_data = json.load(f)
    else:
        hw_data = {}

    return project_config_from_dict(data, hardware_dict=hw_data)
