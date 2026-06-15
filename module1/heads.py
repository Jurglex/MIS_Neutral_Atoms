"""Schedule decoder heads: Omega(t) and Delta(t) reconstruction from latent parameters.

Three architectures are supported:

* **Arch 1** — Spline/peak parameterization (``OmegaHead`` + ``DeltaHead``).
* **Arch 2** — Fourier-series parameterization
  (``FourierOmegaHead`` + ``FourierDeltaHead``).
* **Arch 3** — *Physics-prior* parameterization
  (``MultiplicativeOmegaHead`` + ``MonotoneDeltaHead``):
  Ω(t) is a multiplicative modulation of the analytic trapezoidal baseline,
  and Δ(t) is a monotonically non-decreasing sweep parameterized by positive
  increments.  Designed so that the *only* deviations from the adiabatic
  baseline the network can express are physically meaningful ones.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from config import ControlsConfig, HardwareSpecs, compute_blockade_omega


# ---------------------------------------------------------------------------
# Omega heads
# ---------------------------------------------------------------------------

class OmegaHead(nn.Module):
    """Learnable Omega(t) head (3 latent parameters).

    Reconstructs  Omega(t) = peak * sin^2(envelope)  where the envelope
    is a windowed phase that guarantees Omega(0) = Omega(T) = 0 and
    Omega >= 0 everywhere, by construction.
    """

    N_PARAMS = 3

    def __init__(self, embed_dim: int, controls: ControlsConfig, hidden: int = 64):
        super().__init__()
        self.controls = controls
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.N_PARAMS)
        )

    @property
    def n_params(self) -> int:
        return self.N_PARAMS

    def reconstruct(self, params: torch.Tensor) -> torch.Tensor:
        """params: (B, 3) -> Omega(t): (B, N_t)."""
        peak = torch.sigmoid(params[:, 0]) * self.controls.omega_max
        width = 0.5 + 0.5 * torch.sigmoid(params[:, 1])      # in (0.5, 1.0)
        center = 0.25 + 0.5 * torch.sigmoid(params[:, 2])     # in (0.25, 0.75)

        t = torch.linspace(0.0, 1.0, self.controls.N_t, device=params.device)
        # Boundary mask: guarantees exactly 0 at t=0 and t=T
        boundary = torch.sin(math.pi * t) ** 2

        t_b = t.unsqueeze(0).expand(params.shape[0], -1)      # (B, N_t)
        c = center.unsqueeze(1)
        w = width.unsqueeze(1)
        u = ((t_b - (c - w / 2)) / w).clamp(0.0, 1.0)
        shape = torch.sin(math.pi * u) ** 2

        return peak.unsqueeze(1) * shape * boundary.unsqueeze(0)

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(embed)
        return params, self.reconstruct(params)


class AnalyticOmega(nn.Module):
    """Fixed trapezoidal Ω(t) envelope derived from blockade physics.

    Shape on normalised time t ∈ [0, 1]::

        0 ── onset ── onset+ramp ── 1−onset−ramp ── 1−onset ── 1
        0     0       Ω_peak        Ω_peak          0           0

    Drop-in replacement for OmegaHead / FourierOmegaHead when
    ``learn_omega`` is False.  Contributes zero parameters to the
    action distribution.

    Parameters
    ----------
    embed_dim : int
        Ignored (kept for interface compatibility with learned heads).
    N_t : int
        Number of time-grid points.
    omega_peak : float
        Peak Rabi amplitude (rad/μs), typically computed via
        ``compute_blockade_omega``.
    t_ramp_frac : float
        Ramp duration as a fraction of total time T.
    t_onset_frac : float
        Onset delay as a fraction of total time T.  The same dead zone
        is mirrored at the end: Ω is zero for t ∈ [1−onset, 1].
    """

    def __init__(
        self,
        embed_dim: int,
        N_t: int,
        omega_peak: float,
        t_ramp_frac: float,
        t_onset_frac: float = 0.0,
    ):
        super().__init__()
        self._N_t = N_t

        t = torch.linspace(0.0, 1.0, N_t)
        r = max(t_ramp_frac, 1e-9)
        o = t_onset_frac

        # Piecewise-linear: rise from onset to onset+ramp, hold, fall
        # from 1-onset-ramp to 1-onset, zero outside.
        rise = ((t - o) / r).clamp(0.0, 1.0)
        fall = ((1.0 - o - t) / r).clamp(0.0, 1.0)
        envelope = omega_peak * torch.min(rise, fall)

        self.register_buffer("envelope", envelope)

    @property
    def n_params(self) -> int:
        return 0

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = embed.shape[0]
        omega = self.envelope.unsqueeze(0).expand(B, -1)
        return torch.zeros(B, 0, device=embed.device), omega


class FourierOmegaHead(nn.Module):
    """Learnable Omega(t) via sine-series coefficients (Architecture 2).

    Reconstructs Omega(t) = omega_max * sigmoid(sum_k a_k * sin(k*pi*t)).
    The sine basis guarantees Omega(0) = Omega(T) = 0 by construction
    since sin(k*pi*0) = sin(k*pi*1) = 0 for all integer k.
    Sigmoid keeps the output in [0, omega_max].
    """

    def __init__(self, embed_dim: int, controls: ControlsConfig, hidden: int = 64):
        super().__init__()
        self.controls = controls
        self._n_modes = controls.n_omega_modes
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, self._n_modes)
        )
        t = torch.linspace(0.0, 1.0, controls.N_t)
        ks = torch.arange(1, self._n_modes + 1, dtype=torch.float32)
        basis = torch.sin(math.pi * t.unsqueeze(1) * ks.unsqueeze(0))
        self.register_buffer("basis", basis)

    @property
    def n_params(self) -> int:
        return self._n_modes

    def reconstruct(self, params: torch.Tensor) -> torch.Tensor:
        """params: (B, n_modes) -> Omega(t): (B, N_t) in [0, omega_max]."""
        raw = params @ self.basis.T
        return self.controls.omega_max * torch.sigmoid(raw)

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(embed)
        return params, self.reconstruct(params)


# ---------------------------------------------------------------------------
# Delta heads
# ---------------------------------------------------------------------------

class DeltaHead(nn.Module):
    """Learnable Delta(t) head via spline control points.

    Outputs ``n_delta_knots`` values, linearly interpolates them to the
    full N_t grid, and maps through tanh to enforce [delta_min, delta_max].
    """

    def __init__(self, embed_dim: int, controls: ControlsConfig, hidden: int = 64):
        super().__init__()
        self.controls = controls
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, controls.n_delta_knots),
        )
        knot_t = torch.linspace(0.0, 1.0, controls.n_delta_knots)
        grid_t = torch.linspace(0.0, 1.0, controls.N_t)
        self.register_buffer("interp_matrix", self._build_interp_matrix(knot_t, grid_t))

    @property
    def n_params(self) -> int:
        return self.controls.n_delta_knots

    @staticmethod
    def _build_interp_matrix(knot_t: torch.Tensor, grid_t: torch.Tensor) -> torch.Tensor:
        """Build a (N_t, n_knots) linear-interpolation matrix."""
        n_t = grid_t.shape[0]
        n_knots = knot_t.shape[0]
        M = torch.zeros(n_t, n_knots)
        for i, t in enumerate(grid_t):
            j = torch.searchsorted(knot_t, t).clamp(1, n_knots - 1)
            t0, t1 = knot_t[j - 1], knot_t[j]
            w = (t - t0) / (t1 - t0 + 1e-9)
            M[i, j - 1] = 1 - w
            M[i, j] = w
        return M

    def reconstruct(self, params: torch.Tensor) -> torch.Tensor:
        """params: (B, n_knots) -> Delta(t): (B, N_t) in [delta_min, delta_max]."""
        delta_grid = params @ self.interp_matrix.T
        mid = 0.5 * (self.controls.delta_max + self.controls.delta_min)
        half = 0.5 * (self.controls.delta_max - self.controls.delta_min)
        return mid + half * torch.tanh(delta_grid)

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(embed)
        return params, self.reconstruct(params)


class FourierDeltaHead(nn.Module):
    """Learnable Delta(t) via cosine-series coefficients (Architecture 2).

    Outputs 1 + n_delta_modes parameters: a DC offset plus K cosine
    coefficients.  Reconstructs Delta(t) = a_0 + sum_k a_k * cos(k*pi*t),
    then tanh-clamps to [delta_min, delta_max].
    """

    def __init__(self, embed_dim: int, controls: ControlsConfig, hidden: int = 64):
        super().__init__()
        self.controls = controls
        self._n_modes = controls.n_delta_modes
        self._n_total = 1 + self._n_modes
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self._n_total),
        )
        t = torch.linspace(0.0, 1.0, controls.N_t)
        ks = torch.arange(1, self._n_modes + 1, dtype=torch.float32)
        cos_part = torch.cos(math.pi * t.unsqueeze(1) * ks.unsqueeze(0))
        dc = torch.ones(controls.N_t, 1)
        basis = torch.cat([dc, cos_part], dim=1)
        self.register_buffer("basis", basis)

    @property
    def n_params(self) -> int:
        return self._n_total

    def reconstruct(self, params: torch.Tensor) -> torch.Tensor:
        """params: (B, 1+n_modes) -> Delta(t): (B, N_t) in [delta_min, delta_max]."""
        raw = params @ self.basis.T
        mid = 0.5 * (self.controls.delta_max + self.controls.delta_min)
        half = 0.5 * (self.controls.delta_max - self.controls.delta_min)
        return mid + half * torch.tanh(raw)

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(embed)
        return params, self.reconstruct(params)


# ---------------------------------------------------------------------------
# Architecture 3 — physics-prior heads
# ---------------------------------------------------------------------------

class MultiplicativeOmegaHead(nn.Module):
    """Learnable multiplicative modulation of the analytic Ω(t) envelope (Arch 3).

    Ω(t) = baseline(t) · g_θ(t),  with  g_θ(t) ∈ [g_min, g_max].

    The baseline is the fixed trapezoidal envelope derived from the blockade
    physics (same shape as ``AnalyticOmega`` / ``FixedScheduleBaseline``).
    The network outputs ``n_omega_modes`` Fourier coefficients defining a
    smooth modulation ``g_θ(t) = 1 + tanh(Σ a_k sin(k π t)) · max_gain`` so that:

    * At zero parameters, ``g_θ ≡ 1`` and Ω(t) is exactly the baseline.
    * Ω(t) inherits the baseline's zero boundary conditions
      (Ω(0) = Ω(T) = 0) and trapezoidal envelope.
    * The modulation cannot exceed ``[1 − max_gain, 1 + max_gain]``, so the
      pulse stays within physically reasonable scaling of the baseline.

    Parameters
    ----------
    embed_dim : int
        Graph embedding dimension produced by the encoder.
    controls : ControlsConfig
        Time grid + amplitude bounds.
    hardware : HardwareSpecs
        Physical constants for computing the trapezoidal baseline shape.
    R_b_um : float
        Physical blockade radius in μm (= ``udg.radius * udg.spacing``).
    hidden : int
        Hidden width of the parameter MLP.
    max_gain : float
        Maximum fractional modulation around 1.  E.g. 0.3 ⇒ g ∈ [0.7, 1.3].
    """

    def __init__(
        self,
        embed_dim: int,
        controls: ControlsConfig,
        hardware: HardwareSpecs,
        R_b_um: float,
        hidden: int = 64,
        max_gain: float = 0.3,
    ):
        super().__init__()
        self.controls = controls
        self._n_modes = controls.n_omega_modes
        self._max_gain = max_gain
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self._n_modes),
        )

        omega_peak = compute_blockade_omega(
            hardware.C6, R_b_um, hardware.omega_max,
        )
        T_us = controls.T * 1e6
        t = torch.linspace(0.0, 1.0, controls.N_t)
        r = max(hardware.t_ramp / T_us, 1e-9)
        o = hardware.t_onset / T_us
        rise = ((t - o) / r).clamp(0.0, 1.0)
        fall = ((1.0 - o - t) / r).clamp(0.0, 1.0)
        baseline = omega_peak * torch.min(rise, fall)
        self.register_buffer("baseline_omega", baseline)

        ks = torch.arange(1, self._n_modes + 1, dtype=torch.float32)
        sin_basis = torch.sin(math.pi * t.unsqueeze(1) * ks.unsqueeze(0))
        self.register_buffer("sin_basis", sin_basis)

        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    @property
    def n_params(self) -> int:
        return self._n_modes

    def reconstruct(self, params: torch.Tensor) -> torch.Tensor:
        """params: (B, n_modes) -> Ω(t): (B, N_t) in [0, baseline·(1+max_gain)]."""
        raw = params @ self.sin_basis.T
        modulation = 1.0 + self._max_gain * torch.tanh(raw)
        omega = self.baseline_omega.unsqueeze(0) * modulation
        return omega.clamp(min=0.0, max=self.controls.omega_max)

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(embed)
        return params, self.reconstruct(params)


class MonotoneDeltaHead(nn.Module):
    """Learnable monotone Δ(t) sweep parameterized by positive increments (Arch 3).

    Δ(t) is constructed as a cumulative sum of softplus-positive increments
    plus a learnable starting offset, then affine-mapped onto
    ``[delta_min, delta_max]``::

        Δ(t_k) = delta_min + (delta_max - delta_min) · cumsum(softplus(Δa_k)) / sum

    This guarantees monotone non-decreasing Δ(t) by construction — the policy
    cannot produce a non-adiabatic sweep direction, only modulate the
    *timing* of the sweep.  At zero parameters all increments are equal,
    producing the linear baseline sweep.

    The head outputs ``n_delta_modes`` increment parameters plus one offset.
    """

    def __init__(self, embed_dim: int, controls: ControlsConfig, hidden: int = 64):
        super().__init__()
        self.controls = controls
        n_inc = max(controls.n_delta_modes, 4)
        self._n_inc = n_inc
        self._n_params = n_inc + 1

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self._n_params),
        )

        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

        knot_t = torch.linspace(0.0, 1.0, n_inc + 1)
        grid_t = torch.linspace(0.0, 1.0, controls.N_t)
        M = torch.zeros(controls.N_t, n_inc + 1)
        for i, ti in enumerate(grid_t):
            j = torch.searchsorted(knot_t, ti).clamp(1, n_inc)
            t0, t1 = knot_t[j - 1], knot_t[j]
            w = (ti - t0) / (t1 - t0 + 1e-9)
            M[i, j - 1] = 1 - w
            M[i, j] = w
        self.register_buffer("interp_matrix", M)

    @property
    def n_params(self) -> int:
        return self._n_params

    def reconstruct(self, params: torch.Tensor) -> torch.Tensor:
        """params: (B, n_inc+1) -> Δ(t): (B, N_t) monotone in [delta_min, delta_max]."""
        inc_raw = params[:, : self._n_inc]
        offset = params[:, self._n_inc:]

        increments = torch.nn.functional.softplus(inc_raw + 1.0)
        cum = torch.cumsum(increments, dim=-1)
        normalized = cum / (cum[:, -1:] + 1e-9)

        knot_vals = torch.cat(
            [torch.zeros_like(normalized[:, :1]), normalized],
            dim=-1,
        )
        shift = 0.05 * torch.tanh(offset)
        knot_vals = (knot_vals + shift).clamp(0.0, 1.0)

        delta_grid = knot_vals @ self.interp_matrix.T

        return (
            self.controls.delta_min
            + (self.controls.delta_max - self.controls.delta_min) * delta_grid
        )

    def forward(self, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(embed)
        return params, self.reconstruct(params)
