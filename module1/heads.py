"""Schedule decoder heads: Omega(t) and Delta(t) reconstruction from latent parameters."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from config import ControlsConfig


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
