from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from config import ParamKind


@dataclass(frozen=True)
class GlobalSchedule:
    """Global control schedule using Option A (global Ω and Δ).

    Parameters
    ----------
    omega : NDArray[np.float64]
        Global Rabi amplitudes Ω(t) sampled on a fixed grid of length T.
    delta : NDArray[np.float64]
        Global detunings Δ(t) sampled on a fixed grid of length T.
    dt : float
        Time step between samples in seconds.
    param_kind : ParamKind
        "pwc" or "pwl" (metadata only at this stage).
    """

    omega: NDArray[np.float64]
    delta: NDArray[np.float64]
    dt: float
    param_kind: ParamKind

    @property
    def n_steps(self) -> int:
        """Number of time-grid samples (length of omega / delta arrays)."""
        return int(self.omega.shape[0])

    def validate_shapes(self) -> None:
        assert self.omega.ndim == 1, "omega must be 1D"
        assert self.delta.ndim == 1, "delta must be 1D"
        assert self.omega.shape == self.delta.shape, "omega and delta must match"
