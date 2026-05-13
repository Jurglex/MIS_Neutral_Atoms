"""Concrete QuantumBackend for Amazon Braket AHS (local simulator & Aquila).

This module bridges the project's :class:`GlobalSchedule` representation
(Ω/Δ in rad/μs, time grid in seconds) to the Braket ``TimeSeries`` format
(rad/s, seconds) and handles atom-register construction, hardware-limit
validation, simulation execution, and MIS post-processing.

The ``amazon-braket-sdk`` is imported lazily so the rest of the codebase
works without it installed.
"""
from __future__ import annotations

import time
from typing import Literal

import numpy as np
import networkx as nx

from config import ProjectConfig, HardwareSpecs, compute_blockade_omega
from schedules import GlobalSchedule
from module2.interfaces import BackendResult, DeviceMetadata, Positions, QuantumBackend


# ── Aquila hardware limits (arXiv:2306.11727 §1.5) ─────────────────────
AQUILA_LIMITS = {
    "max_sites": 256,
    "width_m": 75.0e-6,
    "height_m": 76.0e-6,
    "min_spacing_m": 4.0e-6,
    "max_omega_rad_s": 15.8e6,
    "max_delta_rad_s": 125.0e6,
    "max_slew_rad_s2": 250.0e12,
    "max_T_s": 4.0e-6,
}


def _validate_program(
    positions_m: list[tuple[float, float]],
    omega_rad_s: np.ndarray,
    delta_rad_s: np.ndarray,
    times_s: np.ndarray,
    T_s: float,
) -> None:
    """Raise ``ValueError`` if the program violates Aquila limits."""
    errors: list[str] = []

    n = len(positions_m)
    if n > AQUILA_LIMITS["max_sites"]:
        errors.append(f"Too many sites ({n} > {AQUILA_LIMITS['max_sites']})")

    if T_s > AQUILA_LIMITS["max_T_s"] + 1e-12:
        errors.append(
            f"Evolution time {T_s*1e6:.2f} μs > "
            f"{AQUILA_LIMITS['max_T_s']*1e6:.1f} μs limit"
        )

    if np.max(np.abs(omega_rad_s)) > AQUILA_LIMITS["max_omega_rad_s"] * 1.001:
        errors.append(
            f"Ω peak {np.max(np.abs(omega_rad_s))/1e6:.2f} rad/μs > "
            f"{AQUILA_LIMITS['max_omega_rad_s']/1e6:.1f} rad/μs limit"
        )

    if np.max(np.abs(delta_rad_s)) > AQUILA_LIMITS["max_delta_rad_s"] * 1.001:
        errors.append(
            f"|Δ| peak {np.max(np.abs(delta_rad_s))/1e6:.2f} rad/μs > "
            f"{AQUILA_LIMITS['max_delta_rad_s']/1e6:.1f} rad/μs limit"
        )

    dt = np.diff(times_s)
    d_omega = np.diff(omega_rad_s)
    slew = np.where(dt > 0, np.abs(d_omega / dt), 0.0)
    if np.max(slew) > AQUILA_LIMITS["max_slew_rad_s2"] * 1.001:
        errors.append(
            f"Ω slew rate {np.max(slew):.2e} rad/s² > "
            f"{AQUILA_LIMITS['max_slew_rad_s2']:.2e} rad/s² limit"
        )

    coords = np.array(positions_m)
    xs, ys = coords[:, 0], coords[:, 1]
    span_x = xs.max() - xs.min() if len(xs) > 1 else 0.0
    span_y = ys.max() - ys.min() if len(ys) > 1 else 0.0
    if span_x > AQUILA_LIMITS["width_m"] + 1e-9:
        errors.append(f"Register width {span_x*1e6:.1f} μm > {AQUILA_LIMITS['width_m']*1e6:.1f} μm")
    if span_y > AQUILA_LIMITS["height_m"] + 1e-9:
        errors.append(f"Register height {span_y*1e6:.1f} μm > {AQUILA_LIMITS['height_m']*1e6:.1f} μm")

    if n > 1:
        from scipy.spatial.distance import pdist

        dists = pdist(coords)
        if dists.min() < AQUILA_LIMITS["min_spacing_m"] - 1e-9:
            errors.append(
                f"Min atom spacing {dists.min()*1e6:.2f} μm < "
                f"{AQUILA_LIMITS['min_spacing_m']*1e6:.1f} μm limit"
            )

    if errors:
        raise ValueError(
            "Program violates Aquila hardware limits:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


def _schedule_to_braket_timeseries(
    values_rad_us: np.ndarray,
    dt_s: float,
):
    """Convert a project-convention array to a Braket ``TimeSeries``.

    Parameters
    ----------
    values_rad_us : 1-D array
        Signal values in rad/μs (our internal convention).
    dt_s : float
        Time step in seconds.

    Returns
    -------
    TimeSeries
        Braket ``TimeSeries`` with times in seconds and values in rad/s.
    """
    from braket.timings.time_series import TimeSeries

    ts = TimeSeries()
    n = len(values_rad_us)
    for i in range(n):
        t_s = float(i * dt_s)
        val_rad_s = float(values_rad_us[i]) * 1e6  # rad/μs → rad/s
        ts.put(t_s, val_rad_s)
    return ts


class BraketBackend(QuantumBackend):
    """Braket AHS backend — local simulator or QuEra Aquila QPU.

    Parameters
    ----------
    config : ProjectConfig
        Full project configuration (used for metadata only; the actual
        schedule and positions are passed per-call).
    n_shots : int
        Number of measurement shots per run.
    backend_type : ``"simulator"`` | ``"aquila_qpu"``
        Which Braket device to target.
    validate : bool
        If ``True`` (default), check Aquila hardware limits before running.
    """

    def __init__(
        self,
        config: ProjectConfig,
        n_shots: int = 1000,
        backend_type: Literal["simulator", "aquila_qpu"] = "simulator",
        validate: bool = True,
    ):
        device = DeviceMetadata(
            backend="bloqade" if backend_type == "simulator" else "aquila",
            n_sites=0,
            shot_budget=n_shots,
            lattice_spacing_um=config.udg.spacing,
        )
        super().__init__(device)
        self.config = config
        self.n_shots = n_shots
        self.backend_type = backend_type
        self.validate = validate

        # Physical blockade radius: R_b = (C6 / Ω_max)^(1/6) in μm → m.
        # Passing this to the local simulator truncates the Hilbert space to
        # independent-set configurations, giving orders-of-magnitude speedup.
        hw = config.hardware
        R_b_um = (hw.C6 / hw.omega_max) ** (1.0 / 6.0)
        self._blockade_radius_m = R_b_um * 1e-6

    # ── public API ──────────────────────────────────────────────────────

    def estimate_p_mis(
        self,
        schedule: GlobalSchedule,
        graph: nx.Graph,
        positions: Positions,
        *,
        seed: int | None = None,
    ) -> BackendResult:
        """Run an AHS program and return the MIS probability estimate.

        Parameters
        ----------
        schedule : GlobalSchedule
            Ω(t) and Δ(t) arrays in rad/μs with ``dt`` in seconds.
        graph : nx.Graph
            The interaction graph (node IDs must match *positions* keys).
        positions : Positions
            ``{node_id: (x_μm, y_μm)}`` — atom coordinates in **μm**.
        seed : int | None
            RNG seed (local simulator only, currently unused by Braket).
        """
        from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
        from braket.ahs.atom_arrangement import AtomArrangement
        from braket.ahs.driving_field import DrivingField
        from braket.timings.time_series import TimeSeries

        schedule.validate_shapes()
        N_t = schedule.n_steps
        dt_s = schedule.dt
        T_s = dt_s * (N_t - 1)

        # Unit conversions
        omega_rad_s = schedule.omega * 1e6
        delta_rad_s = schedule.delta * 1e6
        times_s = np.arange(N_t) * dt_s

        # Atom register (μm → m)
        node_order = sorted(positions.keys())
        positions_m = [(positions[n][0] * 1e-6, positions[n][1] * 1e-6) for n in node_order]

        if self.validate:
            _validate_program(positions_m, omega_rad_s, delta_rad_s, times_s, T_s)

        # Build Braket objects
        omega_ts = _schedule_to_braket_timeseries(schedule.omega, dt_s)
        delta_ts = _schedule_to_braket_timeseries(schedule.delta, dt_s)
        phi_ts = TimeSeries().put(0.0, 0.0).put(T_s, 0.0)
        drive = DrivingField(amplitude=omega_ts, phase=phi_ts, detuning=delta_ts)

        register = AtomArrangement()
        for pos in positions_m:
            register.add(pos)

        ahs_program = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)

        # Execute
        result = self._run(ahs_program)

        # Post-process
        counts = result.get_counts()
        p_mis, mis_bitstrings, mis_tuples = self._compute_p_mis(graph, counts)
        std_err = BackendResult.binomial_std_err(p_mis, self.n_shots)

        return BackendResult(
            p_mis=p_mis,
            shots=self.n_shots,
            std_err=std_err,
            counts=counts,
            metadata={
                "mis_bitstrings": mis_bitstrings,
                "mis_node_sets": mis_tuples,
                "backend_type": self.backend_type,
            },
        )

    # ── private helpers ─────────────────────────────────────────────────

    def _run(self, ahs_program):
        """Execute the AHS program on the selected backend."""
        if self.backend_type == "simulator":
            from braket.devices import LocalSimulator

            device = LocalSimulator("braket_ahs")
            print(
                f"BraketBackend: running LocalSimulator — "
                f"{self.n_shots} shots, "
                f"blockade_radius={self._blockade_radius_m*1e6:.1f} μm …",
                flush=True,
            )
            t0 = time.time()
            result = device.run(
                ahs_program,
                shots=self.n_shots,
                blockade_radius=self._blockade_radius_m,
            ).result()
            print(f"BraketBackend: done in {time.time() - t0:.1f} s", flush=True)
            return result

        if self.backend_type == "aquila_qpu":
            from braket.aws import AwsDevice

            aquila = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
            discretized = ahs_program.discretize(aquila)
            task = aquila.run(discretized, shots=self.n_shots)
            print(f"BraketBackend: submitted to Aquila — ARN {task.metadata()['quantumTaskArn']}")
            return task.result()

        raise ValueError(f"Unknown backend_type={self.backend_type!r}")

    @staticmethod
    def _compute_p_mis(
        graph: nx.Graph,
        counts: dict[str, int],
    ) -> tuple[float, list[str], list[tuple[int, ...]]]:
        """Extract MIS probability from Braket measurement counts.

        Uses the approximation MIS cardinality from networkx to set the
        target size, then filters bitstrings that are independent sets of
        that cardinality.
        """
        from module2.graph_MIS_utils import find_MIS_probability

        total_shots = sum(counts.values())
        p_mis, mis_bitstrings, mis_tuples = find_MIS_probability(
            graph, counts, total_shots, verbose=False,
        )
        return p_mis, mis_bitstrings, mis_tuples
