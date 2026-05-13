"""Low-level Braket AHS driver.

Takes a piecewise schedule (duration/value arrays) and atom positions,
builds an ``AnalogHamiltonianSimulation``, runs it on the Braket local
simulator or QuEra Aquila, and post-processes the measurement report.

.. note::
   Prefer the higher-level :class:`~module2.braket_backend.BraketBackend`
   for integration with the rest of the pipeline.  This module is kept as
   the original reference implementation.
"""
from __future__ import annotations

import math
import os
import pickle
import time
from datetime import datetime

import numpy as np

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries
from braket.devices import LocalSimulator
from braket.aws import AwsDevice

from module2.graph_MIS_utils import (
    check_independent_set,
    find_MIS_probability,
    regroup_by_ones_counts,
)

# arXiv:2306.11727 Sec. 1.5, p. 16 — Program / geometry limits (Braket-implementable programs)
AQUILA_PROGRAM_GEOMETRY_LIMITS = {
    "max_user_defined_sites_total": 256,
    "max_qubits_filled_sites": 256,
    "max_site_pattern_width_um": 75.0e-6,
    "max_site_pattern_height_um": 76.0e-6,
    "min_distance_between_sites_um": 4.0e-6,
    "min_vertical_spacing_between_rows_um": 4.0e-6,
    "max_rydberg_rabi_frequency_rad_per_us": 15.8e6,
    "max_rabi_slew_rate_rad_per_us2": 250.0e12,
    "max_detuning_abs_rad_per_us": 125.0e6,
    "max_user_defined_evolution_duration_us": 4.0e-6,
}

def simulation(mode, schedule, positions, L, Rb_a, a, t_max, nshots, backend = 'simulator', save = False):
    """
    Returns the measurement report in result_full

    mode: "delta_only", "omega_delta"

    schedule = {'OMEGA_DUR': omega_durations,
    'OMEGA_VALS': omega_values,
    'DELTA_DUR': delta_durations,
    'DELTA_VALS': delta_values}
    """
    
    # constant
    C6 = 2 * np.pi * 862690e6
    max_omega = C6 / (Rb_a * a*1e6) ** 6

    if mode == 'delta_only': # then len(schedule) = 2 {'DELTA_DUR': delta_durations, 'DELTA_VALS': delta_values}
        max_omega = C6 / (Rb_a * a * 1e6) ** 6
        schedule['OMEGA_DUR'] = np.array([0. , 0.2, 3.8, 4. ])
        schedule['OMEGA_VALS'] = np.array([ 0.        , max_omega, max_omega,  0.        ])
    else: 
        max_omega = max(schedule['OMEGA_VALS'])

    errors = []
    if a < AQUILA_PROGRAM_GEOMETRY_LIMITS['min_distance_between_sites_um']:
        errors.append(f"min_distance_between_sites_um : {AQUILA_PROGRAM_GEOMETRY_LIMITS['min_distance_between_sites_um']}")
    if t_max > AQUILA_PROGRAM_GEOMETRY_LIMITS['max_user_defined_evolution_duration_us']:
        errors.append(f"max_user_defined_evolution_duration_us : {AQUILA_PROGRAM_GEOMETRY_LIMITS['max_user_defined_evolution_duration_us']}")
    if L * a > AQUILA_PROGRAM_GEOMETRY_LIMITS['max_site_pattern_width_um']:
        errors.append(f"max_site_pattern_height_um : {AQUILA_PROGRAM_GEOMETRY_LIMITS['max_site_pattern_height_um']}, max_site_pattern_width_um : {AQUILA_PROGRAM_GEOMETRY_LIMITS['max_site_pattern_width_um']}")
    if max_omega > AQUILA_PROGRAM_GEOMETRY_LIMITS['max_rydberg_rabi_frequency_rad_per_us']:
        errors.append(f"max_rydberg_rabi_frequency_rad_per_us : {AQUILA_PROGRAM_GEOMETRY_LIMITS['max_rydberg_rabi_frequency_rad_per_us']}")
    if max(np.abs(schedule['DELTA_VALS'])) > AQUILA_PROGRAM_GEOMETRY_LIMITS['max_detuning_abs_rad_per_us']:
        errors.append(f"max_detuning_abs_rad_per_us : {AQUILA_PROGRAM_GEOMETRY_LIMITS['max_detuning_abs_rad_per_us']}")
    if np.max(np.abs(np.diff(schedule['OMEGA_VALS']) / np.diff(schedule['OMEGA_DUR']))) >= AQUILA_PROGRAM_GEOMETRY_LIMITS['max_rabi_slew_rate_rad_per_us2']:
        errors.append(f"max_rabi_slew_rate_rad_per_us2 : {AQUILA_PROGRAM_GEOMETRY_LIMITS['max_rabi_slew_rate_rad_per_us2']}")

    if errors:
        raise ValueError("Invalid parameters:\n" + "\n".join(f"- {e}" for e in errors))

    Omegas = TimeSeries()
    Deltas = TimeSeries()

    for omega_pair in zip(schedule['OMEGA_DUR'], schedule['OMEGA_VALS']):
        dur, val = omega_pair
        Omegas.put(dur, val)

    for delta_pair in zip(schedule['DELTA_DUR'], schedule['DELTA_VALS']):
        dur, val = delta_pair
        Deltas.put(dur, val)

    Phi = TimeSeries().put(0.0, 0.0).put(t_max, 0.0)
    drive = DrivingField(amplitude=Omegas, phase=Phi, detuning=Deltas)

    register = AtomArrangement()
    for pos in positions:
        register.add(pos)

    ahs_program = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)
    result_full = None

    n_atoms = len(positions)

    if backend == "simulator":
        device = LocalSimulator("braket_ahs")
        print(
            f"Starting LocalSimulator (braket_ahs): {n_atoms} atoms, {nshots} shots.",
            flush=True,
        )
        print(
            "The local AHS solver runs synchronously: there is no progress output until `.result()` "
            "returns. Runtime grows with atom count and shots (often many minutes). "
            "Use a small `nshots` (e.g. 5–20) to sanity-check before scaling up.",
            flush=True,
        )
        start_time = time.time()
        result_full = device.run(ahs_program, shots=nshots).result()
        print(f"The elapsed time = {time.time() - start_time:.2f} seconds")

        if save:
            os.makedirs("sim_result_full", exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sim_result_full/{backend}-nshots-{nshots}-{stamp}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(result_full, f)
        return result_full

    if backend == "aquila_qpu":
        aquila_qpu = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
        discretized_ahs_program = ahs_program.discretize(aquila_qpu)
        task = aquila_qpu.run(discretized_ahs_program, shots=nshots)
        metadata = task.metadata()
        task_arn = metadata["quantumTaskArn"]
        task_status = metadata["status"]

        print(f"ARN: {task_arn}")
        print(f"status: {task_status}")
        return task.result()

    raise ValueError(
        f"Unsupported backend={backend!r}; use 'simulator' or 'aquila_qpu'."
    )

def analyzing_sim_report(result_full, G, nshots):
    """ 
    Takes in a measurement report from the LocalSimulator("braket_ahs"), the graph G and returns the probability of finding an MIS
    adj, positions = get_lattice_UDG(L, Rb_a, a, dropout_rate)
    G = nx.from_numpy_array(adj)
    """
    counts = result_full.get_counts()
    count_ones = regroup_by_ones_counts(counts)

    MIS_probability, qMIS_bitstrings, qMIS_tuples = find_MIS_probability(G, counts, nshots)
    return MIS_probability, qMIS_bitstrings, qMIS_tuples, count_ones