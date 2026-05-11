# MIS on Neutral Atoms

A system for solving the **Maximum Independent Set (MIS)** problem on neutral-atom quantum hardware.
Given an arbitrary graph, the pipeline produces time-dependent control schedules for an analog Hamiltonian simulation (AHS) device, executes them on a quantum backend, and uses the measurement outcomes to train the schedule-generating network.

## High-level pipeline

```
  Graph G            Module 1              Module 2              Module 3
 (NetworkX)    Graph→Schedule Net     Quantum Backend       Learning / Orchestration
─────────┐    ┌─────────────────┐    ┌────────────────┐    ┌────────────────────────┐
         │    │                 │    │                │    │                        │
  G ─────┼───►│ SchedulePolicy  │───►│ QuantumBackend │───►│  Learner               │
         │    │                 │    │                │    │  (REINFORCE / PPO)     │
         │    │   Ω(t), Δ(t)    │    │  p_MIS estimate│    │                        │
         │    └─────────────────┘    └────────────────┘    │  updates model weights │
         │                                                 │  selects next graph    │
         │                                                 └───────────┬────────────┘
         │                                                             │
         └─────────────────────────────────────────────────────────────┘
```

### Inputs and outputs at each stage


| Stage                | Input                                                                                         | Output                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Graph generation** | `UDGConfig` (grid size, spacing, radius, dropout rate, seed)                                  | `nx.Graph` with node positions — a unit-disk graph on a defective square lattice                             |
| **Module 1**         | `nx.Graph` (any graph; UDG or otherwise)                                                      | `GlobalSchedule`: time-discretized arrays Ω(t) and Δ(t) of length `N_t`, plus `dt` and parameterization kind |
| **Module 2**         | `GlobalSchedule` + `nx.Graph` + `DeviceMetadata` (backend type, shot budget, lattice spacing) | `BackendResult`: estimated `p_MIS`, number of shots used, standard error                                     |
| **Module 3**         | Batches of `(graph, p_MIS)` pairs from Module 2, plus the current `SchedulePolicy`            | Updated model weights; next graph to evaluate                                                                |


## Current status

**Module 1 (implemented)** — Graph→Schedule policy network.
The primary model is `SchedulePolicy`: a 3-layer GIN encoder with concat(mean, max, sum) pooling, feeding into a reduced-basis decoder that outputs physically constrained Ω(t) and Δ(t) schedules. Two decoder architectures are available, selected via `architecture` in the config:

- **Architecture 1 (spline-knot):** Ω head uses 3 latent parameters (peak, width, center) reconstructed via a sin² envelope with boundary mask (Ω ≥ 0, Ω(0) = Ω(T) = 0). Δ head outputs 8 spline knots, linearly interpolated and tanh-clamped. Total action dim: 11 (or 8 with `learn_omega=false`).
- **Architecture 2 (Fourier):** Ω head uses K sine coefficients (default 5); sigmoid applied to the summed series ensures Ω ∈ [0, ω_max]. Δ head uses a DC offset + K cosine coefficients (default 8), tanh-clamped. Total action dim: 14 (or 9 with `learn_omega=false`).

A Gaussian policy over the latent parameter space supports REINFORCE/PPO training. Two simpler baselines (`GNNModel`, `AdjacencyMLP`) are also available.

**Module 2 (interface only)** — Quantum backends.
Abstract `QuantumBackend` class defining the `estimate_p_mis` contract. Two backends are planned: a Bloqade simulator and QuEra Aquila hardware. Not implemented yet.

**Module 3 (interface only + REINFORCE scaffold)** — Learning and orchestration.
Abstract `Learner` and `Orchestrator` classes defining the training loop contract. A scaffold `reinforce_step` function is provided but requires a working backend to run.

## Repository structure

```
MIS_Neutral_Atoms/
├── config.py                  # ProjectConfig, ControlsConfig, UDGConfig, HardwareSpecs
├── config.json                # Default experiment parameters
├── hardware_specs.json        # Device constants (C6, limits) and protocol timing
├── schedules.py               # GlobalSchedule dataclass (Ω, Δ, dt, param_kind)
│
├── graphs/
│   └── unit_disk.py           # generate_square_lattice_udg() → (nx.Graph, positions)
│
├── module1/                   # Graph→Schedule models
│   ├── base.py                # ScheduleModel ABC
│   ├── featurize.py           # graph_to_pyg(), Laplacian PE, algebraic connectivity
│   ├── encoder.py             # GINEncoder (3-layer GIN + concat pooling)
│   ├── heads.py               # OmegaHead, AnalyticOmega, DeltaHead, FourierOmegaHead, FourierDeltaHead
│   ├── policy.py              # SchedulePolicy (full model: encoder + heads + value)
│   ├── gnn.py                 # GNNModel (simple baseline)
│   └── adjacency_mlp.py       # AdjacencyMLP (simple baseline)
│
├── module2/                   # Quantum backend interfaces
│   └── interfaces.py          # QuantumBackend ABC, DeviceMetadata, BackendResult
│
├── module3/                   # Learning / orchestration
│   ├── interfaces.py          # Learner ABC, Orchestrator ABC, TrainingConfig
│   └── reinforce.py           # REINFORCE training step scaffold
│
├── visualization/
│   └── graphsample.py         # Sample and plot UDGs from config
│
├── tests/
│   ├── test_config_and_udg.py # Config loading and UDG generation tests
│   └── test_module1_model.py  # Module 1 model / policy tests (25 tests)
│
└── requirements.txt
```

## Module 1 architecture

The encoder is shared; the decoder heads depend on the `architecture` config value.

```
networkx.Graph
    │
    ├─ featurize.py: degree, clustering, Laplacian PE (k=4), λ₂
    │
    ▼
PyG Data object
    │
    ▼
GINEncoder (3 layers, hidden=64)
    │
    ▼
concat(mean, max, sum) pooling → 192-d
    │
    ▼
concat with scalar graph features → 195-d
    │
    ├──────────────────┬────────────────────────────┐
    ▼                  ▼                            ▼
Ω head             Δ head                     Value head
(arch-dependent)   (arch-dependent)            (scalar, for PPO)
    │                  │
    ▼                  ▼
reconstruct Ω(t)   reconstruct Δ(t)
    │                  │
    ▼                  ▼
Ω(t) ∈ ℝ^N_t      Δ(t) ∈ ℝ^N_t
    └──────┬───────────┘
           ▼
     GlobalScheduleomega_cap
```

### Architecture comparison


|                      | **Arch 1 — Spline-knot**                                                | **Arch 2 — Fourier**                                                                  |
| -------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Ω head**           | 3 params (peak, width, center) → sin² envelope × sin²(πt) boundary mask | K_ω sine coefficients (default 5) → Σ aₖ sin(kπt), sigmoid → [0, ω_max]               |
| **Δ head**           | 8 spline knots → linear interpolation, tanh clamp                       | 1 DC + K_δ cosine coefficients (default 8) → tanh clamp                               |
| **Total action dim** | 11 (or 8 w/o Ω)                                                         | 14 (or 9 w/o Ω)                                                                       |
| **Ω boundary cond.** | Enforced by sin²(πt) mask — exact zeros at t=0, t=T                     | Sine basis vanishes at endpoints → raw=0 → sigmoid(0)=0.5·ω_max (soft, not hard zero) |
| **Ω non-negativity** | Guaranteed (sin² is non-negative)                                       | Guaranteed (sigmoid output in [0, ω_max])                                             |
| **Δ bounds**         | tanh clamp to [Δ_min, Δ_max]                                            | tanh clamp to [Δ_min, Δ_max]                                                          |
| **Expressiveness**   | Local control via spline knots                                          | Global control via Fourier modes; inherently smooth                                   |
| **Best for**         | Sharp, localized schedule features                                      | Smooth, globally structured schedules                                                 |


## Key data types

### `GlobalSchedule`

The central data object that flows from Module 1 to Module 2:


| Field        | Type                              | Description                                                                   |
| ------------ | --------------------------------- | ----------------------------------------------------------------------------- |
| `omega`      | `NDArray[float64]` shape `(N_t,)` | Rabi drive amplitude Ω(t) in rad/μs, sampled on a uniform time grid           |
| `delta`      | `NDArray[float64]` shape `(N_t,)` | Detuning Δ(t) in rad/μs, same grid as omega                                   |
| `dt`         | `float`                           | Time step between samples (seconds), derived as `T / (N_t - 1)`               |
| `param_kind` | `"pwc"` or `"pwl"`                | Whether samples represent piecewise-constant values or piecewise-linear knots |
| `n_steps`    | `int` (property)                  | Number of time-grid points (i.e. `len(omega)`)                                |


### `ProjectConfig` / `config.json`

All experiment parameters in one place, loaded from `config.json`:

```json
{
  "backend": "bloqade",
  "controls": {
    "T": 4.0e-06,
    "N_t": 64,
    "param_kind": "pwc",
    "learn_omega": false,
    "architecture": 1,
    "omega_max": 15.8,
    "delta_min": -25.0,
    "delta_max": 25.0,
    "n_delta_knots": 8,
    "n_omega_modes": 5,
    "n_delta_modes": 8,
    "omega_scale": 1.0,
    "omega_cap": null
  },
  "udg": {
    "nx": 6,
    "ny": 6,
    "spacing": 1.0,
    "radius": 2.5,
    "dropout_rate": 0.4,
    "seed": 122
  }
}
```

#### Top-level


| Key       | Type                     | Description                                                                                                     |
| --------- | ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `backend` | `"bloqade"` | `"aquila"` | Which quantum backend to target. `"bloqade"` runs a classical simulation; `"aquila"` submits to QuEra hardware. |


#### `controls` — time grid and drive settings


| Key             | Type              | Default  | Description                                                                                                                                                                                                              |
| --------------- | ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `T`             | `float`           | `4.0e-6` | Total evolution time in seconds.                                                                                                                                                                                         |
| `N_t`           | `int`             | `128`    | Number of time-grid points (including endpoints). The time step is derived as `dt = T / (N_t - 1)`.                                                                                                                      |
| `param_kind`    | `"pwc"` | `"pwl"` | `"pwc"`  | Time parameterization. `"pwc"` = piecewise-constant (each array element is a constant value over one time step). `"pwl"` = piecewise-linear (array elements are knot values, linearly interpolated between grid points). |
| `learn_omega`   | `bool`            | `false`  | Controls what the ML model outputs. When `false`, only Δ(t) is learned and Ω(t) is a fixed analytic sin² envelope at `omega_max`. When `true`, the model outputs both Ω(t) and Δ(t).                                     |
| `architecture`  | `int`             | `1`      | Decoder architecture. `1` = spline-knot (Arch 1), `2` = Fourier-coefficient (Arch 2). Both share the same GIN encoder and value head.                                                                                    |
| `omega_max`     | `float`           | `15.8`   | Maximum Rabi amplitude in rad/μs (Aquila hardware spec). Used as the output bound for the learned Ω head and as the peak of the analytic envelope.                                                                       |
| `delta_min`     | `float`           | `-25.0`  | Lower bound on detuning Δ(t) in rad/μs. The Δ decoder is tanh-clamped to this range.                                                                                                                                     |
| `delta_max`     | `float`           | `25.0`   | Upper bound on detuning Δ(t) in rad/μs.                                                                                                                                                                                  |
| `n_delta_knots` | `int`             | `8`      | Number of spline control points for Δ(t) (Arch 1). Linearly interpolated to the full `N_t` grid. Ignored by Arch 2.                                                                                                      |
| `n_omega_modes` | `int`             | `5`      | Number of sine modes for the Fourier Ω head (Arch 2). Ignored by Arch 1.                                                                                                                                                 |
| `n_delta_modes` | `int`             | `8`      | Number of cosine modes for the Fourier Δ head (Arch 2). Total Δ params = `1 + n_delta_modes` (DC + modes). Ignored by Arch 1.                                                                                            |
| `omega_scale`   | `float`           | `1.0`    | Scale factor for the simple baseline Ω derivation (`omega_scale / radius^6`). Used only by the non-learned baseline models (`GNNModel`, `AdjacencyMLP`).                                                                 |
| `omega_cap`     | `float` | `null`  | `null`   | Optional upper bound on the baseline Ω derivation. `null` means no cap. Used only by the baseline models.                                                                                                                |


#### `udg` — unit-disk graph generation


| Key            | Type    | Default | Description                                                                                                                                              |
| -------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nx`           | `int`   | `8`     | Number of lattice sites along the x-axis.                                                                                                                |
| `ny`           | `int`   | `8`     | Number of lattice sites along the y-axis.                                                                                                                |
| `spacing`      | `float` | `1.0`   | Distance between adjacent lattice sites in μm.                                                                                                           |
| `radius`       | `float` | `1.5`   | Unit-disk connection radius in units of `spacing`. Physical distance = `radius × spacing` μm. Also used as the blockade radius when computing Ω from C₆. |
| `dropout_rate` | `float` | `0.0`   | Fraction of lattice sites to remove. Exactly `round(dropout_rate * nx * ny)` sites are dropped (deterministic count, random selection via `seed`).       |
| `seed`         | `int`   | `0`     | RNG seed for deterministic dropout and graph generation.                                                                                                 |


### `hardware_specs.json` — device constants and protocol timing

Loaded alongside `config.json`.  If the file is missing, built-in Aquila defaults are used.

```json
{
  "C6": 5.42e6,
  "omega_max": 15.8,
  "delta_min": -125.0,
  "delta_max": 125.0,
  "t_ramp": 0.3,
  "t_onset": 0.0
}
```


| Key         | Type    | Default  | Description                                                                           |
| ----------- | ------- | -------- | ------------------------------------------------------------------------------------- |
| `C6`        | `float` | `5.42e6` | Van der Waals coefficient for the Rydberg state (rad/μs · μm⁶). Aquila ⁸⁷Rb |70S₁/₂⟩. |
| `omega_max` | `float` | `15.8`   | Hardware upper bound on global Rabi amplitude Ω (rad/μs).                             |
| `delta_min` | `float` | `-125.0` | Hardware lower bound on global detuning Δ (rad/μs).                                   |
| `delta_max` | `float` | `125.0`  | Hardware upper bound on global detuning Δ (rad/μs).                                   |
| `t_ramp`    | `float` | `0.3`    | Linear ramp-up / ramp-down duration for the trapezoidal Ω envelope (μs).              |
| `t_onset`   | `float` | `0.0`    | Delay before the Ω ramp begins (μs). Usually 0.                                       |


When `learn_omega` is `false`, the analytic Ω peak is computed from the blockade condition: `Ω_peak = min(C₆ / R_b⁶, omega_max)` where `R_b = udg.radius × udg.spacing` (μm).  The envelope is trapezoidal: ramp up over `t_ramp`, hold at `Ω_peak`, ramp down over `t_ramp`.

## Setup

```bash
# Create environment and install dependencies
pip install -r requirements.txt

# Run tests (29 tests covering config, hardware specs, UDG, featurization, heads, and full policy)
python -m pytest tests/ -v
```

All code assumes the repository root is the working directory (imports use `from config import ...`, `from module1 import ...`, etc.).