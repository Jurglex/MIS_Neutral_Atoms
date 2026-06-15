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
| **Module 2**         | `GlobalSchedule` + `nx.Graph` + `positions` (atom coords in μm)                               | `BackendResult`: estimated `p_MIS`, number of shots used, standard error, raw bitstring counts               |
| **Module 3**         | Batches of `(graph, reward)` pairs from Module 2, plus the current `SchedulePolicy`            | Updated model weights; next graph to evaluate                                                                |


## Current status

**Module 1 (implemented)** — Graph→Schedule policy network.
The primary model is `SchedulePolicy`: a 3-layer GIN encoder with concat(mean, max, sum) pooling, feeding into a reduced-basis decoder that outputs physically constrained Ω(t) and Δ(t) schedules. Three decoder architectures are available, selected via `architecture` in the config:

- **Architecture 1 (spline-knot):** Ω head uses 3 latent parameters (peak, width, center) reconstructed via a sin² envelope with boundary mask (Ω ≥ 0, Ω(0) = Ω(T) = 0). Δ head outputs 8 spline knots, linearly interpolated and tanh-clamped. Total action dim: 11 (or 8 with `learn_omega=false`).
- **Architecture 2 (Fourier):** Ω head uses K sine coefficients (default 5); sigmoid applied to the summed series ensures Ω ∈ [0, ω_max]. Δ head uses a DC offset + K cosine coefficients (default 8), tanh-clamped. Total action dim: 14 (or 9 with `learn_omega=false`).
- **Architecture 3 (physics-prior, new in v8):** Ω head is a *multiplicative modulation* of the analytic trapezoidal baseline — Ω(t) = Ω_baseline(t) · (1 + γ · tanh(Σ a_k sin(kπt))), with γ=0.3.  Cannot blow up Ω, preserves Ω(0)=Ω(T)=0 by construction.  Δ head is a *monotone* sweep parameterized by softplus-positive increments mapped affinely to [Δ_min, Δ_max].  Both heads are zero-initialized so at random init the policy produces *exactly* the adiabatic baseline; learning adds only physically meaningful perturbations.  **This is the default architecture.**

A Gaussian policy over the latent parameter space supports REINFORCE/PPO training.  `SchedulePolicy` also inherits from `ScheduleModel`, making it interchangeable with simpler baselines (`FixedScheduleBaseline`, `GNNModel`, `AdjacencyMLP`) in evaluation code.

**Module 2 (implemented)** — Quantum backends.
`BraketBackend` implements the `QuantumBackend` interface using Amazon Braket's AHS simulator and QuEra Aquila QPU.  It converts a `GlobalSchedule` (Ω/Δ in rad/μs) to Braket `TimeSeries` (rad/s), builds an `AtomArrangement` from atom positions (μm → m), validates against Aquila hardware limits, and post-processes measurement counts into a `BackendResult` with p_MIS and standard error.

**Module 3 (implemented, upgraded in v8)** — Learning and orchestration.
The default algorithm is **PPO with paired baselines**, K rollouts per graph, advantage normalization, and optional replay buffer (`module3/ppo.py`).  Training is preceded by a brief **behavioral-cloning pretraining** phase (`module3/pretrain.py`) that snaps the policy mean to the adiabatic baseline and the critic to the measured baseline reward before any RL.  A **residual-α curriculum** (`policy.set_residual_alpha`) scales the deviation from baseline at every forward pass, starting at 5 % and growing to 100 % over the warmup window — a trust region around the baseline that prevents early exploration from destroying the schedule.  The orchestrator runs **probe diagnostics** (`module3/diagnostics.py`) every `diagnostics_every` steps to verify the learned policy is genuinely graph-conditional (correlations of schedule deviation with graph features, conditioning index).  Legacy REINFORCE is still available via `--algorithm reinforce`.

## Repository structure

```
MIS_Neutral_Atoms/
├── config.py                  # ProjectConfig, ControlsConfig, UDGConfig, HardwareSpecs, RewardConfig
├── config.json                # Default experiment parameters
├── hardware_specs.json        # Device constants (C6, limits) and protocol timing
├── schedules.py               # GlobalSchedule dataclass (Ω, Δ, dt, param_kind)
│
├── graphs/
│   └── unit_disk.py           # generate_square_lattice_udg() → (nx.Graph, positions)
│
├── module1/                   # Graph→Schedule models
│   ├── base.py                # ScheduleModel ABC + FixedScheduleBaseline
│   ├── featurize.py           # graph_to_pyg(), Laplacian PE, triangles, algebraic connectivity
│   ├── encoder.py             # GINEncoder (3-layer GIN + concat pooling)
│   ├── heads.py               # OmegaHead, AnalyticOmega, DeltaHead, FourierOmegaHead,
│   │                          # FourierDeltaHead, MultiplicativeOmegaHead, MonotoneDeltaHead
│   ├── policy.py              # SchedulePolicy (encoder + heads + value + residual_alpha)
│   ├── gnn.py                 # GNNModel (simple baseline)
│   └── adjacency_mlp.py       # AdjacencyMLP (simple baseline)
│
├── module2/                   # Quantum backends
│   ├── interfaces.py          # QuantumBackend ABC, DeviceMetadata, BackendResult, Positions
│   ├── braket_backend.py      # BraketBackend: LocalSimulator + Aquila QPU via Braket AHS
│   ├── graph_MIS_utils.py     # MIS post-processing (check IS, find p_MIS from counts)
│   └── schedule_pmis_module.py # Low-level Braket AHS driver (reference implementation)
│
├── module3/                   # Learning / orchestration
│   ├── interfaces.py          # Learner ABC, Orchestrator ABC, TrainingConfig
│   ├── reinforce.py           # Legacy REINFORCE training step
│   ├── ppo.py                 # PPO step: paired baselines, K rollouts, advantage norm
│   ├── replay.py              # FIFO ReplayBuffer for off-policy reuse
│   ├── pretrain.py            # Behavioral-cloning pretraining (policy + critic)
│   ├── diagnostics.py         # Probe: schedule deviation, graph-conditioning index
│   ├── learner.py             # ReinforceLearner (PPO/REINFORCE, BC, replay, curation)
│   ├── orchestrator.py        # TrainingOrchestrator (BC → RL → eval → diag → ckpt)
│   └── backend_adapter.py     # QuantumBackend → reward fn; baseline cache; normalized r
│
├── train.py                   # CLI: python train.py [--algorithm ppo|reinforce --steps N ...]
│
├── visualization/
│   └── graphsample.py         # Sample and plot UDGs from config
│
├── tests/
│   ├── test_config_and_udg.py             # Config loading and UDG generation tests
│   ├── test_module1_model.py              # Module 1 model + policy + warm-start (34 tests)
│   ├── test_integration_m1_m2.py          # End-to-end Graph → M1 → M2 (3 archs)
│   ├── test_module3_training.py           # Core M3 training loop + reward fns (18 tests)
│   └── test_module3_improvements.py       # v8: PPO, Arch 3, BC, replay, diag (27 tests)
│
│   Total: 89 tests, all passing
│
└── requirements.txt
```

## Module 1 architecture

The encoder is shared; the decoder heads depend on the `architecture` config value.

```
networkx.Graph
    │
    ├─ featurize.py: degree, clustering, triangles, Laplacian PE (k=4)
    │                 graph-level: n/64, m/256, λ₂, density
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
concat with scalar graph features → 196-d
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
     GlobalSchedule
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


## Module 2 — Quantum backend

`BraketBackend` is the concrete `QuantumBackend` that bridges Module 1 output to Amazon Braket.

```
GlobalSchedule (rad/μs)         Positions (μm)
        │                           │
        ▼                           ▼
  ┌─ BraketBackend ─────────────────────────┐
  │  1. Unit conversion: rad/μs → rad/s     │
  │     positions: μm → m                   │
  │  2. Build Braket TimeSeries + register   │
  │  3. Validate against Aquila HW limits    │
  │  4. Run LocalSimulator or Aquila QPU     │
  │  5. Post-process: bitstring counts       │
  │     → find IS of target cardinality      │
  │     → p_MIS + binomial std error         │
  └──────────────────────────────────────────┘
        │
        ▼
  BackendResult
    .p_mis     (float)
    .shots     (int)
    .std_err   (float)
    .counts    (dict[str, int])
```

### Usage

```python
from config import load_project_config_json
from graphs.unit_disk import generate_square_lattice_udg
from module1.policy import SchedulePolicy
from module2.braket_backend import BraketBackend

cfg = load_project_config_json()
G, pos = generate_square_lattice_udg(cfg.udg)

policy = SchedulePolicy(cfg)
schedule = policy.make_schedule(G)

backend = BraketBackend(cfg, n_shots=100, backend_type="simulator")
result = backend.estimate_p_mis(schedule, G, pos)
print(f"p_MIS = {result.p_mis:.2%} ± {result.std_err:.2%}")
```

`amazon-braket-sdk` is an optional dependency — the rest of the codebase works without it.

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
    "param_kind": "pwl",
    "learn_omega": false,
    "architecture": 3,
    "omega_max": 15.8,
    "delta_min": -25.0,
    "delta_max": 25.0,
    "n_delta_knots": 8,
    "n_omega_modes": 5,
    "n_delta_modes": 8,
    "warm_start": true,
    "residual_alpha_start": 0.05,
    "residual_alpha_end": 1.0,
    "residual_alpha_warmup_steps": 200,
    "omega_scale": 1.0,
    "omega_cap": null
  },
  "udg": {
    "nx": 6,
    "ny": 6,
    "spacing": 4.0,
    "radius": 2.5,
    "dropout_rate": 0.4,
    "seed": 122
  },
  "reward": {
    "kind": "is_cost_vs_baseline",
    "penalty_U": 3.0,
    "mis_bonus": 0.0,
    "normalize_by_nodes": true,
    "baseline_norm_eps": 1e-3
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
| `architecture`  | `int`             | `3`      | Decoder architecture. `1` = spline-knot (Arch 1), `2` = Fourier-coefficient (Arch 2), `3` = physics-prior (multiplicative Ω × monotone Δ; default and recommended). All three share the same GIN encoder and value head.       |
| `omega_max`     | `float`           | `15.8`   | Maximum Rabi amplitude in rad/μs (Aquila hardware spec). Used as the output bound for the learned Ω head and as the peak of the analytic envelope.                                                                       |
| `delta_min`     | `float`           | `-25.0`  | Lower bound on detuning Δ(t) in rad/μs. The Δ decoder is tanh-clamped to this range.                                                                                                                                     |
| `delta_max`     | `float`           | `25.0`   | Upper bound on detuning Δ(t) in rad/μs.                                                                                                                                                                                  |
| `n_delta_knots` | `int`             | `8`      | Number of spline control points for Δ(t) (Arch 1). Linearly interpolated to the full `N_t` grid. Ignored by Arch 2/3.                                                                                                    |
| `n_omega_modes` | `int`             | `5`      | Number of sine modes for the Ω head (Arch 2 Fourier basis; Arch 3 modulation basis). Ignored by Arch 1.                                                                                                                  |
| `n_delta_modes` | `int`             | `8`      | Number of cosine modes for the Fourier Δ head (Arch 2) / monotone increments for the Δ head (Arch 3, min 4). Ignored by Arch 1.                                                                                          |
| `warm_start`    | `bool`            | `true`   | If true, Arch 1/2 head outputs are interpreted as *corrections* to the fixed adiabatic baseline (trapezoidal Ω + linear Δ sweep). At initialization the model produces the baseline; training learns graph-specific perturbations. Ignored by Arch 3 (residual structure is in the heads themselves). |
| `residual_alpha_start` | `float`    | `0.05`   | Initial scale on the residual correction (Arch 3 always; Arch 1/2 with `warm_start`). `α=0` ⇒ exactly baseline, `α=1` ⇒ unscaled head output. Defines an early trust region around the baseline.                          |
| `residual_alpha_end`   | `float`    | `1.0`    | Final scale after the warmup window. Set equal to `residual_alpha_start` to disable annealing.                                                                                                                            |
| `residual_alpha_warmup_steps` | `int` | `200`    | Number of training steps over which `α` linearly anneals from start → end.                                                                                                                                                |
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


#### `reward` — reward function for training

| Key                  | Type    | Default                  | Description                                                                                                                                                                                            |
| -------------------- | ------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `kind`               | `str`   | `"is_cost_vs_baseline"`  | Reward: `"is_cost"` (raw weighted IS cost), `"is_cost_vs_baseline"` (per-graph normalized improvement over baseline; default), `"p_mis"` (MIS hit rate), or `"composite"` (blend of is_cost + p_mis). |
| `penalty_U`          | `float` | `3.0`                    | Edge-violation penalty for cost-based kinds. Must be > 1. A shot with k selected nodes and v violated edges gets cost k − U·v.                                                                        |
| `mis_bonus`          | `float` | `0.0`                    | Weight λ on the MIS-hit term in `composite` mode: `r = (1−λ)·is_cost + λ·p_mis`. Ignored by the other reward kinds.                                                                                   |
| `normalize_by_nodes` | `bool`  | `true`                   | If true, divide IS cost by the number of nodes so rewards are comparable across graph sizes (independent of the vs-baseline normalization).                                                            |
| `baseline_norm_eps`  | `float` | `1e-3`                   | ε in the denominator of `is_cost_vs_baseline` — prevents division by zero on graphs whose baseline cost is exactly 0.                                                                                 |


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

## Reward function

The reward signal that drives training is configurable via the `reward` section of `config.json`.  Four options:

| Kind                     | Formula                                                                                        | Signal density | Needs \|MIS\|? |
| ------------------------ | ---------------------------------------------------------------------------------------------- | -------------- | -------------- |
| `"is_cost"`              | `(1/shots) Σ [Σᵢ sᵢ − U · Σ_{(i,j)∈E} sᵢsⱼ] / |V|`                                          | Dense          | No             |
| `"is_cost_vs_baseline"`  | `(is_cost_learned − is_cost_baseline) / (|is_cost_baseline| + ε)` (**default**)               | Dense          | No             |
| `"p_mis"`                | Fraction of shots that are IS of maximum cardinality                                           | Very sparse    | Yes (approx)   |
| `"composite"`            | `(1−λ)·is_cost + λ·p_mis`                                                                     | Tunable        | Yes (approx)   |

**`"is_cost"`** counts Rydberg excitations and penalizes edge violations.  Dense gradient signal from every shot, no |MIS| oracle required.  `U` (default 3.0) ensures violations are net-negative.

**`"is_cost_vs_baseline"` (default)** divides the per-graph improvement over the analytic adiabatic baseline by `|is_cost_baseline|`.  Removes graph-scale heterogeneity from the gradient signal — every graph now contributes "fractional improvement over baseline" rather than an absolute reward whose magnitude scales with graph size and density.  Backed by an automatic per-graph baseline cache (one simulator call per graph, reused throughout training).

**`"p_mis"` (legacy)** is the original binary MIS hit rate — useful as an evaluation metric but too sparse for efficient training.

**`"composite"`** blends `is_cost` and `p_mis`: start with `mis_bonus=0` for dense signal, anneal toward 1 to fine-tune for exact MIS hits.

## Module 3 — Training loop

`ReinforceLearner` selects between PPO (`algorithm="ppo"`, default) and legacy REINFORCE (`algorithm="reinforce"`).  PPO is recommended.

### Full pipeline (PPO path)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TrainingOrchestrator.run()                                               │
│                                                                          │
│  ── Pretraining phase ───────────────────────────────────────────────────│
│  1. BC policy:   policy.forward(G).{Ω, Δ}  ≈  baseline(G).{Ω, Δ}          │
│                  (MSE; no simulator calls)                                │
│  2. BC critic:   value_head(G)  ≈  measured baseline reward               │
│                  (1 simulator call per pool graph, cached)                │
│  3. Pool curation: keep G whose baseline ∈ (curation_lo, curation_hi)     │
│                                                                          │
│  ── RL phase ────────────────────────────────────────────────────────────│
│  for step in 1..total_steps:                                             │
│      α ← linear schedule (residual_alpha_start → end over warmup_steps)  │
│      policy.set_residual_alpha(α)                                        │
│                                                                          │
│      graphs = select_batch(train_pool)                                   │
│                                                                          │
│      ┌─────────────────────────────────────────────────────────────┐   │
│      │ ppo_step():                                                  │   │
│      │   1. For each G in graphs:                                  │   │
│      │        for k in 1..K (rollouts_per_graph):                  │   │
│      │            sample schedule → backend → reward                │   │
│      │        evaluate baseline schedule → r_baseline (cached)      │   │
│      │   2. A(G,k) = r_learned − r_baseline                         │   │
│      │   3. Normalize A across batch (zero mean, unit variance)     │   │
│      │   4. For epoch in 1..ppo_epochs:                             │   │
│      │        for minibatch:                                        │   │
│      │            ratio = exp(new_logp − old_logp)                  │   │
│      │            loss = -min(ratio·A, clip(ratio, 1±ε)·A)          │   │
│      │                 + 0.5·MSE(V, returns) − β·H[π]                │   │
│      │            optimizer.step() + grad clip                       │   │
│      │   5. (Optional) store fresh rollouts in ReplayBuffer          │   │
│      └─────────────────────────────────────────────────────────────┘   │
│                                                                          │
│      if step % eval_every: evaluate vs baseline → ckpt best              │
│      if step % diagnostics_every: probe deviation + conditioning index   │
│      if step % pool_refresh: regenerate pool, recurate                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Training quickstart

```bash
# Mock backend (no Braket SDK needed — random rewards for testing the loop)
python train.py --backend mock --steps 100 --batch-size 4

# Braket local simulator, PPO with default settings
python train.py --steps 2000 --shots 50 --batch-size 8

# Legacy REINFORCE instead of PPO
python train.py --algorithm reinforce --steps 2000

# Disable BC pretraining
python train.py --no-bc-pretrain --steps 2000

# Disable paired-baseline advantages (uses normalized rewards instead)
python train.py --no-paired-baseline --steps 2000

# Use a different reward kind
python train.py --reward is_cost --steps 2000

# See all options
python train.py --help
```

### Module 3 hyperparameters

The most important `TrainingConfig` knobs (defaults in parentheses):

| Knob                          | Default | What it controls                                                                                          |
| ----------------------------- | ------- | --------------------------------------------------------------------------------------------------------- |
| `algorithm`                   | `"ppo"` | `"ppo"` or `"reinforce"` (legacy).                                                                        |
| `rollouts_per_graph`          | `4`     | K stochastic rollouts evaluated per graph per gradient step. Higher = lower-variance, costlier.           |
| `ppo_clip`                    | `0.2`   | PPO ratio clip ε.                                                                                         |
| `ppo_epochs`                  | `4`     | Inner epochs over each collected batch of rollouts.                                                       |
| `use_paired_baseline`         | `True`  | Use `r_learned − r_baseline` as the advantage (per-graph adiabatic baseline).                              |
| `advantage_normalization`     | `True`  | Zero-mean, unit-variance the advantages each step.                                                        |
| `bc_pretrain_steps`           | `200`   | Supervised steps matching policy to the baseline schedule. 0 disables.                                    |
| `bc_critic_steps`             | `100`   | Supervised steps matching critic to measured baseline reward. 0 disables.                                 |
| `replay_buffer_size`          | `0`     | FIFO buffer of past rollouts for off-policy reuse. 0 disables.                                            |
| `replay_mix_ratio`            | `0.5`   | Fraction of each PPO minibatch drawn from the replay buffer.                                              |
| `pool_curation`               | `True`  | Filter out graphs whose baseline reward is outside `(curation_lo, curation_hi)`.                          |
| `diagnostics_every`           | `100`   | Steps between probe diagnostics (schedule deviation + conditioning index). 0 disables.                    |
| `init_log_std`                | `-0.5`  | Initial log-σ of the Gaussian policy (higher = more exploration around baseline).                         |

## Setup

```bash
# Create environment and install dependencies
pip install -r requirements.txt

# Run all tests (89 total across 5 test files)
python -m pytest tests/ -v

# Quick Module 1 + 3 unit tests (no Braket needed)
python -m pytest tests/test_module1_model.py tests/test_module3_training.py -v

# Integration tests (requires amazon-braket-sdk)
python tests/test_integration_m1_m2.py
```

All code assumes the repository root is the working directory (imports use `from config import ...`, `from module1 import ...`, etc.).