# Rydberg MIS Schedule Learning — Design Log

A living document tracking architectural decisions, rationale, alternatives considered, and open questions for the GNN→schedule policy network.

---

## 1. Problem framing (locked)

- **Input:** `networkx.Graph` (UDG on defective square lattice, n ∈ [20, 60], but architecture should generalize).
- **Output:** `GlobalSchedule` = (Ω(t), Δ(t)) on N_t ∈ [64, 128] grid. Piecewise-constant or piecewise-linear.
- **Reward:** scalar `p_MIS` (or shaped variant) from finite shots on Bloqade / Aquila. No analytic gradients through physics.
- **Optimizer:** policy gradient (REINFORCE → PPO).
- **Toggle:** `learn_omega ∈ {True, False}`. When False, Ω(t) is analytic and only Δ(t) is learned.
- **Constraints:** Ω(t) ≥ 0, Ω(0) = Ω(T) = 0 (smooth ramp), Δ ∈ [Δ_min, Δ_max], typically swept negative→positive.

---

## 2. Key design decisions

### 2.1 Encoder: GIN + structural node features + concat pooling

**Decision:** 3-layer GIN, hidden 64, node features = [degree, clustering, triangle_count, top-k Laplacian eigenvector entries], pooled with concat(mean, max, sum), augmented with scalar graph features (n, m, λ₂, density).

**Rationale:**

- GIN is WL-expressive; matches MIS structural sensitivity.
- Sum pooling carries graph-size information that mean alone discards — important because optimal total annealing time scales with problem size.
- Laplacian PE breaks symmetries between structurally similar nodes that pure GNNs cannot distinguish.
- λ₂ (algebraic connectivity) is a cheap, physically meaningful summary that correlates with adiabatic time requirements.

**Alternatives rejected:**

- **GCN:** over-smooths at depth >3, weaker on combinatorial tasks.
- **GAT:** extra params without clear MIS benefit at this scale.
- **Adjacency-flatten + MLP:** breaks permutation invariance and size generalization.
- **Pure spectral encoder:** loses local structure that matters for blockade physics.

**Open questions:**

- Is 4 Laplacian eigenvectors enough? Could try 8.
- Should we add a "distance to boundary" feature for defective-lattice UDGs specifically?

### 2.2 Decoder: low-dimensional reduced basis (NOT per-timestep MLP)

**Decision:** Decoder outputs ~11 latent params:

- **Ω head (3 params):** peak amplitude (sigmoid), envelope width, envelope center → reconstruct via `Ω(t) = Ω_max · σ(a) · sin²(...)`.
- **Δ head (8 spline control points):** values at fixed knot times → cubic interpolation, tanh-clamped to [Δ_min, Δ_max].
- Optional: learnable total time T as a 12th param.

**Rationale (the most important decision in the system):**

- Action space dimensionality is the dominant factor for sample efficiency under noisy scalar reward. 11 params vs. 256 raw values is ~20× reduction.
- Smoothness and boundary conditions are enforced *by construction*, not by penalty, so the policy never wastes shots on infeasible schedules.
- Matches inductive bias of adiabatic protocols (smooth, monotone-ish Δ; bump-shaped Ω).
- Khairy et al. (AAAI 2020) found low-dim parameterization wins for QAOA-RL; logic transfers a fortiori here since N_t >> QAOA depth.

**Alternatives considered:**

- **Fourier coeffs (Architecture 2):** now implemented as an alternative — see §3b below. Slightly higher action dim, cleaner sine-basis boundary handling for Ω, but uses sigmoid instead of hard zero at endpoints.
- **Autoregressive GRU per-step (Architecture 3):** maximally expressive, but N_t-dim action space + noisy reward = brutal.
- **Direct MLP → N_t values:** rejected outright.

**Open questions:**

- Should knot times be learnable too, or fixed? Start fixed; revisit if Δ shapes look too constrained.
- For very hard instances (Cain et al.), do we need non-monotone Δ? Spline allows it; just don't enforce monotonicity.

### 2.3 Constraint handling: reparameterization, not activations on raw outputs

**Decision:** Build constraints into the parameterization itself.

- Ω: amplitude × windowed sin² shape × sin²(πt) boundary mask. The boundary mask guarantees Ω(0) = Ω(T) = 0 exactly, regardless of the learned width/center parameters. The windowed shape controls where the pulse is concentrated.
- Δ: tanh maps ℝ → [Δ_min, Δ_max].

**Rationale:** Raw outputs + clipping creates dead-gradient regions and wastes shots. Reparameterization gives nonzero gradient everywhere in feasible space.

**Bug found during integration:** The original windowed sin² envelope did not guarantee Ω(T)=0. When center + width/2 > 1 (common at initialization), the normalized coordinate u at t=1 fell short of 1.0, leaving sin²(πu) ≠ 0. Fixed by multiplying the entire profile by sin²(πt), which is 0 at both endpoints regardless of other parameters.

### 2.4 Policy & training

**Decision:**

- **Policy:** diagonal Gaussian over the latent parameter vector. Mean from network, log_std as learned per-parameter scalar (initialized small, e.g., -1.0).
- **Algorithm:** PPO with shared encoder + value head. Start with REINFORCE + baseline if PPO is overkill for early experiments.
- **Warm start:** initialize so that the mean policy reproduces a linear Ebadi-style ramp. Skips ~thousands of shots of "discover monotone Δ helps."
- **Curriculum:** train on n=20 → 30 → 40 → 60. GNN encoder should transfer; total time T scales with √n approximately.
- **Reward shaping:** see §2.7 for the configurable reward function design.

### 2.7 Configurable reward function

**Decision:** The reward function is selectable via `RewardConfig.kind` in `config.json`. Three options:

1. **`"is_cost"`** (default) — Weighted independent-set cost from raw measurement bitstrings:
   `r = (1/N_shots) Σ_shots [Σ_i s_i − U · Σ_{(i,j)∈E} s_i·s_j]`, optionally normalized by `|V|`.
   Dense gradient signal; no NP-hard oracle needed. Directly aligned with the Rydberg Hamiltonian ground-state objective.

2. **`"p_mis"`** (legacy) — Fraction of shots that are independent sets of maximum cardinality. Extremely sparse for non-trivial graphs; requires an (approximate) |MIS| computation.

3. **`"composite"`** — `r = (1−λ)·r_cost + λ·r_mis`, where λ = `mis_bonus`. Enables curriculum: start with λ≈0 (dense cost signal), anneal to λ→1 (precise MIS targeting).

**Rationale (literature-informed):**

- Pure p_MIS is a near-zero signal on most graphs — the policy gets no gradient telling it which direction is better.  A schedule producing many IS of size |MIS|−1 gets the same reward (zero) as one producing only ground states.
- The IS cost is the quantum Hamiltonian cost function itself (Pichler et al. 2018): in the Δ→+∞ limit, the Rydberg Hamiltonian ground state is the MIS.  Rewarding the RL agent with this cost means it optimizes the same objective the physics is optimizing.
- The penalty U > 1 ensures that adding a conflicting node is always worse than not adding it (for unweighted MIS).  U = 3 works well empirically; too large → heavy penalty dominates and discourages excitation.
- Normalization by |V| keeps rewards comparable across different graph sizes, important for mixed-size training pools.
- Ebadi et al. (2022), Kim et al. (2024) report "approximation ratio" as their figure of merit, which is closely related to the IS cost.
- Byun et al. (2024) use a composite reward that transitions from cost-based to MIS-hit-based during training.

### 2.8 Warm-start / residual parameterization

**Decision:** When `warm_start=True` (default), the decoder heads' outputs are interpreted as *corrections* relative to the fixed adiabatic baseline schedule (trapezoidal Ω + linear Δ sweep):

```
final(t) = baseline(t) + [head(t) − neutral(t)]
```

where `neutral(t)` is what the head would produce for a zero-valued parameter vector (computed once at initialization and stored as a buffer).

At random initialization the MLP outputs are small (Xavier init), so `head(t) ≈ neutral(t)`, corrections ≈ 0, and the model starts producing the baseline schedule.  Training then learns graph-specific perturbations that improve on the baseline.

**Rationale:**

- **The research question becomes sharper:** "Can a GNN learn graph-specific *corrections* to the standard adiabatic protocol?" rather than "Can it rediscover adiabatic physics from scratch?"
- **No wasted training on known physics:** Without warm-start, the model must first discover that Ω should be ~15.8 rad/μs and Δ should sweep negative→positive. This takes thousands of training steps of expensive quantum simulation to learn something we already know.
- **Better optimization landscape:** The model starts in a high-reward region (the baseline is already a reasonable protocol). Gradients from the first training step are already informative.
- **Precedent:** Cain et al. (2023) and Hegde et al. (2023) both use variations of this approach — starting from a known-good protocol and learning perturbations.

**Implementation details:**

- When `learn_omega=False`, warm-start only applies to Delta (Omega is already the baseline via AnalyticOmega).
- The neutral profile is computed via `head.reconstruct(zeros)` — this works for all head types (spline, Fourier) without architecture-specific code.
- The correction magnitude is naturally bounded by the heads' sigmoid/tanh clamping, preventing extreme deviations.
- Final outputs are additionally clamped to physical bounds [0, ω_max] and [Δ_min, Δ_max].

### 2.5 `learn_omega` toggle and analytic Ω envelope

**Decision:** Ω head is a separate `nn.Module`. When `learn_omega=False`, it's swapped for an `AnalyticOmega` module with the same interface (takes graph embedding, returns Ω(t) tensor). Decoder concatenates Ω-from-head and Δ-from-head agnostically.

This keeps the rest of the pipeline unchanged and makes ablation trivial.

### 2.6 Hardware specs and blockade-derived Ω

**Decision:** Device-specific constants and protocol timing live in `hardware_specs.json`, separate from the experiment config.

- `C6` = van der Waals coefficient (5.42×10⁶ rad/μs·μm⁶ for Aquila ⁸⁷Rb |70S₁/₂⟩).
- `omega_max` = hardware Rabi limit (15.8 rad/μs for Aquila).
- `t_ramp`, `t_onset` = trapezoidal envelope parameters (μs).

When `learn_omega=False`, the analytic Ω peak is derived from blockade physics:
`Ω_peak = min(C₆ / R_b⁶, omega_max_hw)` where `R_b = udg.radius × udg.spacing` (μm).

For typical Aquila-scale lattices, C₆/R_b⁶ greatly exceeds the hardware max (e.g. R_b=2.5μm → ~22,000 rad/μs), so Ω is hardware-limited. The formula becomes useful when R_b is large enough (≳8.4μm) that the computed Ω dips below 15.8.

**Rationale:** The sin² envelope used previously was a placeholder. Real adiabatic protocols on Aquila use a piecewise-linear trapezoidal Ω: linear ramp up → flat hold → linear ramp down (see Bloqade tutorials, Ebadi et al.). The trapezoidal shape:
- Matches the actual hardware pulse format (piecewise-linear).
- Has a well-defined hold duration where the system evolves at constant drive.
- Cleanly separates "turning on the drive" from "sweeping detuning."

---

## 3. Architecture summary
## 3a. Architecture 1



```
networkx.Graph
    │
    ├─ feature precomputation (degree, clustering, triangles, Laplacian PE)
    │   graph-level: n/64, m/256, λ₂, density
    ▼
PyG Data object
    │
    ▼
GIN encoder (3 layers, hidden=64)
    │
    ▼
concat(mean, max, sum) pooling → 192-d
    │
    ▼
concat with scalar graph features → 196-d
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Ω head         Δ head         Value head (PPO)
(3 params)     (8 spline pts) (scalar)
    │              │
    ▼              ▼
sin² shape       linear interp
× sin²(πt)      + tanh clamp
boundary
    │              │
    ▼              ▼
Ω(t) ∈ ℝ^Nt    Δ(t) ∈ ℝ^Nt
    └──────┬───────┘
           ▼
     GlobalSchedule
           │
           ▼
     Bloqade / Aquila
           │
           ▼
       p_MIS → reward → PPO update
```

---

## 3b. Architecture 2: Fourier-coefficient decoder

Selected via `config.controls.architecture = 2`. Shares the same GIN encoder, featurization, Gaussian policy, and value head as Arch 1. Only the decoder heads differ.

### Fourier Omega head (`FourierOmegaHead`)

- **Parameters:** `n_omega_modes` sine coefficients (default 5).
- **Reconstruction:** `Ω_raw(t) = Σ_k a_k · sin(kπt)`, then `Ω(t) = ω_max · sigmoid(Ω_raw(t))`.
- **Boundary behavior:** The sine basis vanishes at t=0 and t=T, so `Ω_raw = 0` at endpoints → `sigmoid(0) = 0.5` → `Ω(0) = Ω(T) = ω_max / 2`. This is a *soft* boundary, not a hard zero. It satisfies Ω ≥ 0 and Ω ≤ ω_max everywhere, but does not enforce Ω(0) = Ω(T) = 0 the way Arch 1 does.
- **Trade-off:** Cleaner math (no boundary mask), but the policy must learn to produce small-amplitude endpoints rather than getting them for free.

### Fourier Delta head (`FourierDeltaHead`)

- **Parameters:** 1 DC offset + `n_delta_modes` cosine coefficients (default 8), total 9.
- **Reconstruction:** `Δ_raw(t) = a_0 + Σ_k a_k · cos(kπt)`, then tanh-clamped to `[Δ_min, Δ_max]`.
- **Rationale:** Cosine basis + DC gives a natural offset level (the "starting detuning") plus harmonic corrections. The DC term maps directly to the mean detuning level, which is the single most important parameter physically.

### When to prefer Arch 2

- When smooth, globally structured schedules are expected (e.g. adiabatic sweeps without sharp features).
- When the hard zero boundary condition on Ω is not critical (e.g. hardware already ramps to zero externally).
- As a comparison baseline to measure how much the spline inductive bias helps.

### 2.9 Architecture 3 — physics-prior heads (v8)

**Decision:** New decoder family that bakes the adiabatic baseline directly
into the head architecture, rather than relying on warm-start residuals
around a generic learned head.

- `MultiplicativeOmegaHead`: Ω(t) = Ω_baseline(t) · (1 + γ · tanh(Σ_k a_k sin(k π t))).
  Zero-initialized parameters give g_θ ≡ 1; γ=0.3 bounds the deviation to
  ±30 % of the baseline at every t.  Boundary conditions and trapezoidal
  envelope are inherited from the baseline shape — *cannot* be violated
  by the network.
- `MonotoneDeltaHead`: Δ(t) constructed as cumulative sum of softplus-positive
  increments, then affine-mapped to [Δ_min, Δ_max].  Direction of the sweep
  is fixed; the policy can only modulate *where* it spends time during the
  sweep (slow through the avoided crossing, fast at the edges, etc).

**Rationale:** Warm-start lets the network produce any shape, including
physically destructive ones, before residuals are applied — exploration
during early RL then easily drifts off the baseline and reward collapses.
Arch 3 makes off-baseline drift architecturally impossible: any sampled
schedule is still a valid adiabatic protocol, just with a slightly
different timing or amplitude modulation.  The only deviations the
network can express are exactly the ones that *should* matter physically.

**Cost:** Less expressive than arch 1/2.  If the optimal schedule for
some hard graph is genuinely non-adiabatic (Cain-style diabatic shortcuts),
arch 3 cannot represent it.  This is a deliberate trade-off: we sacrifice
the ceiling to lift the floor and stabilize learning.  Arch 1/2 remain
available for exploratory work.

### 2.10 Algorithm pipeline (v8)

**Decision:** PPO with paired-baseline advantages, multiple rollouts per
graph, advantage normalization, optional replay buffer, BC pretraining,
and residual-α curriculum.

Why each piece, briefly:

- **Paired baselines** (`A(G) = r_learned(G) − r_baseline(G)`) remove the
  dominant variance term in policy gradients on heterogeneous graph pools.
  Standard REINFORCE's EMA baseline cannot track per-graph reward levels
  fast enough when the pool is refreshed periodically.
- **K rollouts per graph** averages out measurement noise in the
  simulator (50 shots → 14 % binomial s.e. on any binary event).
- **Advantage normalization** stabilizes the effective learning rate
  across batches.
- **PPO clipping (ε=0.2)** + 4 inner epochs lets us reuse each simulator
  call several times without destructive updates.
- **Replay buffer** extends that reuse further; PPO clipping bounds the
  effective off-policy distance.
- **BC pretraining** is what makes the first PPO update meaningful: by
  the time RL starts, policy mean ≈ baseline and critic ≈ true baseline
  reward, so the advantage already reflects "did this perturbation help
  for *this* graph?".
- **Residual-α curriculum** starts the policy in a tiny neighborhood of
  the baseline and slowly expands the trust region.  Equivalent to a
  scheduled KL constraint against the baseline.

This is the minimal pipeline that gives both **convergence** (low variance,
stable updates) and **outperformance** (baseline is the *floor*, not a
random init).

### 2.11 Graph-conditioning diagnostics (v8)

To support the central scientific claim — "the learned policy has
internalized graph-dependent physics" — we measure two quantities every
`diagnostics_every` steps on the eval pool:

1. **Schedule-deviation correlations.** Pearson r between per-graph
   L2 schedule deviation from baseline and graph features (n, density,
   λ₂, clustering, mean degree).  Non-zero correlations tell us the
   policy is at least *responding* to graph identity.
2. **Conditioning index.** Variance of mean schedules across graphs
   divided by variance across rollouts on a single graph.  High =
   graph identity matters more than noise; low = the policy collapsed
   to a single mode regardless of input.

The probes are *not* used to drive training — they are evidence in the
eventual write-up.  They tell us whether to claim graph-conditioning or
not, before we try to claim it.

---

## 4. Implementation milestones

1. ~~**Day 1:** Skeleton modules (encoder, heads, schedule reconstruction). Unit-test with random graphs that shapes are correct and gradients flow.~~ **Done.** Split into `featurize.py`, `encoder.py`, `heads.py`, `policy.py`. 29 tests passing (both architectures).
2. ~~**Day 2:** Wire to Bloqade.~~ **Done.** `BraketBackend` converts `GlobalSchedule` → Braket AHS, runs `LocalSimulator("braket_ahs")`, returns `BackendResult`. End-to-end integration verified. Single-graph overfit: can we drive p_MIS up on a fixed n=20 graph with REINFORCE?
3. **Day 3:** Multi-graph training, n=20. Confirm encoder is doing useful work (vs. constant-input baseline).
4. **Day 4:** PPO + value head. Curriculum to n=30, n=40.
5. **Day 5:** Validation on held-out graphs and on Aquila (limited shots).

---

## 5. Open questions / parking lot

- Should we predict per-graph total annealing time T, or fix it per size class?
- Is there value in conditioning on the *target* MIS size (if known classically) as an auxiliary input?
- For Aquila-vs-Bloqade transfer: do we need a noise-aware fine-tuning phase?
- Worth trying contrastive pretraining of the encoder on a graph property prediction task before RL?
- Edge features: should we encode pairwise interaction strength V_ij as edge attributes (use NNConv/GINEConv), or leave it implicit in graph structure?
- `rsample()` vs `sample()` in the policy: currently using `rsample()` (reparameterization trick), which routes gradients through the schedule values, not through logprob. For pure REINFORCE with non-differentiable reward, `sample()` is needed so that logprob carries gradients to the encoder. Decide once the training algorithm is finalized.
- Δ interpolation is currently linear (not cubic spline). Swap in cubic if smoothness matters for Bloqade.
- Warm-start initialization (Ebadi-style linear ramp for Δ) not yet implemented — currently random init.

---

## 6. Changelog

- **v0 (initial):** Architecture 1 specified. GIN + reduced-basis decoder + Gaussian policy + PPO. Awaiting implementation.
- **v1 (integration):** Full implementation integrated into the codebase.
  - `ScheduleConfig` absorbed into `ControlsConfig`; new fields: `omega_max`, `delta_min`, `delta_max`, `n_delta_knots`.
  - Monolith `schedule_policy.py` split into `featurize.py`, `encoder.py`, `heads.py`, `policy.py`.
  - `SchedulePolicy.make_schedule(nx.Graph) → GlobalSchedule` bridges the torch model to the numpy-backed `GlobalSchedule`.
  - Duplicate `GlobalSchedule` removed; single source of truth in `schedules.py`.
  - REINFORCE step moved to `module3/reinforce.py`.
  - Bug fix: Ω boundary condition — added sin²(πt) boundary mask to guarantee Ω(0) = Ω(T) = 0.
  - 17 tests covering featurization, heads, policy forward/sample/make_schedule, gradient flow, and constraint enforcement.
- **v2 (Architecture 2 + config-driven selection):**
  - New config fields: `architecture` (1 or 2), `n_omega_modes` (default 5), `n_delta_modes` (default 8).
  - `FourierOmegaHead`: K sine coefficients → sigmoid → [0, ω_max]. Soft boundary (sigmoid(0) = 0.5·ω_max at endpoints).
  - `FourierDeltaHead`: DC offset + K cosine coefficients → tanh clamp to [Δ_min, Δ_max].
  - `SchedulePolicy.__init_`_ dispatches on `config.controls.architecture` to select Arch 1 or Arch 2 heads.
  - `sample_schedule` uses `n_params > 0` check instead of `isinstance(OmegaHead)` for architecture-agnostic reconstruction.
  - Tests parametrized over both architectures; architecture-specific constraint tests added. 25 tests total.
- **v3 (hardware specs + trapezoidal Ω):**
  - New file `hardware_specs.json` with C₆, hardware Ω/Δ limits, `t_ramp`, `t_onset`.
  - `HardwareSpecs` dataclass added to `config.py`; `ProjectConfig` now includes `hardware` field.
  - `compute_blockade_omega(C6, R_b, omega_max_hw)` derives Ω_peak from blockade physics, capped at hardware max.
  - `AnalyticOmega` rewritten: sin² envelope → trapezoidal (linear ramp-up → hold → linear ramp-down). Parameters: `omega_peak`, `t_ramp_frac`, `t_onset_frac`.
  - `SchedulePolicy` computes Ω_peak and ramp fractions from `HardwareSpecs` + `UDGConfig` when `learn_omega=False`.
  - `UDGConfig.spacing` units clarified as μm (needed for C₆ calculation).
  - 29 tests total (added blockade omega, trapezoidal shape, and onset tests).
- **v4 (Module 2 integration):**
  - `BraketBackend(QuantumBackend)` implemented in `module2/braket_backend.py`. Converts `GlobalSchedule` (rad/μs, seconds) → Braket `TimeSeries` (rad/s, seconds), builds `AtomArrangement` from positions (μm → m), validates against Aquila hardware limits.
  - `QuantumBackend.estimate_p_mis` signature updated: now requires `positions: Positions` (atom coordinates in μm) alongside schedule and graph.
  - `BackendResult` extended with `counts` field (raw bitstring dict) and `binomial_std_err()` helper.
  - Third-party code cleaned up: renamed `graph_MIS_utils (1).py` → `graph_MIS_utils.py`, removed broken `ahs_utils` import from `schedule_pmis_module.py`, fixed `check_independent_set` bug (`'1'` → `'r'` for Braket AHS alphabet), fixed `KeyError` in `regroup_by_ones_dict` when no shots match target cardinality.
  - Bug fix in `graphs/unit_disk.py`: `generate_square_lattice_udg` used `radius / spacing` for neighbor offsets, but `radius` is already in units of `spacing`. Produced zero-edge graphs when `spacing ≠ 1.0`.
  - `amazon-braket-sdk` added to `requirements.txt` as optional dependency; Braket imports are lazy.
  - 5 integration tests (`test_integration_m1_m2.py`) verify Graph → Module 1 → Module 2 → BackendResult end-to-end for both architectures.
- **v5 (Module 3 + literature-informed refinements):**
  - **Literature review:** analyzed Pichler 2018, Ebadi 2022 (Science), Cain 2023, Lukin 2024 (SQS), Schuetz 2024 (compilation toolkit), Hegde 2023 (LSTM schedules), Sohrabizadeh 2024 (GNN p_MIS prediction), and others.
  - **Featurization expanded:**
    - Node features: added per-node triangle count (normalized). Motivated by Schuetz et al. — triangles/cliques correlate with blockade frustration and problem hardness.
    - Graph features: added graph density as 4th scalar. `GRAPH_FEAT_DIM` 3 → 4; node feat dim 2+k → 3+k.
  - **`FixedScheduleBaseline` added** to `module1/base.py`: trapezoidal Ω + linear Δ sweep (Ebadi-style adiabatic protocol). Graph-agnostic — returns the same schedule regardless of input. Serves as the comparison target: any learned policy must outperform this to demonstrate graph-conditioned value.
  - **`SchedulePolicy` now inherits from both `nn.Module` and `ScheduleModel`**, making it interchangeable with baselines in evaluation code.
  - **Module 3 implemented:**
    - `TrainingConfig` expanded with batch_size, eval_graphs, entropy/value-loss coefficients, grad_clip, n_shots, pool settings, checkpoint/log dirs.
    - `reinforce_step` enhanced: added critic value-function loss (MSE), gradient norm tracking, min/max reward logging.
    - `ReinforceLearner(Learner)`: manages policy, optimizer, EMA baseline, graph pool, evaluation against `FixedScheduleBaseline`, checkpointing.
    - `TrainingOrchestrator(Orchestrator)`: main training loop with periodic evaluation, best-model checkpointing, graph pool refresh, JSON history logging.
    - `backend_adapter.py`: bridges `QuantumBackend.estimate_p_mis` to the `(Graph, Schedule) → float` reward function signature. Positions read from `G.graph["positions"]`.
    - `train.py` CLI: `--backend mock` for testing without Braket, `--backend simulator` for real p_MIS. Supports `--steps`, `--shots`, `--batch-size`, `--lr`, etc.
  - 9 new tests in `test_module3_training.py`: pool construction, baseline validation, learner instantiation, single train step, evaluation, checkpoint round-trip, standalone reinforce_step, ScheduleModel isinstance check.
  - 38 total tests across the project (3 config + 26 Module 1 + 9 Module 3), all passing.
- **v6 (configurable reward function):**
  - New `RewardConfig` dataclass in `config.py` with fields: `kind` (`"is_cost"` | `"p_mis"` | `"composite"`), `penalty_U`, `mis_bonus`, `normalize_by_nodes`.
  - Default reward changed from sparse `p_mis` to dense `is_cost`: r = (Σ s_i − U·Σ s_i·s_j) / |V|, averaged over shots. Does not require knowing |MIS|.
  - `backend_adapter.py` rewritten with `_is_cost_from_counts()` and `_compute_reward()`. `make_reward_fn()` now accepts `RewardConfig`.
  - `config.json` extended with `"reward"` section.
  - `train.py` supports `--reward` and `--penalty-U` CLI overrides.
  - 9 new reward function tests (total 47 across the project).
- **v7 (warm-start / residual parameterization):**
  - New `warm_start: bool = True` in `ControlsConfig`.
  - `SchedulePolicy` computes baseline (trapezoidal Ω + linear Δ) and neutral head profiles as buffers.
  - Head outputs reinterpreted as `baseline + (head − neutral)`, clamped to physical bounds.
  - At random init, output ≈ baseline; training learns graph-specific corrections.
  - When `learn_omega=False`, warm-start only applies to Delta (Omega is already baseline).
  - 8 new warm-start tests (omega near baseline, delta closer to baseline, gradient flow, learn_omega=False invariance, physical bounds).  55 total tests.
- **v8 (PPO + physics-prior heads + BC pretraining + diagnostics):**
  Overhauls the learning pipeline to address convergence and outperformance
  failures observed in early training runs. Seven coordinated changes:
  - **Architecture 3 (physics-prior heads).** New `MultiplicativeOmegaHead`
    parameterizes Ω(t) = baseline(t) · g_θ(t) with g_θ ∈ [1−γ, 1+γ] (γ=0.3
    default); cannot blow up Ω, preserves Ω(0)=Ω(T)=0 and the trapezoidal
    envelope by construction.  New `MonotoneDeltaHead` parameterizes Δ(t)
    as the cumulative sum of softplus-positive increments, mapped affinely
    to [Δ_min, Δ_max]; guarantees monotone non-decreasing sweep regardless
    of network output.  Both heads have zero-initialized output layers so
    that at init the policy reproduces the analytic baseline exactly.
    `config.controls.architecture=3` selects them; warm_start is ignored
    for arch 3 (residual structure baked into the heads themselves).
  - **Residual-α curriculum.** New buffer `policy.residual_alpha`
    (initially `residual_alpha_start=0.05`) scales the deviation from
    baseline at every forward pass.  Linearly annealed by the
    orchestrator from `residual_alpha_start` → `residual_alpha_end` over
    `residual_alpha_warmup_steps`.  Acts as a trust region around the
    baseline so early exploration cannot destroy the schedule.
  - **PPO with paired baselines.** New `module3/ppo.py`.  For every graph
    in the batch, K=4 schedule rollouts are sampled *and* the analytic
    baseline is evaluated once on the same graph.  The advantage is
    `A(G) = r_learned(G) − r_baseline(G)` instead of an EMA-tracked
    `r − ema(r)`.  Eliminates per-graph reward heterogeneity that the EMA
    cannot track quickly enough.  Advantages are normalized to
    zero-mean unit-variance within each gradient step.  Standard PPO-clip
    loss with 4 inner epochs over the collected rollouts.
  - **Per-graph normalized reward (`is_cost_vs_baseline`).** New default
    `RewardConfig.kind`.  Returns
    `(r_learned − r_baseline) / (|r_baseline| + ε)` — fractional
    improvement over the adiabatic baseline.  Backed by a
    `BaselineRewardCache` that runs the simulator on the baseline schedule
    exactly once per graph and caches the result.  Removes graph-scale
    effects from the gradient signal.
  - **Behavioral cloning pretraining.** New `module3/pretrain.py`.  Two
    short supervised loops before any RL:
    1. **Policy BC** — MSE between policy.forward(G) and the analytic
       baseline schedule on every pool graph.  No simulator calls.
    2. **Critic BC** — MSE between value_head(G) and the measured
       baseline reward on every pool graph.  One simulator call per pool
       graph (reused later as the paired-baseline cache).
    So RL starts from policy mean ≈ baseline, critic ≈ true baseline
    reward.  First PPO advantage already reflects "did this perturbation
    help?" rather than "is the network even near the baseline?".
  - **Replay buffer for off-policy reuse.** `module3/replay.py` FIFO
    buffer of `(graph, sampled_params, old_logprob, reward, …)` tuples.
    `TrainingConfig.replay_buffer_size > 0` enables it; PPO mixes
    `replay_mix_ratio · batch_rollouts` replayed entries into each
    update.  Importance-sampling correction is implicit in the PPO
    clipping.  Amortizes the cost of each simulator call across several
    gradient steps.
  - **Pool curation + probe diagnostics.** `_curate_pool` filters the
    training pool to graphs whose baseline reward falls in
    `(curation_lo, curation_hi)` — discards trivially solvable / fully
    unsolvable instances that carry no learning signal.
    `module3/diagnostics.py` defines `schedule_deviation_probe` (per-graph
    L2 deviation from baseline + Pearson correlations with graph features
    n, density, λ₂, clustering) and `graph_conditioning_index` (ratio of
    between-graph variance to within-graph variance of the schedules).
    The orchestrator calls these every `diagnostics_every` steps and
    logs the results.  This is the evidence we'll use to claim the
    learned policy is *genuinely* graph-conditional.
  - **Algorithm dispatch.** `TrainingConfig.algorithm ∈ {"reinforce",
    "ppo"}`, default `"ppo"`.  `train.py` adds CLI flags `--algorithm`,
    `--rollouts-per-graph`, `--no-paired-baseline`, `--no-bc-pretrain`.
  - **Tests.** 27 new tests in `test_module3_improvements.py` covering
    Arch 3 head shapes/constraints/baseline match, residual-α schedule,
    `is_cost_vs_baseline` cache + correctness, PPO collect/loss/step,
    paired-baseline advantage equality, BC policy + critic convergence,
    replay buffer FIFO + capacity-0 behavior, diagnostics outputs.
    89 total tests, all passing.
