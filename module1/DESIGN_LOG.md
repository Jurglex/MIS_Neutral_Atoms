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

**Decision:** 3-layer GIN, hidden 64, node features = [degree, clustering, top-k Laplacian eigenvector entries], pooled with concat(mean, max, sum), augmented with scalar graph features (n, m, λ₂).

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
- **Reward shaping:** `r = approx_ratio + α · p_MIS` to get gradient signal on hard instances where p_MIS is sparse.

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
    ├─ feature precomputation (degree, clustering, Laplacian PE, λ₂)
    │
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
concat with scalar graph features → ~195-d
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

---

## 4. Implementation milestones

1. ~~**Day 1:** Skeleton modules (encoder, heads, schedule reconstruction). Unit-test with random graphs that shapes are correct and gradients flow.~~ **Done.** Split into `featurize.py`, `encoder.py`, `heads.py`, `policy.py`. 25 tests passing (both architectures).
2. **Day 2:** Wire to Bloqade. Single-graph overfit: can we drive p_MIS up on a fixed n=20 graph with REINFORCE?
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
