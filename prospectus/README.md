# MIS Prospectus Figures

This folder generates the five chapter figures for the MIS prospectus.
Everything is produced by a single script:

```
prospectus/make_documentation_figs.py
```

Outputs land in `prospectus/figs/MIS/` as both `.pdf` (vector, for LaTeX) and
`.png` (300 dpi raster). Expensive intermediate results (simulator counts,
exact-diag spectra, training histories, BC-pretrained policy states) are
memoized to `prospectus/cache/`.

## Quick start

```bash
# From the repo root.  Generate all five figures with default (light)
# settings.  Figure 1's Braket call is the slow one (~8 min on a 23-atom
# graph at 200 shots); the rest finish in seconds.
python prospectus/make_documentation_figs.py

# Regenerate only one figure.
python prospectus/make_documentation_figs.py --figs 3

# Force re-running cached experiments (drop the cache).
python prospectus/make_documentation_figs.py --regenerate

# Skip every simulator call (use synthetic measurement counts).
python prospectus/make_documentation_figs.py --no-simulator

# Run real RL training for figure 4 (the most expensive option).
python prospectus/make_documentation_figs.py --figs 4 --train-mode quick   # ~10 min
python prospectus/make_documentation_figs.py --figs 4 --train-mode full    # ~30–60 min
```

If running outside Cursor's sandbox you usually need:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python prospectus/make_documentation_figs.py
```

These environment variables avoid the macOS Accelerate-framework SIGFPE
at `numpy` import time.

## What each figure shows

| #   | File                              | What it shows                                                    | Slow?                |
| --- | --------------------------------- | ---------------------------------------------------------------- | -------------------- |
| 1   | `problem_setting.pdf`             | atom array + R_b disk + MIS, baseline schedule, bitstring counts | yes (simulator)      |
| 2   | `gap_profiles.pdf`                | exact-diag ΔE(t) for one dense vs one sparse UDG                 | ~30 s (eigvalsh)     |
| 3   | `architecture_comparison.pdf`     | 20 random-init samples from Arch 1, 2, 3 vs baseline             | no                   |
| 4   | `training_curves.pdf`             | v8 PPO vs legacy REINFORCE (or BC-only preview)                  | depends on mode      |
| 5   | `learned_schedules.pdf`           | per-graph learned schedule after BC pretrain                     | no                   |

### Figure 4 modes

`--train-mode` selects how much compute Figure 4 uses:

* **`bc`** (default, ~5 s) — runs only the behavioral-cloning pretraining for
  Arch 3 (v8 proxy) and Arch 1 (legacy proxy).  The "reward" panel is
  re-interpreted as *baseline fidelity*, showing that Arch 3 starts at
  fidelity ≈ 1 thanks to its physics-prior heads while Arch 1's spline
  initialization sits ≈ 14 % away and is pulled in by BC.

* **`quick`** (~15 min) — runs 30 PPO steps and 30 REINFORCE steps with the
  Braket local simulator on a small pool of 6 UDGs (~12 atoms each) at 50
  shots per evaluation.  Caches the resulting histories so subsequent
  renders are fast.  Both reward streams are converted to the *raw is_cost*
  scale before plotting so the v8 and legacy curves share a y-axis.

* **`full`** (~30–60 min) — 200 steps each on a larger pool of 12 UDGs.
  Recommended for the final prospectus version.

## Global style

All figures share `matplotlib.style.use('seaborn-v0_8-whitegrid')` and the
palette:

| Quantity             | Color     | Code        |
| -------------------- | --------- | ----------- |
| Ω(t)                 | blue      | `#2774AE`   |
| Δ(t)                 | coral     | `#D97706`   |
| MIS nodes            | amber     | `#DAA520`   |
| Baseline             | dashed black                |
| Valid IS / invalid   | green / red                 |
| Diabatic shading     | red @ 15 % alpha            |

Panel letters (`a`, `b`, …) appear in bold sans-serif at the top-left of
each panel.

## Cache invalidation

If you change the script's experiment parameters (e.g. graph seeds, BC
hyperparameters) you should re-run with `--regenerate` to drop the cache.
The cache files in `prospectus/cache/` are pickle files and you may also
delete them manually if needed.
