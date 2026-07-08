---
id: gotcha-eval-rules
title: What will bite you about evaluation (blocked CV, full label set)?
kind: gotcha
status: current
maintainer: agent
sources:
  - ARCHITECTURE.md
  - docs/SPEC-labeling.md
verified:
  commit: 5c2dac7
  date: 2026-07-07
links:
  - overview
  - decision-monitor-normalized-negatives
---
Three eval rules will produce silently invalid results if violated: (1) random
train/test splits are forbidden — autoregressive features leak through them;
(2) the evaluation set must be the FULL unsampled label set — the 3:1 negative
downsampling used for training must NEVER touch eval; (3) the blocked CV gap
must be at least 48 hours — widening when you add longer lookbacks. See the
normative rules at ARCHITECTURE.md §6 and docs/SPEC-labeling.md §6.

## Symptom

- **Random splits**: inflated Brier / PR-AUC numbers that collapse at
  deployment; a model that "beats P.533" in CV but doesn't beat anything
  on held-out time ranges.
- **Sampled eval**: miscalibrated base-rate estimates; PR-AUC artificially
  deflated (or inflated depending on sampling direction) relative to true
  population frequency.
- **Gap too small**: autoregressive history features (max lookback 24 h) reach
  back into the training set across the gap, leaking future information.

## Cause and workaround

Normative rules (ARCHITECTURE.md §6, docs/SPEC-labeling.md §6):

- **Blocked time-series CV only.** Train on months [t0, t1), validate on
  [t1+gap, t2). Multiple folds across seasons and Kp regimes. No shuffle.
- **Gap ≥ 48 h** at current feature set (24 h max AR lookback + 24 h max
  horizon). If you add a feature with longer lookback or a new horizon,
  you MUST widen the gap in `eval/splits.py` — the 48 h figure is at the
  bound, not a comfortable margin.
- **Eval on the full labeled set.** `sample_weight` in the labels Parquet
  is for training objectives and calibrators only. Never filter the eval
  set by that column.
- **Minimum credible result** before calling M2 done: LightGBM beats P.533
  AND climatology on Brier at h=0 and +3h on 20m/15m/10m over ≥3 held-out
  months spanning at least one geomagnetic storm (Kp ≥ 5). Details in
  ARCHITECTURE.md §6 and ROADMAP.md M2/M3.
- **Storm fold requirement**: headline eval MUST include ≥1 storm fold
  (docs/SPEC-labeling.md §6, leakage rule 5). Storms are where climatology
  fails hardest and the model should win biggest.
