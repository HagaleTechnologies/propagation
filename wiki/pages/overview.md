---
id: overview
title: propagation — what is this and where do things live?
kind: overview
status: current
maintainer: agent
sources:
  - README.md
  - ARCHITECTURE.md
  - ROADMAP.md
  - CLAUDE.md
verified:
  commit: 5f75677
  date: 2026-07-14
links:
---
ML-based HF propagation nowcasting: train on real path-report data
(WSPRnet, RBN, PSKReporter) to predict band openings at (tx Maidenhead field,
rx field, band) × 15-minute window, benchmarked honestly against ITU-R P.533
and climatology. The novel methodological contribution is monitor-normalized
negative sampling, which handles observation bias in sparse amateur-radio
reception reports. M0 is complete (PR #11): the WSPRnet→lake→labels→
climatology→eval pipeline runs end-to-end from empty `data/` and the M0
acceptance bar is met. Next step is M1 (vendored P.533 baseline) in
ROADMAP.md.

## Where things live

- `README.md` — thesis, licensing, open/closed boundary
- `ARCHITECTURE.md` — task formulation, pipeline, feature engineering,
  modeling ladder (M-0 → M-3), evaluation design, cqdx contract seam
- `docs/SPEC-labeling.md` — normative label construction spec (v1);
  two independent implementations MUST produce identical labels from this spec
- `docs/SPEC-contract-notes.md` + `contracts/prediction-surface.v1.schema.json`
  — versioned public prediction contract (columnar encoding, integer-quantized)
- `ROADMAP.md` — milestones M0–M4 with acceptance criteria; check M0 before
  any M1+ work
- `baselines/p533/` — vendored ITURHFProp build (not yet present; M1 task)
- `src/propagation/` — implemented M0 pipeline: `data/` (extract, hygiene,
  dedup, lake), `features/` (uptime, universe, labels, sampling), `models/`
  (climatology), `eval/` (splits, metrics, report), `qa/` (checks)
- `scripts/run_m0.py` — end-to-end M0 orchestration; the acceptance artifact

## Start here

1. `overview` (this page) — orient
2. [[gotcha-open-closed-boundary]] — the boundary rule you must not violate
3. [[gotcha-eval-rules]] — why random splits are forbidden here
4. [[decision-monitor-normalized-negatives]] — the methodological core
