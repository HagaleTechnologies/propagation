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
  commit: 808948a
  date: 2026-07-23
links:
  - decision-p533-baseline
  - decision-m1-ladder-exception
  - gotcha-label-leakage-via-nulls
---
ML-based HF propagation nowcasting: train on real path-report data
(WSPRnet, RBN, PSKReporter) to predict band openings at (tx Maidenhead field,
rx field, band) × 15-minute window, benchmarked honestly against ITU-R P.533
and climatology. The novel methodological contribution is monitor-normalized
negative sampling, which handles observation bias in sparse amateur-radio
reception reports. M0 (PR #11), M1 (PR #13), and M2 (PR #20, PR #27) are all
complete: the WSPRnet→lake→labels→climatology→P.533→GBT→eval pipeline runs
end-to-end from empty `data/`, and `scripts/eval_m2.py`'s live acceptance run
confirms GBT genuinely beats both climatology and P.533 on Brier/log-loss
across 20m+10m and 3 held-out 2024 months incl. the Gannon storm — see
[[decision-p533-baseline]], [[decision-m1-ladder-exception]], and
`docs/DECISIONS/0005-m2-acceptance-result.md`. Next step is M3 (scale to
all bands/horizons, RBN + live PSKReporter) in ROADMAP.md.

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
- `baselines/p533/` — vendored ITURHFProp build (M1; `PROVENANCE.md` records
  what was vendored and the license carve-out, see docs/DECISIONS/0001)
- `src/propagation/` — `data/` (extract, hygiene, dedup, lake), `features/`
  (uptime, universe, labels, sampling), `models/` (climatology, p533),
  `eval/` (splits, metrics, report, stratify), `qa/` (checks)
- `scripts/run_m0.py` — end-to-end M0 orchestration; the M0 acceptance artifact
- `scripts/eval_m1.py` — climatology + P.533 headline reports
  (overall/storm/quiet slices); the M1 acceptance artifact
- `scripts/p533_crosscheck.py` — private (non-CI) P.533 agreement spot-check
  against cqdx's independent implementation

## Start here

1. `overview` (this page) — orient
2. [[gotcha-open-closed-boundary]] — the boundary rule you must not violate
3. [[gotcha-eval-rules]] — why random splits are forbidden here; storm-slice
   eval months must be verified, not assumed (docs/DECISIONS/0002)
4. [[decision-monitor-normalized-negatives]] — the methodological core
5. [[gotcha-live-system-bugs]] and [[gotcha-plan-drift-before-merge]] —
   process lessons from M1 worth not re-learning on M2+
