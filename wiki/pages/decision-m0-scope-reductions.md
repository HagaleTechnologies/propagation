---
id: decision-m0-scope-reductions
title: Which M0 scope reductions were deliberate, and when do they expire?
kind: decision-digest
status: current
maintainer: agent
sources:
  - docs/superpowers/plans/2026-07-11-m0-lake-bootstrap.md
  - src/propagation/models/climatology.py
  - src/propagation/features/universe.py
verified:
  commit: 5f75677
  date: 2026-07-14
links:
  - overview
  - gotcha-qa-diurnal-checks
  - gotcha-eval-rules
---
The M0 implementation (PR #11) contains four deliberate, user-approved scope
reductions. Each is documented at its implementation site; none is a bug,
and each has a defined expiry milestone. Future implementers should widen
them at the named milestone, not before, and not silently.

- **Climatology grouping** (`src/propagation/models/climatology.py`):
  ARCHITECTURE §5 specifies grouping by (path-cell, band, hour-of-day,
  month) at similar smoothed SSN. With one training month of one band,
  month/SSN have no variation to group on, so M0 groups by
  `(tx_field, rx_field, band, hour_of_day)` only. **Expires at M3** when
  multi-year history exists.
- **`unlabeled_fraction` formula** (`src/propagation/features/universe.py`):
  the spec requires reporting it but gives no closed form. M0 defines it as
  `1 − n_universe / n_candidates` over active-field candidate pairs per
  (band, date). This is an engineering interpretation, not spec — if a
  second implementation is ever built for the cross-validation requirement
  in docs/SPEC-labeling.md, it must copy this definition or the spec must
  be amended to make it normative.
- **QA checks 3/6/7** (`src/propagation/qa/checks.py`): implemented as real
  precondition gates that report `insufficient_data` with the specific
  unmet precondition (solar-terminator features → M2; F10.7 history → M2;
  a Kp≥5 storm fold → whenever the eval range includes one). Not stubs —
  they run and report honestly.
- **Default train/eval months** (`scripts/run_m0.py`): train=2014-06,
  eval=2014-08. 2014-07 is deliberately skipped as a buffer month because
  adjacent months violate the ≥48h leakage gap ([[gotcha-eval-rules]]) —
  the pipeline now enforces the gap at runtime and will refuse adjacent
  months. Overridable via `--train-month`/`--eval-month`.
