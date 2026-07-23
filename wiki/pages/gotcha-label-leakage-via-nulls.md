---
id: gotcha-label-leakage-via-nulls
title: What will bite you about a leakage audit that only blocklists columns by name?
kind: gotcha
status: current
maintainer: agent
sources:
  - src/propagation/features/matrix.py
  - src/propagation/features/labels.py
  - tests/test_leakage.py
verified:
  commit: 808948a
  date: 2026-07-23
links:
  - decision-p533-baseline
  - overview
---
M2's leakage audit (`tests/test_leakage.py`) blocklisted label columns by
NAME (`open`, definitive Kp) and still let a genuine leak through: a
feature whose innocuous name gave no hint that it was a deterministic
function of the label. The tell wasn't a name match, it was a suspiciously
perfect eval result — GBT scored ~1e-25 Brier, an impossible number for
real propagation forecasting, on the first live M2 acceptance run.

## Symptom

- A model's eval Brier/log-loss comes back too good — near zero, or a
  clean separation between two prediction clusters with nothing in
  between (in this case: every prediction was ~0.0 or exactly 1.0, no
  rows in any of the middle 8 calibration bins).
- The suspiciously good result survives a mocked test suite and even a
  dedicated leakage-audit task, because that audit only checks column
  names against a blocklist, not whether a column's VALUES are a
  deterministic function of the label.

## Cause and workaround

`snr_ft8eq_p50` (`features/labels.py`'s `build_labels()`) is the median
mode-normalized SNR of the spots that constitute the current row's own
`open` observation — null iff `open=0`, non-null iff `open=1`, with zero
exceptions. It ended up in `FEATURE_COLUMNS` (`features/matrix.py`)
because the code reused an already-computed column "as-is" per the M2
plan's own framing, without noticing that CODE reuse and FEATURE
eligibility are different questions — a value computed from the very
thing you're forecasting can never be a legitimate predictive feature,
no matter how convenient it is to reuse.

Workaround, now encoded as two tests in `tests/test_leakage.py`:
1. A direct regression test for this specific column.
2. A general property test: for every column in `FEATURE_COLUMNS`, its
   null-vs-not-null pattern must NOT be a perfect predictor of `open`.
   This catches the same failure mode recurring under a different column
   name, which a name-based blocklist alone cannot. When adding new
   feature sources (M3's RBN/PSKReporter extractors, new history
   relations), ask not just "is this column named like a label" but "is
   this column's value computable only AFTER the label resolves" — if
   so, it's not a feature, no matter what it's called.
