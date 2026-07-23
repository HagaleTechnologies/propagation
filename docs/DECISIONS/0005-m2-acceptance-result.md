# 0005: M2 acceptance run — GBT beats both baselines on 20m and 10m

## Status

Accepted.

## Context

ROADMAP.md's M2 acceptance gate: GBT must beat both climatology and P.533
on Brier/log-loss at h=0, on both 20m and 10m, across >=3 held-out months.
"If it doesn't, stop and diagnose — do not proceed to serving."

Ran `scripts/eval_m2.py` against real WSPRnet + OMNI2 data: train on
2024-01/02/03, eval on 2024-05/07/09 (2024-05 is the May 2024 Gannon
superstorm, verified per [[0002]] as a real storm month; 07/09 add quiet-
month coverage and calendar spread).

The first attempt "passed" with GBT Brier ~1e-25 — an impossible result for
real propagation forecasting, not a genuine win. Root cause:
`snr_ft8eq_p50` in `FEATURE_COLUMNS` (`features/matrix.py`) is the anchor
row's own mode-normalized SNR, computed by `build_labels()` from the exact
spots that define its `open` value — null iff `open=0`, non-null iff
`open=1`, zero exceptions confirmed directly across all 11.1M rows in the
2024-05/20m label set. LightGBM was learning that null/not-null split
instead of any real signal. The existing leakage audit
(`tests/test_leakage.py`) only blocklisted label columns by name, so a
feature that's semantically hindsight-only under an innocuous name slipped
through M2's per-task review, whole-branch review, and the dedicated
leakage-audit task. Fixed in PR #27: dropped the column from
`FEATURE_COLUMNS` (its use inside `add_history_features` for OTHER rows'
history is legitimate and untouched), added a direct regression test plus
a general property test (no `FEATURE_COLUMNS` column's null pattern may
perfectly proxy `open`) to catch a recurrence under a different name.

## Decision

Post-fix, both required bands clear the acceptance bar across all three
held-out months combined (n=31.7M rows for 20m, smaller for 10m given
lower spot volume on that band):

| band | slice | climatology Brier | P.533 Brier | **GBT Brier** | climatology log-loss | P.533 log-loss | **GBT log-loss** |
|---|---|---|---|---|---|---|---|
| 20m | overall | 0.0791 | 0.4032 | **0.0455** | 0.442 | 4.589 | **0.156** |
| 20m | storm   | 0.0724 | 0.4313 | **0.0434** | 0.369 | 5.218 | **0.150** |
| 20m | quiet   | 0.0794 | 0.4018 | **0.0456** | 0.446 | 4.559 | **0.156** |
| 10m | overall | 0.1088 | 0.3097 | **0.0530** | 0.602 | 2.191 | **0.181** |
| 10m | storm   | 0.1142 | 0.3408 | **0.0507** | 0.653 | 2.528 | **0.176** |
| 10m | quiet   | 0.1085 | 0.3083 | **0.0531** | 0.600 | 2.176 | **0.181** |

GBT beats both baselines by a wide margin in every slice on both bands.
Calibration bins are sane post-fix (predicted probability tracks observed
rate closely across all ten bins on both bands), not the degenerate
all-or-nothing split the leaked feature produced. **M2 is accepted.**

## Consequences

- M3 (scale to all HF bands + horizons, RBN extractor, live PSKReporter
  accumulator, storm-window case studies) is unblocked.
- The general "no feature's null pattern may perfectly proxy the label"
  test in `tests/test_leakage.py` should be kept and extended as new
  feature sources are added in M3 — it caught a bug three prior review
  passes (per-task, whole-branch, dedicated leakage audit) all missed.
- **Runtime gotcha for future milestones**: a single band+3-month
  acceptance run takes several hours of real compute, dominated by
  P.533's per-row ITURHFProp subprocess calls (even parallelized per
  PR #23) — budget accordingly when planning M3's wider band/horizon
  sweep, and consider batching or caching P.533 scores per (path, hour,
  frequency, SSN) tuple if M3's sweep makes this cost prohibitive.
