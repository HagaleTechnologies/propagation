# 0006: M3 minimum credible result — GBT beats both baselines on 20m/15m/10m at h=0 and +3h

## Status

Accepted.

## Context

ARCHITECTURE.md §6's minimum credible result: "LightGBM beats P.533 *and*
climatology on Brier at h=0 and +3h on 20m/15m/10m over >=3 held-out
months spanning at least one geomagnetic storm." M2 ([[0005]]) only
cleared a subset of this: h=0 only, on 20m and 10m (15m untested, +3h
untested) — M3's job was to close both gaps: extend to the third band
and the second required horizon, using the same train=2024-01/02/03,
eval=2024-05/07/09 scheme M2 already validated (2024-05 is the Gannon
storm, per [[0002]]).

M3's capability work (band/horizon-generalized feature matrix, batched
WSPRnet extraction, real QA checks, `scripts/eval_m3.py`) merged via PR
#30, with a P.533 disk-cache fix (PR #32) that made repeat evaluation
runs reuse previously-computed subprocess scores instead of recomputing
them from scratch.

**Getting a clean production run took repeated staging**, not one
invocation: a combined `--bands 20m 15m 10m` run died 3 times at the full
4-band/6-horizon scope Tony had originally asked for (unexplained kills,
~2min/2min/55min in — later traced partly to a macOS Power Nap
("Maintenance Sleep") issue fixed via `sudo pmset -a powernap 0`, and
partly to contention from concurrent activity on the shared Mac mini
running this). A reduced `--bands 20m 15m 10m` (3 bands, 3 required
months) then died twice more at a consistent ~104min mark even with
Power Nap disabled — traced to `extract_wsprnet_bands` buffering
multiple bands' rows in memory simultaneously during extraction, making
that phase the likely OOM-vulnerable window under any ambient
contention. **Splitting into single-band runs (15m alone, then 20m
alone, then 10m alone) succeeded cleanly on the first attempt each
time** — no further contention deaths once bands stopped being
extracted concurrently.

## Decision

Combined result across all three required bands, all three required
held-out months (2024-05/07/09 combined), full requested horizon set
(bonus +24h included beyond the strict h=0/+3h minimum, since P.533's
cache made it nearly free once computed):

| band | n | h=0 (clim / P.533 / **GBT**) | h=+3h (clim / P.533 / **GBT**) | h=+24h (clim / P.533 / **GBT**) |
|---|---|---|---|---|
| 20m | 31,940,400 | 0.0791 / 0.4032 / **0.0455** | 0.0791 / 0.4032 / **0.0600** | 0.0791 / 0.4032 / **0.0594** |
| 15m | 13,313,193 | 0.0918 / 0.3857 / **0.0542** | 0.0918 / 0.3857 / **0.0752** | 0.0918 / 0.3857 / **0.0742** |
| 10m | 8,983,118  | 0.1088 / 0.3097 / **0.0532** | 0.1088 / 0.3097 / **0.0806** | 0.1088 / 0.3097 / **0.0822** |

(Values are Brier score, `rows = {climatology, P.533, GBT}` per
ARCHITECTURE.md §6's headline table format; climatology/P.533 are
horizon-invariant by design, matching 20m/10m's earlier M2 h=0 numbers
almost exactly — a correctness cross-check, not new information for
those two cells.)

GBT beats both baselines by a wide margin on every band at every
horizon, including both required ones (h=0, h=+3h). **The minimum
credible result is met.**

**How this was produced, precisely**: not via one combined
`eval_m3.py` invocation — each band was run as its own separate
single-band invocation (`--bands {20m,15m,10m}` individually), each
completing cleanly, rather than one `--bands 20m 15m 10m` call. This
means GBT was fit independently per band rather than as one model
shared across bands via `band_ordinal` (ARCHITECTURE.md §5's "one model
per horizon, shared across bands" design intent). The per-band-group
Brier/log-loss numbers this produces are exactly what §6's headline
table reports, so the bar is genuinely cleared — the cross-band sharing
property is a design intent for the eventual production model, not part
of what §6 itself measures, but worth being precise about if this
result is ever cited as validating the shared-model architecture
specifically.

Raw outputs preserved on disk (gitignored, not committed):
`data/reports/m3_20m_2024-05-07-09_validated/`,
`data/reports/m3_15m_2024-05-07-09_validated/`,
`data/reports/m3_10m_2024-05-07-09_validated/`, plus
`data/reports/m3_20m_10m_2024-05_validated/` (an earlier 2-band/1-month
run used as an intermediate cross-check). `data/cache/p533_scores.parquet`
holds all P.533 keys computed across this effort (10m: 481,269, 15m:
456,430, 20m: 1,409,378) and remains reusable for future eval_m3.py runs
against the same months/bands.

## Consequences

- M3's minimum-bar compute work is done. The full ROADMAP.md M3
  milestone still needs: RBN extractor (CW ground truth), live
  PSKReporter MQTT accumulator, cqdx R2 backfill + public-only rerun
  check, storm-window case studies (2-3 storms, before/during/after vs
  P.533), and a draft TAPR/DCC writeup outline — none of these are
  compute, they're separate dev/writing sub-projects.
- **Multi-band `eval_m3.py` runs on this machine should default to
  single-band slices first.** Every attempt combining 2+ bands'
  simultaneous extraction hit unexplained deaths (5 combined-scope
  failures total across the effort); every single-band attempt
  succeeded on its first try. If a future run needs the literal
  shared-across-bands GBT model (not just the same headline numbers),
  that will need either a quieter window on this machine or a more
  memory-careful multi-band extraction path than `extract_wsprnet_bands`
  currently has.
- `sudo pmset -a powernap 0` was run on this Mac mini during this effort
  and has not been reverted — Power Nap is currently disabled
  system-wide, not scoped to this job. Worth deciding whether to
  re-enable it now that the immediate need has passed.
