# 0008: PRO-8's RBN-inclusive validation attempt 6 — Catalyst slot is cgroup-capped at 12 GiB and OOM-kills the run

## Status

Accepted (outcome recorded). Attempt 6 of the deferred validation (ADR 0007)
ran live and was OOM-killed by the container's cgroup memory cap before
reaching model fitting. PRO-8's second Gherkin scenario remains
**unconfirmed**, not confirmed-and-hidden — the same honest state ADR 0007
recorded, now with the "where does this run" question answered by
measurement: not in a Catalyst slot, and not only because of host RAM.

## Context

ADR 0007 deferred the live `eval_m3.py --include-rbn` validation after five
attempts (three OOM-class crashes, peak RSS 20.3 GB on a contended 12-core
box). The 2026-11-07 ticket comment left one open constraint: "Before
re-queuing, decide where that validation runs; it needs a bigger-memory host
than a Catalyst slot." The PRO-8 implement plan (2026-11-20) tried to resolve
that by measurement: the Catalyst implement slot reported `free -g` at
125 GB total / 107 GB available, 32 cores, no contention — roughly 5x the
20.3 GB recorded peak — and concluded the run could execute in-slot with
`--p533-workers 8` and a scope fallback ladder.

Two session-budget facts also applied (measured, not guessed): the session
had a hard wall-clock deadline ~100 minutes after the code phases, and the
four WSPRnet month archives total ~890M raw CSV lines (2024-01 alone is
13,520,064,926 bytes uncompressed ≈ 218M lines; raw gzip+csv iteration
measured at 549,542 lines/s before the extractor's per-row hygiene/dedup
overhead). Rung 1 (train 2024-01/02/03) therefore could not complete inside
the session window even on a machine with infinite memory — a timeout-class
rung-1 failure determinable a priori — so the ladder's rung 2 (train
2024-03 only, per ADR 0007 option (c)) was the first rung actually
attempted.

Alongside the run, two conformance fixes the run's evidence depends on
landed test-first (16 tests green, full suite 260 passed):

- `_merge_band_spots` now re-applies the shared cross-source `dedup_spots`
  at the merge seam (docs/SPEC-labeling.md sec 1 priority
  wsprnet > rbn), instead of a bare `pl.concat`.
- `_assert_rbn_participation` fails loud (`RuntimeError`) when
  `--include-rbn` yields zero qualifying spots across every requested
  train/eval month, instead of silently proceeding as a WSPRnet-only run.

## The run (attempt 6, 2026-11-20)

Scope, per the ladder's rung 2:

    uv run python scripts/eval_m3.py --bands 20m --horizons 0 3 \
        --train-months 2024-03 --eval-months 2024-05 \
        --data-dir data --include-rbn --p533-workers 8

Pre-flight in the same container: `baselines/p533/bin/iturhfprop` present;
all four WSPRnet monthly archives and all 122 RBN day zips downloaded and
verified (2024-01 = 2,814,538,915 bytes, matching the plan's recorded
sizes); OMNI2/SSN caches pre-fetched; the RBN 2024-05-11 one-day smoke
reproduced the research phase's counts exactly (`n_lines_read=264451`,
`n_parsed=103156`, `n_qualifying=80413`, 919 `no_usable_location` from the
cold DXCC cache — the documented ~0.9% cold-cache yield loss).

What happened:

1. RBN extraction for 2024-03 completed: **3,980,930 qualifying spots**
   (ADR 0007's attempt-3 recorded 4.01M for the same month; the -0.7% delta
   is the cold DXCC cache's `no_usable_location` loss, consistent with the
   smoke run above).
2. RBN extraction for 2024-05 completed: **6,927,719 qualifying spots**
   (ADR 0007 recorded 6.99M; same ~0.9% cold-cache delta). RBN
   participation in the run is therefore proven by the run log — the
   `_assert_rbn_participation` guard passed on real data.
3. ~1-3 minutes into WSPRnet extraction of the 2024-03 archive (the step
   after the last RBN print), the process died silently at **23m17s**
   wall-clock: no traceback, host `free -g` showing 57+ GB free the whole
   time.

Diagnosis (from the container's cgroup controls, read immediately after):

- `/sys/fs/cgroup/memory.max` = **12,884,901,888 bytes (12.0 GiB)** — the
  slot is cgroup-capped at 12 GiB regardless of the host's 125 GB, which is
  what `free -g` reports.
- `/sys/fs/cgroup/memory.peak` = 12,887,134,208 — RSS reached the cap.
- `/sys/fs/cgroup/memory.events`: `oom 4`, **`oom_kill 1`** — the kernel's
  cgroup OOM killer killed exactly one process, the eval.

The kill point is arithmetically consistent with the cap: at that moment the
process held the 2024-03 RBN frame (3.98M rows) in `rbn_by_month`, was
building the 2024-05 frame (`_build_rbn_spots_for_month` accumulates every
day frame in a list, then `pl.concat` copies the whole month again — a ~2x
transient on ~7M rows), and had just started WSPRnet extraction on top. A
direct re-probe of `extract_wsprnet_bands` on the same archive in the same
container ran cleanly until stopped — the failure is purely the 12 GiB
ceiling, not a code crash. This also explains the mechanism behind the
2026-11-07 comment: a Catalyst slot is not a "125 GB host"; it is a 12 GiB
cgroup on one, below the run's recorded 20.3 GB peak before GBT/P.533 even
begin.

## Decision

Record attempt 6 as an environment-capacity failure with a measured root
cause, and resolve the open host question:

1. **Scenario 2 stays unconfirmed.** No RBN-inclusive Brier number exists;
   nothing here changes ADR 0007's "not Done" framing. The headline CSVs
   this ADR's template quotes on a completing run were not produced.
2. **The "where does the validation run" question is answered by
   measurement:** not in a Catalyst slot of this class. The slot's cgroup
   memory cap (12 GiB) is ~60% of the run's recorded peak RSS (20.3 GB,
   ADR 0007 attempt 5) — the run cannot complete there at any scope the
   ladder allows, because `main()` holds every requested month's RBN frame
   simultaneously by design. The validation needs a host with **>= 32 GB
   actually available to the process** (2x the recorded peak), or the
   memory profile fixed first.
3. **Next actions, in order** (ADR 0007's options, now sharpened): (b)
   reduce `main()`'s resident footprint — the concrete, now-identified
   contributors are `rbn_by_month` retaining all months' frames while
   `_build_labels_for_months` re-extracts WSPRnet, and
   `_build_rbn_spots_for_month`'s list-then-concat doubling; building
   labels per month and releasing each month's RBN frame (or spilling
   frames to parquet under `--data-dir`) would cut the peak roughly in
   half. Then rerun rung 2 on a >= 32 GB host with `--p533-workers 8`.
4. The two conformance fixes are **kept**: they are correct per
   docs/SPEC-labeling.md independent of any run, and PRO-10's acceptance
   mode builds on the same merge-seam dedup.

## Consequences

- PRO-8's Gherkin scenario 2 remains open; the ticket is not Done. When a
  conforming run completes, ADR 0008 should be updated (or superseded) with
  the headline tables and per-month counts, and the wiki digest below
  updated with it.
- Attempt 6's byproducts make the next attempt cheaper: all four WSPRnet
  archives and all 122 RBN day zips are staged under `data/raw/`, the
  OMNI2/SSN caches are warm, and the DXCC cache is built — roughly 25
  minutes of per-run setup this attempt did not have to repeat and neither
  does the next one (on a persistent container).
- `free -g` (host view) is the wrong instrument for capacity planning in a
  Catalyst slot; read `/sys/fs/cgroup/memory.max` first. Recorded as a wiki
  gotcha.
- The 15m/10m RBN-inclusive extensions stay deferred under the M3 milestone
  (ROADMAP.md), unchanged by this attempt.

## References

- Run log and cgroup evidence:
  `docs/DECISIONS/attachments/0008-pro8-rbn-inclusive-validation/run.log`,
  `slot-memory-evidence.txt`
- `docs/DECISIONS/0007-pro8-rbn-retrain-validation-deferred.md` — attempts
  1-5 and the deferral this record continues
- `docs/DECISIONS/0006-m3-minimum-credible-result.md` — the WSPR-only bar
  scenario 2 must eventually be measured against
- `docs/SPEC-labeling.md` sec 1 — the cross-source dedup priority the merge
  seam now enforces
