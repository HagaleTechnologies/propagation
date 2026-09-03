# 0007: PRO-8's RBN-included retrain validation deferred — real OOM at reduced scope

## Status

Accepted.

## Context

PRO-8's second acceptance scenario ("a model trained with RBN included
still clears M2's bar") required two things: `eval_m3.py` actually ingesting
RBN data (it never did -- only `extract_rbn`/`download_rbn_archive` existed,
unwired from the eval driver), and a real run confirming the reported Brier
score still beats climatology/P.533. This session implemented the first
part in full (`--include-rbn`, `geo.latlon_to_grid` closing a pipeline gap
where RBN spots had `dx_grid=None` and silently vanished from
`build_universe`/`build_labels` despite passing QA -- see the PR that lands
alongside this doc) and attempted the second, live, five times.

**What happened, in order:**
1. Full scope (train=2024-01/02/03, eval=2024-05, band=20m, h=0,
   `--include-rbn`) ran ~50min through RBN extraction (20.1M qualifying
   spots across 4 months) and WSPRnet/feature-matrix construction, then hit
   `FileNotFoundError` on `baselines/p533/bin/iturhfprop` -- the fresh
   worktree never ran `uv run build-p533`. Environment gap, not a real
   finding.
2. Same scope, after `build-p533`: crashed immediately with
   `dbm.error: db type could not be determined` resolving RBN's DXCC
   lookup cache (`~/.local/cty`) -- built under a different Python's dbm
   backend (3.13) than this worktree now uses (3.12.3, pulled in by PRO-7's
   `.python-version` pin via the rebase onto main between attempts 1 and
   2). Cleared the stale cache; confirmed it rebuilds cleanly. Environment
   gap, not a real finding.
3. Same scope, cache fixed: completed all RBN extraction (matching attempt
   1's counts exactly -- 3.57M/5.49M/4.01M/6.99M spots for
   Jan/Feb/Mar/May), then died with **no traceback** partway into
   WSPRnet+feature-matrix processing. Confirmed via a stray `tail` PID left
   running after the parent shell vanished, and later confirmed directly:
   `PROCESS_EXIT_CODE=137` (SIGKILL) once the orphaned log-tail was killed
   and its pipeline unblocked. A real OOM kill.
4. Reduced scope (train=2024-03 only, same eval/band/horizon): RSS climbed
   steadily to ~8.4GB by the 30min mark, then spiked to 17.7GB with the
   process pinned at 1000%+ CPU (P.533's `ThreadPoolExecutor`, sized
   `os.cpu_count()` = 12 on this machine, forking that many concurrent
   `iturhfprop` subprocesses). System free memory measured at 75MB at that
   point -- killed manually before the kernel could pick an unpredictable
   victim.
5. Same reduced scope, `--p533-workers 3` (new flag, added this session --
   `P533Model` already accepted `max_workers`, just wasn't exposed on the
   CLI): materially better trajectory through the first 25min (RSS ~8.1GB
   vs. the prior attempt's ~8.8GB at the same mark, system free draining
   more slowly), but still spiked catastrophically in the same phase --
   system free went from 10.7GB to 61MB in under a minute, RSS to 20.3GB.
   A watchdog (system free < 3GB -> SIGTERM then SIGKILL, checked every
   60s) killed it proactively this time -- clean recovery, no impact to the
   concurrently-running PRO-9 soak test on the same machine.

**Why this differs from ADR 0006's already-documented instability**: 0006's
kills were WSPRnet-extraction-phase OOM risk from `extract_wsprnet_bands`
buffering multiple bands simultaneously -- mitigated by single-band runs,
which this session's scope already was throughout. What's new here: RBN
roughly **doubles** the row volume feeding `build_universe`/`build_labels`
for a given band/month (RBN's own per-band-per-day counts run
150k-230k/day for 20m -- see attempt 1's per-month totals above -- on top of
WSPRnet's existing volume), and the crash phase has moved downstream, into
GBT fit / P.533 batch scoring rather than WSPRnet extraction itself. Capping
P.533's subprocess pool (attempt 5) measurably slowed the climb but did not
prevent the spike -- the dominant driver appears to be the size of the
feature matrix / GBT training data itself at this combined row count, not
solely P.533's concurrency, though both very likely compound each other.
This session did not have time (or, at 4am, sound judgment) to isolate
which further.

Also relevant, not this repo's fault: a separate long-running job
(`skimmer`/`manta`'s MAN-19 soak harness, `man19-soak`, started 13:34 the
prior day) was consuming 3-4 CPU cores throughout every attempt above --
out of scope to touch (a different ticket's tracked validation), but a real
contributor to the contention this session's attempts were fighting.

## Decision

**Ship the code, defer the live validation.** `--include-rbn` and
`geo.latlon_to_grid` are correct, unit-tested (full coverage of the
column-order/merge logic, the grid derivation, the day/month RBN
aggregation, all with fakes -- no live run required to trust the *code*),
and land in the PR alongside this doc. PRO-8's second Gherkin scenario --
does GBT-with-RBN actually still clear M2's bar -- remains **unconfirmed**,
not confirmed-and-hidden. Ticket stays in whatever state reflects that
honestly (not Done).

## Consequences

- Whoever picks this back up should NOT default to the same full-scope
  invocation that failed 3 of 5 times above. Options, roughly in order of
  effort: (a) retry off-hours / after `man19-soak` (or whatever's
  concurrently running) has finished, watching RSS with the same
  60s-interval/3GB-floor watchdog pattern used in attempt 5 (worth
  extracting to a small reusable script rather than re-inlining); (b)
  profile which specific step in `build_feature_matrix`/`GBTModel.fit`
  drives the spike (attempt 5's data suggests it's not purely P533's
  subprocess count) and fix the actual memory-inefficient step; (c) shrink
  further still -- a single band-month pair each side, accepting an even
  less representative validation, purely to get SOME signal on the
  RBN-inclusion question before committing to a bigger run.
- `--p533-workers` (new CLI flag on `eval_m3.py`, defaults to
  `P533Model`'s own `os.cpu_count()` default, unchanged for every other
  caller) is worth keeping regardless of whether it alone would have been
  sufficient here -- it's a real, previously-missing lever for exactly this
  class of contention, and cost nothing to add.
- The stale-`~/.local/cty`-cache-across-Python-versions gap (attempt 2) is
  a `dxentity` (third-party) library brittleness, not something to fix in
  this repo -- but worth remembering if a future session's Python version
  changes mid-work and RBN extraction throws a `dbm.error`.
