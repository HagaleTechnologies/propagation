---
id: gotcha-qa-diurnal-checks
title: What will bite you about the QA diurnal-ratio checks (1 and 2)?
kind: gotcha
status: current
maintainer: agent
sources:
  - src/propagation/qa/checks.py
  - docs/SPEC-labeling.md
verified:
  commit: 5f75677
  date: 2026-07-14
links:
  - overview
  - decision-m0-scope-reductions
---
QA checks 1 (20m day/night open-rate ratio > 2) and 2 (160m/80m night/day
ratio > 5) look trivial to implement from docs/SPEC-labeling.md §6 and are
not. Two non-obvious requirements, both discovered only when the check was
run against a real month of data (M0 acceptance, train=2014-06):

1. **Distance filter.** The spec's physics intuition holds for skywave
   paths; raw label sets are dominated by short paths where the diurnal
   signal washes out. Check 1 filters to 3000–8000 km before computing the
   ratio (see `_diurnal_ratio_check` in `src/propagation/qa/checks.py`).
2. **Local-time correction.** Bucketing by raw UTC hour is meaningless for
   a globally distributed dataset — "day" at 12:00 UTC is night for half
   the paths. The check converts to path-local time via the path-midpoint
   longitude, using antimeridian-aware longitude averaging (naive averaging
   puts a JA↔W6 path's midpoint over Africa).

Even with both fixes, check 1 honestly reports **fail** on the M0 run
(ratio 1.27 vs the spec's >2). Root cause is documented in PR #11: the
remaining gap needs real solar-geometry features (terminator position,
solar zenith angle), which are M2 scope (`features/solar.py`). Do NOT
"fix" this by loosening the threshold or widening the distance band until
the local-time proxy has been replaced with actual solar geometry — a
persistent honest fail is the designed behavior here.
