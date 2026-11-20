---
id: decision-rbn-inclusive-validation
title: Where does the RBN-inclusive M3 validation run (and why not in a Catalyst slot)?
kind: decision-digest
status: current
maintainer: agent
sources:
  - docs/DECISIONS/0008-pro8-rbn-inclusive-validation.md
  - docs/DECISIONS/0007-pro8-rbn-retrain-validation-deferred.md
verified:
  commit: 8b7fa5e
  date: 2026-11-20
links:
  - gotcha-eval-rules
  - gotcha-catalyst-slot-cgroup-memory
---
PRO-8's second Gherkin scenario (does GBT trained with RBN included still
beat climatology/P.533 on Brier) is still unconfirmed after six live
attempts. Attempt 6 (ADR 0008) finally measured *why* the Catalyst slot
cannot host it: the slot is a 12 GiB cgroup on a 125 GB host, and the kernel
OOM-killed the run (`oom_kill 1`) at minute 23, holding two months of RBN
frames (~11M spots) just as WSPRnet extraction began — before GBT/P.533 even
start, and with the cap at roughly 60% of the 20.3 GB peak ADR 0007 recorded
on contended iron.
The run needs >= 32 GB process-available memory, or the eval driver's
resident footprint halved first (per-month label building instead of
retaining every month's RBN frame; ADR 0007 option (b), sharpened). The
merge-seam cross-source dedup and the fail-loud `--include-rbn`
participation guard it validated along the way are merged and independent of
any completing run.
