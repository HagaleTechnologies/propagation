---
id: gotcha-catalyst-slot-cgroup-memory
title: Why free -g overstates what a Catalyst slot can run (cgroup memory cap)?
kind: gotcha
status: current
maintainer: agent
sources:
  - docs/DECISIONS/0008-pro8-rbn-inclusive-validation.md
verified:
  commit: 8b7fa5e
  date: 2026-11-20
links:
  - decision-rbn-inclusive-validation
---
`free -g` in a Catalyst slot reports the HOST (125 GB on the class PRO-8's
attempt 6 ran on), but the slot process tree lives in a cgroup capped at 12
GiB (`/sys/fs/cgroup/memory.max`). The kernel OOM-kills silently from inside
the cgroup — no traceback, host free memory never looks stressed — which is
exactly how PRO-8's RBN-inclusive eval died at minute 23 (ADR 0008:
`memory.peak` at the cap, `memory.events` `oom_kill 1`). Before sizing any
memory-heavy run (eval_m3.py at multi-month scope is the known case), read
`/sys/fs/cgroup/memory.max`, not `free -g`; and budget peak RSS against that
number. ADR 0007's recorded 20.3 GB peak means multi-month eval runs do not
fit a 12 GiB slot at any scope the ladder allows.
