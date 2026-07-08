---
id: decision-p533-baseline
title: Why is ITURHFProp vendored here rather than calling cqdx's P.533 sidecar?
kind: decision-digest
status: current
maintainer: agent
sources:
  - ARCHITECTURE.md
  - ROADMAP.md
verified:
  commit: 5c2dac7
  date: 2026-07-07
links:
  - overview
  - gotcha-open-closed-boundary
---
The P.533 baseline is implemented by vendoring the ITU's public ITURHFProp C
source directly under `baselines/p533/` (a CLI wrapper scored against path-cells
in the eval harness), NOT by calling cqdx's `apps/propagation-sidecar`. This
keeps the headline model-vs-P.533 comparison independently reproducible without
any cqdx access. The cqdx sidecar wraps the same C library and is used only as
a private cross-check, never as a dependency. Normative rationale in
ARCHITECTURE.md §5 (M-1 modeling ladder entry).

## Digest

**Options considered:**
- Call cqdx's propagation-sidecar (HTTP) for P.533 scores: convenient (already
  running, same inputs), but creates a cqdx dependency in the eval harness.
  Published results could not be reproduced without cqdx access; the open-source
  release is blocked.
- **Vendor ITURHFProp from ITU's public source (chosen)**: build ITURHFProp
  as a standalone CLI under `baselines/p533/`; wrap it to score any
  (path-cell, band, hour, month, SSN). Fully reproducible from public
  ITU sources. The cqdx sidecar wraps the same `P533.c`; numbers should
  agree — that agreement is a private sanity cross-check (~100 paths spot-check
  per ROADMAP.md M1), not a production dependency.

**M1 acceptance criteria** (ROADMAP.md): the headline eval table has two rows
(climatology, P.533) with real numbers, and P.533's storm-time failure is
visible in the Kp ≥ 5 slice. This table is the paper and the launch blog post.

**Modeling ladder order** (ARCHITECTURE.md §5): M-0 climatology must be in place
and evaluated before M-1 P.533 work starts. M-1 must beat M-0 in the eval
harness before M-2 (LightGBM) begins. Milestones are strictly ordered —
do not skip rungs.
