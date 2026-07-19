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
  commit: 5431900
  date: 2026-07-19
links:
  - overview
  - gotcha-open-closed-boundary
  - gotcha-eval-rules
  - gotcha-live-system-bugs
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

**M1 shipped** (PR #13, live-run fixes in #14): `scripts/eval_m1.py` produces
the headline table (climatology vs. P.533, overall/storm/quiet slices) via
`baselines/p533/` + `propagation.models.p533.P533Model`. P.533 does NOT beat
climatology — it is both less accurate and systematically overconfident
(its own highest-confidence bin only opens ~25–52% of the time depending on
eval month, not the ~98% it predicts). The likely cause: P.533's BCR answers
"is this path physically capable," while the labels answer "did a real
operator transmit and a real monitor hear it in this window" — exactly the
observation-bias gap this repo's labeling methodology exists to model, which
a pure physics baseline has no way to see. This is the expected, useful
result: it's the gap M2 (LightGBM, conditioned on real activity/monitor
data) needs to close. See [[gotcha-eval-rules]] for the storm-slice
methodology lesson (0002) and [[gotcha-live-system-bugs]] for bugs the live
acceptance run caught that no mocked test did.

**Modeling ladder order** (ARCHITECTURE.md §5): "Strictly ordered; each rung
must beat the previous on the eval harness before moving on." M1's actual
result does NOT satisfy this literally — P.533 loses badly to climatology,
not beats it. Tony explicitly decided to proceed to M2 anyway rather than
amend the rule or recalibrate P.533 first — see
[[decision-m1-ladder-exception]] and docs/DECISIONS/0003.
