---
id: decision-m1-ladder-exception
title: Why did M2 start despite M1 not beating M0 on the eval harness?
kind: decision-digest
status: current
maintainer: agent
sources:
  - ARCHITECTURE.md
  - docs/DECISIONS/0003-m1-ladder-exception.md
verified:
  commit: f6ae8c3
  date: 2026-07-19
links:
  - decision-p533-baseline
  - overview
---
ARCHITECTURE.md §5 says the modeling ladder is "strictly ordered; each rung
must beat the previous on the eval harness before moving on." M1's actual
result has P.533 losing badly to climatology on Brier, in both live runs
(2014-08 and 2024-05). Tony explicitly decided to proceed to M2 anyway
rather than block on amending the rule or calibrating P.533 first — see
`docs/DECISIONS/0003` for the full reasoning.

## Digest

**Options considered** (raised as an open question during M1's wiki-update
pass, not decided unilaterally by the agent):
- Amend the ladder rule to treat M-1 as a reference/incumbent baseline
  rather than a strict gate, since ROADMAP.md's own M1 acceptance criteria
  only ask for "two rows with real numbers" and storm-slice visibility, not
  "beats M0."
- Calibrate P.533's raw `BCR/100` (e.g. isotonic) before drawing a
  head-to-head conclusion, since Brier conflates discrimination and
  calibration and P.533 is markedly overconfident, not just wrong.
- **Proceed to M2 as-is (chosen).** Tony's call: start M2 now; the ladder
  rule is not being formally amended or re-litigated as part of this
  decision.

**What this means for M2**: LightGBM's own "beats P.533 AND climatology"
acceptance bar (ARCHITECTURE.md §5, ROADMAP.md M2) stands unchanged and
unambiguous — climatology remains the harder baseline to beat given M1's
result, not P.533. Don't assume beating P.533 alone is sufficient.
