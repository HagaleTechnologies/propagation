# 0003: M2 starts despite M1 not beating M0 on the eval harness

## Status

Accepted.

## Context

`ARCHITECTURE.md` §5's modeling ladder is "strictly ordered; each rung must
beat the previous on the eval harness before moving on." M1's live
acceptance runs (2014-08 and, more decisively, the 2024-05 Gannon-storm
re-run) both show ITU-R P.533 (M-1) losing badly to climatology (M-0) on
Brier — overall Brier ~0.40 (P.533) vs. ~0.08 (climatology) in the 2024 run
— not a narrow miss. P.533 is also systematically overconfident: its own
top-confidence bin (predicted 0.9-1.0) only opens 24-31% of the time
depending on eval month, far below what it predicts. See
`wiki/pages/decision-p533-baseline.md` for the qualitative explanation
(P.533 answers "physically capable," the labels answer "a real operator and
monitor observed it," and P.533 has no way to see that gap).

Read literally, ARCHITECTURE.md §5 blocks starting M2 until this is
resolved. This was raised as an open question rather than decided
unilaterally (`wiki/pages/question-p533-worse-than-climatology.md`, now
retired in favor of this ADR).

## Decision

Proceed to M2 now. The ladder rule is not being formally amended, and P.533
is not being recalibrated first — this is a conscious choice to treat M1 as
complete and move forward, not a resolution of whether the rule's literal
wording still fits reality.

## Consequences

- M2's own acceptance bar is unambiguous and unchanged: LightGBM must beat
  BOTH P.533 and climatology (ARCHITECTURE.md §5, ROADMAP.md M2). Given M1's
  result, climatology — not P.533 — is the harder baseline to clear.
- The tension between the ladder rule's literal text and M1's actual result
  is left unresolved. A future pass may want to either amend §5's wording
  (e.g. to treat M-1 as a reference/incumbent rather than a strict gate) or
  revisit P.533 calibration; neither is in scope here.
- `wiki/pages/decision-m1-ladder-exception.md` is the wiki pointer to this
  decision; `wiki/pages/decision-p533-baseline.md` carries the underlying
  M1 result this decision responds to.
