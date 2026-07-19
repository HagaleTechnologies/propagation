# 0002: Storm-slice eval months must be verified against real Kp data, not assumed

## Status

Accepted.

## Context

ROADMAP.md's M1 acceptance criteria require "P.533's storm-time failure
visible in the Kp≥5 slice," and `docs/SPEC-labeling.md` §6 leakage rule 5
requires headline eval to include ≥1 storm fold. M1's first live acceptance
run reused M0's eval month (2014-08) without checking its actual storm
content. `scripts/eval_m1.py`'s storm/quiet split came back nearly flat
(P.533 Brier 0.317 storm vs. 0.321 quiet) — not because the storm-degradation
hypothesis is wrong, but because 2014-08 turned out to contain exactly one
3-hour block at Kp=5.333 all month (the rest of the month maxed at Kp=4.667).
It technically clears "Kp≥5" and "≥1 storm fold," but doesn't contain a real
storm to demonstrate the failure mode with.

A second run against 2024-05 (the May 2024 "Gannon storm," a G5/extreme
event, the strongest since 2003 — max Kp=9.0, 25 blocks ≥Kp5 that month) showed
the expected result clearly: P.533 Brier degraded from 0.396 (quiet) to 0.419
(storm), visible concretely in its own top-confidence bin (predicted
p_open 0.9–1.0: 30.6% actual open rate in quiet conditions vs. 24.1% during
the storm, over a 340k-row sample).

## Decision

Before accepting any milestone's storm-slice eval result, verify the eval
month's actual Kp distribution first — don't assume a month contains a
storm just because it's the month already in hand from a prior milestone,
and don't accept "the slice cleared the Kp≥5 threshold at least once" as
equivalent to "the slice demonstrates storm behavior." Concretely: query
`propagation.eval.stratify.fetch_definitive_kp()` for the candidate month
(and, ideally, the months around it) before running a full eval, and prefer
a month with either several Kp≥5 blocks or at least one Kp≥6+ event.
GFZ's historical archive covers 1932–present, so real candidate storms are
cheap to identify in advance.

## Consequences

- A storm-slice result with a small `n` (few hundred rows) or a small
  quiet-vs-storm gap should be treated as inconclusive on its face, not as
  evidence the model doesn't degrade during storms — check the underlying
  Kp distribution before drawing either conclusion.
- Future milestones (M2/M3, which also carry a storm-fold requirement) should
  pick eval windows the same way: verify real storm content first, rather
  than inheriting whatever window a prior milestone happened to use.
