---
id: question-p533-worse-than-climatology
title: Does M1 clear ARCHITECTURE.md's "each rung must beat the previous" gate?
kind: question
status: draft
maintainer: agent
sources:
  - ARCHITECTURE.md
verified:
  commit: 5431900
  date: 2026-07-19
links:
  - decision-p533-baseline
  - gotcha-eval-rules
---
ARCHITECTURE.md §5 states the modeling ladder is "strictly ordered; each rung
must beat the previous on the eval harness before moving on." M1's actual
live-run result has P.533 (M-1) losing badly to climatology (M-0) on Brier —
not narrowly, and not just in one slice — rather than beating it. Read
literally, this rule blocks starting M2 until that's resolved. It's unclear
whether that block is intended here.

## Current best understanding

Two live acceptance runs (2014-08, 2024-05) both show P.533 substantially
worse than climatology, and systematically overconfident (see
[[decision-p533-baseline]] for the qualitative story: P.533 answers "is this
path physically capable," climatology is fit directly on the same
distribution it's evaluated on and implicitly captures real activity/monitor
patterns P.533 can't see). Plausible resolutions, none yet decided:

- The ladder rule was written assuming a physics baseline would naturally
  beat naive climatology, which turned out false for this dataset — the
  rule may need amending to treat M-1 as a reference/incumbent rather than a
  strict gate, since ROADMAP.md's own M1 acceptance criteria only ask for
  "two rows with real numbers" and storm-slice visibility, not "beats M0."
- P.533's raw BCR/100 is uncalibrated; a post-hoc calibration step (e.g.
  isotonic, the same kind ARCHITECTURE.md §5 already calls for on M-2) might
  close most of the Brier gap without changing the underlying physics call —
  worth trying before concluding the ladder rule is simply wrong.
- This might be treated as the expected, informative result and not a
  blocker at all — decided explicitly rather than silently proceeding either
  way.

This became a live question during M1's wiki-update pass (2026-07-19); it
should be resolved (become a decision-digest, with the ladder rule text
amended if needed) before or during M2 planning, not silently ignored.
