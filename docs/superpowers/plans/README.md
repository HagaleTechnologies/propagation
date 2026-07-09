# Implementation plans

One plan per ROADMAP milestone. Execute strictly in order — each milestone's
acceptance gate must pass before the next begins (M2's `check-m2-gate` exit 0
is a hard precondition for M3). Read `2026-07-09-INTERFACES.md` first: it
pins every cross-milestone name, path, and signature, including the
post-drafting reconciliation amendments.

| plan | milestone | scope |
|---|---|---|
| [INTERFACES](2026-07-09-INTERFACES.md) | — | pinned cross-milestone contract + amendments |
| [M0](2026-07-09-m0-lake-climatology.md) | Lake bootstrap | scaffolding, WSPRnet extractor, hygiene, uptime, labels, QA gates, climatology, eval harness |
| [M1](2026-07-09-m1-p533-baseline.md) | P.533 baseline | vendored ITURHFProp, CLI wrapper, two-row headline, Kp≥5 slice |
| [M2](2026-07-09-m2-features-lightgbm.md) | Learned model | SWPC extractor, 69-column feature matrix, leakage audit, LightGBM, acceptance gate |
| [M3](2026-07-09-m3-scaleout.md) | Scale-out | RBN + PSKReporter + optional cqdx R2, all bands/horizons, public-only rerun, storm case studies, writeup outline |
| [M4](2026-07-09-m4-serving.md) | Serving | surface generation + contract validation, launchd, registry, guarded retrain, scoreboard |

Flagged for owner review before execution:

- **M3 proposes a normative SPEC-labeling v1.1 amendment** (§1.6
  callsign→grid enrichment) — required for RBN, whose archives carry no
  location data. Spec change lands before code; review it first.
- External-format facts (ITURHFProp input cards, RBN CSV layout, PSKReporter
  MQTT payload keys, cqdx R2 timestamp encoding) are marked
  verify-at-execution in each plan rather than guessed.

Each plan executes on its own branch (`feat/m<N>-…`), lands by PR, and uses
subagent-driven development or executing-plans per its header.
