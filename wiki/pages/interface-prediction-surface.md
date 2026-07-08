---
id: interface-prediction-surface
title: What contract does this repo expose for HF band-opening predictions?
kind: interface
status: current
maintainer: agent
sources:
  - contracts/prediction-surface.v1.schema.json
  - docs/SPEC-contract-notes.md
verified:
  commit: 5c2dac7
  date: 2026-07-07
links:
  - overview
  - decision-prediction-contract
---
This repo PRODUCES the `propagation.prediction-surface.v1` contract — a
batch-published JSON surface of HF path-cell openness and expected SNR.
Published every 15 minutes to object storage; consumed by a thin closed-source
adapter inside cqdx (no cqdx code in this repo — contract only). The normative
schema is `contracts/prediction-surface.v1.schema.json`; encoding decisions and
consumer obligations are in `docs/SPEC-contract-notes.md`. The cross-repo
adoption question is dispensa Q-0029.

## Pointers

- **Schema**: `contracts/prediction-surface.v1.schema.json` — discriminator
  field is `"schema": "propagation.prediction-surface.v1"`; validate this
  const before parsing anything.
- **Encoding notes**: `docs/SPEC-contract-notes.md` — columnar parallel arrays,
  integer `p_open_pct` (0–100), integer SNR (FT8-equivalent dB, 100 W reference),
  null = abstain, sparse (missing cell ≠ closed). See [[decision-prediction-contract]].
- **dispensa**: Q-0029 — open question asking cqdx to validate the adapter seam
  and accept the contract. Until adopted by cqdx, this repo is the source of truth.
  If formally adopted, the schema is mirrored into `dispensa/contracts/propagation/`.
- **Version pin**: v1 is current. Breaking changes produce `v2`; dual-publish
  for ≥60 days. Additive fields (new optional columns) allowed without bump.
  See docs/SPEC-contract-notes.md §"Versioning & evolution".
- **Consumer obligations** (summary from docs/SPEC-contract-notes.md):
  validate `schema` const; ignore unknown properties; treat `valid_until` as
  a hard staleness deadline; render null as absence; do not interpolate between
  horizons without labeling it as such.
