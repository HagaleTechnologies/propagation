---
id: gotcha-open-closed-boundary
title: What will bite you if you add a cqdx dependency?
kind: gotcha
status: current
maintainer: agent
sources:
  - README.md
  - ARCHITECTURE.md
verified:
  commit: 5c2dac7
  date: 2026-07-07
links:
  - overview
  - interface-prediction-surface
---
This repo is intended to be open source; cqdx is closed. If you import any
cqdx code, call its internal APIs, or write logic that requires cqdx's private
R2 spot archive to run, you have made the repo unopenable and broken the
fundamental reproducibility guarantee. The symptom is subtle: pipelines work
fine locally (you have cqdx access) and fail silently or mysteriously for
anyone else. This constraint is deliberate and permanent — see README.md
§"Licensing & boundaries".

## Symptom

Any step in the pipeline (extract, feature, train, eval, serve) that requires:
- cqdx R2 credentials or bucket access
- cqdx internal packages or types
- the cqdx propagation-sidecar (the P.533 HTTP endpoint)

will fail for a fresh clone without cqdx access. The model/results cannot be
independently reproduced; the open-source release is blocked.

## Cause and workaround

The boundary rules (ARCHITECTURE.md §3.2 and README.md §"Licensing & boundaries"):

- **Public sources are the only required inputs**: WSPRnet monthly CSV.gz
  archives, RBN daily CSV zips, PSKReporter MQTT (live, accumulates going
  forward), NOAA SWPC JSON/FTP.
- **cqdx R2 archive is an optional accelerant** behind the same extractor
  interface (`src/propagation/data/base.py`). Nothing downstream knows which
  extractor produced a row beyond the `source` column. Published results
  MUST be reproducible from public sources alone — verify this before any
  milestone acceptance.
- **cqdx integration = the JSON contract only** (`contracts/prediction-surface.v1`
  consumed by a thin closed adapter on the cqdx side). This repo neither
  imports cqdx code nor knows cqdx exists beyond that adapter. See
  [[interface-prediction-surface]].
- **P.533 baseline**: vendored ITURHFProp build under `baselines/p533/`
  (not the cqdx propagation-sidecar). The sidecar wraps the same C library;
  it is a cross-check, never a dependency. See [[decision-p533-baseline]].
