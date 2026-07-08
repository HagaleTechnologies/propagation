---
id: subsystem-data-pipeline
title: How does the data pipeline work and why is it pluggable?
kind: subsystem
status: current
maintainer: agent
sources:
  - ARCHITECTURE.md
verified:
  commit: 5c2dac7
  date: 2026-07-07
links:
  - overview
  - gotcha-open-closed-boundary
---
The data pipeline converts heterogeneous amateur-radio spot archives from
multiple public sources (plus one optional private accelerant) into a unified
DuckDB-over-Parquet lake. The pluggable extractor interface is the technical
mechanism that enforces the open/closed boundary: the private cqdx R2 source
is an optional accelerant, never a dependency. The normative architecture is
in ARCHITECTURE.md §§2–3.

## How it works

**Lake layout** (`lake/`, gitignored):
- `spots/band=…/date=…/*.parquet` — spot rows in the common schema
- `receiver_uptime/band=…/date=…/*.parquet` — first-class label artifact
- `space_weather/date=…/*.parquet` — NOAA SWPC (Kp, F10.7, GOES X-ray)
- `labels/band=…/date=…/*.parquet` — openness labels, stored once
  (horizon is a training-time join offset, not a label property)

**Common spot schema**: columnar superset of all sources; key columns include
`source` (enum), `ts` (UTC), `band`, `mode`, `dx_call`/`de_call`,
`dx_grid`/`de_grid` (4-char Maidenhead), derived `dx_lat/lon`/`de_lat/lon`,
`snr_db`, `tx_dbm` (WSPR only), `distance_km`/`bearing_deg`. See
ARCHITECTURE.md §3.1 for the full schema table.

**Extractors** (one Python module per source, `src/propagation/data/`):
- `extract-wsprnet` — WSPRnet monthly CSV.gz archives (public, 2008–present)
- `extract-rbn` — RBN daily CSV zip archives (public, 2009–present)
- `extract-pskreporter` — live MQTT subscriber, accumulates going forward
- `extract-cqdx` (**optional, private**) — cqdx R2 archive via S3-compatible API

**Cross-source dedup**: key `(dx_call, de_call, band, mode, window)`;
source priority `wsprnet > rbn > pskreporter > cqdx`. Applied before labeling
and before any spot-count feature. See docs/SPEC-labeling.md §1.

## Why it is shaped this way

The pluggable interface exists to enforce the open/closed boundary —
see [[gotcha-open-closed-boundary]] and README.md §"Licensing & boundaries".
Nothing downstream may know which extractor produced a row beyond the `source`
column. DuckDB-over-Parquet was chosen for local-first operation (runs on a
Mac, zero infra) with columnar scan performance over years of spots; Parquet
is the interchange format if the lake ever moves to a warehouse. See
ARCHITECTURE.md §3.1 for the storage decision rationale.
