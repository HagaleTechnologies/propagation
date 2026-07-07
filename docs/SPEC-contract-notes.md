# SPEC: Prediction-surface contract notes (v1)

Companion to `contracts/prediction-surface.v1.schema.json`. Status: v1.0
(2026-07-06).

## Encoding decisions

**Columnar, not row-oriented.** **[deviation]** ARCHITECTURE §7 sketches
per-cell objects (`"cells": [{...}, ...]`). At realistic scale (~10–40k active
path-cells × 6 horizons) repeating keys per cell dominates the payload. The
schema instead uses parallel arrays under `columns` — element `i` of every
column describes cell `i`. Rough sizes for 20k cells, 6 horizons: row-objects
~6 MB raw; columnar ~1.6 MB raw, ~250–400 kB gzipped — comfortably under
Cloudflare KV's 25 MB value limit and cheap to ship to browsers. JSON Schema
cannot express "all arrays have length `n_cells`"; that invariant is normative
here and validated by the publisher's tests.

**Integer quantization.** `p_open_pct` is 0–100 integer percent (calibration
finer than 1% is illusory); SNR is integer dB. Halves payload vs floats and
avoids cross-platform float-formatting drift.

**Null = abstain.** A `null` in a per-horizon slot means the model declines to
predict (out-of-universe cell at that horizon, low-confidence SNR). Consumers
must render abstention as "no data", never as 0%.

**Sparse by construction.** Only cells in the activity-gated universe
(SPEC-labeling §2, plus model-scored extensions) appear. Absence of a cell
means "not scored", not "closed" — consumers needing a dense map fall back to
their own default (e.g. climatology or P.533) for missing cells.

**Delivery.** The publisher writes one document per generation cycle (every
15 min) to object storage / HTTP at a stable URL plus a timestamped URL.
Consumers poll the stable URL and check `generated_at`/`valid_until`. Gzip on
the wire is assumed (serve with `Content-Encoding: gzip`).

## Versioning & evolution

- **Additive (allowed within v1, no version bump):** new optional properties
  at any level (e.g. a `p_open_pct_p10`/`p90` uncertainty column, new `model`
  metadata). Consumers MUST ignore unknown properties — the schema keeps
  `additionalProperties: true` everywhere for exactly this reason.
- **Breaking (requires v2):** removing/renaming a required property, changing
  a type or unit, changing `cell_scheme` or the SNR reference, reordering
  semantics of parallel arrays, adding a new *required* property.
- Breaking changes ship as a new file `prediction-surface.v2.schema.json` with
  `"schema": "propagation.prediction-surface.v2"`; the publisher dual-publishes
  v1 and v2 for ≥60 days. The `schema` string is the discriminator — consumers
  hard-match it before parsing anything else.
- New enum members for `band` (e.g. 2m if the project ever covers VHF) are
  treated as **breaking** (closed enum), because consumers index UI by band.
- If cqdx (or anyone) formally adopts the contract, this file and the schema
  are promoted/mirrored into `dispensa` per that repo's ADR process; this repo
  remains the source of truth until then.

## Consumer obligations (summary)

1. Validate `schema` const before parsing.
2. Ignore unknown properties.
3. Treat `valid_until` as a hard staleness deadline.
4. Render `null` as absence, missing cells as "not scored".
5. Do not interpolate between horizons without labeling it as interpolation.
