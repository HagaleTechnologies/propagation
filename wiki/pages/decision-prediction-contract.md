---
id: decision-prediction-contract
title: Why is the prediction surface columnar and integer-quantized?
kind: decision-digest
status: current
maintainer: agent
sources:
  - docs/SPEC-contract-notes.md
  - contracts/prediction-surface.v1.schema.json
verified:
  commit: 5c2dac7
  date: 2026-07-07
links:
  - overview
  - interface-prediction-surface
---
The prediction surface contract (v1) uses columnar parallel arrays (not
per-cell objects) with integer-quantized probabilities and SNR. This encoding
choice was made to keep the payload under Cloudflare KV's 25 MB limit at
realistic scale, avoid float-formatting drift across languages, and stay cheap
to ship to browsers. The normative encoding decisions are in
docs/SPEC-contract-notes.md; the schema is in
contracts/prediction-surface.v1.schema.json.

## Digest

**Options considered:**
- Row-oriented objects per cell (ARCHITECTURE.md §7 sketch): natural to read,
  easy to extend, but repeating keys per cell dominates the payload. At ~20k
  cells × 6 horizons: ~6 MB raw, ~1–2 MB gzipped — marginal for KV.
- **Columnar parallel arrays (chosen)**: all `tx_field[i]`, `rx_field[i]`,
  `p_open_pct[i]` etc. describe cell `i`. Rough savings: ~1.6 MB raw,
  ~250–400 kB gzipped — comfortably under KV's 25 MB and cheap for browsers.
  JSON Schema cannot enforce the "all arrays have length `n_cells`" invariant;
  that's a normative rule, validated by the publisher's tests.

**Integer quantization**: `p_open_pct` is 0–100 integer percent (calibration
finer than 1% is illusory at this stage); SNR is integer dB. Halves payload
vs floats and avoids cross-platform float-formatting drift.

**Null = abstain**: a null in a per-horizon slot means the model declines to
predict. Consumers MUST render null as "no data", never as 0%.

**Sparse by construction**: only activity-gated cells appear. Absence means
"not scored", not "closed" — consumers fall back to climatology or P.533.

**Versioning**: additive changes (new optional columns) are allowed within v1
without a version bump. Breaking changes (type changes, removing required
fields, reordering parallel arrays) require v2 and ≥60 days of dual-publish.
See docs/SPEC-contract-notes.md §"Versioning & evolution" for the full rules.
