# propagation — data-driven HF propagation nowcasting

## Thesis

Every HF propagation tool the amateur radio community uses — VOACAP, ITU-R P.533,
and everything built on them — is an **empirical monthly-median model** whose
ionospheric coefficients were fitted decades ago. They answer "what does a typical
July at this sunspot number look like, on average, for this hour of day?" They are
climatology. They cannot tell you that 15m just opened to Japan twenty minutes ago,
or that today's geomagnetic disturbance has shut down the polar path that the model
says should be open.

Meanwhile, the amateur community operates the largest distributed ionospheric
sounding network ever built and mostly ignores it as a modeling input:

- **PSKReporter**: 500+ path reports *per second* (FT8/FT4/WSPR/CW and more),
  each one a ground-truth measurement — this path, on this band, at this moment,
  with this SNR, was open.
- **WSPRnet**: beacon-grade reports with calibrated power levels, monthly public
  archives back to 2008 — nearly two full solar cycles of history.
- **Reverse Beacon Network**: CW skimmer spots with SNR, archived daily since 2009.

This project trains models on that ground truth to **nowcast and short-horizon
forecast band openings** — P(path open) and expected SNR for (tx region, rx region,
band) over horizons from "right now" to +24 hours — and benchmarks them honestly
against the P.533 baseline. The bet: real-time observations of what the ionosphere
is *actually doing* (autoregressive spot history + live space weather) beat
40-year-old climatology at short horizons, by a lot.

## Why us / why now

- [cqdx](https://cqdx.app) already ingests the PSKReporter MQTT firehose plus RBN
  and cluster telnet feeds (Rust service on Fly.io), and archives **every raw spot**
  to R2 as hourly-partitioned gzipped NDJSON. The training data pipeline's hard
  part is already running in production.
- cqdx also runs an ITU-R P.533-14 engine (`apps/propagation-sidecar`, wrapping
  the vendored ITURHFProp C library) — the baseline to beat is one HTTP call away,
  with identical input conventions.
- A validated model ships directly as a cqdx feature ("15m to JA likely opens
  ~2340Z") and feeds pancetta's autonomous band selection.

## What this is not

- Not a physics ionosphere model (no ray tracing, no IRI assimilation — at least
  not in scope here).
- Not a global TEC map. The unit of prediction is the thing operators care about:
  *can I work that region on that band right now / soon*.

## Deliverables

1. A reproducible research pipeline (extract → features → train → evaluate) with
   a headline **model vs. P.533 vs. climatology** comparison.
2. A serving path: scheduled batch inference publishing band-condition predictions
   that cqdx consumes.
3. A writeup targeted at TAPR/DCC proceedings and/or QEX, with the dataset
   construction method (especially observation-bias handling) as a first-class
   contribution.

## Licensing & boundaries

This repo is intended to be **open source** (MIT OR Apache-2.0 dual license,
final call before first public release). cqdx is a separate, closed product.
The boundary is deliberate and enforced:

- **This repo stands alone.** No cqdx code imports, no cqdx-internal assumptions.
  Everything here — pipelines, features, training, evaluation, model artifacts,
  writeups — must be reproducible by an outsider from **public data sources**:
  WSPRnet monthly archives, RBN daily CSV archives, PSKReporter, NOAA SWPC.
- **cqdx's private R2 spot archive is an accelerant, not a dependency.** It plugs
  in behind the same extractor interface as the public sources (see
  ARCHITECTURE.md §3). Published results must be reproducible from public data.
- **Integration is contract-based.** This repo publishes a versioned JSON
  prediction API contract (ARCHITECTURE.md §7); a thin closed-source adapter on
  the cqdx side consumes it. Model → product coupling never runs the other way.
- **The P.533 baseline is independent.** The benchmark harness builds ITURHFProp
  from ITU's public source directly, so the headline comparison does not require
  cqdx access.
- **Data licensing:** PSKReporter and WSPRnet data come with community usage
  terms (non-commercial spirit, attribution, don't hammer the servers). This repo
  ships *code* and *derived model artifacts*, not redistributed raw spot dumps;
  extraction scripts pull from the canonical public archives and cache locally.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and
[ROADMAP.md](ROADMAP.md) for milestones.
