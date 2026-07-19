# Architecture

Status: v0.1 (initial design, 2026-07-06). Decisions below are made, not open,
unless marked **[research-dependent]**.

---

## 1. Prediction task

### 1.1 Unit of prediction

The **path-cell**: `(tx_field, rx_field, band)` where fields are 2-character
Maidenhead fields (~20° × 10°, 324 worldwide), with 4-character grid squares
retained in the data so we can move to finer cells where density supports it.
Fields are coarse, but they match how operators think ("Europe on 15m", "JA
long-path") and give enough spots per cell per hour to define labels. Cell
granularity is a tunable, not a schema change — the feature store keys on
grid4 and aggregates at query time.

### 1.2 Targets

For each path-cell and 15-minute window `t`, at horizons
`h ∈ {0, +1h, +3h, +6h, +12h, +24h}`:

1. **Openness** (primary): `P(open)` — binary, "open" = at least `k=1` distinct
   (dx_call, de_call) report in the window. Class probabilities, properly
   calibrated, not just labels.
2. **Expected SNR** (secondary): median reported SNR for the window, regression,
   trained only on open windows. Mode-normalized (normative offsets in
   docs/SPEC-labeling.md §4.4) so FT8 at −18 dB and CW at +6 dB land on a
   comparable scale.

`h=0` is a true *nowcast*: given partial real-time observations (spots seen in
the last 15–60 min elsewhere on the band/region) infer current openness of
cells with no direct observation. `h>0` is forecasting.

### 1.3 The observation-bias problem (and the chosen mitigation)

A spot proves a path was open. **The absence of a spot proves almost nothing**:
maybe the path was closed, maybe nobody transmitted, maybe no receiver was
listening. Naive training on "spot = 1, no spot = 0" learns receiver geography,
not propagation. This is the central methodological problem and the part of the
work that is genuinely publishable.

**Decision: monitor-normalized negative sampling.**

- A receiver `R` is **provably monitoring band `b` in window `t`** if `R`
  reported ≥1 spot on band `b` in `[t−30min, t+30min]` (for wideband digital
  monitors like FT8/WSPR receivers, decoding anything proves the receiver was
  up and listening on that band).
- A **valid negative** for path-cell `(TX, RX, b, t)` requires: (a) ≥1 provably
  monitoring receiver in `RX` on band `b`, AND (b) ≥1 station in `TX` provably
  transmitting on band `b` in that window (they were spotted *somewhere* — by
  any receiver worldwide). Both sides active, no report between them → strong
  evidence the path was closed.
- Windows failing (a) or (b) are **unlabeled**, excluded from train and eval.
- WSPR data gets a quality tier of its own: beacons transmit on schedule with
  known power, so the "was anyone transmitting" condition is nearly free.

Receiver-uptime tables (per receiver grid4 × band × window) are a first-class
pipeline artifact, not an afterthought — evaluation validity depends on them.

**[research-dependent]** Whether to weight negatives by monitor count (more
listeners silent = stronger negative) — start unweighted, measure.

---

## 2. System overview

```
                       PUBLIC SOURCES                       PRIVATE ACCELERANT
  ┌────────────┐ ┌───────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────────┐
  │ WSPRnet    │ │ RBN CSV   │ │ PSKReporter│ │ NOAA SWPC│ │ cqdx R2 archive  │
  │ monthly    │ │ daily     │ │ MQTT (live)│ │ GOES     │ │ raw/YYYY-MM-DD/  │
  │ archives   │ │ archives  │ │            │ │ indices  │ │ HH/*.ndjson.gz   │
  └─────┬──────┘ └────┬──────┘ └─────┬──────┘ └────┬─────┘ └────────┬─────────┘
        └─────────────┴──── extractors (one per source, common ─────┘
                             output schema)        │
                                                   ▼
                        ┌────────────────────────────────────┐
                        │  Lake: Parquet, partitioned         │
                        │  band=…/date=…  +  DuckDB views     │
                        ├────────────────────────────────────┤
                        │  features: path-cell × window       │
                        │  matrices + receiver-uptime tables  │
                        ├────────────────────────────────────┤
                        │  models: climatology │ P.533 │ GBT  │
                        ├────────────────────────────────────┤
                        │  eval: blocked time-series CV,      │
                        │  headline vs-P.533 table            │
                        └───────────────┬────────────────────┘
                                        ▼
                        serving: batch inference → versioned
                        JSON prediction surface (§7)
                                        ▼
                     (closed-source cqdx adapter → KV/D1 → UI)
```

## 3. Data pipeline

### 3.1 Analytical store

**Decision: DuckDB over Parquet files.** Local-first (runs on a Mac), zero
infra, columnar scans over years of spots, and Parquet is the interchange
format if anything ever moves to a warehouse. No Postgres, no cloud warehouse.

Layout:

```
lake/
  spots/            band=20m/date=2026-07-06/*.parquet
  receiver_uptime/  band=20m/date=2026-07-06/*.parquet
  space_weather/    date=2026-07-06/*.parquet
  labels/           band=20m/date=…/*.parquet   # stored once; horizon is a
                                                # training-time join offset
                                                # (SPEC-labeling §4.5)
```

Common spot schema (superset of all sources; nullable where a source lacks it):

| column | type | notes |
|---|---|---|
| source | enum | `wsprnet` \| `rbn` \| `pskreporter` \| `cqdx` |
| ts | timestamp | UTC |
| band | text | `160m` … `6m` |
| mode | text | FT8, WSPR, CW, … |
| freq_hz | int64 | |
| dx_call, de_call | text | tx / rx |
| dx_grid, de_grid | text | 4–6 char Maidenhead |
| dx_lat/lon, de_lat/lon | double | derived if only grid |
| snr_db | int16 | as reported (mode-relative) |
| tx_dbm | int16 | WSPR only — calibrated power |
| distance_km, bearing_deg | double | derived |

### 3.2 Extractors (pluggable, one per source)

Each extractor is a small Python module with one job: source format → common
schema Parquet. The interface is trivially pluggable so the private source is
optional:

- `extract-wsprnet`: downloads WSPRnet monthly CSV.gz archives (public, back to
  2008 — ~two solar cycles), converts. **The deep-history backbone.**
- `extract-rbn`: RBN daily CSV zip archives (public, back to 2009). CW/RTTY
  skimmer spots with SNR + WPM.
- `extract-pskreporter`: live MQTT subscriber writing hourly Parquet. For an
  outsider this is the only way to get PSKReporter history (no public bulk
  archive), so it accumulates going forward.
- `extract-cqdx` (**optional, private**): reads Tony's R2 archive
  (`raw/YYYY-MM-DD/HH/{ulid}.ndjson.gz`, gzipped NDJSON of camelCase spot
  records: `timestamp`, `frequency`, `band`, `mode`, `dxCall`, `dxGrid`,
  `dxLat`/`dxLon`, `deCall`, `deGrid`, `deLat`/`deLon`, `snr`, `source`, …) via
  the S3-compatible R2 API. Gives dense multi-mode PSKReporter+RBN+cluster
  history predating the public MQTT subscriber. **Nothing downstream may know
  which extractor produced a row** beyond the `source` column; published
  results must reproduce from public sources alone.

### 3.3 Space weather

From NOAA SWPC (public JSON/FTP): Kp (3-hourly, definitive + est), F10.7
(daily), GOES X-ray flux (1-min, for flare/SID features), solar wind
speed/density/Bz from DSCOVR (1-min). Optional later: GIRO ionosonde foF2/MUF
via DIDBase for assimilation-style features. All resampled onto the 15-minute
window grid with explicit as-of-time semantics (only values *available at
prediction time* — Kp is published with delay; use the estimated series, not
the definitive one, for anything feeding a forecast feature).

## 4. Feature engineering

Per (path-cell, window, horizon):

1. **Path geometry** (static): great-circle distance, bearing, midpoint lat/lon,
   whether the path crosses the auroral oval / geomagnetic latitude of the
   midpoint and control points (P.533 uses control points at 1000 km from each
   terminus — mirror that).
2. **Solar geometry** (computable for any future time — key for h>0): solar
   zenith angle at tx, rx, midpoint, control points; fraction of path in
   daylight; time since sunrise/sunset at midpoint (gray-line features).
3. **Time**: hour-of-day and day-of-year as sin/cos pairs; month.
4. **Space weather** (as-of prediction time): Kp now and lagged 3/6/12/24/48h;
   F10.7 daily + 27-day smoothed (solar rotation); GOES X-ray flux now (flare
   → D-layer absorption); solar wind Bz/speed (storm onset precursors).
5. **Autoregressive spot history** (the nowcasting edge; as-of prediction
   time): spot counts and median SNR for *this* path-cell over trailing 15m/1h/
   3h/24h; same for the reverse path, adjacent cells, adjacent bands (one band
   up/down — MUF is sliding); same-cell-same-hour-yesterday; band-wide global
   activity (controls for contest weekends vs. dead Tuesdays).
6. **Mode normalization** (normative: docs/SPEC-labeling.md §4.4): per-mode
   SNR offsets to a common reference
   (decision: normalize to "FT8-equivalent dB" with fixed published offsets;
   revisit empirically). `tx_dbm` used where present (WSPR) to normalize for
   power. Autoregressive median-SNR features (item 5) use the same
   FT8-equivalent scale.

## 5. Modeling ladder

Strictly ordered; each rung must beat the previous on the eval harness before
moving on. **Exception on record:** M1's real eval result has P.533 losing
badly to climatology, not beating it (`docs/DECISIONS/0003`); M2 proceeded
anyway by explicit decision rather than by satisfying this rule literally.
M2's own bar (beats both M-0 and M-1) is unaffected.

- **M-0 Climatology**: P(open) = historical open-rate for (path-cell, band,
  hour-of-day, month) at similar smoothed SSN. Embarrassingly strong baseline;
  any model that can't beat it is learning nothing beyond seasonality.
- **M-1 ITU-R P.533** (the incumbent): reliability_pct and SNR from ITURHFProp
  for each path-cell (midpoint-to-midpoint, monthly SSN). **Decision:** the
  benchmark harness builds ITURHFProp from ITU's public source as a standalone
  CLI vendored under `baselines/p533/` — the repo does NOT call cqdx's
  propagation-sidecar, so the headline comparison is reproducible without cqdx
  access. (The sidecar wraps the same P533.c, so numbers will agree; that's a
  cross-check, not a dependency.)
- **M-2 Gradient-boosted trees (LightGBM)** on the §4 tabular features. The
  workhorse. One model per horizon (shared across bands, band as a feature),
  binary log-loss objective for openness, quantile objectives for SNR.
  Calibration checked, isotonic post-hoc if needed.
- **M-3 [research-dependent] Spatiotemporal model**: band-conditioned global
  grid maps (ConvGRU / graph NN over grid cells) predicting the whole openness
  field jointly, capturing spatial coherence M-2 can't. Only after M-2 vs
  P.533 results are in.

## 6. Evaluation

- **Splits**: blocked time-series CV — train on months `[t0, t1)`, validate on
  `[t1+gap, t2)`, with gap ≥ max horizon + max autoregressive lookback (≥48h;
  normative rule in docs/SPEC-labeling.md §6) so autoregressive features can't
  leak.
  Multiple folds across seasons and Kp regimes. **No random splits, ever.**
- **Metrics**: Brier score and log-loss (proper scoring, headline), PR-AUC
  (openness is heavily imbalanced on marginal bands), reliability diagrams
  (calibration is the product feature — "73% chance" must mean 73%). SNR: MAE
  + pinball loss on quantiles.
- **Slices**: by band, by horizon, by region-pair, by Kp regime (quiet vs
  storm — storms are where climatology fails hardest and the model should win
  biggest), by season.
- **Headline artifact**: one table, `rows = {climatology, P.533, LightGBM}`,
  `cols = {Brier, log-loss, PR-AUC} × {h=0, +3h, +24h}`, per band group
  (low/mid/high HF). This table is the paper and the launch blog post.
- **Minimum credible result**: LightGBM beats P.533 *and* climatology on Brier
  at h=0 and +3h on 20m/15m/10m over ≥3 held-out months spanning at least one
  geomagnetic storm.

## 7. Serving & the cqdx contract (the public boundary)

**Decision: batch, not real-time inference.** A scheduled job (every 15 min)
scores all path-cells × bands × horizons — it's a few hundred thousand rows
through a GBT, cheap. Runs anywhere Python runs (local Mac via launchd first,
Fly.io later).

The public artifact is a **versioned JSON prediction surface**, defined in
`contracts/prediction-surface.v1.schema.json` in this repo (following the
dispensa cross-repo contract pattern used across Tony's ecosystem; if cqdx
formally adopts it, the contract can be mirrored/promoted into dispensa).

> **Note:** the sketch below is illustrative only and predates the normative
> schema, which uses a **columnar encoding** (parallel arrays, integer
> `p_open_pct`, integer confidence tiers, `horizons_s`, required
> `valid_from`/`valid_until`/`reference`) — see the schema file and
> docs/SPEC-contract-notes.md for the real wire format and why.

```jsonc
{
  "schema": "propagation.prediction-surface.v1",
  "generated_at": "2026-07-06T21:45:00Z",
  "model": { "id": "gbt-m2", "version": "2026.07.03", "trained_through": "2026-06-30" },
  "horizons": [0, 3600, 10800, 86400],          // seconds
  "cells": [
    {
      "tx_field": "EM", "rx_field": "PM", "band": "15m",
      "p_open": [0.91, 0.84, 0.31, 0.66],        // one per horizon
      "snr_db_ft8eq_p50": [-8, -11, null, -14],
      "confidence": "high"                        // data-density tier
    }
  ]
}
```

Consumers pull the surface over HTTP (or read the file from object storage).
A **thin closed-source adapter inside cqdx** fetches it, publishes to KV/D1 on
its schedule, and renders UI ("15m to JA: open now (91%), fading by ~0100Z").
This repo neither imports cqdx code nor knows cqdx exists beyond that adapter;
the contract is the entire interface, semver-versioned (`v1` → `v2` on breaking
change, additive fields allowed within a version).

## 8. Repo layout

Python 3.11+, `uv`-managed (matching the conventions in Tony's `health`/`sbn`
projects), `ruff` + `pytest`. MIT OR Apache-2.0.

```
propagation/
├── README.md / ARCHITECTURE.md / ROADMAP.md
├── pyproject.toml                # uv; deps: duckdb, polars, lightgbm, httpx,
│                                 #   pyarrow, astral (solar geom), scikit-learn
├── contracts/
│   └── prediction-surface.v1.schema.json
├── baselines/
│   └── p533/                     # vendored ITURHFProp source + build script + CLI wrapper
├── src/propagation/
│   ├── data/                     # extractors: wsprnet.py, rbn.py, pskreporter.py,
│   │   │                         #   cqdx_r2.py (optional), swpc.py, base.py (interface)
│   │   └── lake.py               # partitioning, DuckDB view registry
│   ├── features/                 # geometry.py, solar.py, spaceweather.py,
│   │                             #   history.py, labels.py (incl. uptime/negatives)
│   ├── models/                   # climatology.py, p533_baseline.py, gbt.py
│   ├── eval/                     # splits.py, metrics.py, report.py (headline table)
│   └── serving/                  # score.py (batch inference → surface JSON)
├── notebooks/                    # exploration only; nothing load-bearing
├── tests/
└── data/ (gitignored)            # local lake
```

## 9. Publication targets

- **TAPR/DCC proceedings** — natural home, technical ham audience, dataset +
  benchmark framing lands well.
- **QEX (ARRL)** — the "VOACAP vs. learned model" comparison as an article.
- Stretch: URSI / Radio Science if the observation-bias methodology and storm-
  time results are strong.
- The dataset-construction method (monitor-normalized labels from amateur
  reporting networks) is a contribution in its own right — write it up as such.
