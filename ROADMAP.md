# Roadmap

Each milestone has acceptance criteria; don't start the next until they pass.
Bands for early milestones: **20m** (dense, always active) and **10m** (sparse,
solar-dependent — the hard case worth proving early).

## M0 — Lake bootstrap + climatology baseline

- `extract-wsprnet` pulls ≥1 month of WSPRnet archive for 20m into Parquet;
  DuckDB views over it; receiver-uptime table built.
- Labels (openness per path-cell × 15-min window, h=0) generated with
  monitor-normalized negatives; unlabeled fraction reported.
- Climatology baseline (M-0) trained/evaluated on a held-out later month.
- **Accept:** end-to-end `uv run` reproducibility from empty `data/`; Brier +
  reliability diagram for climatology produced by `eval/report.py`.

## M1 — P.533 baseline, standalone

- ITURHFProp vendored + built under `baselines/p533/`; CLI wrapper scores any
  (path-cell, band, hour, month, SSN).
- Scored over M0's eval windows; cross-checked (spot check ~100 paths) against
  cqdx's propagation-sidecar for agreement (private sanity check, not CI).
- **Accept:** headline table now has two rows (climatology, P.533) with real
  numbers; P.533's storm-time failure visible in the Kp≥5 slice.

## M2 — First learned model beats something

- Full feature matrix (§4) incl. space weather (SWPC extractors) and
  autoregressive history features, leakage-audited.
- LightGBM at h=0 and +3h on 20m + 10m, blocked CV.
- **Accept:** GBT beats climatology AND P.533 on Brier/log-loss at h=0 on both
  bands across ≥3 held-out months. If it doesn't, stop and diagnose — do not
  proceed to serving.

## M3 — Scale out + minimum credible result

- Extend to all HF bands, horizons {0, 1h, 3h, 6h, 12h, 24h}; add RBN extractor
  (CW ground truth) and live PSKReporter MQTT accumulator.
- Backfill from cqdx R2 archive via optional extractor (private accelerant);
  verify public-only rerun still clears M2's bar.
- Storm-window case studies (pick 2–3 storms in the data; before/during/after
  maps model vs P.533).
- **Accept:** the §6 "minimum credible result" holds; draft TAPR/DCC writeup
  outline exists.

## M4 — Serving + cqdx integration

- `serving/score.py` emits `prediction-surface.v1` JSON every 15 min (launchd
  on the Mac first); contract schema + docs published in `contracts/`
  (mirrored to dispensa if adopted ecosystem-wide).
- cqdx-side (closed) adapter consumes it → KV/D1 → "band outlook" UI; pancetta
  band-selection experiment fed from the same surface.
- Model registry discipline: surface carries model id/version/trained-through;
  weekly retrain job.
- **Accept:** live predictions visible in cqdx against real-time truth; a
  simple live scoreboard (yesterday's predictions vs. what actually happened)
  runs continuously — the ongoing honesty check.

## Later / research backlog

- M-3 spatiotemporal model (grid-field prediction).
- GIRO ionosonde foF2 assimilation features.
- Finer cells (grid4) where density allows; long-path handling.
- Public model artifact releases + a small "current conditions" public page.
- GOES X-ray flux feature (flare → D-layer absorption, ARCHITECTURE.md §4
  item 4): deferred out of M2 — its historical archive is NetCDF-based and
  meaningfully heavier to integrate than OMNI2 (which already covers M2's
  other space-weather features in one place). Stays in scope for the
  project; add when there's a concrete reason to prioritize the integration
  work, not silently dropped.
