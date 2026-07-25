# M3 sub-project: band/horizon expansion — design

**Status:** approved, pre-implementation
**Parent:** ROADMAP.md M3 ("Extend to all HF bands, horizons {0,1h,3h,6h,12h,24h}")
**Scope note:** M3 is decomposed into independent sub-projects (band/horizon
expansion, RBN extractor, live PSKReporter accumulator, cqdx R2 backfill,
storm case studies + writeup). This spec covers band/horizon expansion only.

## 1. Goal

Generalize the M2 pipeline — currently proven on 20m+10m at h=0 only — to all
11 HF bands (160m,80m,60m,40m,30m,20m,17m,15m,12m,10m,6m) and horizons
{0,1h,3h,6h,12h,24h}, per ARCHITECTURE.md §5's "one model per horizon, shared
across bands, band as a feature."

**Definition of done:** the pipeline *supports* arbitrary band lists and all
six horizons end-to-end, proven by unit tests plus a smoke run on a band/
horizon combination beyond what M2 already validated. The full 11×6
historical production run and headline table are explicitly deferred to a
follow-up (see §6).

## 2. Horizon mechanics

Every M2 feature is currently computed as-of `window_start` (h=0 only).
Features split into two classes by whether their value is knowable in
advance of the forecast:

- **Future-computable** — time-of-day (`add_time_features`), solar geometry
  (`add_solar_features`), static path geometry (`add_geometry_features`).
  These stay anchored at `window_start` (the target window being forecast) —
  unchanged. Solar position at a future time is deterministic; this is
  exactly why solar.py's docstring already says it's "needed for h>0."
- **As-of-now** — space weather (`add_spaceweather_features`) and
  autoregressive spot history (`add_history_features`). These must anchor at
  `prediction_time = window_start − horizon_hours`, not `window_start`, or
  the model leaks future space-weather/activity data into a forecast.

Changes:
- `build_feature_matrix(labels, full_history, omni, horizon_hours: float =
  0.0)` computes `prediction_time` and threads it to the as-of-now builders.
- `add_spaceweather_features` and `add_history_features` key their asof
  join / rolling-window anchor off `prediction_time` instead of
  `window_start`.
- `blocked_cv_gap_hours(max_horizon_hours, max_ar_lookback_hours)` (already
  generic) gets called with each model's real horizon instead of the
  hardcoded `3.0` in `eval_m2.py`.
- Labels remain stored once per SPEC-labeling.md's existing note ("horizon is
  a training-time join offset") — no schema change to `labels/`.

## 3. Band mechanics

- Add `band` to `FEATURE_COLUMNS` as an ordinal feature using the existing
  `_BAND_ORDER` list in `history.py` (already lists all 11 bands for the
  adjacent-band history feature). LightGBM handles an integer-encoded
  ordinal fine — no one-hot needed.
- `extract_wsprnet` currently filters one band per call over a full-month
  archive scan; scanning 11× per month re-reads the same file 11 times.
  Extend it to accept a band list and flush all requested bands' Parquet in
  one pass.
- Labels/receiver-uptime/universe stay band-scoped per SPEC-labeling.md (no
  change to their per-band logic), built per band then concatenated into one
  shared frame before feature-matrix construction and training — this is
  what makes "shared across bands" real.

## 4. QA gates

Checks 3 (`grayline_40m`), 6 (`solar_cycle`), 7 (`storm_response`) in
`qa/checks.py` are currently stubbed `"insufficient_data"`; their docstrings
say they were blocked on solar-geometry / space-weather features that M2
already shipped. Implement the real logic for all three now:
- Check 3: 40m gray-line open-rate peak near the terminator, using
  `midpoint_hours_since_terminator` from `features/solar.py`.
- Check 6: monthly 10m DX open-rate vs F10.7 correlation > 0.5, using
  `f107_daily`/`f107_smoothed_27d` from `features/spaceweather.py`.
- Check 7: Kp≥6 trans-polar open-rate ≤ 50% of Kp≤2 matched baseline, using
  `kp_now` from `features/spaceweather.py`.

Checks 1/2/4 (`diurnal_20m`, `lowband_diurnal`, `sporadic_e`) already filter
by band internally and generalize without change — they just get exercised
against a wider label set.

## 5. Reporting

ARCHITECTURE.md §6's headline table groups bands low/mid/high HF:
- low = {160m, 80m, 60m, 40m}
- mid = {30m, 20m, 17m, 15m}
- high = {12m, 10m, 6m}

New `scripts/eval_m3.py` (matching the `eval_m1.py`/`eval_m2.py` convention)
drives the headline report for an arbitrary band list × horizon list, rather
than hardcoding 20m/10m at h=0 like `eval_m2.py` does. `eval_m2.py` is left
as-is (it's the M2 acceptance artifact of record).

## 6. Testing & validation

- `test_history.py` / `test_matrix.py` / `test_spaceweather.py`: horizon-shift
  correctness — as-of-now features anchor at `prediction_time`, not
  `window_start`, for h>0.
- `test_leakage.py`: new property — as-of-now features must never see data
  after `prediction_time` for h>0 (extends the existing Δ_avail
  boundary-exactness tests to the horizon-shifted case).
- New test for band-ordinal encoding in `FEATURE_COLUMNS`.
- New test for batched multi-band `extract_wsprnet`: output equivalence to N
  separate single-band calls, run against a synthetic multi-band fixture.
- `qa/test_checks.py`: real pass/fail assertions for checks 3/6/7 (currently
  only "insufficient_data" is tested).
- Smoke run: real WSPRnet data for a band/horizon combination beyond M2's
  proven 20m/10m@h=0 — e.g. add 40m and one non-zero horizon (+6h) — to prove
  the end-to-end path works. Not a full 11×6 historical run.

## 7. Explicitly out of scope

Deferred to their own sub-projects/follow-ups:
- Full 11-band × 6-horizon historical training/eval production run and the
  ARCHITECTURE.md §6 headline table as a completed artifact.
- RBN extractor, live PSKReporter MQTT accumulator, cqdx R2 backfill.
- Storm-window case studies, TAPR/DCC writeup outline.
