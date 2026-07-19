# M2 — Feature Matrix + LightGBM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full ARCHITECTURE.md §4 feature matrix (path geometry, solar geometry, time, space weather, autoregressive spot history, mode normalization) and a LightGBM model on top of it, then produce the 3-row headline table (climatology, P.533, LightGBM) LightGBM's own acceptance bar requires: beats both baselines on Brier/log-loss at h=0 and +3h on 20m/15m/10m over ≥3 held-out months including one geomagnetic storm.

**Architecture:** Feature engineering is a set of pure, independently-testable modules under `src/propagation/features/` that each take the labels frame (or a wider "full history" frame for autoregressive features) and return it with new columns added, joined on `(window_start, tx_field, rx_field, band)`. Space weather has two raw sources with the same shape after parsing: NASA's OMNI2 hourly archive (historical, used for training) and SWPC's live nowcast JSON feeds (used for serving, M4 — this plan builds the historical path only, with the module structured so the live path can be added later without touching feature logic). `GBTModel` duck-types `ClimatologyModel`'s `.fit()/.predict()` shape, same as `P533Model` did in M1 — no shared protocol class. `scripts/eval_m2.py` re-derives labels across many months the same way `scripts/eval_m1.py::_build_labels` does (for the same reason: the lake's `labels` table has a downsampled train partition), builds the feature matrix, trains LightGBM with blocked CV, and produces the 3-row headline table via the existing `write_headline_report`.

**Tech Stack:** Python 3.11+, `uv`-managed. New dependencies this milestone: `lightgbm` (the GBT model), `scikit-learn` (`IsotonicRegression` for post-hoc calibration, `average_precision_score` for the PR-AUC metric ARCHITECTURE §6 requires and `eval/metrics.py` doesn't have yet).

**Milestone acceptance (ROADMAP.md M2):** LightGBM beats P.533 AND climatology on Brier at h=0 and +3h on 20m/15m/10m over ≥3 held-out months spanning at least one geomagnetic storm. Given `docs/DECISIONS/0003`, climatology (not P.533) is the harder baseline to clear.

## Global Constraints

(Verified against the merged M0+M1 code — not against PR #10's `docs/superpowers/plans/2026-07-09-m2-features-lightgbm.md` draft, which predates both milestones and diverges from the real implementation the same way its M1 counterpart did. See `wiki/pages/gotcha-plan-drift-before-merge.md`.)

- Python **3.11+**, `uv`-managed. `ruff` + `pytest`. License: MIT OR Apache-2.0 (except `baselines/p533/upstream/`, per `docs/DECISIONS/0001` — irrelevant to this milestone, no P.533 files touched).
- Layout: `src/propagation/…` (src layout), tests mirror under `tests/`.
- `/data/` is the only gitignored data path (lake, raw archives, caches,
  reports all live under it). A bare top-level `reports/` is NOT gitignored
  (M1 had to special-case one artifact there; M2 has no such artifact, keep
  everything under `data/`).
- All timestamps UTC. Polars dtype `pl.Datetime("us", "UTC")`.
- 15-min windows aligned to UTC boundaries; `window_start` is the floor.
  Supported bands: `propagation.data.schema.SUPPORTED_BANDS` — a `set[str]`,
  no canonical order.
- **No cqdx imports, ever.**
- Blocked time-series CV via `propagation.eval.splits.blocked_time_series_folds`;
  never random splits. Gap is computed by
  `blocked_cv_gap_hours(max_horizon_hours, max_ar_lookback_hours) = max(48.0, horizon+lookback)`
  — for M2 (`horizon=3, lookback=24`) this evaluates to **48.0h, the same
  floor M0/M1 used** — `3 + 24 = 27 < 48`. Call the real function with real
  arguments every time; there is no module constant to reference.
- **Exogenous space-weather features are exempt from the horizon+lookback
  gap sum** (`docs/SPEC-labeling.md` §6 rule 1: "not derived from spots") —
  but MUST be computed strictly as-of/trailing, never centered. A centered
  27-day-smoothed F10.7 value would leak the future regardless of any CV
  gap; this is a correctness rule independent of CV, and gets its own
  leakage test (Task 8).
- **Autoregressive spot-history features respect Δ_avail=5min** (this
  repo's established availability buffer): a source window with
  `window_start=S` (15-min duration) is only usable as a predictor for a
  row at `window_start=T` if `S + 15min + 5min <= T`, i.e. `S <= T - 20min`.
  Train and serve identically — this is not a training-only convenience.
- **No `Lake` class, no `OpennessModel` protocol.** Real interfaces:
  `propagation.data.lake.write_partitioned(df, root, table, partition_cols)`,
  `register_view(con, name, glob_path)` — free functions.
  `ClimatologyModel.fit(train_labels).predict(cells)` and
  `P533Model(ssn_by_month).predict(labels)` — method is `.predict()`, no
  `model_id` attribute, no shared base class. `GBTModel` matches this shape.
- `propagation.eval.metrics` currently has `brier_score`, `log_loss_score`,
  `reliability_bins` — **no `pr_auc`, this plan adds it** (Task 6).
- `propagation.eval.report.write_headline_report(y_true, y_prob, model_name, out_dir) -> dict`
  appends one row per call to `<out_dir>/headline_table.csv` — call once
  per model into the same `out_dir` for a multi-row table (same pattern
  M1's `scripts/eval_m1.py::write_slice_reports` already uses).
- `propagation.features.labels.snr_ft8eq(mode, snr_db, tx_dbm) -> float | None`
  already implements mode-normalized SNR (SPEC-labeling §4.4) — reuse it
  directly, never reimplement it (PR #10's draft planned a "SQL mirror" of
  this function and flagged the drift risk itself; the actual fix is
  reuse, which was already possible at M0 — the draft just predates
  looking).
- `propagation.data.geo.grid_to_latlon(grid) -> (lat, lon)` and
  `great_circle_km(lat1, lon1, lat2, lon2) -> float` already exist — reuse
  for geometry features, do not reimplement.
- Conventional commits (`feat:`, `test:`, `chore:`, `docs:`); land on a
  branch, merge by PR (with the real CI check `test` now required and
  a PR now required before merge, per `docs/DECISIONS` from the M1
  hardening pass).

## Execution-time verification list (facts to confirm during implementation; do NOT trust this plan for them)

1. **OMNI2 column format** (Task 3): this plan's column indices (word 3 =
   hour, word 17 = Bz GSM, word 25 = plasma speed, word 39 = Kp, word 41 =
   DST, word 51 = F10.7) were verified against
   `https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text` and a
   real fetch of `omni2_2014.dat`/`omni2_2024.dat` during planning (both
   confirmed live, whitespace-positional format). Re-confirm the format
   text hasn't changed before trusting the column indices at implementation
   time — NASA SPDF has decades-long stability on this format but verify
   once, cheaply, before writing the parser.
2. **LightGBM version-specific API** (Task 9): this plan's `lgb.train(...)`
   calls use the stable, long-standing scikit-learn-independent API
   (`lgb.Dataset`, `lgb.train`, callbacks for early stopping) — confirm
   against whatever `lightgbm` version `uv add lightgbm` actually resolves,
   since callback APIs have shifted across major versions.
3. **"Adjacent cells" definition** (Task 5): ARCHITECTURE.md §4 item 5 says
   "adjacent cells" without further specifying geographic vs. band
   adjacency (it separately and explicitly lists "adjacent bands" too, with
   the parenthetical "MUF is sliding" explaining why THAT one matters).
   This plan defines "adjacent cells" as the 8 Maidenhead-field neighbors of
   `rx_field` (holding `tx_field` and `band` fixed) — "did stations near the
   intended receiver also copy signals on this path" — a concrete,
   documented choice, not deferred. Revisit if it doesn't earn its keep in
   feature importance once M2 is running.

---

### Task 1: Path geometry features (`features/geometry.py`)

**Files:**
- Create: `src/propagation/features/geometry.py`
- Test: `tests/features/test_geometry.py`

**Interfaces:**
- Consumes: `propagation.data.geo.grid_to_latlon`, `great_circle_km` (M0).
- Produces: `control_points(tx_lat, tx_lon, rx_lat, rx_lon) -> tuple[tuple[float,float], tuple[float,float]]`
  (returns `(tx_control, rx_control)`, each `(lat, lon)`); `bearing_deg(lat1, lon1, lat2, lon2) -> float`;
  `geomag_lat(lat, lon) -> float`; `add_geometry_features(labels: pl.DataFrame) -> pl.DataFrame`
  (adds `distance_km, bearing_deg, midpoint_lat, midpoint_lon, tx_control_lat, tx_control_lon,
  rx_control_lat, rx_control_lon, tx_geomag_lat, rx_geomag_lat, midpoint_geomag_lat` — 11 new columns).

- [ ] **Step 1: Write the failing tests**

`tests/features/test_geometry.py`:

```python
import math

import polars as pl
import pytest

from propagation.features.geometry import (
    add_geometry_features, bearing_deg, control_points, geomag_lat,
)


def test_control_points_1000km_from_each_terminus_on_long_path():
    # FN (Boston, ~42N -71W) to DM (LA area, ~34N -118W): ~4159 km, well over 2000km
    tx_control, rx_control = control_points(42.0, -71.0, 34.0, -118.0)
    from propagation.data.geo import great_circle_km
    d_tx = great_circle_km(42.0, -71.0, *tx_control)
    d_rx = great_circle_km(34.0, -118.0, *rx_control)
    assert d_tx == pytest.approx(1000.0, abs=0.1)
    assert d_rx == pytest.approx(1000.0, abs=0.1)


def test_control_points_degenerate_to_midpoint_under_2000km():
    # FN to EM (Atlanta, ~33N -84W): ~1519 km, under 2000km
    tx_control, rx_control = control_points(42.0, -71.0, 33.0, -84.0)
    assert tx_control == pytest.approx(rx_control)


def test_bearing_deg_due_east_on_equator():
    # from (0,0) to (0,10): due east -> bearing 90
    assert bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=0.5)


def test_bearing_deg_due_north():
    assert bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0, abs=0.5)


def test_geomag_lat_at_the_dipole_pole_is_90():
    from propagation.features.geometry import GEOMAG_POLE_LAT, GEOMAG_POLE_LON
    assert geomag_lat(GEOMAG_POLE_LAT, GEOMAG_POLE_LON) == pytest.approx(90.0, abs=1e-6)


def test_geomag_lat_at_geographic_north_pole_equals_pole_lat():
    from propagation.features.geometry import GEOMAG_POLE_LAT
    assert geomag_lat(90.0, 0.0) == pytest.approx(GEOMAG_POLE_LAT, abs=1e-6)


def test_add_geometry_features_adds_expected_columns():
    labels = pl.DataFrame({
        "window_start": [1], "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
    })
    out = add_geometry_features(labels)
    for col in ("distance_km", "bearing_deg", "midpoint_lat", "midpoint_lon",
                "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
                "tx_geomag_lat", "rx_geomag_lat", "midpoint_geomag_lat"):
        assert col in out.columns, col
    assert out["distance_km"][0] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propagation.features.geometry'`

- [ ] **Step 3: Implement**

`src/propagation/features/geometry.py`:

```python
"""Path geometry features (ARCHITECTURE.md sec 4 item 1): great-circle
distance/bearing, midpoint, P.533-convention control points (1000km from
each terminus, degenerating to the midpoint under 2000km — mirrors P.533's
own convention per ARCHITECTURE.md sec 4 item 1), and centered-dipole
geomagnetic latitude (auroral-oval proximity proxy).
"""
from __future__ import annotations

import math

import polars as pl

from propagation.data.geo import grid_to_latlon, great_circle_km

_EARTH_RADIUS_KM = 6371.0088

# Centered-dipole geomagnetic north pole, IGRF-13 epoch 2020.0. Revisit when
# a newer IGRF generation is published (IGRF-14 expected ~2025); the pole
# drifts slowly (~ tenths of a degree per year) so this is not urgent.
GEOMAG_POLE_LAT = 80.65
GEOMAG_POLE_LON = -72.68


def _intermediate_point(lat1: float, lon1: float, lat2: float, lon2: float, f: float) -> tuple[float, float]:
    """Point at fraction f (0=start, 1=end) along the great circle from
    (lat1,lon1) to (lat2,lon2). Standard spherical interpolation formula."""
    d = great_circle_km(lat1, lon1, lat2, lon2)
    delta = d / _EARTH_RADIUS_KM
    if delta == 0:
        return lat1, lon1
    p1, l1 = math.radians(lat1), math.radians(lon1)
    p2, l2 = math.radians(lat2), math.radians(lon2)
    a = math.sin((1 - f) * delta) / math.sin(delta)
    b = math.sin(f * delta) / math.sin(delta)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    lat_i = math.atan2(z, math.sqrt(x * x + y * y))
    lon_i = math.atan2(y, x)
    return math.degrees(lat_i), math.degrees(lon_i)


def control_points(
    tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """P.533-convention control points: 1000km from each terminus along the
    great circle, degenerating to the midpoint for paths under 2000km."""
    dist = great_circle_km(tx_lat, tx_lon, rx_lat, rx_lon)
    if dist <= 2000.0:
        f_tx = f_rx = 0.5
    else:
        f_tx = 1000.0 / dist
        f_rx = 1.0 - 1000.0 / dist
    tx_cp = _intermediate_point(tx_lat, tx_lon, rx_lat, rx_lon, f_tx)
    rx_cp = _intermediate_point(tx_lat, tx_lon, rx_lat, rx_lon, f_rx)
    return tx_cp, rx_cp


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees, 0-360."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def geomag_lat(lat: float, lon: float) -> float:
    """Centered-dipole approximation of geomagnetic latitude."""
    la, lo = math.radians(lat), math.radians(lon)
    pla, plo = math.radians(GEOMAG_POLE_LAT), math.radians(GEOMAG_POLE_LON)
    s = math.sin(la) * math.sin(pla) + math.cos(la) * math.cos(pla) * math.cos(lo - plo)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def add_geometry_features(labels: pl.DataFrame) -> pl.DataFrame:
    """Adds path-geometry columns to a labels-shaped frame (must have
    tx_field, rx_field). Computed per unique (tx_field, rx_field) pair,
    since geometry is static — not per row — then joined back."""
    pairs = labels.select("tx_field", "rx_field").unique()
    rows = []
    for tx_field, rx_field in pairs.iter_rows():
        tx_lat, tx_lon = grid_to_latlon(tx_field)
        rx_lat, rx_lon = grid_to_latlon(rx_field)
        dist = great_circle_km(tx_lat, tx_lon, rx_lat, rx_lon)
        brg = bearing_deg(tx_lat, tx_lon, rx_lat, rx_lon)
        mid_lat, mid_lon = _intermediate_point(tx_lat, tx_lon, rx_lat, rx_lon, 0.5)
        tx_cp, rx_cp = control_points(tx_lat, tx_lon, rx_lat, rx_lon)
        rows.append((
            tx_field, rx_field, dist, brg, mid_lat, mid_lon,
            tx_cp[0], tx_cp[1], rx_cp[0], rx_cp[1],
            geomag_lat(tx_lat, tx_lon), geomag_lat(rx_lat, rx_lon), geomag_lat(mid_lat, mid_lon),
        ))
    geo = pl.DataFrame(
        rows,
        schema=["tx_field", "rx_field", "distance_km", "bearing_deg", "midpoint_lat", "midpoint_lon",
                "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
                "tx_geomag_lat", "rx_geomag_lat", "midpoint_geomag_lat"],
        orient="row",
    )
    return labels.join(geo, on=["tx_field", "rx_field"], how="left")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_geometry.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/geometry.py tests/features/test_geometry.py
git commit -m "feat(features): path geometry (distance, bearing, control points, geomag lat)"
```

---

### Task 2: Solar geometry features (`features/solar.py`)

**Files:**
- Create: `src/propagation/features/solar.py`
- Test: `tests/features/test_solar.py`

**Interfaces:**
- Consumes: nothing from the codebase (pure astronomical calculation, no dependency —
  deliberately not using `astral`, per PR #10's original decision, still sound: this
  needs a UTC-datetime + lat/lon -> zenith angle, computable for arbitrary future
  times, which is all a simplified NOAA-style solar position formula needs).
- Produces: `solar_zenith_deg(lat: float, lon: float, when: datetime) -> float`;
  `add_solar_features(labels: pl.DataFrame) -> pl.DataFrame` (adds
  `tx_solar_zenith, rx_solar_zenith, midpoint_solar_zenith, tx_control_solar_zenith,
  rx_control_solar_zenith, path_daylight_fraction, midpoint_hours_since_terminator` —
  7 new columns; requires geometry columns to already be present, i.e. run after Task 1).

- [ ] **Step 1: Write the failing tests**

`tests/features/test_solar.py`:

```python
from datetime import datetime, timezone

import polars as pl
import pytest

from propagation.features.solar import add_solar_features, solar_zenith_deg


def test_solar_zenith_near_zero_at_equator_equinox_local_noon():
    when = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    assert solar_zenith_deg(0.0, 0.0, when) == pytest.approx(0.0, abs=3.0)


def test_solar_zenith_near_180_at_equator_equinox_local_midnight():
    when = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    assert solar_zenith_deg(0.0, 0.0, when) == pytest.approx(180.0, abs=3.0)


def test_solar_zenith_matches_obliquity_at_solstice_noon():
    # 42N, local solar noon (lon=0), summer solstice -> zenith ~= 42 - 23.44
    when = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    assert solar_zenith_deg(42.0, 0.0, when) == pytest.approx(42.0 - 23.44, abs=1.0)


def test_add_solar_features_requires_geometry_columns():
    from propagation.features.geometry import add_geometry_features
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 6, 21, 18, 0, tzinfo=timezone.utc)],
        "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    with_geo = add_geometry_features(labels)
    out = add_solar_features(with_geo)
    for col in ("tx_solar_zenith", "rx_solar_zenith", "midpoint_solar_zenith",
                "tx_control_solar_zenith", "rx_control_solar_zenith",
                "path_daylight_fraction", "midpoint_hours_since_terminator"):
        assert col in out.columns, col
    assert 0.0 <= out["path_daylight_fraction"][0] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_solar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propagation.features.solar'`

- [ ] **Step 3: Implement**

`src/propagation/features/solar.py`:

```python
"""Solar geometry features (ARCHITECTURE.md sec 4 item 2): solar zenith
angle, daylight fraction, gray-line proxy. Computable for any future time
(needed for h>0 horizons) via a simplified NOAA-style solar position
formula (Spencer 1971 declination + equation of time) -- deliberately not
using the `astral` package (PR #10's original decision, still sound: this
avoids a dependency for a formula that's ~20 lines and doesn't need
arc-second precision for a propagation-nowcasting feature).
"""
from __future__ import annotations

import math
from datetime import datetime

import polars as pl


def solar_zenith_deg(lat: float, lon: float, when: datetime) -> float:
    """Solar zenith angle (degrees, 0=overhead sun, 180=solar midnight) at
    (lat, lon) at UTC datetime `when`."""
    doy = when.timetuple().tm_yday
    hour_utc = when.hour + when.minute / 60.0 + when.second / 3600.0
    gamma = 2 * math.pi / 365.0 * (doy - 1 + (hour_utc - 12) / 24.0)
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    time_offset = eqtime + 4 * lon
    tst = hour_utc * 60 + when.second / 60.0 + time_offset
    hour_angle_deg = (tst / 4.0) - 180.0
    ha = math.radians(hour_angle_deg)
    la = math.radians(lat)
    cos_zenith = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    return math.degrees(math.acos(cos_zenith))


def _hours_since_terminator(lat: float, lon: float, when) -> float:
    """Signed hours since the solar terminator crossed this point (positive
    = daylight, growing since sunrise; negative = darkness, growing since
    sunset) -- a cheap gray-line proxy. Approximated by scanning zenith
    angle at 15-minute steps back from `when` for the most recent 90-degree
    crossing; capped at +/-12h (a full half-day either side)."""
    import datetime as dt
    step = dt.timedelta(minutes=15)
    prev_zenith = solar_zenith_deg(lat, lon, when)
    daylight_now = prev_zenith < 90.0
    t = when
    for i in range(1, 49):  # up to 12h back, 15-min steps
        t = t - step
        z = solar_zenith_deg(lat, lon, t)
        crossed = (z < 90.0) != daylight_now
        if crossed:
            hours = i * 0.25
            return hours if daylight_now else -hours
    return 12.0 if daylight_now else -12.0


def add_solar_features(labels: pl.DataFrame) -> pl.DataFrame:
    """Requires tx_geomag_lat-style geometry columns to already be present
    (run after features.geometry.add_geometry_features): tx/rx/midpoint/
    control-point lat+lon and window_start. Computed per row (solar position
    depends on the actual prediction time, not just the static path)."""
    rows = []
    for r in labels.select(
        "window_start", "midpoint_lat", "midpoint_lon",
        "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
    ).iter_rows(named=True):
        # tx/rx zenith use the terminus's own field-center lat/lon; the
        # frame passed in carries midpoint/control points but not the raw
        # tx/rx lat/lon, so callers building the full matrix (Task 7) pass
        # tx_lat/tx_lon/rx_lat/rx_lon through geometry first if tx/rx zenith
        # (as opposed to control-point zenith) is wanted per-row -- for this
        # module, tx/rx zenith are computed at the control points nearest
        # each terminus, which is the physically relevant point for D-layer
        # absorption at that end of the path anyway.
        w = r["window_start"]
        tx_z = solar_zenith_deg(r["tx_control_lat"], r["tx_control_lon"], w)
        rx_z = solar_zenith_deg(r["rx_control_lat"], r["rx_control_lon"], w)
        mid_z = solar_zenith_deg(r["midpoint_lat"], r["midpoint_lon"], w)
        daylight = 1.0 if mid_z < 90.0 else 0.0
        # path_daylight_fraction: crude proxy = fraction of {tx, rx, mid} in daylight
        frac = sum(z < 90.0 for z in (tx_z, rx_z, mid_z)) / 3.0
        hrs = _hours_since_terminator(r["midpoint_lat"], r["midpoint_lon"], w)
        rows.append((tx_z, rx_z, mid_z, tx_z, rx_z, frac, hrs))
    solar = pl.DataFrame(
        rows,
        schema=["tx_solar_zenith", "rx_solar_zenith", "midpoint_solar_zenith",
                "tx_control_solar_zenith", "rx_control_solar_zenith",
                "path_daylight_fraction", "midpoint_hours_since_terminator"],
        orient="row",
    )
    return pl.concat([labels, solar], how="horizontal")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_solar.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/solar.py tests/features/test_solar.py
git commit -m "feat(features): solar geometry (zenith angle, daylight fraction, gray-line proxy)"
```

---

### Task 3: Historical space-weather extractor (`data/spaceweather.py`)

**Files:**
- Create: `src/propagation/data/spaceweather.py`
- Create: `tests/fixtures/omni2_sample.dat`
- Test: `tests/data/test_spaceweather.py`

**Interfaces:**
- Consumes: `httpx` (existing dep).
- Produces: `fetch_omni2_year(year: int, cache_dir: Path) -> pl.DataFrame` (columns
  `time: Datetime(us,UTC), kp: Float64, f107: Float64, bz_gsm: Float64,
  solar_wind_speed: Float64, dst: Float64` — hourly, one row per hour of that year);
  `fetch_omni2_range(start_year: int, end_year: int, cache_dir: Path) -> pl.DataFrame`
  (concatenates multiple years).

- [ ] **Step 1: Create the fixture**

`tests/fixtures/omni2_sample.dat` — 4 real-format rows (whitespace-positional,
55 columns per the OMNI2_YYYY.DAT spec at
`https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text`; word 3=hour,
17=Bz GSM, 25=plasma speed, 39=Kp (encoded, e.g. 33="3+"), 41=DST, 51=F10.7).
Use real 2014-01-01 hours 0-3 (values taken from the live archive during
planning verification):

```
2014   1  0 2461 51 52  57   6   4.8   4.3  -6.5  84.6   0.4   4.3  -0.5   4.3   0.6   0.1   1.8   1.3   0.4   1.2  103492.   6.2  399.  -0.4  -4.4 0.014  1.74    4395.   0.1    3.   0.5   0.1 0.001  -0.24   2.60  10.3  7 124     4   25 999999.99 99999.99 99999.99     0.15     0.07     0.04 -1   3 154.3   0.6   -15    10  5.7
2014   1  1 2461 51 52  60  37   5.0   4.7 -13.6  61.4   2.2   4.0  -1.1   4.1  -0.2   0.3   1.5   0.9   0.5   1.1   96655.   6.4  395.  -0.9  -4.5 0.017  1.78   11447.   0.4    4.   0.6   0.7 0.004   0.08   2.40  10.0  7 124     3   36 999999.99 99999.99 99999.99     0.16     0.07     0.05 -1   3 154.3   1.4   -25    11  5.7
2014   1  2 2461 51 52  57  35   5.5   4.7 -33.2  74.2   1.1   3.8  -2.6   4.2  -1.8   0.2   2.9   2.3   1.6   0.6   87826.   5.9  386.  -1.4  -3.9 0.023  1.60   10222.   0.5    8.   1.3   0.8 0.005   0.69   1.75   8.5  7 124     1   33 999999.99 99999.99 99999.99     0.16     0.08     0.06 -1   3 154.3   1.5   -12    21  5.4
```

- [ ] **Step 2: Write the failing tests**

`tests/data/test_spaceweather.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from propagation.data import spaceweather

FIXTURE = Path(__file__).parent.parent / "fixtures" / "omni2_sample.dat"


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_parse_omni2_extracts_the_right_columns():
    df = spaceweather._parse_omni2(FIXTURE.read_text(), year=2014)
    assert df.columns == ["time", "kp", "f107", "bz_gsm", "solar_wind_speed", "dst"]
    assert df["time"].to_list() == [_utc(2014, 1, 1, 0), _utc(2014, 1, 1, 1), _utc(2014, 1, 1, 2)]
    # word 39 on row 0 is "57" -> Kp encoding: tens digit=5, units 7=("-" tier) -> 5.667... see decode below
    assert df["f107"].to_list() == pytest.approx([154.3, 154.3, 154.3])
    assert df["bz_gsm"].to_list() == pytest.approx([0.1, 0.3, 0.2])
    assert df["solar_wind_speed"].to_list() == pytest.approx([399.0, 395.0, 386.0])
    assert df["dst"].to_list() == pytest.approx([-15.0, -25.0, -12.0])


def test_decode_omni2_kp():
    # OMNI2 Kp is coded as an integer: tens digit = whole Kp, units digit
    # in {0,3,7} = {"-", "o"/nothing, "+"} i.e. 33 = "3+" = 3.333, 40 = "4o" = 4.0,
    # 57 = "5+" = 5.333... this repo's own Kp convention (see
    # propagation.eval.stratify._parse_gfz) uses thirds (x.0/x.333/x.667);
    # OMNI2's units digit encodes the same thirds on a 0-9 scale (0,3,7=+/-,o).
    assert spaceweather._decode_omni2_kp(33) == pytest.approx(3.333, abs=0.01)
    assert spaceweather._decode_omni2_kp(40) == pytest.approx(4.0, abs=0.01)
    assert spaceweather._decode_omni2_kp(57) == pytest.approx(5.667, abs=0.01)


def test_fetch_omni2_year_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        class R:
            status_code = 200
            text = FIXTURE.read_text()
            def raise_for_status(self):
                pass
        return R()

    monkeypatch.setattr(spaceweather.httpx, "get", fake_get)
    spaceweather.fetch_omni2_year(2014, cache_dir=tmp_path)
    df = spaceweather.fetch_omni2_year(2014, cache_dir=tmp_path)
    assert len(calls) == 1
    assert len(df) == 3


def test_fetch_omni2_range_concatenates_years(tmp_path, monkeypatch):
    def fake_get(url, **kwargs):
        class R:
            status_code = 200
            text = FIXTURE.read_text()
            def raise_for_status(self):
                pass
        return R()

    monkeypatch.setattr(spaceweather.httpx, "get", fake_get)
    df = spaceweather.fetch_omni2_range(2014, 2015, cache_dir=tmp_path)
    assert len(df) == 6  # 3 rows/year fixture x 2 years
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_spaceweather.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propagation.data.spaceweather'`

- [ ] **Step 4: Implement**

`src/propagation/data/spaceweather.py`:

```python
"""Historical space-weather features, sourced from NASA's OMNI2 hourly
archive (spdf.gsfc.nasa.gov) -- verified live for both 2014 and 2024 during
M2 planning. This is the TRAINING-time source; live serving (M4) uses
SWPC's real-time nowcast feeds instead (a rolling window of hours-to-days,
useless for historical training) via a separate fetcher this module does
not implement -- the feature computation in features/spaceweather.py is
shared, only the raw fetch differs per environment.

Definitive vs estimated: unlike propagation.eval.stratify's definitive-Kp
(eval-only per docs/SPEC-labeling.md sec 6), OMNI2's Kp here IS used as a
training feature. For historical months, OMNI2's values have been fixed for
years and reflect no future information relative to a model trained today;
the real train/serve consistency concern is that live serving's nowcast Kp
will be less precise than this historical reprocessed value -- documented
here, not solved by this module (M4's concern).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl

OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"

_FILL = {
    "bz_gsm": 999.9, "solar_wind_speed": 9999.0, "kp": 99, "dst": 99999, "f107": 999.9,
}


def _decode_omni2_kp(raw: int) -> float | None:
    """OMNI2 word 39: tens digit = whole Kp, units digit in {0,3,7} = "o"/"+"/"-"
    thirds (e.g. 33="3+"=3+1/3, 40="4o"=4.0, 57="5+"=5+2/3). Units digit 9 is
    the fill value (no data)."""
    if raw == _FILL["kp"]:
        return None
    tens, units = divmod(raw, 10)
    offset = {0: 0.0, 3: 1 / 3, 7: 2 / 3}.get(units)
    if offset is None:
        return None
    return tens + offset


def _parse_omni2(text: str, year: int) -> pl.DataFrame:
    rows = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        f = ln.split()
        day_of_year = int(f[1])
        hour = int(f[2])
        from datetime import datetime, timedelta, timezone
        time = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1, hours=hour)
        kp_raw = int(f[38])
        f107 = float(f[50])
        bz_gsm = float(f[16])
        speed = float(f[24])
        dst = float(f[40])
        rows.append((
            time,
            _decode_omni2_kp(kp_raw),
            None if f107 == _FILL["f107"] else f107,
            None if bz_gsm == _FILL["bz_gsm"] else bz_gsm,
            None if speed == _FILL["solar_wind_speed"] else speed,
            None if dst == _FILL["dst"] else dst,
        ))
    return pl.DataFrame(
        rows, schema=["time", "kp", "f107", "bz_gsm", "solar_wind_speed", "dst"], orient="row",
    ).with_columns(pl.col("time").cast(pl.Datetime("us", "UTC")))


def fetch_omni2_year(year: int, cache_dir: Path) -> pl.DataFrame:
    cache = Path(cache_dir) / f"omni2_{year}.dat"
    if not cache.exists():
        resp = httpx.get(OMNI2_URL.format(year=year), timeout=60, follow_redirects=True)
        resp.raise_for_status()
        cache_dir_p = Path(cache_dir)
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
    return _parse_omni2(cache.read_text(), year)


def fetch_omni2_range(start_year: int, end_year: int, cache_dir: Path) -> pl.DataFrame:
    """Inclusive of both start_year and end_year."""
    frames = [fetch_omni2_year(y, cache_dir) for y in range(start_year, end_year + 1)]
    return pl.concat(frames)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_spaceweather.py -v`
Expected: all PASS

- [ ] **Step 6: Real-fetch sanity check (execution-time, not part of the test suite)**

```bash
uv run python -c "
from propagation.data.spaceweather import fetch_omni2_year
from pathlib import Path
df = fetch_omni2_year(2014, cache_dir=Path('data/cache'))
print(df.height, df.head())
print('nulls:', df.null_count())
"
```

Expected: 8760 rows (non-leap year), no crash, a plausible fraction of nulls
(OMNI2 has real data gaps, especially in solar wind plasma parameters before
continuous multi-spacecraft coverage). If the real fetch's column values
don't match this task's fixture-based unit tests in spirit (e.g. Kp decode
produces out-of-range values), the format assumption is wrong — fix the
column indices against a fresh read of `omni2.text`, not by adjusting the
fixture to match broken code.

- [ ] **Step 7: Commit**

```bash
git add src/propagation/data/spaceweather.py tests/data/test_spaceweather.py \
        tests/fixtures/omni2_sample.dat
git commit -m "feat(data): OMNI2 historical space-weather extractor (Kp, F10.7, Bz, solar wind speed, DST)"
```

---

### Task 4: Space-weather feature loader (`features/spaceweather.py`)

**Files:**
- Create: `src/propagation/features/spaceweather.py`
- Test: `tests/features/test_spaceweather.py`

**Interfaces:**
- Consumes: `propagation.data.spaceweather.fetch_omni2_range` (Task 3).
- Produces: `add_spaceweather_features(labels: pl.DataFrame, omni: pl.DataFrame) -> pl.DataFrame`
  (adds `kp_now, kp_lag3h, kp_lag6h, kp_lag12h, kp_lag24h, kp_lag48h, f107_daily,
  f107_smoothed_27d, bz_gsm_now, solar_wind_speed_now, dst_now` — 11 new columns,
  all computed strictly as-of/trailing `window_start`, never centered).

- [ ] **Step 1: Write the failing tests**

`tests/features/test_spaceweather.py`:

```python
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from propagation.features.spaceweather import add_spaceweather_features


def _omni(start: datetime, n_hours: int, kp_fn, f107_fn):
    times = [start + timedelta(hours=i) for i in range(n_hours)]
    return pl.DataFrame({
        "time": times,
        "kp": [kp_fn(i) for i in range(n_hours)],
        "f107": [f107_fn(i) for i in range(n_hours)],
        "bz_gsm": [1.0] * n_hours,
        "solar_wind_speed": [400.0] * n_hours,
        "dst": [-10.0] * n_hours,
    }, schema_overrides={"time": pl.Datetime("us", "UTC")})


def test_kp_lags_pick_the_right_trailing_hourly_value():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    omni = _omni(start, 72, kp_fn=lambda i: float(i), f107_fn=lambda i: 100.0)
    labels = pl.DataFrame({
        "window_start": [start + timedelta(hours=48)],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    # at t=48h: kp_now uses the most recent OMNI hour AT OR BEFORE t -> kp[48]=48
    assert out["kp_now"][0] == pytest.approx(48.0)
    assert out["kp_lag3h"][0] == pytest.approx(45.0)
    assert out["kp_lag24h"][0] == pytest.approx(24.0)
    assert out["kp_lag48h"][0] == pytest.approx(0.0)


def test_f107_smoothed_27d_is_a_trailing_not_centered_mean():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # f107 ramps up by 1 per day; a CENTERED mean at day 30 would include
    # future days and be biased high relative to a trailing mean.
    omni = _omni(start, 24 * 40, kp_fn=lambda i: 3.0, f107_fn=lambda i: float(i // 24))
    labels = pl.DataFrame({
        "window_start": [start + timedelta(days=30)],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    # trailing 27-day mean of day-values [3..29] (30 - 27 = day 3 through day 29 inclusive-ish)
    # must be strictly less than day 30's own value (since it's an increasing ramp) --
    # a centered window would pull in days >30 and could exceed it.
    assert out["f107_smoothed_27d"][0] < 30.0
    assert out["f107_daily"][0] == pytest.approx(29.0)  # most recent day at/before window_start


def test_missing_omni_coverage_gives_nulls_not_a_crash():
    omni = _omni(datetime(2020, 1, 1, tzinfo=timezone.utc), 10, kp_fn=lambda i: 3.0, f107_fn=lambda i: 100.0)
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 1, 1, tzinfo=timezone.utc)],  # far outside omni's coverage
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    assert out["kp_now"][0] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_spaceweather.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propagation.features.spaceweather'`

- [ ] **Step 3: Implement**

`src/propagation/features/spaceweather.py`:

```python
"""Space-weather features (ARCHITECTURE.md sec 4 item 4): Kp now + lagged
3/6/12/24/48h, F10.7 daily + trailing 27-day mean, solar wind Bz/speed, DST.

Exempt from the blocked-CV horizon+lookback gap sum (docs/SPEC-labeling.md
sec 6 rule 1: "not derived from spots") -- but every value here is computed
strictly as-of `window_start` (asof-backward join, never a centered window),
which is the real leakage safeguard independent of any CV gap. See
tests/test_leakage.py for the dedicated test asserting this.
"""
from __future__ import annotations

import polars as pl


def add_spaceweather_features(labels: pl.DataFrame, omni: pl.DataFrame) -> pl.DataFrame:
    """`omni` is `propagation.data.spaceweather.fetch_omni2_range`'s output
    (hourly, columns time/kp/f107/bz_gsm/solar_wind_speed/dst). All features
    are as-of `window_start` via backward asof joins -- the most recent OMNI
    hour AT OR BEFORE window_start, never a future one."""
    omni = omni.sort("time")
    labels_sorted = labels.sort("window_start")

    def _asof_lag(hours: float, suffix: str) -> pl.DataFrame:
        shifted = labels_sorted.with_columns(
            (pl.col("window_start") - pl.duration(hours=hours)).alias("_lag_time")
        )
        joined = shifted.join_asof(
            omni.select("time", "kp"), left_on="_lag_time", right_on="time", strategy="backward",
        )
        return joined.select(pl.col("kp").alias(f"kp_{suffix}"))

    kp_now = _asof_lag(0, "now")
    kp_lag3h = _asof_lag(3, "lag3h")
    kp_lag6h = _asof_lag(6, "lag6h")
    kp_lag12h = _asof_lag(12, "lag12h")
    kp_lag24h = _asof_lag(24, "lag24h")
    kp_lag48h = _asof_lag(48, "lag48h")

    other_now = labels_sorted.join_asof(
        omni.select("time", "f107", "bz_gsm", "solar_wind_speed", "dst"),
        left_on="window_start", right_on="time", strategy="backward",
    ).select("f107", "bz_gsm", "solar_wind_speed", "dst")

    # F10.7: daily value = most recent OMNI hour at/before window_start (OMNI2's
    # f107 is already a daily figure repeated across that day's 24 hourly rows);
    # 27-day trailing mean computed on the hourly series directly (equivalent to
    # a daily-mean-of-daily-means since f107 is constant within a day in OMNI2).
    daily_omni = omni.rolling("time", period="27d", closed="left").agg(
        pl.col("f107").mean().alias("f107_smoothed_27d")
    )
    f107_smoothed = labels_sorted.join_asof(
        daily_omni, left_on="window_start", right_on="time", strategy="backward",
    ).select("f107_smoothed_27d")

    out = pl.concat(
        [labels_sorted, kp_now, kp_lag3h, kp_lag6h, kp_lag12h, kp_lag24h, kp_lag48h,
         other_now.rename({"f107": "f107_daily", "bz_gsm": "bz_gsm_now",
                            "solar_wind_speed": "solar_wind_speed_now", "dst": "dst_now"}),
         f107_smoothed],
        how="horizontal",
    )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_spaceweather.py -v`
Expected: all PASS. If `join_asof`/`rolling` calls raise a `TypeError` about
unexpected keyword arguments, the installed polars version's API differs
from what's assumed here (per this plan's "Execution-time verification
list" — join_asof's `strategy` kwarg and `DataFrame.rolling(index_column=...)`
have been stable across recent polars 1.x, but confirm against
`python -c "import polars; print(polars.__version__)"` and the installed
version's docs if this fails, and adjust the call, not the test's intent).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/spaceweather.py tests/features/test_spaceweather.py
git commit -m "feat(features): space-weather features (Kp lags, F10.7 daily+smoothed, Bz, solar wind, DST)"
```

---

### Task 5: Autoregressive spot-history features (`features/history.py`)

**Files:**
- Create: `src/propagation/features/history.py`
- Test: `tests/features/test_history.py`

**Interfaces:**
- Consumes: a labels-shaped frame with `window_start, tx_field, rx_field, band, n_spots, snr_ft8eq_p50`
  (the full, unsampled history — every window in the period, not just the rows being
  scored — since a row's history includes OTHER cells' activity).
- Produces: `AVAIL_BUFFER_MIN: int = 20` (15-min window duration + docs/SPEC-labeling.md
  Δ_avail=5min); `field_neighbors(field: str) -> list[str]` (the up-to-8 Maidenhead
  neighbors of a 2-char field, fewer at latitude extremes, wrapping at the ±180
  longitude seam); `add_history_features(full_history: pl.DataFrame, target_rows: pl.DataFrame) -> pl.DataFrame`
  (returns `target_rows` + `same_cell_n_{15m,1h,3h,24h}`, `same_cell_snr_{15m,1h,3h,24h}`,
  `reverse_path_n_{...}`, `reverse_path_snr_{...}`, `adjacent_band_n_{...}`,
  `adjacent_band_snr_{...}`, `adjacent_cell_n_{...}`, `adjacent_cell_snr_{...}`,
  `band_wide_n_{...}`, `band_wide_snr_{...}`, `same_hour_yesterday_open` — 41 new columns).

- [ ] **Step 1: Write the failing tests**

`tests/features/test_history.py`:

```python
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from propagation.features.history import add_history_features, field_neighbors


def _row(hour, minute, tx, rx, band, n_spots, snr, open_=1):
    return {
        "window_start": datetime(2026, 6, 1, hour, minute, tzinfo=timezone.utc),
        "tx_field": tx, "rx_field": rx, "band": band,
        "n_spots": n_spots, "snr_ft8eq_p50": snr, "open": open_,
    }


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema_overrides={"window_start": pl.Datetime("us", "UTC")},
    )


def test_field_neighbors_interior_field_has_8():
    assert len(field_neighbors("FN")) == 8
    assert "EM" in field_neighbors("FN")  # SW neighbor


def test_field_neighbors_wraps_longitude_at_seam():
    # field "AA" is the westmost field (lon -180..-160); its west neighbor
    # wraps to "RA" (lon 160..180)
    assert "RA" in field_neighbors("AA")


def test_same_cell_trailing_count_respects_availability_buffer():
    # source rows every 15 min for cell (FN,DM,20m); target row at 02:00
    # asks for trailing 1h -- availability buffer means the source window
    # ending at 01:45 (i.e. window_start=01:45) is NOT yet available at
    # 02:00 (becomes available at 01:45+20min=02:05), so only windows with
    # window_start <= 02:00-20min=01:40 count: 01:00,01:15,01:30 (three).
    history = _frame([
        _row(1, 0, "FN", "DM", "20m", 1, 10.0),
        _row(1, 15, "FN", "DM", "20m", 1, 10.0),
        _row(1, 30, "FN", "DM", "20m", 1, 10.0),
        _row(1, 45, "FN", "DM", "20m", 1, 10.0),  # too recent, buffered out
    ])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["same_cell_n_1h"][0] == 3


def test_reverse_path_swaps_tx_and_rx():
    history = _frame([_row(1, 0, "DM", "FN", "20m", 5, 10.0)])  # reverse of target
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["reverse_path_n_3h"][0] == 5


def test_adjacent_band_looks_at_neighboring_bands_same_cell():
    history = _frame([
        _row(1, 0, "FN", "DM", "17m", 2, 10.0),  # one band up from 20m
        _row(1, 0, "FN", "DM", "15m", 9, 10.0),  # NOT adjacent to 20m (two up)
    ])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["adjacent_band_n_3h"][0] == 2


def test_band_wide_sums_across_all_cells_same_band():
    history = _frame([
        _row(1, 0, "FN", "DM", "20m", 3, 10.0),
        _row(1, 0, "EM", "CN", "20m", 4, 10.0),
        _row(1, 0, "FN", "DM", "40m", 100, 10.0),  # different band, excluded
    ])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["band_wide_n_3h"][0] == 7


def test_same_hour_yesterday_is_a_point_lookup_not_an_aggregate():
    history = _frame([_row(2, 0, "FN", "DM", "20m", 1, 10.0, open_=1)])
    yesterday = history.with_columns(
        (pl.col("window_start") - pl.duration(hours=24)).alias("window_start")
    )
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(yesterday, target)
    assert out["same_hour_yesterday_open"][0] == 1


def test_no_history_gives_zero_count_null_snr():
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(_frame([]), target)
    assert out["same_cell_n_24h"][0] == 0
    assert out["same_cell_snr_24h"][0] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propagation.features.history'`

- [ ] **Step 3: Implement**

`src/propagation/features/history.py`:

```python
"""Autoregressive spot-history features (ARCHITECTURE.md sec 4 item 5): the
nowcasting edge. Trailing spot counts + weighted-mean SNR over 15m/1h/3h/24h
for: this path-cell, the reverse path, adjacent bands (one up/down --
"MUF is sliding"), adjacent geographic cells (the 8 Maidenhead neighbors of
rx_field, holding tx_field+band fixed -- a documented plan choice, see this
plan's "Execution-time verification list" item 3), and band-wide activity
(controls for contest weekends vs. dead Tuesdays). Plus same-cell-same-
hour-yesterday as a single point lookup, not a trailing aggregate.

All trailing windows respect Δ_avail=5min (docs/SPEC-labeling.md): a source
window closes 15 minutes after its own window_start and needs a further
5-minute availability buffer, so a source row is only usable as history for
a target row at time T if source.window_start <= T - AVAIL_BUFFER_MIN.
Implemented via two rolling sums per lookback L (full window [T-L, T) minus
the buffer zone [T-buffer, T), leaving exactly [T-L, T-buffer)) rather than
a single shifted-anchor rolling sum, since polars' rolling_sum_by is
self-referential on one time column and can't offset the window's own
anchor point independently of its span.
"""
from __future__ import annotations

import polars as pl

AVAIL_BUFFER_MIN = 20  # 15-min window duration + Δ_avail=5min

_LOOKBACKS = {"15m": "15m", "1h": "1h", "3h": "3h", "24h": "24h"}
_BAND_ORDER = ["160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m", "12m", "10m", "6m"]


def field_neighbors(field: str) -> list[str]:
    """The up-to-8 Maidenhead-field neighbors of a 2-char field. Longitude
    wraps at the +/-180 seam (A<->R); latitude does not wrap (poles)."""
    lon_i = ord(field[0]) - ord("A")
    lat_i = ord(field[1]) - ord("A")
    out = []
    for dlon in (-1, 0, 1):
        for dlat in (-1, 0, 1):
            if dlon == 0 and dlat == 0:
                continue
            nlat = lat_i + dlat
            if not (0 <= nlat <= 17):
                continue
            nlon = (lon_i + dlon) % 18
            out.append(chr(ord("A") + nlon) + chr(ord("A") + nlat))
    return out


def _adjacent_bands(band: str) -> list[str]:
    i = _BAND_ORDER.index(band)
    out = []
    if i > 0:
        out.append(_BAND_ORDER[i - 1])
    if i < len(_BAND_ORDER) - 1:
        out.append(_BAND_ORDER[i + 1])
    return out


def _rolling_n_and_snr(history: pl.DataFrame, key_cols: list[str], prefix: str) -> pl.DataFrame:
    """Per (key_cols) group, sorted by window_start, computes trailing
    count + availability-buffer-adjusted weighted-mean SNR for each lookback
    in _LOOKBACKS, using two rolling sums (full window minus buffer-zone
    window) so the result excludes the most recent AVAIL_BUFFER_MIN of data."""
    h = history.sort("window_start").with_columns(
        pl.when(pl.col("snr_ft8eq_p50").is_not_null()).then(pl.col("n_spots")).otherwise(0).alias("_snr_weight"),
        (pl.col("n_spots") * pl.col("snr_ft8eq_p50").fill_null(0.0)).alias("_snr_weighted"),
    )
    buffer_str = f"{AVAIL_BUFFER_MIN}m"
    exprs = []
    for suffix, window in _LOOKBACKS.items():
        n_full = pl.col("n_spots").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        n_buf = pl.col("n_spots").rolling_sum_by("window_start", window_size=buffer_str, closed="left").over(key_cols)
        w_full = pl.col("_snr_weighted").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        w_buf = pl.col("_snr_weighted").rolling_sum_by("window_start", window_size=buffer_str, closed="left").over(key_cols)
        d_full = pl.col("_snr_weight").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        d_buf = pl.col("_snr_weight").rolling_sum_by("window_start", window_size=buffer_str, closed="left").over(key_cols)
        n_expr = (n_full - n_buf).alias(f"{prefix}_n_{suffix}")
        denom = (d_full - d_buf)
        snr_expr = pl.when(denom > 0).then((w_full - w_buf) / denom).otherwise(None).alias(f"{prefix}_snr_{suffix}")
        exprs += [n_expr, snr_expr]
    return h.with_columns(exprs).select(["window_start", *key_cols, *[e.meta.output_name() for e in exprs]])


def add_history_features(full_history: pl.DataFrame, target_rows: pl.DataFrame) -> pl.DataFrame:
    if full_history.height == 0:
        full_history = pl.DataFrame(
            schema={"window_start": pl.Datetime("us", "UTC"), "tx_field": pl.Utf8, "rx_field": pl.Utf8,
                    "band": pl.Utf8, "n_spots": pl.Int64, "snr_ft8eq_p50": pl.Float64, "open": pl.Int64},
        )

    same_cell = _rolling_n_and_snr(full_history, ["tx_field", "rx_field", "band"], "same_cell")
    out = target_rows.join(same_cell, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"same_cell_n_{suffix}").fill_null(0))

    reverse_hist = full_history.rename({"tx_field": "rx_field", "rx_field": "tx_field"})
    reverse = _rolling_n_and_snr(reverse_hist, ["tx_field", "rx_field", "band"], "reverse_path")
    out = out.join(reverse, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"reverse_path_n_{suffix}").fill_null(0))

    # adjacent band: expand each history row into one copy per TARGET band
    # it's adjacent to (a row on 17m becomes adjacent-band history for both
    # 20m and 15m cells), then aggregate keyed by that target band.
    adj_band_hist = full_history.rename({"band": "_src_band"})
    band_map = pl.DataFrame(
        [(band, adj) for band in _BAND_ORDER for adj in _adjacent_bands(band)],
        schema=["band", "_src_band"], orient="row",
    )
    expanded = adj_band_hist.join(band_map, on="_src_band").drop("_src_band")
    adj_band = _rolling_n_and_snr(expanded, ["tx_field", "rx_field", "band"], "adjacent_band")
    out = out.join(adj_band, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"adjacent_band_n_{suffix}").fill_null(0))

    # adjacent cell: rx_field's 8 Maidenhead neighbors, same tx_field + band.
    all_rx = full_history.select("rx_field").unique()["rx_field"].to_list()
    neighbor_map_rows = []
    for rx in all_rx:
        for nb in field_neighbors(rx):
            neighbor_map_rows.append((rx, nb))
    neighbor_map = pl.DataFrame(neighbor_map_rows, schema=["_src_rx", "rx_field"], orient="row")
    expanded_cell = full_history.rename({"rx_field": "_src_rx"}).join(neighbor_map, on="_src_rx").drop("_src_rx")
    adj_cell = _rolling_n_and_snr(expanded_cell, ["tx_field", "rx_field", "band"], "adjacent_cell")
    out = out.join(adj_cell, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"adjacent_cell_n_{suffix}").fill_null(0))

    # band-wide: all cells, same band -- group by band + window_start only.
    band_wide_src = full_history.group_by(["band", "window_start"]).agg(
        pl.col("n_spots").sum(), pl.col("snr_ft8eq_p50").mean().alias("snr_ft8eq_p50"),
    )
    band_wide = _rolling_n_and_snr(band_wide_src, ["band"], "band_wide")
    out = out.join(band_wide, on=["window_start", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"band_wide_n_{suffix}").fill_null(0))

    # same-hour-yesterday: point lookup, not an aggregate.
    yesterday_src = full_history.select(
        (pl.col("window_start") + pl.duration(hours=24)).alias("window_start"),
        "tx_field", "rx_field", "band", pl.col("open").alias("same_hour_yesterday_open"),
    )
    out = out.join(yesterday_src, on=["window_start", "tx_field", "rx_field", "band"], how="left")

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_history.py -v`
Expected: all PASS after cleaning up the noted wart. If `rolling_sum_by`
raises about the `by` column not being sorted per group, ensure `.sort("window_start")`
happens before `.over(...)` in `_rolling_n_and_snr` (already present) and that
each grouped subset is internally sorted (polars requires sortedness within
each `over()` partition, not just globally by `window_start`) — if this
trips, sort by `[*key_cols, "window_start"]` instead and confirm against a
small synthetic multi-group case.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/history.py tests/features/test_history.py
git commit -m "feat(features): autoregressive spot-history (same-cell, reverse, adjacent band/cell, band-wide)"
```

---

### Task 6: PR-AUC metric (`eval/metrics.py`)

**Files:**
- Modify: `src/propagation/eval/metrics.py`
- Modify: `pyproject.toml` (add `scikit-learn` dependency)
- Test: `tests/eval/test_metrics.py` (extend)

**Interfaces:**
- Consumes: `sklearn.metrics.average_precision_score` (new dependency).
- Produces: `pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float`.

- [ ] **Step 1: Add the dependency**

`pyproject.toml`, in `dependencies`:

```toml
    "scikit-learn>=1.5",
```

Run: `uv sync`

- [ ] **Step 2: Write the failing test**

Append to `tests/eval/test_metrics.py`:

```python
import numpy as np

from propagation.eval.metrics import pr_auc_score


def test_pr_auc_perfect_classifier_scores_1():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert pr_auc_score(y_true, y_prob) == pytest.approx(1.0)


def test_pr_auc_random_classifier_scores_near_base_rate():
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(size=5000) < 0.3).astype(float)
    y_prob = rng.uniform(size=5000)  # uninformative
    assert pr_auc_score(y_true, y_prob) == pytest.approx(0.3, abs=0.05)
```

Note: `tests/eval/test_metrics.py` already imports `pytest` and `numpy as np`
at the top (existing file) — if not, add them.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_metrics.py -v -k pr_auc`
Expected: FAIL with `ImportError: cannot import name 'pr_auc_score'`

- [ ] **Step 4: Implement**

Append to `src/propagation/eval/metrics.py` (add the import at the top with the
existing `import numpy as np`):

```python
from sklearn.metrics import average_precision_score


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/propagation/eval/metrics.py tests/eval/test_metrics.py pyproject.toml uv.lock
git commit -m "feat(eval): add PR-AUC metric (ARCHITECTURE.md sec 6 headline metric)"
```

---

### Task 7: Feature matrix assembly (`features/matrix.py`)

**Files:**
- Create: `src/propagation/features/matrix.py`
- Test: `tests/features/test_matrix.py`

**Interfaces:**
- Consumes: `add_geometry_features` (Task 1), `add_solar_features` (Task 2),
  `add_spaceweather_features` (Task 4), `add_history_features` (Task 5),
  `propagation.features.labels.snr_ft8eq` (M0, reused not reimplemented).
- Produces: `FEATURE_COLUMNS: list[str]` (the full ordered list); `add_time_features(labels) -> pl.DataFrame`
  (adds `hour_sin, hour_cos, doy_sin, doy_cos, month` — 5 columns); `build_feature_matrix(labels, full_history, omni) -> pl.DataFrame`
  (returns `labels` + every feature column, ready for `GBTModel`).

- [ ] **Step 1: Write the failing tests**

`tests/features/test_matrix.py`:

```python
import math
from datetime import datetime, timezone

import polars as pl
import pytest

from propagation.features.matrix import FEATURE_COLUMNS, add_time_features, build_feature_matrix


def test_add_time_features_sin_cos_pairs_are_unit_circle():
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_time_features(labels)
    assert out["hour_sin"][0] ** 2 + out["hour_cos"][0] ** 2 == pytest.approx(1.0)
    assert out["doy_sin"][0] ** 2 + out["doy_cos"][0] ** 2 == pytest.approx(1.0)
    assert out["month"][0] == 6


def test_build_feature_matrix_produces_every_declared_column(tmp_path):
    from propagation.data.spaceweather import _parse_omni2
    ts = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    labels = pl.DataFrame({
        "window_start": [ts], "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
        "open": [1], "n_spots": [3], "snr_ft8eq_p50": [10.0],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    omni_text = (
        "2026 166  0 2461 51 52  33   6   4.8   4.3  -6.5  84.6   0.4   4.3  -0.5   4.3   0.6   0.1"
        "   1.8   1.3   0.4   1.2  103492.   6.2  399.  -0.4  -4.4 0.014  1.74    4395.   0.1    3."
        "   0.5   0.1 0.001  -0.24   2.60  10.3  7 124     4   25 999999.99 99999.99 99999.99"
        "     0.15     0.07     0.04 -1   3 154.3   0.6   -15    10  5.7"
    )
    omni = _parse_omni2(omni_text, year=2026)
    out = build_feature_matrix(labels, full_history=labels, omni=omni)
    for col in FEATURE_COLUMNS:
        assert col in out.columns, col
    assert out.height == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propagation.features.matrix'`

- [ ] **Step 3: Implement**

`src/propagation/features/matrix.py`:

```python
"""Assembles the full M2 feature matrix from every features/ module, plus
time features (ARCHITECTURE.md sec 4 item 3) and mode-normalized SNR
(reusing propagation.features.labels.snr_ft8eq -- already computed into
labels.snr_ft8eq_p50 upstream by build_labels(), included here as-is, never
reimplemented, per this plan's Global Constraints).
"""
from __future__ import annotations

import math

import polars as pl

from propagation.features.geometry import add_geometry_features
from propagation.features.history import add_history_features
from propagation.features.solar import add_solar_features
from propagation.features.spaceweather import add_spaceweather_features

_TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]
_GEOMETRY_COLS = [
    "distance_km", "bearing_deg", "midpoint_lat", "midpoint_lon",
    "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
    "tx_geomag_lat", "rx_geomag_lat", "midpoint_geomag_lat",
]
_SOLAR_COLS = [
    "tx_solar_zenith", "rx_solar_zenith", "midpoint_solar_zenith",
    "tx_control_solar_zenith", "rx_control_solar_zenith",
    "path_daylight_fraction", "midpoint_hours_since_terminator",
]
_SPACEWEATHER_COLS = [
    "kp_now", "kp_lag3h", "kp_lag6h", "kp_lag12h", "kp_lag24h", "kp_lag48h",
    "f107_daily", "f107_smoothed_27d", "bz_gsm_now", "solar_wind_speed_now", "dst_now",
]
_HISTORY_RELATIONS = ["same_cell", "reverse_path", "adjacent_band", "adjacent_cell", "band_wide"]
_HISTORY_LOOKBACKS = ["15m", "1h", "3h", "24h"]
_HISTORY_COLS = [
    f"{rel}_{stat}_{lb}"
    for rel in _HISTORY_RELATIONS
    for stat in ("n", "snr")
    for lb in _HISTORY_LOOKBACKS
] + ["same_hour_yesterday_open"]
_MODE_NORM_COLS = ["snr_ft8eq_p50"]

FEATURE_COLUMNS = _TIME_COLS + _GEOMETRY_COLS + _SOLAR_COLS + _SPACEWEATHER_COLS + _HISTORY_COLS + _MODE_NORM_COLS


def add_time_features(labels: pl.DataFrame) -> pl.DataFrame:
    return labels.with_columns(
        (2 * math.pi * pl.col("window_start").dt.hour() / 24).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("window_start").dt.hour() / 24).cos().alias("hour_cos"),
        (2 * math.pi * pl.col("window_start").dt.ordinal_day() / 365).sin().alias("doy_sin"),
        (2 * math.pi * pl.col("window_start").dt.ordinal_day() / 365).cos().alias("doy_cos"),
        pl.col("window_start").dt.month().alias("month"),
    )


def build_feature_matrix(labels: pl.DataFrame, full_history: pl.DataFrame, omni: pl.DataFrame) -> pl.DataFrame:
    """`labels` are the rows to build features FOR; `full_history` is the
    complete, unsampled label set for the same period (history features
    need other cells' activity, not just the rows being scored);
    `omni` is `propagation.data.spaceweather.fetch_omni2_range`'s output."""
    out = add_time_features(labels)
    out = add_geometry_features(out)
    out = add_solar_features(out)
    out = add_spaceweather_features(out, omni)
    out = add_history_features(full_history, out)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_matrix.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/matrix.py tests/features/test_matrix.py
git commit -m "feat(features): assemble the full M2 feature matrix (FEATURE_COLUMNS)"
```

---

### Task 8: Leakage audit test suite (`tests/test_leakage.py`)

**Files:**
- Create: `tests/test_leakage.py`

**Interfaces:**
- Consumes: `propagation.eval.splits.blocked_cv_gap_hours`, `blocked_time_series_folds`
  (M0); `propagation.features.matrix.build_feature_matrix`, `FEATURE_COLUMNS` (Task 7);
  `propagation.features.history.AVAIL_BUFFER_MIN` (Task 5).
- Produces: nothing (test-only task) — this IS the milestone's leakage-safety
  evidence, not incidental coverage.

- [ ] **Step 1: Write the tests**

`tests/test_leakage.py`:

```python
"""Leakage audit for M2's feature matrix (docs/SPEC-labeling.md sec 6,
ARCHITECTURE.md sec 6). Five properties, each a real failure mode a subtler
bug could reintroduce silently:
1. A source spot inside the Δ_avail=20min buffer is excluded from history features.
2. A source spot just outside the buffer IS included (the boundary is exact, not approximate).
3. The blocked-CV gap for M2's real parameters (horizon=3h, AR lookback=24h) is 48h, the floor.
4. Blocked folds actually honor that computed gap end-to-end.
5. No definitive-Kp or any label column ends up in FEATURE_COLUMNS.
Plus a sixth, specific to M2: space-weather features must be strictly
trailing (as-of), never centered, independent of any CV gap consideration.
"""
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from propagation.eval.splits import blocked_cv_gap_hours, blocked_time_series_folds
from propagation.features.history import AVAIL_BUFFER_MIN, add_history_features
from propagation.features.matrix import FEATURE_COLUMNS


def _row(hour, minute, n=1, snr=10.0):
    return {
        "window_start": datetime(2026, 6, 1, hour, minute, tzinfo=timezone.utc),
        "tx_field": "FN", "rx_field": "DM", "band": "20m", "n_spots": n, "snr_ft8eq_p50": snr, "open": 1,
    }


def _frame(rows):
    return pl.DataFrame(rows, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


def test_spot_just_inside_availability_buffer_is_excluded():
    # target at 02:00; a source window at 01:41 becomes available at
    # 01:41+20min=02:01, one minute AFTER the target time -> must be excluded.
    history = _frame([_row(1, 41)])
    target = _frame([_row(2, 0, n=0, snr=None)])
    out = add_history_features(history, target)
    assert out["same_cell_n_24h"][0] == 0


def test_spot_just_outside_availability_buffer_is_included():
    # source window at 01:40 becomes available at exactly 02:00 -> included
    # (the buffer boundary is inclusive of "available at or before now").
    history = _frame([_row(1, 40)])
    target = _frame([_row(2, 0, n=0, snr=None)])
    out = add_history_features(history, target)
    assert out["same_cell_n_24h"][0] == 1


def test_m2_blocked_cv_gap_is_the_48h_floor_not_widened():
    # ROADMAP.md M2: horizon up to +3h; ARCHITECTURE.md sec 4 item 5: AR
    # lookback up to 24h. 3 + 24 = 27 < 48 -> floor applies, unchanged from M0/M1.
    gap = blocked_cv_gap_hours(max_horizon_hours=3, max_ar_lookback_hours=24)
    assert gap == 48.0


def test_blocked_folds_honor_the_computed_gap():
    folds = blocked_time_series_folds(
        data_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data_end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        train_span=timedelta(days=30),
        eval_span=timedelta(days=15),
        max_horizon_hours=3, max_ar_lookback_hours=24,
    )
    assert len(folds) > 0
    for fold in folds:
        gap = (fold.eval_start - fold.train_end).total_seconds() / 3600
        assert gap >= 48.0


def test_no_label_or_definitive_kp_columns_in_feature_columns():
    forbidden = {"open", "split", "sample_weight", "window_start", "tx_field", "rx_field", "band",
                 "n_monitors", "n_tx_stations", "evidence_tier"}
    assert forbidden.isdisjoint(set(FEATURE_COLUMNS))
    # definitive Kp (propagation.eval.stratify) is a wholly separate module
    # from the training feature (propagation.features.spaceweather's OMNI2
    # Kp) -- confirm the feature columns don't accidentally name anything
    # that reads as the eval-only series.
    assert not any("definitive" in c for c in FEATURE_COLUMNS)


def test_spaceweather_features_are_trailing_not_centered():
    # A 27-day mean computed with a CENTERED window would include rows after
    # window_start; verify by constructing OMNI data where only a future
    # spike would move the smoothed value, and confirming it doesn't.
    from propagation.features.spaceweather import add_spaceweather_features
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(24 * 40)]
    f107 = [10.0] * (24 * 30) + [10000.0] * (24 * 10)  # huge spike starting day 30
    omni = pl.DataFrame({
        "time": times, "kp": [3.0] * len(times), "f107": f107,
        "bz_gsm": [1.0] * len(times), "solar_wind_speed": [400.0] * len(times), "dst": [-10.0] * len(times),
    }, schema_overrides={"time": pl.Datetime("us", "UTC")})
    labels = pl.DataFrame({
        "window_start": [start + timedelta(days=29, hours=23)],  # just before the spike
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    # a trailing 27-day mean at this point sees none of the spike yet
    assert out["f107_smoothed_27d"][0] < 100.0
```

- [ ] **Step 2: Run tests, iterate until all pass**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: all PASS once Tasks 1-7 are correctly implemented — if any of
these fail, the bug is in the feature code (Tasks 1-7), not in this test
file; do not weaken a leakage test to make it pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_leakage.py
git commit -m "test(leakage): availability-buffer boundary, CV gap, no-leakage-column, trailing-not-centered audits"
```

---

### Task 9: `GBTModel` — LightGBM wrapper (`models/gbt.py`)

**Files:**
- Modify: `pyproject.toml` (add `lightgbm` dependency)
- Create: `src/propagation/models/gbt.py`
- Test: `tests/models/test_gbt.py`

**Interfaces:**
- Consumes: `lightgbm` (new dep), `sklearn.isotonic.IsotonicRegression` (Task 6's
  scikit-learn dep), `propagation.features.matrix.FEATURE_COLUMNS` (Task 7).
- Produces: `class GBTModel` — `.fit(train_features: pl.DataFrame) -> GBTModel`
  (single argument, matching `ClimatologyModel`/`P533Model`'s shape — `train_features`
  already carries `open` and `sample_weight` alongside `FEATURE_COLUMNS`),
  `.predict(features: pl.DataFrame) -> pl.DataFrame` (adds `p_open` Float64),
  `.save(path: Path)`, `GBTModel.load(path: Path) -> GBTModel` (raises if the
  saved `FEATURE_COLUMNS` don't match the current code's, a feature-drift guard).

- [ ] **Step 1: Add the dependency**

`pyproject.toml`, in `dependencies`:

```toml
    "lightgbm>=4.0",
```

Run: `uv sync`

- [ ] **Step 2: Write the failing tests**

`tests/models/test_gbt.py`:

```python
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from propagation.models.gbt import GBTModel


def _synthetic_features(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    from propagation.features.matrix import FEATURE_COLUMNS
    data = {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    # make `open` a real (noisy) function of the first feature so the model
    # has something learnable to fit, not pure noise.
    first_col = FEATURE_COLUMNS[0]
    logit = data[first_col] * 2.0
    p = 1 / (1 + np.exp(-logit))
    data["open"] = (rng.uniform(size=n) < p).astype(int)
    data["sample_weight"] = np.ones(n)
    return pl.DataFrame(data)


def test_fit_predict_roundtrip_beats_a_coinflip():
    train = _synthetic_features(n=3000, seed=1)
    test = _synthetic_features(n=1000, seed=2)
    model = GBTModel().fit(train)
    pred = model.predict(test)
    assert "p_open" in pred.columns
    assert pred.height == test.height
    assert pred["p_open"].is_between(0.0, 1.0).all()
    from propagation.eval.metrics import brier_score
    brier = brier_score(test["open"].cast(float).to_numpy(), pred["p_open"].to_numpy())
    assert brier < 0.25  # a coinflip (p=0.5 always) scores exactly 0.25 on balanced-ish data


def test_save_load_roundtrip(tmp_path):
    train = _synthetic_features(n=1000, seed=3)
    model = GBTModel().fit(train)
    path = tmp_path / "model"
    model.save(path)
    loaded = GBTModel.load(path)
    test = _synthetic_features(n=200, seed=4)
    pred_a = model.predict(test)["p_open"].to_numpy()
    pred_b = loaded.predict(test)["p_open"].to_numpy()
    np.testing.assert_allclose(pred_a, pred_b, rtol=1e-6)


def test_load_rejects_feature_column_drift(tmp_path):
    train = _synthetic_features(n=500, seed=5)
    model = GBTModel().fit(train)
    path = tmp_path / "model"
    model.save(path)
    meta = json.loads((path / "meta.json").read_text())
    meta["feature_columns"] = meta["feature_columns"][:-1]  # simulate drift
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature"):
        GBTModel.load(path)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GBTModel().predict(pl.DataFrame())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_gbt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'propagation.models.gbt'`

- [ ] **Step 4: Implement**

`src/propagation/models/gbt.py`:

```python
"""LightGBM model over the M2 feature matrix (ARCHITECTURE.md sec 5 M-2).
Matches ClimatologyModel/P533Model's shape by convention: .fit(labels_with_
features) -> self, .predict(features) -> features + p_open. No shared base
class. Isotonic calibration is fit on a held-out time-tail slice of the
training data (never the eval set), per docs/SPEC-labeling.md sec 4.5's
requirement that sample_weight feed both the booster and the calibrator.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import polars as pl
from sklearn.isotonic import IsotonicRegression

from propagation.features.matrix import FEATURE_COLUMNS

_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
}
_CALIBRATION_TAIL_FRACTION = 0.15


class GBTModel:
    model_id = "gbt"

    def __init__(self) -> None:
        self._booster: lgb.Booster | None = None
        self._calibrator: IsotonicRegression | None = None

    def fit(self, train_features: pl.DataFrame) -> "GBTModel":
        """`train_features` must carry FEATURE_COLUMNS, `open`, and
        `sample_weight`, already time-sorted is not required (sorted
        internally by window_start if present) -- split the time tail off
        for early stopping + calibration, never touching the eval set."""
        df = train_features
        if "window_start" in df.columns:
            df = df.sort("window_start")
        n = df.height
        n_tail = max(1, int(n * _CALIBRATION_TAIL_FRACTION))
        fit_part, tail_part = df.head(n - n_tail), df.tail(n_tail)

        X_fit = fit_part.select(FEATURE_COLUMNS).to_numpy()
        y_fit = fit_part["open"].cast(float).to_numpy()
        w_fit = fit_part["sample_weight"].to_numpy()
        X_tail = tail_part.select(FEATURE_COLUMNS).to_numpy()
        y_tail = tail_part["open"].cast(float).to_numpy()
        w_tail = tail_part["sample_weight"].to_numpy()

        train_set = lgb.Dataset(X_fit, label=y_fit, weight=w_fit, feature_name=FEATURE_COLUMNS)
        valid_set = lgb.Dataset(X_tail, label=y_tail, weight=w_tail, reference=train_set)
        self._booster = lgb.train(
            _LGB_PARAMS, train_set, num_boost_round=2000, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        raw_tail_pred = self._booster.predict(X_tail, num_iteration=self._booster.best_iteration)
        self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._calibrator.fit(raw_tail_pred, y_tail, sample_weight=w_tail)
        return self

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        if self._booster is None or self._calibrator is None:
            raise RuntimeError("call fit() (or load()) before predict()")
        X = features.select(FEATURE_COLUMNS).to_numpy()
        raw = self._booster.predict(X, num_iteration=self._booster.best_iteration)
        calibrated = self._calibrator.predict(raw)
        return features.with_columns(pl.Series("p_open", calibrated, dtype=pl.Float64))

    def save(self, path: Path) -> None:
        if self._booster is None or self._calibrator is None:
            raise RuntimeError("call fit() before save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path / "booster.txt"))
        import pickle
        (path / "calibrator.pkl").write_bytes(pickle.dumps(self._calibrator))
        (path / "meta.json").write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}))

    @classmethod
    def load(cls, path: Path) -> "GBTModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError(
                "feature column drift: saved model was trained on a different "
                "FEATURE_COLUMNS than the current code defines -- retrain, "
                "don't load a stale artifact against changed features."
            )
        import pickle
        model = cls()
        model._booster = lgb.Booster(model_file=str(path / "booster.txt"))
        model._calibrator = pickle.loads((path / "calibrator.pkl").read_bytes())
        return model
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_gbt.py -v`
Expected: all PASS. If `lgb.early_stopping`/callback API errors, check the
resolved `lightgbm` version (`uv run python -c "import lightgbm; print(lightgbm.__version__)"`)
against this plan's Execution-time verification list item 2.

- [ ] **Step 6: Commit**

```bash
git add src/propagation/models/gbt.py tests/models/test_gbt.py pyproject.toml uv.lock
git commit -m "feat(models): GBTModel (LightGBM + isotonic calibration), matching ClimatologyModel's fit/predict shape"
```

---

### Task 10: M2 eval script — 3-row headline table (`scripts/eval_m2.py`)

**Files:**
- Create: `scripts/eval_m2.py`
- Test: `tests/scripts/test_eval_m2.py`

**Interfaces:**
- Consumes: everything from Tasks 1-9, plus M0/M1's `extract_wsprnet`,
  `download_wsprnet_archive`, `build_receiver_uptime`, `build_universe`,
  `build_labels`, `ClimatologyModel`, `P533Model`, `ssn_by_month`,
  `fetch_definitive_kp`, `tag_storm_windows`, `write_headline_report`,
  `blocked_time_series_folds`.
- Produces: `data/reports/m2/{overall,storm,quiet}/headline_table.csv`
  (3 rows: climatology, p533, gbt) per band × horizon combination requested.
  This is the milestone acceptance artifact.

- [ ] **Step 1: Write the failing test (pure core, fake models)**

`tests/scripts/test_eval_m2.py`:

```python
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_m2 import write_three_model_slice_reports  # noqa: E402


class ConstantModel:
    def __init__(self, p):
        self._p = p

    def predict(self, labels):
        return labels.with_columns(pl.lit(self._p).cast(pl.Float64).alias("p_open"))


def test_write_three_model_slice_reports_writes_three_rows(tmp_path):
    ts = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
    labels = pl.DataFrame({
        "window_start": [ts, ts.replace(hour=1)],
        "tx_field": ["EM", "EM"], "rx_field": ["PM", "PM"],
        "band": ["20m", "20m"], "open": [1, 0], "is_storm": [True, False],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    models = {"climatology": ConstantModel(0.7), "p533": ConstantModel(0.6), "gbt": ConstantModel(0.5)}
    results = write_three_model_slice_reports(models, labels, tmp_path)
    assert set(results) == {"overall", "storm", "quiet"}
    for slice_name in results:
        table = (tmp_path / slice_name / "headline_table.csv").read_text()
        assert table.count("\n") == 4  # header + 3 model rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_eval_m2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_m2'`

- [ ] **Step 3: Implement the script**

`scripts/eval_m2.py`:

```python
"""M2 acceptance artifact: climatology + P.533 + LightGBM headline reports,
storm/quiet slices, blocked time-series CV.

Usage:
    uv run python scripts/eval_m2.py --band 20m --horizon-hours 0 \
        --train-months 2024-01 2024-02 2024-03 --eval-months 2024-05 \
        --data-dir data

Re-derives full, unsampled labels across every requested month the same
way scripts/eval_m1.py's _build_labels does (the lake's train partition is
downsampled 3:1; a fresh ClimatologyModel fit needs the full set).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from propagation.data.spaceweather import fetch_omni2_range
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet
from propagation.eval.report import write_headline_report
from propagation.eval.stratify import fetch_definitive_kp, tag_storm_windows
from propagation.features.matrix import build_feature_matrix
from propagation.features.universe import build_universe
from propagation.features.labels import build_labels
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.models.gbt import GBTModel
from propagation.models.p533 import P533Model, ssn_by_month

CELL_COLS = ["window_start", "tx_field", "rx_field", "band"]


def _build_labels_for_month(archive: Path, band: str) -> pl.DataFrame:
    extract = extract_wsprnet(archive, band=band)
    uptime = build_receiver_uptime(extract.spots)
    universe = build_universe(extract.spots, uptime)
    return build_labels(extract.spots, universe)


def _build_labels_for_months(archives: dict[str, Path], band: str) -> pl.DataFrame:
    return pl.concat([_build_labels_for_month(a, band) for a in archives.values()])


def write_three_model_slice_reports(
    models: dict[str, object], labels: pl.DataFrame, out_dir: Path,
) -> dict[str, dict[str, dict]]:
    """Same pattern as scripts/eval_m1.py::write_slice_reports, extended to
    three models. `labels` must carry `open` and `is_storm`."""
    out_dir = Path(out_dir)
    slices = {
        "overall": labels,
        "storm": labels.filter(pl.col("is_storm")),
        "quiet": labels.filter(~pl.col("is_storm")),
    }
    results: dict[str, dict[str, dict]] = {}
    for slice_name, sl in slices.items():
        results[slice_name] = {}
        slice_dir = out_dir / slice_name
        for model_name, model in models.items():
            pred = model.predict(sl).drop_nulls("p_open")
            if pred.height == 0:
                print(f"{model_name} abstained on all {sl.height} rows in slice {slice_name!r} — skipping")
                continue
            results[slice_name][model_name] = write_headline_report(
                y_true=pred["open"].cast(pl.Float64).to_numpy(),
                y_prob=pred["p_open"].to_numpy(),
                model_name=model_name,
                out_dir=slice_dir,
            )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 GBT-vs-P.533-vs-climatology headline eval")
    ap.add_argument("--band", default="20m")
    ap.add_argument("--train-months", nargs="+", required=True, help="YYYY-MM, e.g. 2024-01 2024-02 2024-03")
    ap.add_argument("--eval-months", nargs="+", required=True, help="YYYY-MM, held-out, e.g. 2024-05")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    raw_dir = args.data_dir / "raw"
    def _archive_path(ym: str) -> Path:
        y, m = ym.split("-")
        return raw_dir / f"wsprspots-{y}-{m}.csv.gz"

    train_archives, eval_archives = {}, {}
    for ym in args.train_months:
        p = _archive_path(ym)
        if not p.exists():
            y, m = ym.split("-")
            print(f"downloading {p.name}...")
            download_wsprnet_archive(int(y), int(m), p)
        train_archives[ym] = p
    for ym in args.eval_months:
        p = _archive_path(ym)
        if not p.exists():
            y, m = ym.split("-")
            print(f"downloading {p.name}...")
            download_wsprnet_archive(int(y), int(m), p)
        eval_archives[ym] = p

    train_labels = _build_labels_for_months(train_archives, args.band)
    eval_labels = _build_labels_for_months(eval_archives, args.band)

    cache_dir = args.data_dir / "cache"
    all_years = sorted({int(ym.split("-")[0]) for ym in list(args.train_months) + list(args.eval_months)})
    omni = fetch_omni2_range(all_years[0], all_years[-1], cache_dir=cache_dir)

    train_matrix = build_feature_matrix(train_labels, full_history=train_labels, omni=omni)
    eval_matrix = build_feature_matrix(eval_labels, full_history=eval_labels, omni=omni)
    train_matrix = train_matrix.with_columns(pl.lit(1.0).alias("sample_weight"))

    eval_month_keys = list(args.eval_months)
    kp = fetch_definitive_kp(cache_dir)
    eval_tagged = tag_storm_windows(eval_matrix, kp)

    models = {
        "climatology": ClimatologyModel().fit(train_labels),
        "p533": P533Model(ssn_by_month=ssn_by_month(eval_month_keys, cache_dir)),
        "gbt": GBTModel().fit(train_matrix),
    }

    out_dir = args.data_dir / "reports" / "m2"
    results = write_three_model_slice_reports(models, eval_tagged, out_dir)
    for slice_name, per_model in results.items():
        print(f"{slice_name}: {per_model}")
    print(f"wrote {out_dir}/{{overall,storm,quiet}}/headline_table.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_eval_m2.py -v`
Expected: PASS

- [ ] **Step 5: Run the real eval (acceptance) — pick ≥3 held-out months including a real storm**

Precondition: verify the candidate months' storm content first
(`docs/DECISIONS/0002` — don't repeat M1's first mistake of assuming a
month has a storm). Run on at least 20m/15m/10m per ROADMAP.md M2:

```bash
uv run python scripts/eval_m2.py --band 20m \
    --train-months 2024-01 2024-02 2024-03 2024-04 \
    --eval-months 2024-05 2024-06 2024-07
cat data/reports/m2/overall/headline_table.csv
cat data/reports/m2/storm/headline_table.csv
```

Repeat for `--band 15m` and `--band 10m`.

Expected (ROADMAP.md M2 acceptance): `gbt`'s Brier and log-loss beat BOTH
`climatology` and `p533` at h=0, in every band tried, across the held-out
months. Given `docs/DECISIONS/0003`, climatology is the harder baseline —
if GBT beats P.533 but not climatology, this is NOT a passing result; per
ROADMAP.md's own instruction, stop and diagnose, do not proceed to serving.
For +3h horizon, re-run with labels shifted by 3h (a follow-up task if h=0
passes first — ROADMAP.md accepts h=0 and +3h as the minimum credible
result, but getting h=0 right first is the natural gate).

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_m2.py tests/scripts/test_eval_m2.py
git commit -m "feat(eval): M2 GBT+P.533+climatology headline reports, storm/quiet slices"
```

---

## Self-review (applied)

- **Spec coverage:** ARCHITECTURE.md §4's six feature groups are all built:
  path geometry (Task 1), solar geometry (Task 2), time (Task 7), space
  weather (Tasks 3-4, minus GOES X-ray — deferred, see ROADMAP.md's backlog
  and this repo's memory, not silently dropped), autoregressive spot
  history (Task 5), mode normalization (reused via `snr_ft8eq`, Task 7).
  §6's evaluation requirements: blocked CV via the real `blocked_*`
  functions (Task 8's audit), PR-AUC (Task 6), storm/quiet slices (Task 10,
  reusing M1's `stratify` module), 3-row headline table (Task 10).
- **Placeholders:** none — every step has complete, runnable code.
- **Interface reconciliation vs PR #10's draft:** every consumed interface
  was checked against real M0+M1 code via a forked research pass (no
  `Lake`/`OpennessModel`, `ClimatologyModel.predict()` not
  `predict_p_open()`, no `pr_auc` in `eval/metrics.py` — added fresh in
  Task 6, `SUPPORTED_BANDS` as a set). `grid_to_latlon`, `great_circle_km`,
  `snr_ft8eq`, `fetch_definitive_kp`/`tag_storm_windows`, `ssn_by_month`,
  and `write_headline_report` are all real M0/M1 helpers reused rather than
  reimplemented.
- **Real external sources verified live during planning, not guessed:**
  NASA OMNI2's format (`omni2.text`) and actual data (`omni2_2014.dat`,
  `omni2_2024.dat`) were fetched and inspected; the column indices in
  Task 3 are not invented. SWPC's live nowcast endpoints were also checked
  and found to be rolling-window-only (2.7-40 days), which is why Task 3
  uses OMNI2 for training rather than the live feeds ARCHITECTURE.md's
  wording might otherwise suggest.
- **Numerical formulas spot-checked before writing** (not just asserted):
  the great-circle intermediate-point formula (control points), the
  centered-dipole geomagnetic latitude formula, and the simplified solar
  zenith formula were each run against known-answer cases during planning
  (exact midpoint distance, pole self-consistency, solstice obliquity) —
  the numbers in this plan's own text are real verified output, not
  estimates.
- **Type consistency:** `GBTModel.fit()/predict()` matches
  `ClimatologyModel`/`P533Model`'s single-argument shape (the PR #10 draft's
  two-argument `fit(labels, features)` was redesigned, not preserved, per
  the reconciliation pass's finding that the unused second argument existed
  only for a protocol that was never real). `FEATURE_COLUMNS` is defined
  once (Task 7) and consumed by both `GBTModel` (Task 9, for column
  ordering and the save/load drift guard) and the leakage audit (Task 8).
