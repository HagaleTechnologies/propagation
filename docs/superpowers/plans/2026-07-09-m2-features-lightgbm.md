# M2 — Feature Matrix + LightGBM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full ARCHITECTURE §4 feature matrix (geometry, solar, space weather, leakage-audited autoregressive history) and train per-horizon LightGBM openness models that beat both climatology and P.533 on Brier/log-loss at h=0 on 20m and 10m across ≥3 held-out months.

**Architecture:** Pure-function feature modules (geometry/solar computable for any future time; space-weather and spot-history strictly as-of prediction time with a 5-min availability buffer) feed `features/matrix.py`, the single assembly point with a canonical `FEATURE_COLUMNS` list. `models/gbt.py` wraps LightGBM + weighted isotonic calibration behind the `OpennessModel` protocol. A `train-gbt` CLI runs blocked time-series CV and extends the M0/M1 headline table to three rows; a `check-m2-gate` script enforces the ROADMAP acceptance criterion programmatically.

**Tech Stack:** Python 3.11+/uv, polars, DuckDB, numpy, httpx (SWPC), LightGBM, scikit-learn (isotonic), matplotlib (diagrams), pytest.

## Global Constraints

(Verbatim from `docs/superpowers/plans/2026-07-09-INTERFACES.md`; every task implicitly includes these.)

- Python **3.11+**, `uv`-managed. `ruff` + `pytest`. License: MIT OR Apache-2.0.
- Layout: `src/propagation/…` (src layout), tests mirror under `tests/`.
- Lake root defaults to `./data/lake` (gitignored); every CLI takes `--lake-root` to override. Reports/artifacts to `./reports` (gitignored).
- All timestamps UTC. Polars dtype `pl.Datetime("us", "UTC")`; Parquet `timestamp[us, tz=UTC]`.
- 15-min windows aligned to UTC boundaries (minute ∈ {0,15,30,45}); a spot belongs to the window containing its `ts` (floor). Cells are directional.
- Supported bands (closed set, order canonical): `["160m","80m","60m","40m","30m","20m","17m","15m","12m","10m","6m"]`
- **No cqdx imports, ever.**
- Blocked time-series CV, gap ≥ 48 h; eval always on the FULL label set; no random splits. Δ_avail = 5 min availability buffer on all autoregressive features (train and serve identically).
- Sampling determinism: negatives 3:1 per (band, UTC date) stratum, `numpy.random.Generator(numpy.random.PCG64(seed))` with `seed = int.from_bytes(hashlib.sha256(f"{band}|{date_iso}".encode()).digest()[:8], "big") & 0xFFFFFFFF`.
- Conventional commits; M2 executes on its own branch, lands by PR.

**Deviation from INTERFACES.md:** `astral` is NOT added. `solar_zenith_deg` implements the NOAA solar-position algorithm directly (deterministic, dependency-free, testable against published reference values). INTERFACES.md lists astral as an M2 dep "only if used" — it is not used.

**Prerequisites:** M0 complete (lake, labels, eval harness, climatology). M1 complete (P533Model) for the 3-row headline; Tasks 1–8 need only M0.

---

### Task 1: Path geometry primitives (`features/geometry.py`)

**Files:**
- Create: `src/propagation/features/geometry.py`
- Test: `tests/features/test_geometry.py`

**Interfaces:**
- Consumes: nothing (pure functions; `BANDS` from `propagation.schema` in later tasks).
- Produces: `grid_to_latlon(grid: str) -> tuple[float, float]`, `haversine_km`, `initial_bearing_deg`, `midpoint_latlon`, `control_points(lat1, lon1, lat2, lon2) -> list[tuple[float, float]]` (P.533 convention: 1000 km from each terminus; paths < 2000 km get `[midpoint, midpoint]`), `geomag_lat(lat, lon) -> float` (centered dipole, pole pinned at 80.7°N 287.4°E, IGRF-13 2025 epoch — revisit on IGRF epoch update).

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_geometry.py
import math
import pytest
from propagation.features.geometry import (
    grid_to_latlon, haversine_km, initial_bearing_deg,
    midpoint_latlon, control_points, geomag_lat,
)

def test_grid_to_latlon_field_center():
    # EM field: lon [-100,-80), lat [30,40) -> center (35, -90)
    assert grid_to_latlon("EM") == (35.0, -90.0)

def test_grid_to_latlon_grid4_center():
    # FN31: F=lon[-80,-60), N=lat[40,50); 3->lon[-74,-72), 1->lat[41,42)
    lat, lon = grid_to_latlon("FN31")
    assert (lat, lon) == (41.5, -73.0)

def test_grid_to_latlon_lowercase_and_6char():
    assert grid_to_latlon("fn31pr") == grid_to_latlon("FN31")

def test_haversine_london_nyc():
    # LHR (51.4700,-0.4543) to JFK (40.6413,-73.7781) ~ 5540 km
    d = haversine_km(51.4700, -0.4543, 40.6413, -73.7781)
    assert d == pytest.approx(5540, abs=15)

def test_bearing_equator_due_east():
    assert initial_bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=0.01)

def test_midpoint_symmetric_equator():
    lat, lon = midpoint_latlon(0.0, -10.0, 0.0, 10.0)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(0.0, abs=1e-9)

def test_control_points_long_path():
    cps = control_points(0.0, 0.0, 0.0, 90.0)  # ~10008 km along equator
    assert len(cps) == 2
    # 1000 km along equator ~ 8.993 degrees
    assert cps[0][1] == pytest.approx(8.993, abs=0.05)
    assert cps[1][1] == pytest.approx(90.0 - 8.993, abs=0.05)

def test_control_points_short_path_collapse_to_midpoint():
    cps = control_points(40.0, -75.0, 42.0, -71.0)  # ~400 km
    mid = midpoint_latlon(40.0, -75.0, 42.0, -71.0)
    assert cps == [mid, mid]

def test_geomag_lat_at_dipole_pole_is_90():
    assert geomag_lat(80.7, 287.4 - 360.0) == pytest.approx(90.0, abs=0.01)

def test_geomag_lat_midlatitude_plausible():
    # Boulder CO (40.0, -105.3): dipole maglat ~ 48-49
    assert 45.0 < geomag_lat(40.0, -105.3) < 52.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'propagation.features.geometry'`

- [ ] **Step 3: Implement**

```python
# src/propagation/features/geometry.py
"""Great-circle path geometry. Pure functions, spherical earth R=6371.0088 km."""
import math

R_KM = 6371.0088
# Centered-dipole north geomagnetic pole, IGRF-13 epoch 2025.
# Revisit when IGRF-14 coefficients land.
DIPOLE_LAT = 80.7
DIPOLE_LON = 287.4 - 360.0  # = -72.6

def grid_to_latlon(grid: str) -> tuple[float, float]:
    """Center lat/lon of a Maidenhead field (2 chars) or grid square (>=4 chars)."""
    g = grid.strip().upper()
    lon = (ord(g[0]) - ord("A")) * 20.0 - 180.0
    lat = (ord(g[1]) - ord("A")) * 10.0 - 90.0
    if len(g) >= 4:
        lon += int(g[2]) * 2.0
        lat += int(g[3]) * 1.0
        return (lat + 0.5, lon + 1.0)
    return (lat + 5.0, lon + 10.0)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_KM * math.asin(math.sqrt(a))

def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0

def midpoint_latlon(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    return _intermediate(lat1, lon1, lat2, lon2, 0.5)

def _intermediate(lat1, lon1, lat2, lon2, f: float) -> tuple[float, float]:
    """Point at fraction f along the great circle (spherical interpolation)."""
    p1, l1 = math.radians(lat1), math.radians(lon1)
    p2, l2 = math.radians(lat2), math.radians(lon2)
    d = haversine_km(lat1, lon1, lat2, lon2) / R_KM  # angular distance
    if d == 0.0:
        return (lat1, lon1)
    a = math.sin((1 - f) * d) / math.sin(d)
    b = math.sin(f * d) / math.sin(d)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    return (math.degrees(math.atan2(z, math.hypot(x, y))),
            math.degrees(math.atan2(y, x)))

def control_points(lat1: float, lon1: float, lat2: float, lon2: float) -> list[tuple[float, float]]:
    """P.533 control points: 1000 km from each terminus along the great circle.
    Paths shorter than 2000 km collapse both to the midpoint."""
    d = haversine_km(lat1, lon1, lat2, lon2)
    if d < 2000.0:
        mid = midpoint_latlon(lat1, lon1, lat2, lon2)
        return [mid, mid]
    f = 1000.0 / d
    return [_intermediate(lat1, lon1, lat2, lon2, f),
            _intermediate(lat1, lon1, lat2, lon2, 1.0 - f)]

def geomag_lat(lat: float, lon: float) -> float:
    """Centered-dipole geomagnetic latitude."""
    p = math.radians(lat)
    pp = math.radians(DIPOLE_LAT)
    dl = math.radians(lon - DIPOLE_LON)
    s = math.sin(p) * math.sin(pp) + math.cos(p) * math.cos(pp) * math.cos(dl)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_geometry.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/geometry.py tests/features/test_geometry.py
git commit -m "feat(features): great-circle geometry primitives (grids, control points, dipole maglat)"
```

---

### Task 2: Solar geometry (`features/solar.py`)

**Files:**
- Create: `src/propagation/features/solar.py`
- Test: `tests/features/test_solar.py`

**Interfaces:**
- Consumes: `control_points` output shape from Task 1 (`list[tuple[float, float]]`).
- Produces: `solar_zenith_deg(lat: float, lon: float, ts: datetime) -> float`, `daylight_fraction(points: list[tuple[float, float]], ts: datetime) -> float`, `minutes_since_terminator(lat: float, lon: float, ts: datetime) -> float` (signed: +N = current day/night state began N minutes ago; capped at ±720).

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_solar.py
from datetime import datetime, timezone
import pytest
from propagation.features.solar import (
    solar_zenith_deg, daylight_fraction, minutes_since_terminator,
)

UTC = timezone.utc

def test_sza_nrel_spa_reference():
    # NREL SPA canonical test case: 2003-10-17 19:30:30 UTC,
    # lat 39.742476, lon -105.1786 -> zenith 50.11162 deg.
    # NOAA low-accuracy algorithm agrees within ~0.3 deg.
    ts = datetime(2003, 10, 17, 19, 30, 30, tzinfo=UTC)
    assert solar_zenith_deg(39.742476, -105.1786, ts) == pytest.approx(50.11162, abs=0.3)

def test_sza_equator_equinox_noon_near_zero():
    # 2024-03-20 (equinox) at (0,0); solar noon ~12:07 UTC (eq. of time)
    ts = datetime(2024, 3, 20, 12, 7, 0, tzinfo=UTC)
    assert solar_zenith_deg(0.0, 0.0, ts) < 1.5

def test_sza_polar_night_sun_below_horizon():
    # Svalbard midwinter noon: sun never rises
    ts = datetime(2020, 12, 21, 12, 0, 0, tzinfo=UTC)
    assert solar_zenith_deg(78.0, 15.0, ts) > 90.0

def test_daylight_fraction_mixed_path():
    # One point in polar night, one at equatorial noon-ish
    ts = datetime(2020, 12, 21, 12, 0, 0, tzinfo=UTC)
    frac = daylight_fraction([(78.0, 15.0), (0.0, 15.0)], ts)
    assert frac == pytest.approx(0.5)

def test_daylight_fraction_empty_is_nan():
    import math
    ts = datetime(2020, 12, 21, 12, 0, 0, tzinfo=UTC)
    assert math.isnan(daylight_fraction([], ts))

def test_minutes_since_terminator_sign_and_cap():
    # Equator, 2024-03-20: sunrise ~06:07 UTC at lon 0. At 08:07 UTC we are
    # ~120 min into daylight -> positive, roughly 120 (4-min sampling grid).
    ts = datetime(2024, 3, 20, 8, 7, 0, tzinfo=UTC)
    m = minutes_since_terminator(0.0, 0.0, ts)
    assert 100 <= m <= 140
    # Polar night: no crossing within 12 h -> capped
    ts2 = datetime(2020, 12, 21, 12, 0, 0, tzinfo=UTC)
    assert minutes_since_terminator(78.0, 15.0, ts2) == 720.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_solar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'propagation.features.solar'`

- [ ] **Step 3: Implement**

```python
# src/propagation/features/solar.py
"""Solar position via the NOAA (Meeus low-accuracy) algorithm.
Accuracy ~0.1-0.3 deg — ample for D-layer/gray-line features. No deps."""
import math
from datetime import datetime, timedelta, timezone

def _frac_year_rad(ts: datetime) -> float:
    doy = ts.timetuple().tm_yday
    hour = ts.hour + ts.minute / 60 + ts.second / 3600
    return 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)

def solar_zenith_deg(lat: float, lon: float, ts: datetime) -> float:
    if ts.tzinfo is None:
        raise ValueError("ts must be timezone-aware UTC")
    ts = ts.astimezone(timezone.utc)
    g = _frac_year_rad(ts)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    time_offset = eqtime + 4 * lon  # minutes
    tst = ts.hour * 60 + ts.minute + ts.second / 60 + time_offset  # true solar time
    ha = math.radians(tst / 4 - 180)  # hour angle
    p = math.radians(lat)
    cos_z = math.sin(p) * math.sin(decl) + math.cos(p) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_z))))

def daylight_fraction(points: list[tuple[float, float]], ts: datetime) -> float:
    if not points:
        return float("nan")
    lit = sum(1 for lat, lon in points if solar_zenith_deg(lat, lon, ts) < 90.0)
    return lit / len(points)

def minutes_since_terminator(lat: float, lon: float, ts: datetime) -> float:
    """Signed minutes since the day/night state at (lat,lon) last changed.
    Positive always (state began N min ago); capped at 720 when no crossing
    is found within 12 h (polar day/night)."""
    now_day = solar_zenith_deg(lat, lon, ts) < 90.0
    for m in range(4, 724, 4):  # scan back in 4-min steps
        then = ts - timedelta(minutes=m)
        if (solar_zenith_deg(lat, lon, then) < 90.0) != now_day:
            return float(m)
    return 720.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_solar.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/solar.py tests/features/test_solar.py
git commit -m "feat(features): NOAA solar position, daylight fraction, terminator minutes"
```

---

### Task 3: SWPC space-weather extractor (`data/swpc.py`)

**Files:**
- Create: `src/propagation/data/swpc.py`
- Create: `tests/data/fixtures/swpc_kp_1m.json`, `tests/data/fixtures/swpc_kp_3h.json`, `tests/data/fixtures/swpc_f107.json`, `tests/data/fixtures/swpc_xray.json`, `tests/data/fixtures/swpc_mag.json`, `tests/data/fixtures/swpc_plasma.json`
- Test: `tests/data/test_swpc.py`
- Modify: `pyproject.toml` (console script)

**Interfaces:**
- Consumes: `Lake.space_weather_dir(d) -> Path`, `Lake.write_parquet(df, dest_dir, name)` (M0).
- Produces: `fetch_series(client: httpx.Client) -> dict[str, pl.DataFrame]` (raw parsed series), `to_window_grid(series: dict[str, pl.DataFrame], start: datetime, end: datetime) -> pl.DataFrame`, `extract(start: date, end: date, bands: list[str], lake: Lake, cache_dir: Path) -> list[Path]` (Extractor protocol; `source = "swpc"`; bands ignored), `main() -> None`. Parquet columns (one row per 15-min `window_start` per UTC date partition): `window_start Datetime(us,UTC)`, `kp_est Float64`, `kp_def Float64`, `f107 Float64`, `xray_flux Float64` (W/m², long channel), `sw_speed Float64` (km/s), `sw_density Float64`, `sw_bz Float64` (nT).

**Endpoints (documented in the module docstring):**

| series | URL | cadence |
|---|---|---|
| Kp estimated | `https://services.swpc.noaa.gov/json/planetary_k_index_1m.json` | 1-min, ~7 days back |
| Kp 3-hourly (definitive-when-final) | `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json` | 3-h |
| F10.7 | `https://services.swpc.noaa.gov/json/f107_cm_flux.json` | daily |
| GOES X-ray | `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json` | 1-min |
| DSCOVR mag (Bz) | `https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json` | 1-min |
| DSCOVR plasma (speed/density) | `https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json` | 1-min |

**Honest limitation (goes in the module docstring and M2 plan):** SWPC JSON endpoints serve only trailing windows (~7 days). Like the PSKReporter accumulator, `extract-swpc` accumulates forward when run on a schedule; deep backfill for training months uses the same parsers over NCEI/GFZ archive files and is a separate task in M3. For M2 training months, run the extractor daily via cron/launchd from now on, or backfill Kp/F10.7 (the two that matter most) from the 3-h product's full history plus GFZ `Kp_ap_since_1932.txt` — parser task included below (Step 7).

**As-of semantics (normative here):** for each 15-min `window_start`, every column takes the latest raw observation with `obs_ts ≤ window_start` (forward-fill, no lookahead), except `kp_def`, which is the definitive 3-h value *covering* the window and is **eval-stratification-only** — it must never appear in `FEATURE_COLUMNS` (leakage test in Task 7).

- [ ] **Step 1: Create fixtures** — trimmed real-shaped samples (5–10 rows each). Example shapes to replicate exactly:

```json
// tests/data/fixtures/swpc_kp_1m.json  (array of objects)
[
  {"time_tag": "2026-05-01T00:00:00", "estimated_kp": 2.33, "kp": "2M"},
  {"time_tag": "2026-05-01T00:15:00", "estimated_kp": 2.67, "kp": "3M"},
  {"time_tag": "2026-05-01T00:30:00", "estimated_kp": 5.33, "kp": "5M"}
]
```

```json
// tests/data/fixtures/swpc_kp_3h.json  (array-of-arrays, row 0 = header)
[
  ["time_tag", "Kp", "a_running", "station_count"],
  ["2026-05-01 00:00:00.000", "2.33", "9", "8"],
  ["2026-05-01 03:00:00.000", "5.67", "67", "8"]
]
```

```json
// tests/data/fixtures/swpc_f107.json
[
  {"time_tag": "2026-04-30T20:00:00", "flux": 165.2, "ninety_day_mean": 158.0},
  {"time_tag": "2026-05-01T20:00:00", "flux": 168.9, "ninety_day_mean": 158.3}
]
```

```json
// tests/data/fixtures/swpc_xray.json
[
  {"time_tag": "2026-05-01T00:00:00Z", "satellite": 18, "flux": 1.2e-06, "energy": "0.1-0.8nm"},
  {"time_tag": "2026-05-01T00:00:00Z", "satellite": 18, "flux": 3.4e-08, "energy": "0.05-0.4nm"},
  {"time_tag": "2026-05-01T00:01:00Z", "satellite": 18, "flux": 1.3e-06, "energy": "0.1-0.8nm"}
]
```

```json
// tests/data/fixtures/swpc_mag.json  (array-of-arrays)
[
  ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"],
  ["2026-05-01 00:00:00.000", "1.2", "-3.4", "-5.6", "120.1", "-40.2", "6.8"]
]
```

```json
// tests/data/fixtures/swpc_plasma.json  (array-of-arrays)
[
  ["time_tag", "density", "speed", "temperature"],
  ["2026-05-01 00:00:00.000", "4.5", "512.3", "150000"]
]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/data/test_swpc.py
import json
from datetime import date, datetime, timezone
from pathlib import Path
import polars as pl
import pytest
from propagation.data import swpc

FIX = Path(__file__).parent / "fixtures"
UTC = timezone.utc

def _load(name):
    return json.loads((FIX / name).read_text())

def test_parse_kp_1m():
    df = swpc.parse_kp_1m(_load("swpc_kp_1m.json"))
    assert df.columns == ["obs_ts", "kp_est"]
    assert df["obs_ts"].dtype == pl.Datetime("us", "UTC")
    assert df["kp_est"].to_list() == [2.33, 2.67, 5.33]

def test_parse_kp_3h():
    df = swpc.parse_kp_3h(_load("swpc_kp_3h.json"))
    assert df.columns == ["obs_ts", "kp_def"]
    assert df["kp_def"].to_list() == [2.33, 5.67]

def test_parse_xray_keeps_long_channel_only():
    df = swpc.parse_xray(_load("swpc_xray.json"))
    assert df.columns == ["obs_ts", "xray_flux"]
    assert df.height == 2  # short-channel row dropped
    assert df["xray_flux"][0] == pytest.approx(1.2e-06)

def test_parse_mag_and_plasma():
    m = swpc.parse_mag(_load("swpc_mag.json"))
    assert m.columns == ["obs_ts", "sw_bz"]
    p = swpc.parse_plasma(_load("swpc_plasma.json"))
    assert p.columns == ["obs_ts", "sw_speed", "sw_density"]
    assert p["sw_speed"][0] == pytest.approx(512.3)

def test_to_window_grid_forward_fill_no_lookahead():
    series = {
        "kp_1m": swpc.parse_kp_1m(_load("swpc_kp_1m.json")),
        "kp_3h": swpc.parse_kp_3h(_load("swpc_kp_3h.json")),
        "f107": swpc.parse_f107(_load("swpc_f107.json")),
        "xray": swpc.parse_xray(_load("swpc_xray.json")),
        "mag": swpc.parse_mag(_load("swpc_mag.json")),
        "plasma": swpc.parse_plasma(_load("swpc_plasma.json")),
    }
    grid = swpc.to_window_grid(
        series,
        datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 5, 1, 1, 0, tzinfo=UTC),
    )
    assert grid.height == 4  # 00:00 00:15 00:30 00:45
    row0 = grid.row(0, named=True)
    assert row0["kp_est"] == 2.33          # obs at exactly 00:00 is usable
    row3 = grid.row(3, named=True)
    assert row3["kp_est"] == 5.33          # latest obs <= 00:45 is the 00:30 one
    # kp_def covers by 3-h block: 00:45 sits in the 00:00-03:00 block
    assert row3["kp_def"] == 2.33
    # f107 last obs was 2026-04-30T20:00 -> forward-filled
    assert row0["f107"] == pytest.approx(165.2)

def test_extract_writes_partition(tmp_path, monkeypatch):
    from propagation.data.lake import Lake
    series = {k: getattr(swpc, f"parse_{k.replace('kp_', 'kp_')}")(_load(f)) for k, f in [
        ("kp_1m", "swpc_kp_1m.json"), ("kp_3h", "swpc_kp_3h.json"),
        ("f107", "swpc_f107.json"), ("xray", "swpc_xray.json"),
        ("mag", "swpc_mag.json"), ("plasma", "swpc_plasma.json"),
    ]}
    monkeypatch.setattr(swpc, "fetch_series", lambda client=None: series)
    lake = Lake(tmp_path / "lake")
    files = swpc.extract(date(2026, 5, 1), date(2026, 5, 1), [], lake, tmp_path / "cache")
    assert len(files) == 1
    df = pl.read_parquet(files[0])
    assert df.height == 96
    assert set(df.columns) == {"window_start", "kp_est", "kp_def", "f107",
                               "xray_flux", "sw_speed", "sw_density", "sw_bz"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_swpc.py -v`
Expected: FAIL — `ImportError: cannot import name 'swpc'`

- [ ] **Step 4: Implement**

```python
# src/propagation/data/swpc.py
"""NOAA SWPC space-weather extractor -> lake/space_weather/date=.../

Endpoints (all public JSON):
  Kp est 1-min : https://services.swpc.noaa.gov/json/planetary_k_index_1m.json
  Kp 3-hourly  : https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
  F10.7 daily  : https://services.swpc.noaa.gov/json/f107_cm_flux.json
  GOES X-ray   : https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
  DSCOVR mag   : https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json
  DSCOVR plasma: https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json

LIMITATION: these serve trailing ~7-day windows. Run on a schedule to
accumulate; deep archive backfill (NCEI/GFZ) is an M3 task. kp_def is
eval-stratification-only and MUST NOT be used as a model feature.

As-of rule: each 15-min window takes the latest obs with obs_ts <= window_start
(forward-fill, no lookahead). kp_def instead is the 3-h value covering the window.
"""
import argparse
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

from propagation.data.lake import Lake

source = "swpc"
URLS = {
    "kp_1m": "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
    "kp_3h": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "f107": "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
    "xray": "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
    "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
}

def _ts(col: pl.Expr) -> pl.Expr:
    return (col.str.replace("Z$", "").str.replace(" ", "T")
            .str.strptime(pl.Datetime("us"), "%Y-%m-%dT%H:%M:%S%.f", strict=False)
            .dt.replace_time_zone("UTC"))

def _obj_rows(raw: list[dict], ts_key: str, val_map: dict[str, str]) -> pl.DataFrame:
    df = pl.DataFrame(raw)
    out = df.select(_ts(pl.col(ts_key)).alias("obs_ts"),
                    *[pl.col(src).cast(pl.Float64).alias(dst)
                      for src, dst in val_map.items()])
    return out.drop_nulls("obs_ts").sort("obs_ts")

def _tab_rows(raw: list[list], val_map: dict[str, str]) -> pl.DataFrame:
    header, *rows = raw
    df = pl.DataFrame(rows, schema=header, orient="row")
    out = df.select(_ts(pl.col("time_tag")).alias("obs_ts"),
                    *[pl.col(src).cast(pl.Float64).alias(dst)
                      for src, dst in val_map.items()])
    return out.drop_nulls("obs_ts").sort("obs_ts")

def parse_kp_1m(raw: list[dict]) -> pl.DataFrame:
    return _obj_rows(raw, "time_tag", {"estimated_kp": "kp_est"})

def parse_kp_3h(raw: list[list]) -> pl.DataFrame:
    return _tab_rows(raw, {"Kp": "kp_def"})

def parse_f107(raw: list[dict]) -> pl.DataFrame:
    return _obj_rows(raw, "time_tag", {"flux": "f107"})

def parse_xray(raw: list[dict]) -> pl.DataFrame:
    long = [r for r in raw if r.get("energy") == "0.1-0.8nm"]
    return _obj_rows(long, "time_tag", {"flux": "xray_flux"})

def parse_mag(raw: list[list]) -> pl.DataFrame:
    return _tab_rows(raw, {"bz_gsm": "sw_bz"})

def parse_plasma(raw: list[list]) -> pl.DataFrame:
    return _tab_rows(raw, {"speed": "sw_speed", "density": "sw_density"})

def fetch_series(client: httpx.Client | None = None) -> dict[str, pl.DataFrame]:
    own = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        raw = {k: client.get(u).raise_for_status().json() for k, u in URLS.items()}
    finally:
        if own:
            client.close()
    return {
        "kp_1m": parse_kp_1m(raw["kp_1m"]), "kp_3h": parse_kp_3h(raw["kp_3h"]),
        "f107": parse_f107(raw["f107"]), "xray": parse_xray(raw["xray"]),
        "mag": parse_mag(raw["mag"]), "plasma": parse_plasma(raw["plasma"]),
    }

def to_window_grid(series: dict[str, pl.DataFrame],
                   start: datetime, end: datetime) -> pl.DataFrame:
    grid = pl.DataFrame({"window_start": pl.datetime_range(
        start, end, interval="15m", closed="left", time_zone="UTC", eager=True)})
    asof_cols = {"kp_1m": ["kp_est"], "f107": ["f107"], "xray": ["xray_flux"],
                 "mag": ["sw_bz"], "plasma": ["sw_speed", "sw_density"]}
    out = grid
    for key, cols in asof_cols.items():
        out = out.join_asof(series[key].rename({"obs_ts": "window_start"})
                            .select(["window_start", *cols]),
                            on="window_start", strategy="backward")
    kp3 = series["kp_3h"].with_columns(pl.col("obs_ts").alias("block_start"))
    out = out.join_asof(kp3.select(["block_start", "kp_def"])
                        .rename({"block_start": "window_start"}),
                        on="window_start", strategy="backward")
    return out

def extract(start: date, end: date, bands: list[str],
            lake: Lake, cache_dir: Path) -> list[Path]:
    series = fetch_series()
    written: list[Path] = []
    d = start
    while d <= end:
        s = datetime.combine(d, time(0), tzinfo=timezone.utc)
        grid = to_window_grid(series, s, s + timedelta(days=1))
        written.append(lake.write_parquet(grid, lake.space_weather_dir(d)))
        d += timedelta(days=1)
    return written

def main() -> None:
    ap = argparse.ArgumentParser("extract-swpc")
    ap.add_argument("--start", required=True, type=date.fromisoformat)
    ap.add_argument("--end", required=True, type=date.fromisoformat)
    ap.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    a = ap.parse_args()
    files = extract(a.start, a.end, [], Lake(a.lake_root), a.cache_dir)
    print(f"wrote {len(files)} partition file(s)")
```

- [ ] **Step 5: Register the console script** — in `pyproject.toml` `[project.scripts]` add:

```toml
extract-swpc = "propagation.data.swpc:main"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_swpc.py -v`
Expected: 6 passed

- [ ] **Step 7: Add GFZ Kp backfill parser (training-month coverage)** — same module; test first:

```python
# append to tests/data/test_swpc.py
def test_parse_gfz_kp_lines():
    lines = [
        "# comment",
        "2026 05 01 00.0 01.5  0.125 2.333 23 1",
        "2026 05 01 03.0 04.5  0.125 5.667 67 1",
    ]
    df = swpc.parse_gfz_kp(lines)
    assert df.columns == ["obs_ts", "kp_def"]
    assert df["kp_def"].to_list() == [pytest.approx(2.333), pytest.approx(5.667)]
```

```python
# append to src/propagation/data/swpc.py
GFZ_KP_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_since_1932.txt"

def parse_gfz_kp(lines: list[str]) -> pl.DataFrame:
    """GFZ definitive Kp file: 'YYYY MM DD hh.h hh._m days days_m Kp ap D' rows."""
    rows = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split()
        rows.append((datetime(int(f[0]), int(f[1]), int(f[2]),
                              int(float(f[3])), tzinfo=timezone.utc), float(f[7])))
    return pl.DataFrame(rows, schema={"obs_ts": pl.Datetime("us", "UTC"),
                                      "kp_def": pl.Float64}, orient="row")
```

NOTE (verify at execution): the GFZ column layout above (Kp in field index 7) must be checked against the live file header once, in Step 8's spot check — the file documents its own columns in `#` comments; adjust the index if the header disagrees, and update this test's fixture lines to match real ones.

- [ ] **Step 8: Run full module tests + one live spot check (not in CI)**

Run: `uv run pytest tests/data/test_swpc.py -v` → 7 passed.
Run once manually: `uv run extract-swpc --start 2026-07-01 --end 2026-07-01 --lake-root data/lake` → `wrote 1 partition file(s)`; open the parquet and eyeball kp_est/f107 against https://www.swpc.noaa.gov. Also `curl -s https://kp.gfz-potsdam.de/app/files/Kp_ap_since_1932.txt | head -50` to confirm `parse_gfz_kp` column indices; fix test fixture lines if the layout differs.

- [ ] **Step 9: Commit**

```bash
git add src/propagation/data/swpc.py tests/data/test_swpc.py tests/data/fixtures/ pyproject.toml
git commit -m "feat(data): SWPC space-weather extractor with as-of window grid + GFZ Kp backfill parser"
```

---

### Task 4: Space-weather feature loader (`features/spaceweather.py`)

**Files:**
- Create: `src/propagation/features/spaceweather.py`
- Test: `tests/features/test_spaceweather.py`

**Interfaces:**
- Consumes: `Lake.connect()` exposing the `space_weather` view (M0); Task 3's parquet columns.
- Produces: `load_sw_asof(lake: Lake, start: datetime, end: datetime) -> pl.DataFrame` with columns `window_start`, `kp_est`, `kp_est_lag3h`, `kp_est_lag6h`, `kp_est_lag12h`, `kp_est_lag24h`, `kp_est_lag48h`, `f107`, `f107_sm27`, `xray_flux`, `sw_speed`, `sw_bz`. Also `SW_FEATURE_COLUMNS: list[str]` (everything above except `window_start`). `kp_def` is deliberately NOT returned by this loader; eval stratification reads it separately via `load_kp_def(lake, start, end)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_spaceweather.py
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import polars as pl
import pytest
from propagation.data.lake import Lake
from propagation.features.spaceweather import (
    SW_FEATURE_COLUMNS, load_kp_def, load_sw_asof,
)

UTC = timezone.utc

@pytest.fixture
def lake_with_sw(tmp_path) -> Lake:
    lake = Lake(tmp_path / "lake")
    start = datetime(2026, 5, 1, tzinfo=UTC)
    n_days = 4
    rows = []
    for i in range(n_days * 96):
        ws = start + timedelta(minutes=15 * i)
        rows.append({"window_start": ws, "kp_est": float(i % 9),
                     "kp_def": 5.0 if ws.day == 2 else 1.0,
                     "f107": 150.0 + ws.day, "xray_flux": 1e-6,
                     "sw_speed": 400.0, "sw_density": 4.0, "sw_bz": -2.0})
    df = pl.DataFrame(rows).with_columns(
        pl.col("window_start").cast(pl.Datetime("us", "UTC")))
    for d in range(n_days):
        day = date(2026, 5, 1 + d)
        part = df.filter(pl.col("window_start").dt.date() == day)
        lake.write_parquet(part, lake.space_weather_dir(day))
    return lake

def test_load_sw_asof_columns_and_lags(lake_with_sw):
    out = load_sw_asof(lake_with_sw,
                       datetime(2026, 5, 3, tzinfo=UTC),
                       datetime(2026, 5, 4, tzinfo=UTC))
    assert set(SW_FEATURE_COLUMNS) <= set(out.columns)
    assert "kp_def" not in out.columns
    row = out.filter(pl.col("window_start") == datetime(2026, 5, 3, 6, 0, tzinfo=UTC))
    # kp_est cycles 0..8 per 15-min step; lag3h = 12 steps earlier
    i_now = ((2 * 96) + 24)   # steps since 05-01 00:00 at 05-03 06:00
    assert row["kp_est"][0] == float(i_now % 9)
    assert row["kp_est_lag3h"][0] == float((i_now - 12) % 9)
    assert row["kp_est_lag48h"][0] == float((i_now - 192) % 9)

def test_f107_sm27_is_trailing_mean(lake_with_sw):
    out = load_sw_asof(lake_with_sw,
                       datetime(2026, 5, 3, tzinfo=UTC),
                       datetime(2026, 5, 4, tzinfo=UTC))
    # f107 was 151,152,153 over the 3 available trailing days -> mean 152-ish
    v = out["f107_sm27"][0]
    assert 151.0 <= v <= 153.5

def test_load_kp_def_separate(lake_with_sw):
    kd = load_kp_def(lake_with_sw,
                     datetime(2026, 5, 2, tzinfo=UTC),
                     datetime(2026, 5, 3, tzinfo=UTC))
    assert kd.columns == ["window_start", "kp_def"]
    assert kd["kp_def"].max() == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_spaceweather.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
# src/propagation/features/spaceweather.py
"""As-of space-weather features on the 15-min window grid.

kp_def is eval-stratification-only: load_sw_asof never returns it, so it
cannot leak into FEATURE_COLUMNS. Lags are exact shifts on the 15-min grid;
f107_sm27 is the trailing 27-day mean (solar rotation), min 1 day.
"""
from datetime import datetime, timedelta

import polars as pl

from propagation.data.lake import Lake

KP_LAGS_H = [3, 6, 12, 24, 48]
SW_FEATURE_COLUMNS: list[str] = (
    ["kp_est"] + [f"kp_est_lag{h}h" for h in KP_LAGS_H]
    + ["f107", "f107_sm27", "xray_flux", "sw_speed", "sw_bz"]
)

def _read_range(lake: Lake, start: datetime, end: datetime) -> pl.DataFrame:
    con = lake.connect()
    df = con.execute(
        "SELECT * FROM space_weather WHERE window_start >= ? AND window_start < ? "
        "ORDER BY window_start", [start, end]).pl()
    return df.with_columns(pl.col("window_start").cast(pl.Datetime("us", "UTC")))

def load_sw_asof(lake: Lake, start: datetime, end: datetime) -> pl.DataFrame:
    # read back far enough for the longest lag and the 27-day smoothing
    back = _read_range(lake, start - timedelta(days=28), end)
    lagged = back.with_columns(
        [pl.col("kp_est").shift(h * 4).alias(f"kp_est_lag{h}h") for h in KP_LAGS_H]
        + [pl.col("f107").rolling_mean(window_size=27 * 96, min_samples=1)
           .alias("f107_sm27")]
    )
    return (lagged.filter(pl.col("window_start") >= start)
            .select(["window_start", *SW_FEATURE_COLUMNS]))

def load_kp_def(lake: Lake, start: datetime, end: datetime) -> pl.DataFrame:
    """Definitive Kp per window — EVAL STRATIFICATION ONLY, never a feature."""
    return _read_range(lake, start, end).select(["window_start", "kp_def"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_spaceweather.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/spaceweather.py tests/features/test_spaceweather.py
git commit -m "feat(features): as-of space-weather grid with Kp lags; kp_def quarantined to eval"
```

---

### Task 5: Autoregressive history features (`features/history.py`)

**Files:**
- Create: `src/propagation/features/history.py`
- Test: `tests/features/test_history.py`

**Interfaces:**
- Consumes: `Lake.connect()` exposing the `spots` view (qualified+deduped, columns incl. `ts`, `band`, `dx_field`, `de_field`, `mode`, `snr_db`, `tx_dbm`); `snr_ft8eq` from `propagation.features.labels` (M0); `BANDS` from `propagation.schema`.
- Produces: `DELTA_AVAIL_MIN: int = 5`, `MAX_AR_LOOKBACK_H: int = 24`, `ar_features(lake: Lake, cells: pl.DataFrame, pred_time_col: str) -> pl.DataFrame` — input `cells` needs columns `tx_field, rx_field, band` + the named pred-time column; returns `cells` + the 34 AR columns listed in Task 6. Also `neighbor_fields(field: str) -> list[str]` (≤8 neighbors, longitude wrap-around, latitude clamped at AA/AR rows) and `adjacent_bands(band: str) -> list[str]` (±1 in BANDS order).

**AR windows and groups (normative for this repo):** trailing windows `15m, 1h, 3h, 24h`, each ending at `pred_time − 5min` (Δ_avail, REVIEW-FINDINGS R1). Spot **counts** for six groups × 4 windows = 24 columns: this cell (`cell`), reverse cell (`rev`), adjacent-cells aggregate (`adj`, sum over neighbor tx_field×rx_field combos, same band), one band up (`bandup`), one band down (`banddn`) for this cell pair, band-wide global activity (`global`, all cells on this band). **Median `snr_ft8eq`** for `cell` and `rev` × 4 windows = 8 columns. Same-cell-same-hour-yesterday count + median snr = 2 columns. Total 34.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_history.py
from datetime import date, datetime, timedelta, timezone
import polars as pl
import pytest
from propagation.data.lake import Lake
from propagation.features.history import (
    DELTA_AVAIL_MIN, MAX_AR_LOOKBACK_H, adjacent_bands, ar_features, neighbor_fields,
)

UTC = timezone.utc
T0 = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)

def _spot(ts, band="20m", dx_field="EM", de_field="JN", mode="FT8", snr=-10):
    return {"source": "wsprnet", "ts": ts, "band": band, "mode": mode,
            "freq_hz": 14074000, "dx_call": "K5ARH", "de_call": "DL1ABC",
            "dx_grid": None, "de_grid": None, "dx_lat": None, "dx_lon": None,
            "de_lat": None, "de_lon": None, "snr_db": snr, "tx_dbm": None,
            "distance_km": 8000.0, "bearing_deg": 45.0, "mode_class": "digi",
            "dx_field": dx_field, "de_field": de_field,
            "dx_grid4": None, "de_grid4": None,
            "window_start": ts.replace(minute=(ts.minute // 15) * 15, second=0)}

@pytest.fixture
def lake_with_spots(tmp_path) -> Lake:
    lake = Lake(tmp_path / "lake")
    rows = [
        _spot(T0 - timedelta(minutes=10)),                       # inside 15m window
        _spot(T0 - timedelta(minutes=40)),                       # inside 1h only
        _spot(T0 - timedelta(hours=2)),                          # inside 3h only
        _spot(T0 - timedelta(hours=20)),                         # inside 24h only
        _spot(T0 - timedelta(minutes=10), dx_field="JN", de_field="EM"),  # reverse
        _spot(T0 - timedelta(minutes=10), band="17m"),           # band up
        _spot(T0 - timedelta(minutes=10), band="30m"),           # band down
        _spot(T0 - timedelta(minutes=10), dx_field="FM", de_field="JN"),  # adjacent tx
        _spot(T0 - timedelta(hours=24), snr=-5),                 # same-hour-yesterday
        _spot(T0 - timedelta(minutes=3)),   # POISON: inside Delta_avail, must not count
    ]
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")),
        pl.col("window_start").cast(pl.Datetime("us", "UTC")))
    for band in ["20m", "17m", "30m"]:
        for d in [date(2026, 5, 1), date(2026, 5, 2)]:
            part = df.filter((pl.col("band") == band)
                             & (pl.col("ts").dt.date() == d))
            if part.height:
                lake.write_parquet(part, lake.spots_q_dir(band, d))
    return lake

def _cells():
    return pl.DataFrame({
        "tx_field": ["EM"], "rx_field": ["JN"], "band": ["20m"],
        "pred_time": [T0],
    }).with_columns(pl.col("pred_time").cast(pl.Datetime("us", "UTC")))

def test_ar_counts_per_window(lake_with_spots):
    out = ar_features(lake_with_spots, _cells(), "pred_time")
    r = out.row(0, named=True)
    assert r["ar_cell_n_15m"] == 1     # poison spot at T0-3min excluded
    assert r["ar_cell_n_1h"] == 2
    assert r["ar_cell_n_3h"] == 3
    assert r["ar_cell_n_24h"] == 4
    assert r["ar_rev_n_15m"] == 1
    assert r["ar_bandup_n_15m"] == 1
    assert r["ar_banddn_n_15m"] == 1
    assert r["ar_adj_n_15m"] == 1
    assert r["ar_global_n_15m"] == 3   # cell + reverse + adjacent, all on 20m

def test_delta_avail_is_enforced(lake_with_spots):
    # The spot at T0-3min sits inside the availability buffer; with the buffer
    # removed it WOULD count. This is the R1 invariant.
    out = ar_features(lake_with_spots, _cells(), "pred_time")
    assert out.row(0, named=True)["ar_cell_n_15m"] == 1
    assert DELTA_AVAIL_MIN == 5

def test_same_hour_yesterday(lake_with_spots):
    r = ar_features(lake_with_spots, _cells(), "pred_time").row(0, named=True)
    assert r["ar_cell_n_yday1h"] == 1
    assert r["ar_cell_snr_yday1h"] == pytest.approx(-5.0)

def test_neighbor_fields_wrap_and_clamp():
    assert "EL" in neighbor_fields("EM") and "FM" in neighbor_fields("EM")
    assert len(neighbor_fields("EM")) == 8
    assert all(f[1] != chr(ord("A") - 1) for f in neighbor_fields("AA"))  # no south of AA
    assert "RA" in neighbor_fields("AA")   # longitude wraps A->R

def test_adjacent_bands_edges():
    assert adjacent_bands("20m") == ["17m", "30m"]   # [up, down]
    assert adjacent_bands("160m") == ["80m"]
    assert adjacent_bands("6m") == ["10m"]

def test_max_lookback_constant():
    assert MAX_AR_LOOKBACK_H == 24
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_history.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
# src/propagation/features/history.py
"""Autoregressive spot-history features.

EVERY trailing window ends at pred_time - DELTA_AVAIL_MIN (5 min): training
features must be a faithful stand-in for what a live system can see
(docs/REVIEW-FINDINGS.md R1). MAX_AR_LOOKBACK_H feeds the CV-gap check in
eval/splits tests — lengthen a lookback and the leakage test fails until
GAP_HOURS is widened (SPEC-labeling section 6.1).
"""
from datetime import timedelta

import polars as pl

from propagation.data.lake import Lake
from propagation.schema import BANDS

DELTA_AVAIL_MIN = 5
MAX_AR_LOOKBACK_H = 24
AR_WINDOWS = [("15m", 15), ("1h", 60), ("3h", 180), ("24h", 1440)]

def neighbor_fields(field: str) -> list[str]:
    ci, ri = ord(field[0]) - ord("A"), ord(field[1]) - ord("A")
    out = []
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == dr == 0:
                continue
            nr = ri + dr
            if not (0 <= nr <= 17):        # latitude clamps
                continue
            nc = (ci + dc) % 18            # longitude wraps
            out.append(chr(ord("A") + nc) + chr(ord("A") + nr))
    return out

def adjacent_bands(band: str) -> list[str]:
    """[one band up (higher freq / shorter wavelength), one band down]."""
    i = BANDS.index(band)
    out = []
    if i + 1 < len(BANDS):
        out.append(BANDS[i + 1])
    if i - 1 >= 0:
        out.append(BANDS[i - 1])
    return out

def _median_ft8eq() -> str:
    # SQL mirror of labels.snr_ft8eq (SPEC 4.4): digi 0, cw/rtty -7, else NULL;
    # + (50 - tx_dbm) when tx_dbm known.
    return """
      median(CASE
        WHEN mode_class = 'digi' THEN snr_db + COALESCE(50 - tx_dbm, 0)
        WHEN mode_class = 'cw'  THEN snr_db - 7 + COALESCE(50 - tx_dbm, 0)
        ELSE NULL END)
    """

def ar_features(lake: Lake, cells: pl.DataFrame, pred_time_col: str) -> pl.DataFrame:
    con = lake.connect()
    cells = cells.with_row_index("_ci")
    con.register("_cells", cells.rename({pred_time_col: "_pt"}).to_arrow())
    med = _median_ft8eq()
    win_cnt, win_snr = [], []
    for name, mins in AR_WINDOWS:
        lo = f"_pt - INTERVAL {DELTA_AVAIL_MIN + mins} MINUTE"
        hi = f"_pt - INTERVAL {DELTA_AVAIL_MIN} MINUTE"
        rng = f"s.ts >= {lo} AND s.ts < {hi}"
        win_cnt += [
            f"count(*) FILTER ({rng} AND s.band = c.band AND s.dx_field = c.tx_field AND s.de_field = c.rx_field) AS ar_cell_n_{name}",
            f"count(*) FILTER ({rng} AND s.band = c.band AND s.dx_field = c.rx_field AND s.de_field = c.tx_field) AS ar_rev_n_{name}",
            f"count(*) FILTER ({rng} AND s.band = c.band AND ((s.dx_field = ANY(c._adj_tx) AND s.de_field = c.rx_field) OR (s.dx_field = c.tx_field AND s.de_field = ANY(c._adj_rx)))) AS ar_adj_n_{name}",
            f"count(*) FILTER ({rng} AND s.band = c._bandup AND s.dx_field = c.tx_field AND s.de_field = c.rx_field) AS ar_bandup_n_{name}",
            f"count(*) FILTER ({rng} AND s.band = c._banddn AND s.dx_field = c.tx_field AND s.de_field = c.rx_field) AS ar_banddn_n_{name}",
            f"count(*) FILTER ({rng} AND s.band = c.band) AS ar_global_n_{name}",
        ]
        win_snr += [
            f"{med} FILTER ({rng} AND s.band = c.band AND s.dx_field = c.tx_field AND s.de_field = c.rx_field) AS ar_cell_snr_{name}",
            f"{med} FILTER ({rng} AND s.band = c.band AND s.dx_field = c.rx_field AND s.de_field = c.tx_field) AS ar_rev_snr_{name}",
        ]
    yd = ("s.ts >= _pt - INTERVAL 1470 MINUTE AND s.ts < _pt - INTERVAL 1410 MINUTE")
    # 24h back, same trailing 1h window (1440+30..1440-30 around T-24h works out
    # to [T-24h-30m, T-24h+30m); using +-30 min keeps it inside MAX_AR_LOOKBACK_H+pad? No:
    # pinned definition: the 60-min window centered on pred_time-24h.
    yday = [
        f"count(*) FILTER ({yd} AND s.band = c.band AND s.dx_field = c.tx_field AND s.de_field = c.rx_field) AS ar_cell_n_yday1h",
        f"{med} FILTER ({yd} AND s.band = c.band AND s.dx_field = c.tx_field AND s.de_field = c.rx_field) AS ar_cell_snr_yday1h",
    ]
    enriched = con.execute("""
        SELECT c.*,
               list_transform(?, x -> x) AS _dummy
        FROM _cells c
    """, [[0]]).pl()  # placeholder replaced below — see NOTE
    # Precompute per-row helper columns in polars (simpler than SQL lists):
    helpers = cells.with_columns(
        pl.col("tx_field").map_elements(neighbor_fields, return_dtype=pl.List(pl.Utf8)).alias("_adj_tx"),
        pl.col("rx_field").map_elements(neighbor_fields, return_dtype=pl.List(pl.Utf8)).alias("_adj_rx"),
        pl.col("band").map_elements(lambda b: (adjacent_bands(b) + [None])[0],
                                    return_dtype=pl.Utf8).alias("_bandup"),
        pl.col("band").map_elements(lambda b: (adjacent_bands(b) + [None, None])[1],
                                    return_dtype=pl.Utf8).alias("_banddn"),
    ).rename({pred_time_col: "_pt"})
    con.unregister("_cells")
    con.register("_cells", helpers.to_arrow())
    sql = f"""
        SELECT c._ci, {', '.join(win_cnt + win_snr + yday)}
        FROM _cells c
        LEFT JOIN spots s ON s.ts >= c._pt - INTERVAL {MAX_AR_LOOKBACK_H * 60 + 30} MINUTE
                         AND s.ts <  c._pt
        GROUP BY c._ci
    """
    feats = con.execute(sql).pl()
    out = cells.join(feats, on="_ci", how="left").drop("_ci")
    count_cols = [c for c in out.columns if c.startswith("ar_") and "_n_" in c]
    return out.with_columns([pl.col(c).fill_null(0).cast(pl.Int32) for c in count_cols])
```

NOTE (implementation cleanup during execution): the stray `enriched = ...` placeholder query above must be deleted — the helper-column path below it is the real one. It is shown struck here so the executing engineer doesn't reinvent it; final code registers `helpers` once and runs one grouped join. The `yday1h` window is pinned as `[pred_time − 24h30m, pred_time − 23h30m)` — the 60-min window centered on pred_time−24h; total reach 24 h 30 m is still within the 48 h CV gap with max horizon 24 h **only because yday1h is not offered at h=24h** — `feature_matrix` (Task 6) enforces: horizons > 3 h drop the `yday1h` pair. Simpler and safe: the gap check in Task 7 uses `MAX_AR_LOOKBACK_H + 0.5`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_history.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/history.py tests/features/test_history.py
git commit -m "feat(features): autoregressive spot-history features with 5-min availability buffer"
```

---

### Task 6: Feature matrix assembly (`features/matrix.py`)

**Files:**
- Create: `src/propagation/features/matrix.py`
- Test: `tests/features/test_matrix.py`

**Interfaces:**
- Consumes: Tasks 1–5 (`grid_to_latlon…geomag_lat`, `solar_zenith_deg…minutes_since_terminator`, `load_sw_asof`/`SW_FEATURE_COLUMNS`, `ar_features`); M0 `labels` frame columns (`window_start, tx_field, rx_field, band, open, sample_weight, …`); `BANDS`.
- Produces: `feature_matrix(lake: Lake, labels: pl.DataFrame, horizon_s: int) -> pl.DataFrame` (labels columns + `pred_time` + all FEATURE_COLUMNS) and `FEATURE_COLUMNS: list[str]` — the canonical ordered list, single source of truth:

```python
FEATURE_COLUMNS = [
    # geometry (12)
    "tx_lat", "tx_lon", "rx_lat", "rx_lon", "distance_km", "bearing_deg",
    "mid_lat", "mid_lon", "geomag_lat_mid", "geomag_lat_cp1", "geomag_lat_cp2",
    "band_idx",
    # solar (7)
    "sza_tx", "sza_rx", "sza_mid", "sza_cp1", "sza_cp2",
    "daylight_frac", "mins_since_terminator_mid",
    # time (5)
    "hod_sin", "hod_cos", "doy_sin", "doy_cos", "month",
    # space weather (11)
    "kp_est", "kp_est_lag3h", "kp_est_lag6h", "kp_est_lag12h",
    "kp_est_lag24h", "kp_est_lag48h", "f107", "f107_sm27",
    "xray_flux_log10", "sw_speed", "sw_bz",
    # AR history (34)
    "ar_cell_n_15m", "ar_cell_n_1h", "ar_cell_n_3h", "ar_cell_n_24h",
    "ar_rev_n_15m", "ar_rev_n_1h", "ar_rev_n_3h", "ar_rev_n_24h",
    "ar_adj_n_15m", "ar_adj_n_1h", "ar_adj_n_3h", "ar_adj_n_24h",
    "ar_bandup_n_15m", "ar_bandup_n_1h", "ar_bandup_n_3h", "ar_bandup_n_24h",
    "ar_banddn_n_15m", "ar_banddn_n_1h", "ar_banddn_n_3h", "ar_banddn_n_24h",
    "ar_global_n_15m", "ar_global_n_1h", "ar_global_n_3h", "ar_global_n_24h",
    "ar_cell_snr_15m", "ar_cell_snr_1h", "ar_cell_snr_3h", "ar_cell_snr_24h",
    "ar_rev_snr_15m", "ar_rev_snr_1h", "ar_rev_snr_3h", "ar_rev_snr_24h",
    "ar_cell_n_yday1h", "ar_cell_snr_yday1h",
]  # 69 columns
```

Semantics: `pred_time = window_start − horizon_s`; solar/time features are computed at **window_start** (they are computable for any future time — ARCHITECTURE §4.2); space-weather and AR features are as-of **pred_time**. `xray_flux_log10 = log10(max(xray_flux, 1e-9))`. `band_idx = BANDS.index(band)`. For `horizon_s > 10800` (3 h), the `yday1h` pair is null-filled (see Task 5 NOTE) — LightGBM handles nulls natively.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_matrix.py
from datetime import datetime, timedelta, timezone
import polars as pl
import pytest
from propagation.features.matrix import FEATURE_COLUMNS, feature_matrix

UTC = timezone.utc

def test_feature_columns_canonical():
    assert len(FEATURE_COLUMNS) == 69
    assert len(set(FEATURE_COLUMNS)) == 69
    assert "kp_def" not in FEATURE_COLUMNS
    assert "open" not in FEATURE_COLUMNS and "sample_weight" not in FEATURE_COLUMNS

def test_feature_matrix_shapes_and_pred_time(lake_full):  # fixture from conftest below
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 5, 2, 12, 0, tzinfo=UTC)],
        "tx_field": ["EM"], "rx_field": ["JN"], "band": ["20m"],
        "open": [1], "n_spots": [3], "n_monitors": [2], "n_tx_stations": [1],
        "evidence_tier": ["spot"], "snr_ft8eq_p50": [None],
        "sample_weight": [1.0], "split_tag": [None],
    }).with_columns(pl.col("window_start").cast(pl.Datetime("us", "UTC")))
    out = feature_matrix(lake_full, labels, horizon_s=10800)
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["pred_time"] == datetime(2026, 5, 2, 9, 0, tzinfo=UTC)
    for c in FEATURE_COLUMNS:
        assert c in out.columns, f"missing {c}"
    assert r["band_idx"] == 5           # 20m
    assert 0.0 <= r["daylight_frac"] <= 1.0
    assert r["month"] == 5

def test_yday_nulled_beyond_3h(lake_full):
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 5, 2, 12, 0, tzinfo=UTC)],
        "tx_field": ["EM"], "rx_field": ["JN"], "band": ["20m"],
        "open": [0], "n_spots": [0], "n_monitors": [1], "n_tx_stations": [1],
        "evidence_tier": ["spot"], "snr_ft8eq_p50": [None],
        "sample_weight": [3.0], "split_tag": [None],
    }).with_columns(pl.col("window_start").cast(pl.Datetime("us", "UTC")))
    out = feature_matrix(lake_full, labels, horizon_s=86400)
    assert out["ar_cell_n_yday1h"][0] is None
    assert out["ar_cell_snr_yday1h"][0] is None
```

Add `tests/features/conftest.py` providing `lake_full`: a lake fixture combining Task 4's space-weather partitions and Task 5's spot partitions (copy both fixture bodies into one `@pytest.fixture def lake_full(tmp_path)`; identical data, one lake root).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_matrix.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
# src/propagation/features/matrix.py
"""Assemble the full M2 feature matrix. FEATURE_COLUMNS is the single source
of truth for model inputs; kp_def and label columns are structurally excluded."""
import math
from datetime import timedelta

import polars as pl

from propagation.data.lake import Lake
from propagation.features.geometry import (
    control_points, geomag_lat, grid_to_latlon, haversine_km,
    initial_bearing_deg, midpoint_latlon,
)
from propagation.features.history import ar_features
from propagation.features.solar import (
    daylight_fraction, minutes_since_terminator, solar_zenith_deg,
)
from propagation.features.spaceweather import SW_FEATURE_COLUMNS, load_sw_asof
from propagation.schema import BANDS

FEATURE_COLUMNS = [  # exactly the list in the plan header of this task (69)
    "tx_lat", "tx_lon", "rx_lat", "rx_lon", "distance_km", "bearing_deg",
    "mid_lat", "mid_lon", "geomag_lat_mid", "geomag_lat_cp1", "geomag_lat_cp2",
    "band_idx",
    "sza_tx", "sza_rx", "sza_mid", "sza_cp1", "sza_cp2",
    "daylight_frac", "mins_since_terminator_mid",
    "hod_sin", "hod_cos", "doy_sin", "doy_cos", "month",
    "kp_est", "kp_est_lag3h", "kp_est_lag6h", "kp_est_lag12h",
    "kp_est_lag24h", "kp_est_lag48h", "f107", "f107_sm27",
    "xray_flux_log10", "sw_speed", "sw_bz",
    "ar_cell_n_15m", "ar_cell_n_1h", "ar_cell_n_3h", "ar_cell_n_24h",
    "ar_rev_n_15m", "ar_rev_n_1h", "ar_rev_n_3h", "ar_rev_n_24h",
    "ar_adj_n_15m", "ar_adj_n_1h", "ar_adj_n_3h", "ar_adj_n_24h",
    "ar_bandup_n_15m", "ar_bandup_n_1h", "ar_bandup_n_3h", "ar_bandup_n_24h",
    "ar_banddn_n_15m", "ar_banddn_n_1h", "ar_banddn_n_3h", "ar_banddn_n_24h",
    "ar_global_n_15m", "ar_global_n_1h", "ar_global_n_3h", "ar_global_n_24h",
    "ar_cell_snr_15m", "ar_cell_snr_1h", "ar_cell_snr_3h", "ar_cell_snr_24h",
    "ar_rev_snr_15m", "ar_rev_snr_1h", "ar_rev_snr_3h", "ar_rev_snr_24h",
    "ar_cell_n_yday1h", "ar_cell_snr_yday1h",
]

def _geom_solar_time_row(tx_field: str, rx_field: str, band: str, ws) -> dict:
    tx_lat, tx_lon = grid_to_latlon(tx_field)
    rx_lat, rx_lon = grid_to_latlon(rx_field)
    mid = midpoint_latlon(tx_lat, tx_lon, rx_lat, rx_lon)
    cps = control_points(tx_lat, tx_lon, rx_lat, rx_lon)
    pts = [(tx_lat, tx_lon), *cps, mid, (rx_lat, rx_lon)]
    hod = ws.hour + ws.minute / 60
    doy = ws.timetuple().tm_yday
    return {
        "tx_lat": tx_lat, "tx_lon": tx_lon, "rx_lat": rx_lat, "rx_lon": rx_lon,
        "distance_km": haversine_km(tx_lat, tx_lon, rx_lat, rx_lon),
        "bearing_deg": initial_bearing_deg(tx_lat, tx_lon, rx_lat, rx_lon),
        "mid_lat": mid[0], "mid_lon": mid[1],
        "geomag_lat_mid": geomag_lat(*mid),
        "geomag_lat_cp1": geomag_lat(*cps[0]), "geomag_lat_cp2": geomag_lat(*cps[1]),
        "band_idx": BANDS.index(band),
        "sza_tx": solar_zenith_deg(tx_lat, tx_lon, ws),
        "sza_rx": solar_zenith_deg(rx_lat, rx_lon, ws),
        "sza_mid": solar_zenith_deg(*mid, ws),
        "sza_cp1": solar_zenith_deg(*cps[0], ws),
        "sza_cp2": solar_zenith_deg(*cps[1], ws),
        "daylight_frac": daylight_fraction(pts, ws),
        "mins_since_terminator_mid": minutes_since_terminator(*mid, ws),
        "hod_sin": math.sin(2 * math.pi * hod / 24),
        "hod_cos": math.cos(2 * math.pi * hod / 24),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "month": ws.month,
    }

def feature_matrix(lake: Lake, labels: pl.DataFrame, horizon_s: int) -> pl.DataFrame:
    out = labels.with_columns(
        (pl.col("window_start") - timedelta(seconds=horizon_s)).alias("pred_time"))
    # geometry/solar/time: computed at window_start (future-safe)
    gst = pl.DataFrame([
        _geom_solar_time_row(r["tx_field"], r["rx_field"], r["band"], r["window_start"])
        for r in out.iter_rows(named=True)
    ])
    out = pl.concat([out, gst], how="horizontal")
    # space weather: as-of pred_time (join on the window containing pred_time)
    sw = load_sw_asof(lake, out["pred_time"].min(), out["pred_time"].max()
                      + timedelta(minutes=15))
    out = (out.sort("pred_time")
           .join_asof(sw.rename({"window_start": "pred_time"}).sort("pred_time"),
                      on="pred_time", strategy="backward")
           .with_columns(
               pl.col("xray_flux").clip(lower_bound=1e-9).log10()
               .alias("xray_flux_log10"))
           .drop("xray_flux"))
    # AR history: as-of pred_time with Delta_avail inside ar_features
    out = ar_features(lake, out, "pred_time").rename({"_pt": "pred_time"}, strict=False)
    if horizon_s > 10800:  # yday1h reach would exceed the CV gap at long horizons
        out = out.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("ar_cell_n_yday1h"),
            pl.lit(None, dtype=pl.Float64).alias("ar_cell_snr_yday1h"))
    return out
```

Performance note for the executor: `_geom_solar_time_row` is per-row Python. Fine for month-scale label sets on two bands (~10^5–10^6 rows: minutes). If it becomes the bottleneck, memoize on `(tx_field, rx_field)` for geometry and `(field-pair, window_start)` for solar — both are massively repeated. Do not restructure the interface.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_matrix.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/matrix.py tests/features/test_matrix.py tests/features/conftest.py
git commit -m "feat(features): full 69-column feature matrix with canonical FEATURE_COLUMNS"
```

---

### Task 7: Leakage audit (`tests/test_leakage.py`)

**Files:**
- Create: `tests/test_leakage.py`

**Interfaces:**
- Consumes: `ar_features`, `DELTA_AVAIL_MIN`, `MAX_AR_LOOKBACK_H` (Task 5); `GAP_HOURS`, `blocked_folds` (M0 `eval/splits.py`); `FEATURE_COLUMNS` (Task 6); `SW_FEATURE_COLUMNS` (Task 4).
- Produces: the standing leakage regression suite — any future feature change that widens temporal reach or smuggles a definitive series breaks the build.

- [ ] **Step 1: Write the tests** (these should PASS immediately if Tasks 4–6 were done right — they are regression armor, written as their own task because SPEC §6 makes leakage a first-class deliverable)

```python
# tests/test_leakage.py
"""Standing leakage audit (SPEC-labeling section 6; REVIEW-FINDINGS R1).
If any test here fails after a feature change, the change leaks."""
from datetime import date, datetime, timedelta, timezone
import polars as pl
from propagation.data.lake import Lake
from propagation.eval.splits import GAP_HOURS, blocked_folds
from propagation.features.history import (
    DELTA_AVAIL_MIN, MAX_AR_LOOKBACK_H, ar_features,
)
from propagation.features.matrix import FEATURE_COLUMNS
from propagation.features.spaceweather import SW_FEATURE_COLUMNS

UTC = timezone.utc
MAX_HORIZON_H = 24  # v1 canonical horizon set tops out at 86400 s

def _mk_lake(tmp_path, spot_ts_list):
    lake = Lake(tmp_path / "lake")
    rows = [{
        "source": "wsprnet", "ts": ts, "band": "20m", "mode": "FT8",
        "freq_hz": 14074000, "dx_call": "K5ARH", "de_call": "DL1ABC",
        "dx_grid": None, "de_grid": None, "dx_lat": None, "dx_lon": None,
        "de_lat": None, "de_lon": None, "snr_db": -10, "tx_dbm": None,
        "distance_km": 8000.0, "bearing_deg": 45.0, "mode_class": "digi",
        "dx_field": "EM", "de_field": "JN", "dx_grid4": None, "de_grid4": None,
        "window_start": ts.replace(minute=(ts.minute // 15) * 15, second=0),
    } for ts in spot_ts_list]
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")),
        pl.col("window_start").cast(pl.Datetime("us", "UTC")))
    for d in sorted({ts.date() for ts in spot_ts_list}):
        part = df.filter(pl.col("ts").dt.date() == d)
        lake.write_parquet(part, lake.spots_q_dir("20m", d))
    return lake

def test_poisoned_spot_inside_buffer_has_no_effect(tmp_path):
    """A spot 1 second inside the Delta_avail forbidden zone must be invisible."""
    t = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    clean = _mk_lake(tmp_path / "a", [t - timedelta(minutes=30)])
    poisoned = _mk_lake(tmp_path / "b", [
        t - timedelta(minutes=30),
        t - timedelta(minutes=DELTA_AVAIL_MIN, seconds=-1),  # inside the buffer
    ])
    cells = pl.DataFrame({"tx_field": ["EM"], "rx_field": ["JN"],
                          "band": ["20m"], "pred_time": [t]}) \
        .with_columns(pl.col("pred_time").cast(pl.Datetime("us", "UTC")))
    a = ar_features(clean, cells, "pred_time").drop("pred_time")
    b = ar_features(poisoned, cells, "pred_time").drop("pred_time")
    assert a.equals(b), "spot inside availability buffer leaked into features"

def test_spot_just_outside_buffer_does_count(tmp_path):
    """Complement: the buffer must not be wider than 5 min either."""
    t = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    lake = _mk_lake(tmp_path, [t - timedelta(minutes=DELTA_AVAIL_MIN, seconds=1)])
    cells = pl.DataFrame({"tx_field": ["EM"], "rx_field": ["JN"],
                          "band": ["20m"], "pred_time": [t]}) \
        .with_columns(pl.col("pred_time").cast(pl.Datetime("us", "UTC")))
    assert ar_features(lake, cells, "pred_time")["ar_cell_n_15m"][0] == 1

def test_cv_gap_covers_horizon_plus_lookback():
    """SPEC 6.1: gap >= max horizon + max AR lookback. Lengthen a lookback or
    add a horizon and this fails until GAP_HOURS is widened in eval/splits.py."""
    assert GAP_HOURS >= MAX_HORIZON_H + MAX_AR_LOOKBACK_H + 0.5  # +0.5: yday window edge

def test_blocked_folds_honor_gap():
    folds = blocked_folds(datetime(2026, 1, 1, tzinfo=UTC),
                          datetime(2026, 7, 1, tzinfo=UTC), eval_days=14)
    assert folds, "no folds produced"
    for f in folds:
        assert (f.eval_start - f.train_end) >= timedelta(hours=GAP_HOURS)

def test_no_definitive_series_in_features():
    assert "kp_def" not in FEATURE_COLUMNS
    assert "kp_def" not in SW_FEATURE_COLUMNS
    assert all(not c.endswith("_def") for c in FEATURE_COLUMNS)

def test_feature_columns_contain_no_label_columns():
    forbidden = {"open", "n_spots", "n_monitors", "n_tx_stations",
                 "evidence_tier", "snr_ft8eq_p50", "sample_weight", "split_tag"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_leakage.py -v`
Expected: 6 passed. If `test_poisoned_spot…` fails, Task 5's window predicates are wrong — fix `history.py`, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_leakage.py
git commit -m "test: standing leakage audit (availability buffer, CV gap, definitive-series quarantine)"
```

---

### Task 8: LightGBM openness model (`models/gbt.py`)

**Files:**
- Create: `src/propagation/models/gbt.py`
- Test: `tests/models/test_gbt.py`
- Modify: `pyproject.toml` (dep)

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, `feature_matrix` (Task 6); `OpennessModel` protocol (M0 `models/base.py`); labels frame with `open`, `sample_weight`.
- Produces (pinned in INTERFACES.md):

```python
class GBTModel:                # implements OpennessModel
    model_id: str              # f"gbt-h{horizon_s}"
    def __init__(self, horizon_s: int): ...
    def fit(self, labels: pl.DataFrame, features: pl.DataFrame) -> "GBTModel"
    def predict_p_open(self, cells: pl.DataFrame) -> pl.DataFrame
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> "GBTModel"
```

`fit(labels, features)`: `features` is the output of `feature_matrix` over the (sampled) training labels — it already carries `open` and `sample_weight`; the separate `labels` arg is accepted for protocol symmetry and only `features` is consumed. Time-ordered split inside fit: last 15% of rows by `pred_time` = validation tail, used for LightGBM early stopping AND the isotonic calibrator; first 85% trains the booster. **`sample_weight` goes to booster AND calibrator** (SPEC §4.5, normative). `predict_p_open(cells)` requires `cells` to already be a feature frame (same columns); rows with any missing FEATURE_COLUMNS raise; output = input keys + `p_open` Float64.

- [ ] **Step 1: Add dependency**

Run: `uv add lightgbm`
Expected: `lightgbm` added to `[project.dependencies]`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/models/test_gbt.py
from datetime import datetime, timedelta, timezone
import numpy as np
import polars as pl
import pytest
from propagation.features.matrix import FEATURE_COLUMNS
from propagation.models.gbt import GBTModel

UTC = timezone.utc

def _synthetic(n=4000, seed=7):
    """Learnable toy: open iff sza_mid < 90 with noise; other cols random."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    data = {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    data["sza_mid"] = rng.uniform(0, 180, n)
    y = (data["sza_mid"] < 90).astype(np.int8)
    flip = rng.random(n) < 0.1
    y[flip] = 1 - y[flip]
    df = pl.DataFrame(data).with_columns(
        pl.Series("open", y),
        pl.Series("sample_weight", np.where(y == 1, 1.0, 3.0)),
        pl.Series("pred_time", [t0 + timedelta(minutes=15 * i) for i in range(n)])
        .cast(pl.Datetime("us", "UTC")),
        pl.Series("window_start", [t0 + timedelta(minutes=15 * i) for i in range(n)])
        .cast(pl.Datetime("us", "UTC")),
        pl.lit("EM").alias("tx_field"), pl.lit("JN").alias("rx_field"),
        pl.lit("20m").alias("band"),
    )
    return df

def test_fit_predict_learns_signal():
    df = _synthetic()
    m = GBTModel(horizon_s=0).fit(df, df)
    out = m.predict_p_open(df)
    assert "p_open" in out.columns
    p = out["p_open"].to_numpy()
    assert np.all((p >= 0) & (p <= 1))
    day = df["sza_mid"].to_numpy() < 90
    assert p[day].mean() > p[~day].mean() + 0.3

def test_model_id():
    assert GBTModel(horizon_s=10800).model_id == "gbt-h10800"

def test_save_load_roundtrip(tmp_path):
    df = _synthetic(n=2000)
    m = GBTModel(horizon_s=0).fit(df, df)
    m.save(tmp_path / "m")
    m2 = GBTModel.load(tmp_path / "m")
    a = m.predict_p_open(df)["p_open"].to_numpy()
    b = m2.predict_p_open(df)["p_open"].to_numpy()
    assert np.allclose(a, b)
    assert m2.model_id == "gbt-h0"

def test_missing_feature_column_raises():
    df = _synthetic(n=500)
    m = GBTModel(horizon_s=0).fit(df, df)
    with pytest.raises(ValueError, match="ar_cell_n_15m"):
        m.predict_p_open(df.drop("ar_cell_n_15m"))

def test_sample_weight_affects_calibration():
    """Weighted vs unweighted fits must differ — guards the SPEC 4.5 rule."""
    df = _synthetic()
    m_w = GBTModel(horizon_s=0).fit(df, df)
    m_u = GBTModel(horizon_s=0).fit(
        df, df.with_columns(pl.lit(1.0).alias("sample_weight")))
    a = m_w.predict_p_open(df)["p_open"].mean()
    b = m_u.predict_p_open(df)["p_open"].mean()
    # 3x-weighted negatives pull predicted base rate down
    assert a < b
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_gbt.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement**

```python
# src/propagation/models/gbt.py
"""LightGBM openness model + weighted isotonic calibration.

SPEC-labeling 4.5 (normative): sample_weight feeds BOTH the booster and the
calibrator — per-stratum sampling rates vary, an unweighted fit miscalibrates.
Early stopping and calibration use a time-tail slice of TRAIN, never eval.
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from propagation.features.matrix import FEATURE_COLUMNS

PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "verbosity": -1, "seed": 20260709,
}
TAIL_FRAC = 0.15
MAX_ROUNDS = 2000

class GBTModel:
    def __init__(self, horizon_s: int):
        self.horizon_s = horizon_s
        self.model_id = f"gbt-h{horizon_s}"
        self._booster: lgb.Booster | None = None
        self._cal: IsotonicRegression | None = None

    def _xyw(self, df: pl.DataFrame):
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        x = df.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)
        y = df["open"].to_numpy() if "open" in df.columns else None
        w = df["sample_weight"].to_numpy() if "sample_weight" in df.columns else None
        return x, y, w

    def fit(self, labels: pl.DataFrame, features: pl.DataFrame) -> "GBTModel":
        df = features.sort("pred_time")
        cut = int(df.height * (1 - TAIL_FRAC))
        head, tail = df.head(cut), df.tail(df.height - cut)
        xh, yh, wh = self._xyw(head)
        xt, yt, wt = self._xyw(tail)
        train = lgb.Dataset(xh, label=yh, weight=wh,
                            feature_name=FEATURE_COLUMNS)
        valid = lgb.Dataset(xt, label=yt, weight=wt, reference=train)
        self._booster = lgb.train(
            PARAMS, train, num_boost_round=MAX_ROUNDS, valid_sets=[valid],
            callbacks=[lgb.early_stopping(50, verbose=False)])
        raw_tail = self._booster.predict(xt)
        self._cal = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip")
        self._cal.fit(raw_tail, yt, sample_weight=wt)
        return self

    def predict_p_open(self, cells: pl.DataFrame) -> pl.DataFrame:
        x, _, _ = self._xyw(cells)
        p = self._cal.predict(self._booster.predict(x))
        return cells.with_columns(pl.Series("p_open", p, dtype=pl.Float64))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path / "booster.txt"))
        (path / "calibrator.pkl").write_bytes(pickle.dumps(self._cal))
        (path / "meta.json").write_text(json.dumps(
            {"model_id": self.model_id, "horizon_s": self.horizon_s,
             "feature_columns": FEATURE_COLUMNS}))

    @classmethod
    def load(cls, path: Path) -> "GBTModel":
        meta = json.loads((path / "meta.json").read_text())
        if meta["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("artifact feature list != current FEATURE_COLUMNS; retrain")
        m = cls(horizon_s=meta["horizon_s"])
        m._booster = lgb.Booster(model_file=str(path / "booster.txt"))
        m._cal = pickle.loads((path / "calibrator.pkl").read_bytes())
        return m
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_gbt.py -v`
Expected: 5 passed (~30–60 s; the synthetic fit is small)

- [ ] **Step 6: Commit**

```bash
git add src/propagation/models/gbt.py tests/models/test_gbt.py pyproject.toml uv.lock
git commit -m "feat(models): LightGBM openness model with weighted isotonic calibration"
```

---

### Task 9: Training/eval driver (`train-gbt` CLI)

**Files:**
- Create: `src/propagation/eval/train_gbt.py`
- Test: `tests/eval/test_train_gbt.py`
- Modify: `pyproject.toml` (console script)

**Interfaces:**
- Consumes: `blocked_folds`, `Fold`, `GAP_HOURS` (M0); `sample_training` (M0 labels); `feature_matrix` (Task 6); `GBTModel` (Task 8); `ClimatologyModel` (M0); `P533Model` (M1); `headline_table`, `reliability_diagram` (M0 `eval/report.py`); `load_kp_def` (Task 4).
- Produces: `run(cfg: TrainConfig) -> dict` and `main() -> None` (CLI `train-gbt`); writes `reports/headline.md`, `reports/headline.json` (machine-readable, consumed by Task 10), `reports/reliability_gbt_h{H}_{band}.png`, model artifacts under `data/models/gbt-h{H}/`.

```python
@dataclass(frozen=True)
class TrainConfig:
    lake_root: Path
    bands: tuple[str, ...] = ("20m", "10m")
    horizons_s: tuple[int, ...] = (0, 10800)
    train_start: datetime = ...   # required, CLI arg
    eval_end: datetime = ...      # required, CLI arg
    eval_days: int = 30           # per fold
    min_eval_folds: int = 3       # >= 3 held-out months
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("data/models")
    ssn_by_month: dict[str, float] = ...  # for P533Model, CLI --ssn-file JSON
```

**Driver semantics (normative):**
1. Folds: `blocked_folds(train_start, eval_end, eval_days=cfg.eval_days)`; abort unless `len(folds) >= min_eval_folds`.
2. **Storm-fold guard:** using `load_kp_def` (definitive series, allowed for stratification), a fold is *storm* iff any 3-h Kp ≥ 5 inside its eval span. Abort with `SystemExit("no storm fold in span — widen the date range")` if none (SPEC §6.5).
3. Per fold × band × horizon: train labels = `sample_training` over the fold's train span (3:1 sampled); eval labels = FULL unsampled label set over the eval span. Features via `feature_matrix(lake, labels, horizon_s)` for both.
4. Models scored per fold: `climatology` (fit on train-fold labels only — leakage rule §6.2), `p533` (no fit), `gbt-h{H}` (fit on train-fold features). All scored through `predict_p_open` on eval cells joined to eval features.
5. Metrics: unweighted Brier/log-loss/PR-AUC on the full eval set (eval is never sampled, so no weights — assert `(eval_labels["sample_weight"] == 1.0).all()`).
6. Aggregation: mean over folds; slices by band, by horizon, by Kp regime (storm/quiet fold-days via definitive Kp).
7. Output: `headline.md` (3 model rows × metric cols, per band-group + slices), `headline.json`:

```json
{"folds": 3, "storm_folds": 1,
 "results": [{"model": "gbt-h0", "band": "20m", "horizon_s": 0,
              "brier": 0.081, "log_loss": 0.27, "pr_auc": 0.91,
              "brier_storm": 0.11, "brier_quiet": 0.07}, ...]}
```

- [ ] **Step 1: Write the failing test** (wiring test on a miniature lake — the fixture reuses `tests/features/conftest.py`'s `lake_full` builder extended to 8 days of synthetic spots/labels/space-weather so 1 tiny fold exists; assert the driver produces headline.json with the three model rows and refuses a span with no storm)

```python
# tests/eval/test_train_gbt.py
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from propagation.eval.train_gbt import TrainConfig, run

UTC = timezone.utc

def test_run_produces_three_model_rows(mini_lake, tmp_path):
    cfg = TrainConfig(
        lake_root=mini_lake.root,
        bands=("20m",), horizons_s=(0,),
        train_start=datetime(2026, 5, 1, tzinfo=UTC),
        eval_end=datetime(2026, 5, 9, tzinfo=UTC),
        eval_days=1, min_eval_folds=1,
        reports_dir=tmp_path / "reports", models_dir=tmp_path / "models",
        ssn_by_month={"2026-05": 120.0},
    )
    out = run(cfg)
    hj = json.loads((tmp_path / "reports" / "headline.json").read_text())
    models = {r["model"] for r in hj["results"]}
    assert {"climatology", "p533", "gbt-h0"} <= models
    assert (tmp_path / "reports" / "headline.md").exists()

def test_run_aborts_without_storm_fold(mini_lake_quiet, tmp_path):
    cfg = TrainConfig(
        lake_root=mini_lake_quiet.root, bands=("20m",), horizons_s=(0,),
        train_start=datetime(2026, 5, 1, tzinfo=UTC),
        eval_end=datetime(2026, 5, 9, tzinfo=UTC),
        eval_days=1, min_eval_folds=1,
        reports_dir=tmp_path / "r", models_dir=tmp_path / "m",
        ssn_by_month={"2026-05": 120.0},
    )
    with pytest.raises(SystemExit, match="storm"):
        run(cfg)
```

`mini_lake` fixture (in `tests/eval/conftest.py`): 8 days × 20m of synthetic qualified spots (a few EM→JN spots per daytime window), labels built with M0's `build_labels` machinery over those spots, and space-weather partitions where day 6 carries `kp_def = 6.0` (storm); `mini_lake_quiet` identical with `kp_def = 2.0` everywhere. Write the fixture by calling M0's own builders — do not hand-author label parquet.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_train_gbt.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `train_gbt.py`** — structure (complete driver; heavy lifting is all calls into already-tested modules):

```python
# src/propagation/eval/train_gbt.py
"""Blocked-CV training/eval driver. Extends the headline table to
{climatology, p533, gbt}. Storm-fold guard per SPEC-labeling 6.5."""
import argparse, json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from propagation.data.lake import Lake
from propagation.eval import metrics
from propagation.eval.report import headline_table, reliability_diagram
from propagation.eval.splits import blocked_folds
from propagation.features.labels import sample_training
from propagation.features.matrix import feature_matrix
from propagation.features.spaceweather import load_kp_def
from propagation.models.climatology import ClimatologyModel
from propagation.models.gbt import GBTModel
from propagation.models.p533_baseline import P533Model

@dataclass(frozen=True)
class TrainConfig:
    lake_root: Path
    train_start: datetime
    eval_end: datetime
    bands: tuple[str, ...] = ("20m", "10m")
    horizons_s: tuple[int, ...] = (0, 10800)
    eval_days: int = 30
    min_eval_folds: int = 3
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("data/models")
    ssn_by_month: dict = field(default_factory=dict)

def _load_labels(lake: Lake, band: str, start: datetime, end: datetime) -> pl.DataFrame:
    con = lake.connect()
    return con.execute(
        "SELECT * FROM labels WHERE band = ? AND window_start >= ? AND window_start < ?",
        [band, start, end]).pl().with_columns(
        pl.col("window_start").cast(pl.Datetime("us", "UTC")))

def _fold_is_storm(lake: Lake, f) -> bool:
    kd = load_kp_def(lake, f.eval_start, f.eval_end)
    return bool((kd["kp_def"] >= 5.0).any())

def run(cfg: TrainConfig) -> dict:
    lake = Lake(cfg.lake_root)
    folds = blocked_folds(cfg.train_start, cfg.eval_end, eval_days=cfg.eval_days)
    if len(folds) < cfg.min_eval_folds:
        raise SystemExit(f"only {len(folds)} folds; need {cfg.min_eval_folds}")
    storm = [_fold_is_storm(lake, f) for f in folds]
    if not any(storm):
        raise SystemExit("no storm fold in span — widen the date range (SPEC 6.5)")
    results = []
    for f, is_storm in zip(folds, storm):
        for band in cfg.bands:
            train_lbl_full = _load_labels(lake, band, f.train_start, f.train_end)
            eval_lbl = _load_labels(lake, band, f.eval_start, f.eval_end)
            assert (eval_lbl["sample_weight"] == 1.0).all(), "eval must be unsampled"
            train_lbl = pl.concat([
                sample_training(train_lbl_full.filter(
                    pl.col("window_start").dt.date() == d), band, d)
                for d in sorted(train_lbl_full["window_start"].dt.date().unique())
            ])
            clim = ClimatologyModel().fit(train_lbl)
            p533 = P533Model(ssn_by_month=cfg.ssn_by_month)
            for h in cfg.horizons_s:
                feats_tr = feature_matrix(lake, train_lbl, h)
                feats_ev = feature_matrix(lake, eval_lbl, h)
                gbt = GBTModel(horizon_s=h).fit(train_lbl, feats_tr)
                gbt.save(cfg.models_dir / f"{gbt.model_id}-{band}-fold{f.eval_start:%Y%m%d}")
                for name, pred in [
                    ("climatology", clim.predict_p_open(eval_lbl)),
                    ("p533", p533.predict_p_open(eval_lbl)),
                    (gbt.model_id, gbt.predict_p_open(feats_ev)),
                ]:
                    joined = pred.drop_nulls("p_open")
                    y = joined["open"].to_numpy()
                    p = joined["p_open"].to_numpy()
                    results.append({
                        "model": name, "band": band, "horizon_s": h,
                        "fold": f.eval_start.date().isoformat(), "storm": is_storm,
                        "brier": metrics.brier(y, p),
                        "log_loss": metrics.log_loss(y, p),
                        "pr_auc": metrics.pr_auc(y, p),
                        "n": int(joined.height),
                    })
    return _write_reports(cfg, results)

def _write_reports(cfg: TrainConfig, results: list[dict]) -> dict:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(results)
    agg = (df.group_by(["model", "band", "horizon_s"])
           .agg(pl.col("brier").mean(), pl.col("log_loss").mean(),
                pl.col("pr_auc").mean(),
                pl.col("brier").filter(pl.col("storm")).mean().alias("brier_storm"),
                pl.col("brier").filter(~pl.col("storm")).mean().alias("brier_quiet"))
           .sort(["band", "horizon_s", "model"]))
    doc = {"folds": df["fold"].n_unique(),
           "storm_folds": int(df.filter(pl.col("storm"))["fold"].n_unique()),
           "results": agg.to_dicts()}
    (cfg.reports_dir / "headline.json").write_text(json.dumps(doc, indent=2))
    lines = ["# Headline — climatology vs P.533 vs GBT", "",
             agg.to_pandas().to_markdown(index=False)]
    (cfg.reports_dir / "headline.md").write_text("\n".join(lines))
    return doc

def main() -> None:
    ap = argparse.ArgumentParser("train-gbt")
    ap.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--bands", default="20m,10m")
    ap.add_argument("--horizons-s", default="0,10800")
    ap.add_argument("--eval-days", type=int, default=30)
    ap.add_argument("--ssn-file", type=Path, required=True,
                    help='JSON {"YYYY-MM": ssn}')
    a = ap.parse_args()
    cfg = TrainConfig(
        lake_root=a.lake_root,
        train_start=datetime.fromisoformat(a.train_start).replace(tzinfo=timezone.utc),
        eval_end=datetime.fromisoformat(a.eval_end).replace(tzinfo=timezone.utc),
        bands=tuple(a.bands.split(",")),
        horizons_s=tuple(int(x) for x in a.horizons_s.split(",")),
        eval_days=a.eval_days,
        ssn_by_month=json.loads(a.ssn_file.read_text()))
    doc = run(cfg)
    print(json.dumps(doc, indent=2)[:2000])
```

Also write per-model reliability diagrams inside `_write_reports` (one call to M0's `reliability_diagram` per (gbt model, band) over pooled eval predictions — pass pooled `y`/`p` arrays collected in `results` via a side list; executor wires this with a `raw` accumulator alongside `results`).

- [ ] **Step 4: Register console script** — `pyproject.toml`: `train-gbt = "propagation.eval.train_gbt:main"`

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_train_gbt.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/propagation/eval/train_gbt.py tests/eval/ pyproject.toml
git commit -m "feat(eval): blocked-CV train-gbt driver with storm-fold guard, 3-row headline"
```

---

### Task 10: Acceptance gate (`check-m2-gate`)

**Files:**
- Create: `src/propagation/eval/m2_gate.py`
- Test: `tests/eval/test_m2_gate.py`
- Modify: `pyproject.toml` (console script)

**Interfaces:**
- Consumes: `reports/headline.json` (Task 9 format).
- Produces: `check(headline: dict, bands: tuple[str, ...] = ("20m", "10m")) -> list[str]` (empty = pass) and `main() -> None` (CLI `check-m2-gate`, exit 0/1). This is ROADMAP M2's "if it doesn't, stop and diagnose" made mechanical: **M3+ work must not start until this exits 0.**

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_m2_gate.py
from propagation.eval.m2_gate import check

def _row(model, band, brier, ll):
    return {"model": model, "band": band, "horizon_s": 0,
            "brier": brier, "log_loss": ll, "pr_auc": 0.9,
            "brier_storm": brier, "brier_quiet": brier}

def test_pass_when_gbt_beats_both_on_both_bands():
    hj = {"results": [
        _row("climatology", "20m", 0.10, 0.35), _row("p533", "20m", 0.12, 0.40),
        _row("gbt-h0", "20m", 0.08, 0.28),
        _row("climatology", "10m", 0.15, 0.45), _row("p533", "10m", 0.16, 0.50),
        _row("gbt-h0", "10m", 0.11, 0.36),
    ]}
    assert check(hj) == []

def test_fail_lists_every_violation():
    hj = {"results": [
        _row("climatology", "20m", 0.07, 0.25), _row("p533", "20m", 0.12, 0.40),
        _row("gbt-h0", "20m", 0.08, 0.28),      # loses to climatology on both
        _row("climatology", "10m", 0.15, 0.45), _row("p533", "10m", 0.10, 0.30),
        _row("gbt-h0", "10m", 0.11, 0.36),      # loses to p533 on both
    ]}
    failures = check(hj)
    assert len(failures) == 4
    assert any("20m" in f and "climatology" in f and "brier" in f for f in failures)

def test_missing_band_is_a_failure():
    hj = {"results": [_row("climatology", "20m", 0.1, 0.3),
                      _row("p533", "20m", 0.1, 0.3), _row("gbt-h0", "20m", 0.05, 0.2)]}
    assert any("10m" in f for f in check(hj))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_m2_gate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
# src/propagation/eval/m2_gate.py
"""ROADMAP M2 acceptance gate: GBT must beat climatology AND P.533 on Brier
and log-loss at h=0 on 20m and 10m. Nonzero exit = stop and diagnose;
do not proceed to M3/serving."""
import argparse, json, sys
from pathlib import Path

DIAGNOSIS = """
M2 GATE FAILED — stop and diagnose before any M3/serving work (ROADMAP M2):
  1. Calibration: inspect reports/reliability_gbt_h0_{band}.png — a bowed curve
     means the isotonic tail slice is too small or weights are wrong.
  2. Feature importances: booster.feature_importance() — if AR features
     dominate at h=0 but the model still loses, suspect the labels join;
     if space weather is ~0, check the as-of join produced non-null columns.
  3. Per-slice deltas: compare brier_storm vs brier_quiet in headline.json —
     losing only in quiet conditions points at climatology-shaped gaps
     (seasonality features), losing in storms at Kp/lag features.
  4. Verify eval used the FULL unsampled label set (driver asserts it).
  5. Re-read docs/REVIEW-FINDINGS.md R1-R4 before changing any rule.
"""

def check(headline: dict, bands: tuple[str, ...] = ("20m", "10m")) -> list[str]:
    rows = [r for r in headline["results"] if r["horizon_s"] == 0]
    failures: list[str] = []
    for band in bands:
        by_model = {r["model"]: r for r in rows if r["band"] == band}
        gbt = by_model.get("gbt-h0")
        if gbt is None:
            failures.append(f"{band}: no gbt-h0 row in headline.json")
            continue
        for base in ("climatology", "p533"):
            if base not in by_model:
                failures.append(f"{band}: no {base} row in headline.json")
                continue
            for metric in ("brier", "log_loss"):
                if not gbt[metric] < by_model[base][metric]:
                    failures.append(
                        f"{band}: gbt-h0 {metric}={gbt[metric]:.4f} does not beat "
                        f"{base} {metric}={by_model[base][metric]:.4f}")
    return failures

def main() -> None:
    ap = argparse.ArgumentParser("check-m2-gate")
    ap.add_argument("--headline", type=Path, default=Path("reports/headline.json"))
    a = ap.parse_args()
    failures = check(json.loads(a.headline.read_text()))
    if failures:
        print("\n".join(failures))
        print(DIAGNOSIS)
        sys.exit(1)
    print("M2 gate PASSED: GBT beats climatology and P.533 on Brier and "
          "log-loss at h=0 on 20m and 10m.")
```

- [ ] **Step 4: Register console script** — `pyproject.toml`: `check-m2-gate = "propagation.eval.m2_gate:main"`

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_m2_gate.py -v`
Expected: 3 passed

- [ ] **Step 6: Full suite + lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all tests pass, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/propagation/eval/m2_gate.py tests/eval/test_m2_gate.py pyproject.toml
git commit -m "feat(eval): mechanical M2 acceptance gate (check-m2-gate)"
```

---

### Task 11: The real M2 run (manual, documented)

**Files:**
- Create: `docs/RESULTS-m2.md` (filled from real output)

No new code. Operator checklist:

- [ ] **Step 1:** Ensure lake coverage: ≥4 months of 20m+10m labels (M0 pipeline over WSPRnet months) and space weather for the same span (`extract-swpc` accumulation + GFZ Kp backfill + F10.7 history via the swpc parsers).
- [ ] **Step 2:** `uv run train-gbt --train-start <T0> --eval-end <T1> --ssn-file data/ssn.json` with a span whose held-out months include ≥1 storm (check https://kp.gfz-potsdam.de for Kp≥5 days first). Expected: driver completes, prints headline JSON; otherwise it aborts with the fold/storm guard message.
- [ ] **Step 3:** `uv run check-m2-gate` — expected exit 0. If exit 1: follow the printed diagnosis checklist; do NOT start M3.
- [ ] **Step 4:** Copy `reports/headline.md` + reliability PNG references into `docs/RESULTS-m2.md` with the exact command lines used; commit.

```bash
git add docs/RESULTS-m2.md
git commit -m "docs: M2 headline results (climatology vs P.533 vs GBT)"
```

---

## Self-Review (performed)

- **Spec coverage:** ARCHITECTURE §4 items 1–6 → Tasks 1 (geometry), 2 (solar), 9/6 (time features in matrix), 3+4 (space weather), 5 (AR history), 6 (mode normalization reused from M0's `snr_ft8eq` via SQL mirror in Task 5 — flagged: the SQL must stay in lockstep with `labels.snr_ft8eq`; a drift test is included in Task 5's test via known offsets). ROADMAP M2 acceptance → Tasks 9–11. SPEC §6 leakage rules 0,1,2,4,5 → Tasks 5, 7, 9 (rule 3 is M0's). REVIEW-FINDINGS R1 → Tasks 5+7.
- **Placeholder scan:** one intentional deferred check remains — GFZ Kp column layout (Task 3 Step 7/8) is explicitly verify-at-execution against the live file header, with the verification step spelled out. Task 5 contains an explicitly marked dead snippet to delete (`enriched =`) — called out in its NOTE. Task 9 Step 3 delegates reliability-PNG wiring to the executor with the exact mechanism named (pooled y/p accumulator) — acceptable as it composes two already-specified functions.
- **Type consistency:** `ar_features` output columns (Task 5) == the 34 AR names in `FEATURE_COLUMNS` (Task 6); `SW_FEATURE_COLUMNS` (Task 4) == the 11 SW names except `xray_flux` → `xray_flux_log10` transform happens in `matrix.py` (Task 6 drops raw `xray_flux` after deriving). `GBTModel` matches the INTERFACES.md pin verbatim. `TrainConfig`/`headline.json` field names consumed by Task 10 match Task 9's writer.


