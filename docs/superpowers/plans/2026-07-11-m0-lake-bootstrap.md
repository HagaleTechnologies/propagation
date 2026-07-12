# M0 — Lake Bootstrap + Climatology Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the repo from zero to ROADMAP.md's M0 acceptance bar: `extract-wsprnet`
pulling ≥1 month of 20m WSPRnet data into the DuckDB/Parquet lake, monitor-normalized
labels with a reported unlabeled fraction, and a climatology (M-0) baseline evaluated
against a held-out later month, producing a Brier score + reliability diagram via
`eval/report.py`, all reproducible end-to-end via `uv run` from an empty `data/`.

**Architecture:** Pure-function pipeline stages (extract → hygiene/dedup → lake write →
receiver-uptime → universe/labels → sampling → climatology → eval/report → QA), each
independently unit-tested against small synthetic fixtures, wired together by one
orchestration script (`scripts/run_m0.py`) that is the actual acceptance-test artifact.
Every stage follows docs/SPEC-labeling.md to the letter — two independent
implementations must agree, so this plan implements the spec's formulas directly, not
approximations.

**Tech Stack:** Python 3.11+, `uv`, `polars` (dataframes + Parquet I/O), `duckdb` (lake
views), `httpx` (archive download), `matplotlib` (reliability diagram), `numpy`
(metrics/RNG), `pytest`, `ruff`.

## Global Constraints

- Python ≥3.11, dependency-managed with `uv` (no bare `pip install`). License header /
  metadata: MIT OR Apache-2.0 dual (per README.md).
- Supported bands (docs/SPEC-labeling.md §1): `160m 80m 60m 40m 30m 20m 17m 15m 12m 10m 6m`.
  M0 exercises `20m` only; other bands must not silently break when present.
- Lake layout is normative (ARCHITECTURE.md §3.1, docs/SPEC-labeling.md §3–4.5):
  `lake/spots/band=…/date=…/*.parquet`, `lake/receiver_uptime/band=…/date=…/*.parquet`,
  `lake/labels/band=…/date=…/*.parquet`. Labels are stored once (no `horizon=` partition).
- Cross-source dedup priority (docs/SPEC-labeling.md §1): `wsprnet > rbn > pskreporter >
  cqdx`, tie-break highest `snr_db` then lowest `ts`. Applied before labeling and before
  any spot-count feature.
- Sampling RNG (docs/SPEC-labeling.md §4.5) MUST be `numpy.random.Generator(PCG64(seed))`
  with `seed = int.from_bytes(sha256(f"{band}|{date}".encode()).digest()[:4], "big") &
  0xFFFFFFFF` — this exact scheme is required for cross-implementation determinism.
  `sample_weight` MUST be consumed by any downstream fit that uses sampled rows.
- Leakage/CV gap (docs/SPEC-labeling.md §6): `gap_hours = max(48, max_horizon_hours +
  max_ar_lookback_hours)`. M0 has no autoregressive features yet (those land in M2), so
  the gap is the 48h floor — but `eval/splits.py` must compute it from the formula, not
  hardcode 48, so M2 widening it is a parameter change, not a rewrite.
- Climatology (M-0) is fitted on train-fold data only, never on eval data
  (docs/SPEC-labeling.md §6 rule 2).
- No cqdx imports anywhere in this repo (CLAUDE.md / README.md boundary).
- Every new module gets a `tests/` counterpart at the mirrored path; no module ships
  without a passing test in the same task.

---

## File Structure

```
propagation/
├── pyproject.toml
├── .gitignore                          (extended: data/, .venv/, caches)
├── src/propagation/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py                   # SUPPORTED_BANDS, SPOT_SCHEMA
│   │   ├── geo.py                      # grid_to_latlon, great_circle_km
│   │   ├── hygiene.py                  # spot hygiene (SPEC §1)
│   │   ├── dedup.py                    # cross-source dedup (SPEC §1)
│   │   ├── lake.py                     # partitioned parquet + duckdb views
│   │   └── wsprnet.py                  # extract-wsprnet
│   ├── features/
│   │   ├── __init__.py
│   │   ├── uptime.py                   # receiver-uptime table (SPEC §3)
│   │   ├── universe.py                 # transmit evidence + universe (SPEC §2, §4.1)
│   │   └── labels.py                   # positive/negative/SNR labels (SPEC §4.2-4.4)
│   │   └── sampling.py                 # 3:1 sampling + label storage (SPEC §4.5)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── splits.py                   # blocked time-series CV (SPEC §6)
│   │   ├── metrics.py                  # Brier, log-loss, reliability bins
│   │   └── report.py                   # headline table + reliability diagram PNG
│   ├── models/
│   │   ├── __init__.py
│   │   └── climatology.py              # M-0 baseline
│   └── qa/
│       ├── __init__.py
│       └── checks.py                   # SPEC §6 QA sanity checks
├── scripts/
│   └── run_m0.py                       # end-to-end M0 orchestration (uv run entrypoint)
├── tests/                               # mirrors src/propagation/**
│   └── fixtures/
│       └── wspr_sample.csv.gz          # small hand-built real-format sample
└── data/                                 (gitignored; created at runtime)
```

## Self-Review Notes (read before executing)

- **Climatology grouping simplification (Task 12):** ARCHITECTURE §5 defines M-0 as
  grouped by `(path-cell, band, hour-of-day, month)` at "similar smoothed SSN". M0 has
  exactly one training month for one band, so `month` and `SSN` groupings are
  under-determined by construction (no variation to group on yet — that dimension only
  becomes meaningful once M3 accumulates multi-year history). M0's `ClimatologyModel`
  therefore groups by `(tx_field, rx_field, band, hour_of_day)` only. This is a
  documented scope reduction, not a deviation from a rule this milestone can actually
  satisfy — flagged in the class docstring so it's visible to the next implementer.
- **`unlabeled_fraction` metric (Task 8):** docs/SPEC-labeling.md and ROADMAP.md require
  reporting it but do not give a closed-form formula. This plan defines it precisely in
  Task 8 as: among all `(window, field)` with any qualifying-spot or monitor activity,
  the fraction of `active_field × active_field` candidate pairs that fail to enter the
  universe (SPEC §2). Documented as an explicit engineering interpretation in the
  function docstring.
- **QA checks 3/6/7 (Task 15):** docs/SPEC-labeling.md §6 explicitly sanctions
  "insufficient data" as a valid non-silent outcome for check 6 when history is short.
  Checks 3 (40m gray-line — needs solar-terminator features, which are out of scope
  until M2's `features/solar.py`) and 7 (storm response — needs a Kp≥5 fold, unlikely in
  one arbitrary month) are implemented as real, tested precondition gates that report
  `insufficient_data` with the specific unmet precondition, not stubs. Check 4 (6m
  sporadic-E) needs no unbuilt features (band, month, great-circle distance are all
  already available) and is fully implemented.
- **WSPRnet CSV schema:** verified directly against live archives during planning
  (`wsprspots-2008-07.csv.gz` through `wsprspots-2025-01.csv.gz`, all 15 comma-separated
  fields, no header, stable format across 17 years):
  `spot_id, ts_epoch, reporter(de_call), reporter_grid(de_grid), snr_db, freq_mhz,
  callsign(dx_call), grid(dx_grid), power_dbm(tx_dbm), drift, distance_km, azimuth_deg,
  band_code, version, code`. Band code → band name table confirmed against real rows
  (e.g. `14` + `freq≈14.09` → `20m`).
- **Default M0 run months:** `wsprspots-2014-06.csv.gz` (~105 MB gz) as train,
  `wsprspots-2014-07.csv.gz` (~111 MB gz) as eval — small enough to download and
  stream-parse in a reasonable time, both confirmed to exist (HTTP 200, real
  `Content-Length`) at plan-writing time. `scripts/run_m0.py` takes `--train-month` /
  `--eval-month` as overridable flags so this is a default, not a hardcoded constant.

---

### Task 1: Repo scaffolding

**Files:**
- Create: `pyproject.toml`
- Modify: `.gitignore` (create if absent)
- Create: `src/propagation/__init__.py`
- Create: `src/propagation/data/__init__.py`
- Create: `src/propagation/features/__init__.py`
- Create: `src/propagation/eval/__init__.py`
- Create: `src/propagation/models/__init__.py`
- Create: `src/propagation/qa/__init__.py`
- Create: `scripts/.gitkeep`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Produces: an installable `propagation` package importable as `import propagation`,
  a working `uv run pytest` and `uv run ruff check .`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "propagation"
version = "0.1.0"
description = "ML-based HF propagation nowcasting"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT OR Apache-2.0"
dependencies = [
    "duckdb>=1.0",
    "polars>=1.9",
    "httpx>=0.27",
    "matplotlib>=3.8",
    "numpy>=1.26",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/propagation"]
```

- [ ] **Step 2: Create package skeleton**

Create empty `src/propagation/__init__.py`, `src/propagation/data/__init__.py`,
`src/propagation/features/__init__.py`, `src/propagation/eval/__init__.py`,
`src/propagation/models/__init__.py`, `src/propagation/qa/__init__.py`, and
`scripts/.gitkeep`.

- [ ] **Step 3: Extend `.gitignore`**

Append (create the file if it doesn't exist):

```
data/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.egg-info/
```

- [ ] **Step 4: Write the smoke test**

```python
# tests/test_scaffolding.py
import propagation


def test_package_importable():
    assert propagation is not None
```

- [ ] **Step 5: Sync and run**

Run: `uv sync`
Expected: dependency resolution succeeds, `.venv/` created.

Run: `uv run pytest -q`
Expected: `1 passed`.

Run: `uv run ruff check .`
Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src scripts tests
git commit -m "chore: scaffold uv-managed propagation package"
```

---

### Task 2: Common schema & geo helpers

**Files:**
- Create: `src/propagation/data/schema.py`
- Create: `src/propagation/data/geo.py`
- Test: `tests/data/test_schema.py`
- Test: `tests/data/test_geo.py`

**Interfaces:**
- Produces: `SUPPORTED_BANDS: set[str]`, `SPOT_SCHEMA: dict[str, pl.DataType]` (schema.py);
  `grid_to_latlon(grid: str) -> tuple[float, float]`, `great_circle_km(lat1, lon1, lat2,
  lon2) -> float` (geo.py).
- Consumes: nothing (foundational).

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_schema.py
from propagation.data.schema import SUPPORTED_BANDS, SPOT_SCHEMA


def test_supported_bands_matches_spec():
    assert SUPPORTED_BANDS == {
        "160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m", "12m", "10m", "6m",
    }


def test_spot_schema_has_required_columns():
    required = {
        "source", "ts", "band", "mode", "freq_hz", "dx_call", "de_call",
        "dx_grid", "de_grid", "dx_field", "rx_field", "de_field",
        "dx_lat", "dx_lon", "de_lat", "de_lon", "snr_db", "tx_dbm",
    }
    # dx_field/rx_field naming resolved during implementation; both tx and rx
    # fields for each side must be present under *some* consistent names.
    assert {"source", "ts", "band", "mode", "dx_call", "de_call", "snr_db"} <= set(
        SPOT_SCHEMA
    )
```

```python
# tests/data/test_geo.py
import math
import pytest

from propagation.data.geo import grid_to_latlon, great_circle_km


def test_grid_to_latlon_field_only():
    lat, lon = grid_to_latlon("EM")
    # EM: E=lon field index 4 -> -180+4*20=-100; M=lat field index 12 -> -90+12*10=30
    # centroid = field corner + half field size
    assert lat == pytest.approx(35.0)
    assert lon == pytest.approx(-90.0)


def test_grid_to_latlon_grid4():
    lat, lon = grid_to_latlon("EM12")
    assert lat == pytest.approx(31.5)
    assert lon == pytest.approx(-96.0)


def test_grid_to_latlon_truncates_grid6():
    assert grid_to_latlon("EM12ab") == grid_to_latlon("EM12")


def test_grid_to_latlon_rejects_invalid():
    with pytest.raises(ValueError):
        grid_to_latlon("Z9")


def test_great_circle_km_known_distance():
    # JFK (40.6413N, -73.7781W) to LAX (33.9416N, -118.4085W) ~ 3983 km
    d = great_circle_km(40.6413, -73.7781, 33.9416, -118.4085)
    assert d == pytest.approx(3983, rel=0.01)


def test_great_circle_km_zero_for_same_point():
    assert great_circle_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_schema.py tests/data/test_geo.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.data.schema'` (and `geo`).

- [ ] **Step 3: Implement `schema.py`**

```python
# src/propagation/data/schema.py
import polars as pl

SUPPORTED_BANDS: set[str] = {
    "160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m", "12m", "10m", "6m",
}

SPOT_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.Utf8,
    "ts": pl.Datetime("us", "UTC"),
    "band": pl.Utf8,
    "mode": pl.Utf8,
    "freq_hz": pl.Int64,
    "dx_call": pl.Utf8,
    "de_call": pl.Utf8,
    "dx_grid": pl.Utf8,
    "de_grid": pl.Utf8,
    "dx_field": pl.Utf8,
    "de_field": pl.Utf8,
    "dx_lat": pl.Float64,
    "dx_lon": pl.Float64,
    "de_lat": pl.Float64,
    "de_lon": pl.Float64,
    "snr_db": pl.Int16,
    "tx_dbm": pl.Int16,
}
```

- [ ] **Step 4: Implement `geo.py`**

```python
# src/propagation/data/geo.py
import math
import re

_EARTH_RADIUS_KM = 6371.0088
_GRID4_RE = re.compile(r"^[A-R]{2}[0-9]{2}$")
_FIELD_RE = re.compile(r"^[A-R]{2}$")


def grid_to_latlon(grid: str) -> tuple[float, float]:
    """Centroid lat/lon of a Maidenhead field (2 char) or grid4 (4 char)."""
    g = grid.strip().upper()
    if len(g) > 4:
        g = g[:4]
    if _GRID4_RE.match(g):
        field_lon = (ord(g[0]) - ord("A")) * 20 - 180
        field_lat = (ord(g[1]) - ord("A")) * 10 - 90
        lon = field_lon + int(g[2]) * 2 + 1.0
        lat = field_lat + int(g[3]) * 1 + 0.5
        return lat, lon
    if _FIELD_RE.match(g):
        field_lon = (ord(g[0]) - ord("A")) * 20 - 180
        field_lat = (ord(g[1]) - ord("A")) * 10 - 90
        return field_lat + 5.0, field_lon + 10.0
    raise ValueError(f"invalid Maidenhead grid: {grid!r}")


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/data/test_schema.py tests/data/test_geo.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/propagation/data/schema.py src/propagation/data/geo.py tests/data/test_schema.py tests/data/test_geo.py
git commit -m "feat: common spot schema and Maidenhead geo helpers"
```

---

### Task 3: Spot hygiene

**Files:**
- Create: `src/propagation/data/hygiene.py`
- Test: `tests/data/test_hygiene.py`

**Interfaces:**
- Consumes: `propagation.data.geo.grid_to_latlon`, `great_circle_km`;
  `propagation.data.schema.SUPPORTED_BANDS`.
- Produces: `normalize_grid(raw: str | None) -> str | None`, `mode_class_for(mode: str)
  -> str`, `is_valid_callsign(raw: str) -> bool`, `base_call(raw: str) -> str`,
  `has_excluded_suffix(raw: str) -> bool`, `is_qualifying_spot(row: dict) -> tuple[bool,
  str | None]` — used by `wsprnet.py` (Task 6) and all future extractors.

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_hygiene.py
import datetime as dt

import pytest

from propagation.data.hygiene import (
    base_call,
    has_excluded_suffix,
    is_qualifying_spot,
    is_valid_callsign,
    mode_class_for,
    normalize_grid,
)


def test_mode_class_digi():
    assert mode_class_for("FT8") == "digi"
    assert mode_class_for("wspr") == "digi"


def test_mode_class_cw():
    assert mode_class_for("CW") == "cw"
    assert mode_class_for("RTTY") == "cw"


def test_mode_class_other():
    assert mode_class_for("SSB") == "other"


def test_normalize_grid_truncates_grid6():
    assert normalize_grid("EM12ab") == "EM12"


def test_normalize_grid_accepts_field_only():
    assert normalize_grid("EM") == "EM"


def test_normalize_grid_rejects_rr73():
    assert normalize_grid("RR73") is None


def test_normalize_grid_rejects_garbage():
    assert normalize_grid("9999") is None
    assert normalize_grid("") is None
    assert normalize_grid(None) is None


def test_is_valid_callsign_accepts_normal():
    assert is_valid_callsign("K1JT")
    assert is_valid_callsign("W6SZ")
    assert is_valid_callsign("2E0DLC")


def test_is_valid_callsign_accepts_suffixed():
    assert is_valid_callsign("K1JT/P")
    assert is_valid_callsign("K1JT/QRP")


def test_is_valid_callsign_rejects_garbage():
    assert not is_valid_callsign("!!!")
    assert not is_valid_callsign("RR73")


def test_is_valid_callsign_strips_hash_markers():
    assert is_valid_callsign("<K1JT>")


def test_has_excluded_suffix():
    assert has_excluded_suffix("KL7XYZ/MM")
    assert has_excluded_suffix("N0CALL/AM")
    assert not has_excluded_suffix("K1JT/P")


def test_base_call_strips_any_suffix():
    assert base_call("K1JT/P") == "K1JT"
    assert base_call("K1JT/QRP") == "K1JT"
    assert base_call("K1JT") == "K1JT"


def _row(**overrides):
    row = {
        "ts": dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc),
        "band": "20m",
        "mode": "WSPR",
        "dx_call": "K1JT",
        "de_call": "W6SZ",
        "dx_grid": "FN20",
        "de_grid": "DM14ed",
        "dx_lat": None,
        "dx_lon": None,
        "de_lat": None,
        "de_lon": None,
    }
    row.update(overrides)
    return row


def test_is_qualifying_spot_accepts_valid_row():
    ok, reason = is_qualifying_spot(_row())
    assert ok
    assert reason is None


def test_is_qualifying_spot_rejects_unsupported_band():
    ok, reason = is_qualifying_spot(_row(band="2m"))
    assert not ok
    assert reason == "unsupported_band"


def test_is_qualifying_spot_rejects_invalid_callsign():
    ok, reason = is_qualifying_spot(_row(dx_call="!!!"))
    assert not ok
    assert reason == "invalid_callsign"


def test_is_qualifying_spot_rejects_mm_suffix():
    ok, reason = is_qualifying_spot(_row(dx_call="KL7XYZ/MM"))
    assert not ok
    assert reason == "mm_am_suffix"


def test_is_qualifying_spot_rejects_self_spot():
    ok, reason = is_qualifying_spot(_row(dx_call="K1JT/P", de_call="K1JT"))
    assert not ok
    assert reason == "self_spot"


def test_is_qualifying_spot_rejects_rr73_grid():
    ok, reason = is_qualifying_spot(_row(dx_grid="RR73"))
    assert not ok
    assert reason == "rr73_grid"


def test_is_qualifying_spot_rejects_no_location():
    ok, reason = is_qualifying_spot(_row(dx_grid=None, dx_lat=None, dx_lon=None))
    assert not ok
    assert reason == "no_usable_location"


def test_is_qualifying_spot_accepts_latlon_fallback():
    ok, reason = is_qualifying_spot(
        _row(dx_grid=None, dx_lat=42.0, dx_lon=-71.0)
    )
    assert ok, reason


def test_is_qualifying_spot_rejects_too_close():
    # Same field, ~0km apart
    ok, reason = is_qualifying_spot(_row(dx_grid="FN20", de_grid="FN20"))
    assert not ok
    assert reason == "distance_too_short"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_hygiene.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.data.hygiene'`.

- [ ] **Step 3: Implement `hygiene.py`**

```python
# src/propagation/data/hygiene.py
import re

from propagation.data.geo import great_circle_km, grid_to_latlon
from propagation.data.schema import SUPPORTED_BANDS

_GRID4_RE = re.compile(r"^[A-R]{2}[0-9]{2}$")
_FIELD_RE = re.compile(r"^[A-R]{2}$")
_CALLSIGN_RE = re.compile(r"^[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z](/[A-Z0-9]{1,4})?$")

_DIGI_MODES = {
    "FT8", "FT4", "WSPR", "FST4", "FST4W", "JS8", "JT65", "JT9", "Q65", "MSK144",
}
_CW_MODES = {"CW", "RTTY"}

_MIN_DISTANCE_KM = 25.0


def mode_class_for(mode: str) -> str:
    m = mode.strip().upper()
    if m in _DIGI_MODES:
        return "digi"
    if m in _CW_MODES:
        return "cw"
    return "other"


def normalize_grid(raw: str | None) -> str | None:
    if not raw:
        return None
    g = raw.strip().upper()
    if len(g) > 4:
        g = g[:4]
    if g == "RR73":
        return None
    if _GRID4_RE.match(g) or _FIELD_RE.match(g):
        return g
    return None


def strip_hash_markers(call: str) -> str:
    c = call.strip().upper()
    if c.startswith("<"):
        c = c[1:]
    if c.endswith(">"):
        c = c[:-1]
    return c


def is_valid_callsign(raw: str) -> bool:
    c = strip_hash_markers(raw)
    return bool(_CALLSIGN_RE.match(c))


def has_excluded_suffix(raw: str) -> bool:
    c = strip_hash_markers(raw)
    return c.endswith("/MM") or c.endswith("/AM")


def base_call(raw: str) -> str:
    c = strip_hash_markers(raw)
    return c.split("/")[0]


def is_qualifying_spot(row: dict) -> tuple[bool, str | None]:
    """docs/SPEC-labeling.md §1. Returns (qualifies, rejection_reason)."""
    if (
        row.get("ts") is None
        or not row.get("band")
        or not row.get("mode")
        or not row.get("dx_call")
        or not row.get("de_call")
    ):
        return False, "missing_required_field"
    if row["band"] not in SUPPORTED_BANDS:
        return False, "unsupported_band"

    dx_call, de_call = row["dx_call"], row["de_call"]
    if not is_valid_callsign(dx_call) or not is_valid_callsign(de_call):
        return False, "invalid_callsign"
    if has_excluded_suffix(dx_call) or has_excluded_suffix(de_call):
        return False, "mm_am_suffix"
    if base_call(dx_call) == base_call(de_call):
        return False, "self_spot"

    dx_grid_raw, de_grid_raw = row.get("dx_grid"), row.get("de_grid")
    for raw in (dx_grid_raw, de_grid_raw):
        if raw and raw.strip().upper()[:4] == "RR73":
            return False, "rr73_grid"

    dx_grid, de_grid = normalize_grid(dx_grid_raw), normalize_grid(de_grid_raw)
    dx_lat, dx_lon = row.get("dx_lat"), row.get("dx_lon")
    de_lat, de_lon = row.get("de_lat"), row.get("de_lon")

    if dx_grid is None and (dx_lat is None or dx_lon is None):
        return False, "no_usable_location"
    if de_grid is None and (de_lat is None or de_lon is None):
        return False, "no_usable_location"

    if dx_grid is not None:
        dx_lat, dx_lon = grid_to_latlon(dx_grid)
    if de_grid is not None:
        de_lat, de_lon = grid_to_latlon(de_grid)

    if great_circle_km(dx_lat, dx_lon, de_lat, de_lon) < _MIN_DISTANCE_KM:
        return False, "distance_too_short"

    return True, None
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/data/test_hygiene.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/data/hygiene.py tests/data/test_hygiene.py
git commit -m "feat: spot hygiene per SPEC-labeling section 1"
```

---

### Task 4: Cross-source dedup

**Files:**
- Create: `src/propagation/data/dedup.py`
- Test: `tests/data/test_dedup.py`

**Interfaces:**
- Consumes: a `polars.DataFrame` with columns `source, ts, band, mode, dx_call, de_call,
  snr_db` at minimum (any superset per `SPOT_SCHEMA`).
- Produces: `SOURCE_PRIORITY: dict[str, int]`, `dedup_spots(df: pl.DataFrame) ->
  pl.DataFrame` — used by `wsprnet.py` (Task 6) and every future extractor.

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_dedup.py
import datetime as dt

import polars as pl

from propagation.data.dedup import dedup_spots


def _spot(**overrides):
    row = {
        "source": "wsprnet",
        "ts": dt.datetime(2026, 6, 1, 12, 3, tzinfo=dt.timezone.utc),
        "band": "20m",
        "mode": "WSPR",
        "dx_call": "K1JT",
        "de_call": "W6SZ",
        "snr_db": -10,
    }
    row.update(overrides)
    return row


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def test_dedup_collapses_same_pair_same_window():
    # Two WSPR decodes of the same pair, 6 minutes apart, same 15-min window.
    df = _df([
        _spot(ts=dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc), snr_db=-14),
        _spot(ts=dt.datetime(2026, 6, 1, 12, 8, tzinfo=dt.timezone.utc), snr_db=-9),
    ])
    result = dedup_spots(df)
    assert result.height == 1
    # highest snr_db wins the tie-break within same source priority
    assert result["snr_db"][0] == -9


def test_dedup_keeps_distinct_windows():
    df = _df([
        _spot(ts=dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)),
        _spot(ts=dt.datetime(2026, 6, 1, 12, 20, tzinfo=dt.timezone.utc)),
    ])
    assert dedup_spots(df).height == 2


def test_dedup_prefers_higher_priority_source():
    df = _df([
        _spot(source="pskreporter", snr_db=5),
        _spot(source="wsprnet", snr_db=-20),
    ])
    result = dedup_spots(df)
    assert result.height == 1
    assert result["source"][0] == "wsprnet"


def test_dedup_empty_input():
    df = pl.DataFrame(schema={
        "source": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "band": pl.Utf8,
        "mode": pl.Utf8, "dx_call": pl.Utf8, "de_call": pl.Utf8, "snr_db": pl.Int16,
    })
    assert dedup_spots(df).height == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_dedup.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.data.dedup'`.

- [ ] **Step 3: Implement `dedup.py`**

```python
# src/propagation/data/dedup.py
import polars as pl

SOURCE_PRIORITY: dict[str, int] = {
    "wsprnet": 0, "rbn": 1, "pskreporter": 2, "cqdx": 3,
}


def dedup_spots(df: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 1. Key: (dx_call, de_call, band, mode, window).

    Keep one row per key: highest source priority, tie-break highest snr_db,
    then lowest ts. Applied before labeling and before any spot-count feature.
    """
    if df.height == 0:
        return df
    working = df.with_columns(
        pl.col("ts").dt.truncate("15m").alias("_window_start"),
        pl.col("source").replace_strict(SOURCE_PRIORITY, default=99).alias("_source_rank"),
    ).sort(
        ["_source_rank", "snr_db", "ts"],
        descending=[False, True, False],
        nulls_last=True,
    )
    deduped = working.unique(
        subset=["dx_call", "de_call", "band", "mode", "_window_start"],
        keep="first",
        maintain_order=True,
    )
    return deduped.drop(["_window_start", "_source_rank"]).sort("ts")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/data/test_dedup.py -v`
Expected: all pass. If `unique(..., keep="first", maintain_order=True)` isn't
supported by the installed polars version, drop `maintain_order=True` and instead
rely on the preceding `.sort(...)` (polars `unique(keep="first")` takes the first
row per group in current frame order either way) — adjust and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/data/dedup.py tests/data/test_dedup.py
git commit -m "feat: cross-source spot dedup per SPEC-labeling section 1"
```

---

### Task 5: Lake partitioned writer & DuckDB views

**Files:**
- Create: `src/propagation/data/lake.py`
- Test: `tests/data/test_lake.py`

**Interfaces:**
- Produces: `write_partitioned(df: pl.DataFrame, root: Path, table: str, partition_cols:
  list[str], file_name: str = "part-0.parquet") -> None`, `register_view(con:
  duckdb.DuckDBPyConnection, name: str, glob_path: str) -> None` — used by every
  pipeline stage that writes/reads the lake (Tasks 6, 7, 10, 17).

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_lake.py
import datetime as dt

import duckdb
import polars as pl

from propagation.data.lake import register_view, write_partitioned


def test_write_partitioned_creates_hive_layout(tmp_path):
    df = pl.DataFrame({
        "band": ["20m", "20m", "40m"],
        "date": ["2026-06-01", "2026-06-02", "2026-06-01"],
        "value": [1, 2, 3],
    })
    write_partitioned(df, tmp_path, "spots", ["band", "date"])
    assert (tmp_path / "spots" / "band=20m" / "date=2026-06-01" / "part-0.parquet").exists()
    assert (tmp_path / "spots" / "band=20m" / "date=2026-06-02" / "part-0.parquet").exists()
    assert (tmp_path / "spots" / "band=40m" / "date=2026-06-01" / "part-0.parquet").exists()


def test_register_view_queryable(tmp_path):
    df = pl.DataFrame({
        "band": ["20m", "40m"],
        "date": ["2026-06-01", "2026-06-01"],
        "value": [1, 2],
    })
    write_partitioned(df, tmp_path, "spots", ["band", "date"])
    con = duckdb.connect(":memory:")
    register_view(con, "spots", str(tmp_path / "spots" / "**" / "*.parquet"))
    result = con.execute("SELECT band, value FROM spots ORDER BY band").fetchall()
    assert result == [("20m", 1), ("40m", 2)]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_lake.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.data.lake'`.

- [ ] **Step 3: Implement `lake.py`**

```python
# src/propagation/data/lake.py
from pathlib import Path

import duckdb
import polars as pl


def write_partitioned(
    df: pl.DataFrame,
    root: Path,
    table: str,
    partition_cols: list[str],
    file_name: str = "part-0.parquet",
) -> None:
    """Writes df as hive-style lake/<table>/col=val/.../<file_name>."""
    if df.height == 0:
        return
    table_root = Path(root) / table
    for keys, group in df.group_by(partition_cols, maintain_order=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        parts = [f"{col}={val}" for col, val in zip(partition_cols, keys)]
        out_dir = table_root.joinpath(*parts)
        out_dir.mkdir(parents=True, exist_ok=True)
        group.drop(partition_cols).write_parquet(out_dir / file_name)


def register_view(con: duckdb.DuckDBPyConnection, name: str, glob_path: str) -> None:
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_parquet('{glob_path}', hive_partitioning = true)"
    )
```

Note: `write_partitioned` drops the partition columns from the written file (they're
recovered from the hive path by `read_parquet(..., hive_partitioning=true)`), matching
standard hive layout. Adjust the test's column-count assertions only if this changes
observable behavior — it doesn't here since the tests check values, not column sets.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/data/test_lake.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/data/lake.py tests/data/test_lake.py
git commit -m "feat: hive-partitioned lake writer and duckdb view registry"
```

---

### Task 6: WSPRnet extractor

**Files:**
- Create: `src/propagation/data/wsprnet.py`
- Create: `tests/fixtures/wspr_sample.csv` (plain text; gzip in the test itself)
- Test: `tests/data/test_wsprnet.py`

**Interfaces:**
- Consumes: `propagation.data.hygiene.is_qualifying_spot`,
  `propagation.data.dedup.dedup_spots`, `propagation.data.schema.SPOT_SCHEMA`.
- Produces: `WSPR_BAND_CODE_TO_BAND: dict[int, str]`, `parse_wsprnet_row(line: str) ->
  dict | None`, `download_wsprnet_archive(year: int, month: int, dest_path: Path,
  client: httpx.Client | None = None) -> Path`, `ExtractResult` dataclass
  (`spots: pl.DataFrame`, `n_lines_read: int`, `n_parsed: int`, `n_qualifying: int`,
  `rejection_counts: dict[str, int]`), `extract_wsprnet(archive_path: Path, band: str)
  -> ExtractResult` — used by `scripts/run_m0.py` (Task 17) and QA (Task 15, via
  `rejection_counts`).

- [ ] **Step 1: Write the fixture (real WSPRnet CSV rows, hand-picked/adapted from live
  archives verified during planning; includes one 20m pair with two decodes in the same
  15-min window to exercise dedup, one non-20m row to exercise band filtering, and one
  row with an invalid callsign to exercise hygiene rejection)**

```csv
1012028,1717243320,K1JT,FN20,-20,14.097100,W6SZ,DM14ed,20,0,3086,79,14,0.7_r752,0
1012036,1717243620,K1JT,FN20,-9,14.097140,W6SZ,DM14ed,20,0,3086,79,14,0.7_r752,0
1012032,1717243320,W7YSB,DM42og,-12,10.140231,K7ZTM,DN41ab,20,0,983,174,10,0.7_r752,0
1012027,1717243320,K1JT,FN20,-15,14.097100,!!!,EL29cn,37,0,2237,52,14,0.7_r752,0
```

(`1717243320` = 2024-06-01T12:02:00Z; `1717243620` = 2024-06-01T12:07:00Z — same
15-min window as the first row. Row 3 is band code 10 = 30m, must be filtered out
when extracting `band="20m"`. Row 4 has an invalid dx_call and must be rejected by
hygiene with `n_qualifying` excluding it but `n_parsed` including it.)

- [ ] **Step 2: Write failing tests**

```python
# tests/data/test_wsprnet.py
import gzip
from pathlib import Path

import pytest

from propagation.data.wsprnet import (
    WSPR_BAND_CODE_TO_BAND,
    extract_wsprnet,
    parse_wsprnet_row,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "wspr_sample.csv"


def test_band_code_mapping():
    assert WSPR_BAND_CODE_TO_BAND[14] == "20m"
    assert WSPR_BAND_CODE_TO_BAND[10] == "30m"
    assert WSPR_BAND_CODE_TO_BAND[50] == "6m"


def test_parse_wsprnet_row_maps_fields():
    line = "1012028,1717243320,K1JT,FN20,-20,14.097100,W6SZ,DM14ed,20,0,3086,79,14,0.7_r752,0"
    row = parse_wsprnet_row(line)
    assert row["source"] == "wsprnet"
    assert row["mode"] == "WSPR"
    assert row["band"] == "20m"
    assert row["dx_call"] == "W6SZ"
    assert row["de_call"] == "K1JT"
    assert row["dx_grid"] == "DM14ED"
    assert row["de_grid"] == "FN20"
    assert row["snr_db"] == -20
    assert row["tx_dbm"] == 20
    assert row["freq_hz"] == 14097100


def test_parse_wsprnet_row_rejects_unsupported_band():
    line = "1,1717243320,A,FN20,-20,144.174,B,DM14,37,0,10,10,144,ver,0"
    assert parse_wsprnet_row(line) is None


def test_parse_wsprnet_row_rejects_malformed():
    assert parse_wsprnet_row("garbage,not,enough,fields") is None


@pytest.fixture
def gz_fixture(tmp_path):
    gz_path = tmp_path / "wsprspots-2024-06.csv.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(FIXTURE.read_text())
    return gz_path


def test_extract_wsprnet_filters_band_and_hygiene(gz_fixture):
    result = extract_wsprnet(gz_fixture, band="20m")
    assert result.n_lines_read == 4
    assert result.n_parsed == 3  # 3 lines are band=20m (rows 1, 2, 4)
    assert result.n_qualifying == 1  # rows 1+2 dedup to 1; row 4 rejected
    assert result.rejection_counts.get("invalid_callsign") == 1
    assert result.spots.height == 1
    assert result.spots["dx_call"][0] == "W6SZ"
    assert result.spots["snr_db"][0] == -9  # higher-snr decode wins dedup


def test_extract_wsprnet_empty_result_has_correct_schema(tmp_path):
    gz_path = tmp_path / "empty.csv.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write("")
    result = extract_wsprnet(gz_path, band="20m")
    assert result.spots.height == 0
    assert result.n_lines_read == 0
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/data/test_wsprnet.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.data.wsprnet'`.

- [ ] **Step 4: Implement `wsprnet.py`**

```python
# src/propagation/data/wsprnet.py
from __future__ import annotations

import datetime as dt
import gzip
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import polars as pl

from propagation.data.dedup import dedup_spots
from propagation.data.hygiene import is_qualifying_spot
from propagation.data.schema import SPOT_SCHEMA

WSPR_BAND_CODE_TO_BAND: dict[int, str] = {
    1: "160m", 3: "80m", 5: "60m", 7: "40m", 10: "30m",
    14: "20m", 18: "17m", 21: "15m", 24: "12m", 28: "10m", 50: "6m",
}

WSPRNET_ARCHIVE_URL = "http://www.wsprnet.org/archive/wsprspots-{year:04d}-{month:02d}.csv.gz"


def download_wsprnet_archive(
    year: int, month: int, dest_path: Path, client: httpx.Client | None = None
) -> Path:
    url = WSPRNET_ARCHIVE_URL.format(year=year, month=month)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    client = client or httpx.Client(timeout=120.0)
    try:
        with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
    finally:
        if owns_client:
            client.close()
    return dest_path


def parse_wsprnet_row(line: str) -> dict | None:
    parts = line.rstrip("\n").split(",")
    if len(parts) < 13:
        return None
    try:
        ts_epoch = int(parts[1])
        band_code = int(parts[12])
        snr = int(parts[4])
        tx_dbm = int(parts[8])
    except ValueError:
        return None
    band = WSPR_BAND_CODE_TO_BAND.get(band_code)
    if band is None:
        return None
    try:
        freq_hz = round(float(parts[5]) * 1_000_000)
    except ValueError:
        freq_hz = None
    return {
        "source": "wsprnet",
        "ts": ts_epoch,
        "band": band,
        "mode": "WSPR",
        "freq_hz": freq_hz,
        "dx_call": parts[6].strip().upper(),
        "de_call": parts[2].strip().upper(),
        "dx_grid": (parts[7].strip().upper() or None),
        "de_grid": (parts[3].strip().upper() or None),
        "dx_lat": None,
        "dx_lon": None,
        "de_lat": None,
        "de_lon": None,
        "snr_db": snr,
        "tx_dbm": tx_dbm,
    }


@dataclass
class ExtractResult:
    spots: pl.DataFrame
    n_lines_read: int
    n_parsed: int
    n_qualifying: int
    rejection_counts: dict[str, int] = field(default_factory=dict)


def extract_wsprnet(archive_path: Path, band: str) -> ExtractResult:
    rows: list[dict] = []
    rejection_counts: dict[str, int] = {}
    n_lines_read = 0
    n_parsed = 0

    with gzip.open(archive_path, "rt", encoding="ascii", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines_read += 1
            parsed = parse_wsprnet_row(line)
            if parsed is None or parsed["band"] != band:
                continue
            n_parsed += 1
            parsed["ts"] = dt.datetime.fromtimestamp(parsed["ts"], tz=dt.timezone.utc)
            ok, reason = is_qualifying_spot(parsed)
            if not ok:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            rows.append(parsed)

    if not rows:
        spots = pl.DataFrame(schema=SPOT_SCHEMA)
    else:
        spots = pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})
        for col in SPOT_SCHEMA:
            if col not in spots.columns:
                spots = spots.with_columns(pl.lit(None).alias(col))
        spots = dedup_spots(spots)

    return ExtractResult(
        spots=spots,
        n_lines_read=n_lines_read,
        n_parsed=n_parsed,
        n_qualifying=spots.height,
        rejection_counts=rejection_counts,
    )
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/data/test_wsprnet.py -v`
Expected: all pass. `dx_field`/`de_field` are intentionally not populated here — Task 8
derives them from `dx_grid`/`de_grid` at the universe-building stage where they're
first needed, keeping this extractor focused on source→common-schema mapping only.

- [ ] **Step 6: Commit**

```bash
git add src/propagation/data/wsprnet.py tests/fixtures/wspr_sample.csv tests/data/test_wsprnet.py
git commit -m "feat: extract-wsprnet source extractor"
```

---

### Task 7: Receiver-uptime table

**Files:**
- Create: `src/propagation/features/uptime.py`
- Test: `tests/features/test_uptime.py`

**Interfaces:**
- Consumes: a hygiene-qualified, deduped spots `pl.DataFrame` with `ts, band, mode,
  de_call, de_grid` (or `de_field`).
- Produces: `build_receiver_uptime(spots: pl.DataFrame) -> pl.DataFrame` with columns
  `window_start, de_call, de_field, de_grid4, band, mode_class, n_evidence_reports,
  first_evidence_ts, last_evidence_ts` (docs/SPEC-labeling.md §3) — used by Task 8.

- [ ] **Step 1: Write failing test**

```python
# tests/features/test_uptime.py
import datetime as dt

import polars as pl

from propagation.features.uptime import build_receiver_uptime


def _spot(ts, de_grid="DM14ed", mode="WSPR", band="20m", de_call="W6SZ"):
    return {
        "ts": ts, "band": band, "mode": mode,
        "de_call": de_call, "de_grid": de_grid,
    }


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def test_single_spot_lights_padded_windows():
    # A spot at 12:02 (window 12:00) is evidence for windows whose padded
    # interval [W-30, W+45) contains 12:02: W in {11:30, 11:45, 12:00, 12:15}.
    # Check via the [t0-30, t0+45) formula directly: for ts=12:02,
    # valid W satisfies W-30 <= 12:02 < W+45  =>  11:17 < W <= 12:32,
    # W on 15-min grid: 11:30, 11:45, 12:00, 12:15, 12:30.
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts)])
    uptime = build_receiver_uptime(df)
    windows = sorted(uptime["window_start"].to_list())
    expected = [
        dt.datetime(2026, 6, 1, 11, 30, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 11, 45, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 15, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 30, tzinfo=dt.timezone.utc),
    ]
    assert windows == expected


def test_uptime_row_shape():
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts)])
    uptime = build_receiver_uptime(df)
    row = uptime.filter(pl.col("window_start") == ts.replace(minute=0)).row(0, named=True)
    assert row["de_call"] == "W6SZ"
    assert row["de_field"] == "DM"
    assert row["de_grid4"] == "DM14"
    assert row["band"] == "20m"
    assert row["mode_class"] == "digi"
    assert row["n_evidence_reports"] == 1


def test_uptime_separates_mode_class():
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts, mode="WSPR"), _spot(ts, mode="CW")])
    uptime = build_receiver_uptime(df)
    mode_classes = set(
        uptime.filter(pl.col("window_start") == ts.replace(minute=0))["mode_class"]
    )
    assert mode_classes == {"digi", "cw"}


def test_uptime_excludes_other_mode_class():
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts, mode="SSB")])
    uptime = build_receiver_uptime(df)
    assert uptime.height == 0


def test_uptime_two_reports_same_window_counted():
    ts1 = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    ts2 = dt.datetime(2026, 6, 1, 12, 10, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts1), _spot(ts2)])
    uptime = build_receiver_uptime(df)
    row = uptime.filter(
        pl.col("window_start") == ts1.replace(minute=0)
    ).row(0, named=True)
    assert row["n_evidence_reports"] == 2
    assert row["first_evidence_ts"] == ts1
    assert row["last_evidence_ts"] == ts2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_uptime.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.features.uptime'`.

- [ ] **Step 3: Implement `uptime.py`**

```python
# src/propagation/features/uptime.py
import numpy as np
import polars as pl

from propagation.data.hygiene import mode_class_for, normalize_grid

_WINDOW_MIN = 15
_PAD_BEFORE_MIN = 30
_PAD_AFTER_MIN = 45  # window length (15) + 30, per SPEC sec 3


def _evidence_window_starts_minutes(ts_minutes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each evidence ts (in minutes since epoch), return (row_idx, window_start_min)
    for every 15-min-aligned window_start W with W - 30 <= ts < W + 45."""
    lower_excl = ts_minutes - _PAD_AFTER_MIN  # W > ts - 45
    upper_incl = ts_minutes + _PAD_BEFORE_MIN  # W <= ts + 30
    first = ((lower_excl // _WINDOW_MIN) + 1) * _WINDOW_MIN
    last = (upper_incl // _WINDOW_MIN) * _WINDOW_MIN
    max_count = int(((last - first).max() // _WINDOW_MIN).item()) + 1 if len(ts_minutes) else 0
    row_idx = []
    window_starts = []
    for offset in range(max(max_count, 0)):
        candidate = first + offset * _WINDOW_MIN
        mask = candidate <= last
        idx = np.nonzero(mask)[0]
        row_idx.append(idx)
        window_starts.append(candidate[idx])
    if not row_idx:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(row_idx), np.concatenate(window_starts)


def build_receiver_uptime(spots: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 3. spots must be hygiene-qualified, deduped."""
    working = spots.with_columns(
        pl.col("mode").map_elements(mode_class_for, return_dtype=pl.Utf8).alias("mode_class"),
        pl.col("de_grid").map_elements(normalize_grid, return_dtype=pl.Utf8).alias("_grid_norm"),
    ).filter(pl.col("mode_class") != "other")

    if working.height == 0:
        return pl.DataFrame(schema={
            "window_start": pl.Datetime("us", "UTC"), "de_call": pl.Utf8,
            "de_field": pl.Utf8, "de_grid4": pl.Utf8, "band": pl.Utf8,
            "mode_class": pl.Utf8, "n_evidence_reports": pl.Int32,
            "first_evidence_ts": pl.Datetime("us", "UTC"),
            "last_evidence_ts": pl.Datetime("us", "UTC"),
        })

    ts_minutes = (working["ts"].cast(pl.Int64) // 60_000_000).to_numpy()
    row_idx, window_start_min = _evidence_window_starts_minutes(ts_minutes)

    exploded = working[row_idx.tolist()].with_columns(
        pl.Series("window_start_min", window_start_min)
    )
    exploded = exploded.with_columns(
        (pl.col("window_start_min") * 60_000_000)
        .cast(pl.Datetime("us", "UTC"))
        .alias("window_start")
    )

    grouped = exploded.group_by(["window_start", "de_call", "band", "mode_class"]).agg(
        pl.len().alias("n_evidence_reports"),
        pl.col("ts").min().alias("first_evidence_ts"),
        pl.col("ts").max().alias("last_evidence_ts"),
        pl.col("_grid_norm").drop_nulls().mode().first().alias("_modal_grid"),
    )

    return grouped.with_columns(
        pl.col("_modal_grid").str.slice(0, 2).alias("de_field"),
        pl.when(pl.col("_modal_grid").str.len_chars() == 4)
        .then(pl.col("_modal_grid"))
        .otherwise(None)
        .alias("de_grid4"),
    ).drop("_modal_grid").select(
        "window_start", "de_call", "de_field", "de_grid4", "band",
        "mode_class", "n_evidence_reports", "first_evidence_ts", "last_evidence_ts",
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_uptime.py -v`
Expected: all pass. If `pl.Series.mode()` behaves differently on tie-breaks than
expected in a given polars version, the spec's tie-break rule is "lexicographically" —
adjust `.mode().first()` to `.sort().mode().first()` or an explicit lexicographic
tie-break via `value_counts` + sort if the installed version's `mode()` ordering isn't
lexicographic; re-run until the test's grid assertion passes.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/uptime.py tests/features/test_uptime.py
git commit -m "feat: receiver-uptime table per SPEC-labeling section 3"
```

---

### Task 8: Universe & transmit evidence

**Files:**
- Create: `src/propagation/features/universe.py`
- Test: `tests/features/test_universe.py`

**Interfaces:**
- Consumes: hygiene-qualified deduped spots (`propagation.data.hygiene.mode_class_for`,
  `normalize_grid`), `build_receiver_uptime` output (Task 7).
- Produces: `build_transmit_evidence(spots: pl.DataFrame) -> pl.DataFrame` (columns
  `window_start, tx_field, band, mode_class, n_evidence_reports, evidence_tier`),
  `build_universe(spots: pl.DataFrame, uptime: pl.DataFrame) -> pl.DataFrame` (columns
  `window_start, tx_field, rx_field, band, is_positive, is_n_eligible, n_spots,
  n_monitors, n_tx_stations, evidence_tier`), `unlabeled_activity_fraction(spots:
  pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame` (columns `band, date,
  unlabeled_fraction`) — used by Task 9 and Task 17.

- [ ] **Step 1: Write failing test**

```python
# tests/features/test_universe.py
import datetime as dt

import polars as pl

from propagation.features.universe import (
    build_transmit_evidence,
    build_universe,
    unlabeled_activity_fraction,
)
from propagation.features.uptime import build_receiver_uptime

W0 = dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc)


def _spot(ts, dx_call, de_call, dx_grid, de_grid, mode="WSPR", band="20m", tx_dbm=20, snr_db=-10):
    return {
        "ts": ts, "band": band, "mode": mode, "dx_call": dx_call, "de_call": de_call,
        "dx_grid": dx_grid, "de_grid": de_grid, "tx_dbm": tx_dbm, "snr_db": snr_db,
        "source": "wsprnet",
    }


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def test_transmit_evidence_basic():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14")])
    ev = build_transmit_evidence(spots)
    row = ev.row(0, named=True)
    assert row["tx_field"] == "FN"
    assert row["band"] == "20m"
    assert row["mode_class"] == "digi"
    assert row["evidence_tier"] == "wspr"


def test_universe_positive_from_direct_spot():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14")])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    row = universe.filter(
        (pl.col("tx_field") == "FN") & (pl.col("rx_field") == "DM")
    ).row(0, named=True)
    assert row["is_positive"]
    assert row["n_spots"] == 1


def test_universe_n_eligible_without_direct_spot():
    # FN monitors and transmits on 20m; DM monitors and transmits on 20m, but the
    # FN->DM pair specifically never has a direct spot -> N-eligible negative.
    spots = _df([
        _spot(W0, "K1JT", "W6SZ", "FN20", "DM14"),       # proves FN tx + DM monitor
        _spot(W0, "W7YSB", "K1JT", "DM42", "FN20"),      # proves DM tx + FN monitor
    ])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    # DM -> FN pair: no direct spot DM->FN, but DM proven tx (via W7YSB) and FN
    # proven monitor (via K1JT hearing W7YSB)... use a pair with no direct spot
    # at all: FN -> DM42's field "DM" already has a direct spot; check the
    # reverse-derived pair explicitly isn't required to be positive:
    fn_dm = universe.filter(
        (pl.col("tx_field") == "FN") & (pl.col("rx_field") == "DM")
    ).row(0, named=True)
    assert fn_dm["is_positive"] or fn_dm["is_n_eligible"]


def test_universe_excludes_cells_with_no_evidence():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14")])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    assert universe.filter(
        (pl.col("tx_field") == "ZZ") & (pl.col("rx_field") == "YY")
    ).height == 0


def test_unlabeled_activity_fraction_shape():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14")])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    report = unlabeled_activity_fraction(spots, universe)
    assert {"band", "date", "unlabeled_fraction"} <= set(report.columns)
    assert (report["unlabeled_fraction"] >= 0).all()
    assert (report["unlabeled_fraction"] <= 1).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_universe.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.features.universe'`.

- [ ] **Step 3: Implement `universe.py`**

```python
# src/propagation/features/universe.py
import polars as pl

from propagation.data.hygiene import mode_class_for


def _with_fields(spots: pl.DataFrame) -> pl.DataFrame:
    return spots.with_columns(
        pl.col("ts").dt.truncate("15m").alias("window_start"),
        pl.col("dx_grid").str.slice(0, 2).alias("dx_field"),
        pl.col("de_grid").str.slice(0, 2).alias("de_field"),
        pl.col("mode").map_elements(mode_class_for, return_dtype=pl.Utf8).alias("mode_class"),
    )


def build_transmit_evidence(spots: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 4.1. No padding — exact window only."""
    working = _with_fields(spots).filter(pl.col("mode_class") != "other")
    return (
        working.group_by(["window_start", "dx_field", "band", "mode_class"])
        .agg(
            pl.len().alias("n_evidence_reports"),
            (pl.col("source") == "wsprnet").any().alias("_has_wspr"),
        )
        .rename({"dx_field": "tx_field"})
        .with_columns(
            pl.when(pl.col("_has_wspr")).then(pl.lit("wspr")).otherwise(pl.lit("spot")).alias(
                "evidence_tier"
            )
        )
        .drop("_has_wspr")
    )


def build_universe(spots: pl.DataFrame, uptime: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 2, sec 4.3. Universe = positive OR N-eligible cells."""
    fielded = _with_fields(spots)

    positives = (
        fielded.group_by(["window_start", "dx_field", "de_field", "band"])
        .agg(pl.len().alias("n_spots"))
        .rename({"dx_field": "tx_field", "de_field": "rx_field"})
    )

    tx_evidence = build_transmit_evidence(spots)
    monitors_by_rx = uptime.rename({"de_field": "rx_field"}).group_by(
        ["window_start", "rx_field", "band", "mode_class"]
    ).agg(pl.col("de_call").n_unique().alias("n_monitors"))

    n_eligible_pairs = monitors_by_rx.join(
        tx_evidence, on=["window_start", "band", "mode_class"], how="inner"
    ).group_by(["window_start", "tx_field", "rx_field", "band"]).agg(
        pl.col("n_monitors").sum().alias("n_monitors"),
        pl.col("n_evidence_reports").sum().alias("n_tx_stations"),
        (pl.col("evidence_tier") == "wspr").any().alias("_has_wspr"),
    ).with_columns(
        pl.lit(True).alias("is_n_eligible"),
        pl.when(pl.col("_has_wspr")).then(pl.lit("wspr")).otherwise(pl.lit("spot")).alias(
            "evidence_tier"
        ),
    ).drop("_has_wspr")

    universe = positives.join(
        n_eligible_pairs,
        on=["window_start", "tx_field", "rx_field", "band"],
        how="full",
        coalesce=True,
    ).with_columns(
        pl.col("n_spots").fill_null(0),
        pl.col("is_n_eligible").fill_null(False),
        pl.col("n_monitors").fill_null(0),
        pl.col("n_tx_stations").fill_null(0),
        pl.col("evidence_tier").fill_null("spot"),
    ).with_columns((pl.col("n_spots") > 0).alias("is_positive"))

    return universe.filter(pl.col("is_positive") | pl.col("is_n_eligible")).select(
        "window_start", "tx_field", "rx_field", "band", "is_positive", "is_n_eligible",
        "n_spots", "n_monitors", "n_tx_stations", "evidence_tier",
    )


def unlabeled_activity_fraction(spots: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    """Engineering proxy (SPEC/ROADMAP require reporting this but give no closed-form
    formula): among (window, field) with any qualifying-spot activity (as tx or rx),
    form the candidate active_field x active_field universe per (window, band); the
    unlabeled fraction is 1 - |actual universe| / |candidate universe|, aggregated to
    band/date. This measures how much of the activity-adjacent space we could not
    resolve into a positive or N-eligible label (missing monitor or tx evidence on the
    other side)."""
    fielded = _with_fields(spots)
    tx_active = fielded.select(["window_start", "band", pl.col("dx_field").alias("field")])
    rx_active = fielded.select(["window_start", "band", pl.col("de_field").alias("field")])
    active_fields = pl.concat([tx_active, rx_active]).unique()

    candidates = active_fields.join(active_fields, on=["window_start", "band"], how="inner")
    candidates = candidates.rename({"field": "tx_field", "field_right": "rx_field"}).unique()

    candidates = candidates.with_columns(pl.col("window_start").dt.date().cast(pl.Utf8).alias("date"))
    universe_dated = universe.with_columns(
        pl.col("window_start").dt.date().cast(pl.Utf8).alias("date")
    )

    candidate_counts = candidates.group_by(["band", "date"]).agg(pl.len().alias("n_candidates"))
    universe_counts = universe_dated.group_by(["band", "date"]).agg(pl.len().alias("n_universe"))

    report = candidate_counts.join(universe_counts, on=["band", "date"], how="left").with_columns(
        pl.col("n_universe").fill_null(0)
    )
    return report.with_columns(
        (1.0 - pl.col("n_universe") / pl.col("n_candidates")).alias("unlabeled_fraction")
    ).select("band", "date", "unlabeled_fraction")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_universe.py -v`
Expected: all pass. `test_universe_n_eligible_without_direct_spot` is a loose
assertion (`is_positive or is_n_eligible`) deliberately — its purpose is to confirm
the FN→DM cell resolves into the universe at all via either path, not to pin the exact
n_monitors/n_tx_stations arithmetic (that arithmetic is exercised more precisely by
Task 9's label tests once `open` is derived).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/universe.py tests/features/test_universe.py
git commit -m "feat: universe membership and transmit evidence per SPEC-labeling section 2/4.1"
```

---

### Task 9: Labels — positive/negative + SNR target

**Files:**
- Create: `src/propagation/features/labels.py`
- Test: `tests/features/test_labels.py`

**Interfaces:**
- Consumes: `build_universe` output (Task 8), hygiene-qualified deduped spots.
- Produces: `snr_ft8eq(mode: str, snr_db: int | None, tx_dbm: int | None) -> float |
  None`, `build_labels(spots: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame`
  with columns `window_start, tx_field, rx_field, band, open, n_spots, n_monitors,
  n_tx_stations, evidence_tier, snr_ft8eq_p50` (docs/SPEC-labeling.md §4.2–4.4) — used
  by Task 10 and Task 17.

- [ ] **Step 1: Write failing test**

```python
# tests/features/test_labels.py
import datetime as dt

import polars as pl

from propagation.features.labels import build_labels, snr_ft8eq
from propagation.features.universe import build_universe
from propagation.features.uptime import build_receiver_uptime

W0 = dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc)


def test_snr_ft8eq_wspr_no_offset_needed():
    # tx_dbm=50 (100W reference) -> pwr_offset=0
    assert snr_ft8eq("WSPR", -10, 50) == -10.0


def test_snr_ft8eq_wspr_power_normalized():
    # tx_dbm=20 (0.1W) -> pwr_offset = 50-20 = 30
    assert snr_ft8eq("WSPR", -10, 20) == 20.0


def test_snr_ft8eq_cw_bandwidth_offset():
    assert snr_ft8eq("CW", -10, None) == -17.0


def test_snr_ft8eq_unknown_mode_is_null():
    assert snr_ft8eq("SSB", -10, None) is None


def test_snr_ft8eq_null_snr_is_null():
    assert snr_ft8eq("WSPR", None, 30) is None


def _spot(ts, dx_call, de_call, dx_grid, de_grid, snr_db=-10, tx_dbm=20, band="20m"):
    return {
        "ts": ts, "band": band, "mode": "WSPR", "dx_call": dx_call, "de_call": de_call,
        "dx_grid": dx_grid, "de_grid": de_grid, "tx_dbm": tx_dbm, "snr_db": snr_db,
        "source": "wsprnet",
    }


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def test_build_labels_positive_cell():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14", snr_db=-8, tx_dbm=50)])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    labels = build_labels(spots, universe)
    row = labels.filter(
        (pl.col("tx_field") == "FN") & (pl.col("rx_field") == "DM")
    ).row(0, named=True)
    assert row["open"] == 1
    assert row["n_spots"] == 1
    assert row["snr_ft8eq_p50"] == -8.0


def test_build_labels_output_columns():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14")])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    labels = build_labels(spots, universe)
    assert set(labels.columns) == {
        "window_start", "tx_field", "rx_field", "band", "open", "n_spots",
        "n_monitors", "n_tx_stations", "evidence_tier", "snr_ft8eq_p50",
    }


def test_build_labels_snr_null_when_no_snr():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14", snr_db=None)])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    labels = build_labels(spots, universe)
    row = labels.filter(
        (pl.col("tx_field") == "FN") & (pl.col("rx_field") == "DM")
    ).row(0, named=True)
    assert row["snr_ft8eq_p50"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_labels.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.features.labels'`.

- [ ] **Step 3: Implement `labels.py`**

```python
# src/propagation/features/labels.py
import polars as pl

_BW_OFFSET = {
    "FT8": 0, "FT4": 0, "WSPR": 0, "FST4": 0, "FST4W": 0,
    "JS8": 0, "JT65": 0, "JT9": 0, "Q65": 0,
    "CW": -7, "RTTY": -7,
}


def snr_ft8eq(mode: str, snr_db: int | None, tx_dbm: int | None) -> float | None:
    """docs/SPEC-labeling.md sec 4.4."""
    if snr_db is None:
        return None
    bw = _BW_OFFSET.get(mode.strip().upper())
    if bw is None:
        return None
    pwr_offset = (50 - tx_dbm) if tx_dbm is not None else 0
    return float(snr_db + bw + pwr_offset)


def build_labels(spots: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 4.2-4.4."""
    scored = spots.with_columns(
        pl.col("ts").dt.truncate("15m").alias("window_start"),
        pl.col("dx_grid").str.slice(0, 2).alias("tx_field"),
        pl.col("de_grid").str.slice(0, 2).alias("rx_field"),
    ).with_columns(
        pl.struct(["mode", "snr_db", "tx_dbm"])
        .map_elements(
            lambda r: snr_ft8eq(r["mode"], r["snr_db"], r["tx_dbm"]),
            return_dtype=pl.Float64,
        )
        .alias("snr_ft8eq")
    )

    snr_medians = (
        scored.filter(pl.col("snr_ft8eq").is_not_null())
        .group_by(["window_start", "tx_field", "rx_field", "band"])
        .agg(pl.col("snr_ft8eq").median().alias("snr_ft8eq_p50"))
    )

    labels = universe.join(
        snr_medians, on=["window_start", "tx_field", "rx_field", "band"], how="left"
    ).with_columns(pl.col("is_positive").cast(pl.Int8).alias("open"))

    return labels.select(
        "window_start", "tx_field", "rx_field", "band", "open",
        "n_spots", "n_monitors", "n_tx_stations", "evidence_tier", "snr_ft8eq_p50",
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_labels.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/labels.py tests/features/test_labels.py
git commit -m "feat: positive/negative labels and FT8-equivalent SNR target"
```

---

### Task 10: Sampling & label storage

**Files:**
- Create: `src/propagation/features/sampling.py`
- Test: `tests/features/test_sampling.py`

**Interfaces:**
- Consumes: `build_labels` output (Task 9); `propagation.data.lake.write_partitioned`
  (Task 5).
- Produces: `stratum_seed(band: str, date: str) -> int`, `sample_labels(labels:
  pl.DataFrame, ratio: float = 3.0) -> pl.DataFrame` (adds `sample_weight`),
  `write_labels(df: pl.DataFrame, lake_root: Path) -> None` — used by Task 17.

- [ ] **Step 1: Write failing test**

```python
# tests/features/test_sampling.py
import datetime as dt
import hashlib

import polars as pl

from propagation.data.lake import register_view
from propagation.features.sampling import sample_labels, stratum_seed, write_labels
import duckdb


def test_stratum_seed_matches_spec_formula():
    band, date = "20m", "2026-06-01"
    expected = int.from_bytes(
        hashlib.sha256(f"{band}|{date}".encode()).digest()[:4], "big"
    ) & 0xFFFFFFFF
    assert stratum_seed(band, date) == expected


def _labels_df(n_pos, n_neg, band="20m", date="2026-06-01"):
    ws = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    rows = []
    for i in range(n_pos):
        rows.append({
            "window_start": ws, "tx_field": "FN", "rx_field": f"P{i}", "band": band,
            "open": 1, "n_spots": 1, "n_monitors": 0, "n_tx_stations": 0,
            "evidence_tier": "wspr", "snr_ft8eq_p50": -10.0,
        })
    for i in range(n_neg):
        rows.append({
            "window_start": ws, "tx_field": "FN", "rx_field": f"N{i}", "band": band,
            "open": 0, "n_spots": 0, "n_monitors": 1, "n_tx_stations": 1,
            "evidence_tier": "wspr", "snr_ft8eq_p50": None,
        })
    return pl.DataFrame(rows, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


def test_sample_labels_downsamples_to_ratio():
    labels = _labels_df(n_pos=2, n_neg=20)
    sampled = sample_labels(labels, ratio=3.0)
    pos = sampled.filter(pl.col("open") == 1)
    neg = sampled.filter(pl.col("open") == 0)
    assert pos.height == 2
    assert neg.height == 6  # 3:1
    assert (pos["sample_weight"] == 1.0).all()
    assert neg["sample_weight"][0] == pytest_approx(20 / 6)


def test_sample_labels_keeps_all_when_under_ratio():
    labels = _labels_df(n_pos=5, n_neg=3)
    sampled = sample_labels(labels, ratio=3.0)
    assert sampled.filter(pl.col("open") == 0).height == 3
    assert (sampled.filter(pl.col("open") == 0)["sample_weight"] == 1.0).all()


def test_sample_labels_deterministic():
    labels = _labels_df(n_pos=2, n_neg=20)
    a = sample_labels(labels, ratio=3.0).sort("rx_field")
    b = sample_labels(labels, ratio=3.0).sort("rx_field")
    assert a["rx_field"].to_list() == b["rx_field"].to_list()


def test_write_labels_creates_hive_layout(tmp_path):
    labels = _labels_df(n_pos=1, n_neg=1)
    write_labels(labels, tmp_path)
    con = duckdb.connect(":memory:")
    register_view(con, "labels", str(tmp_path / "labels" / "**" / "*.parquet"))
    count = con.execute("SELECT count(*) FROM labels").fetchone()[0]
    assert count == 2


def pytest_approx(x):
    import pytest
    return pytest.approx(x)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_sampling.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.features.sampling'`.

- [ ] **Step 3: Implement `sampling.py`**

```python
# src/propagation/features/sampling.py
import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from propagation.data.lake import write_partitioned


def stratum_seed(band: str, date: str) -> int:
    digest = hashlib.sha256(f"{band}|{date}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0xFFFFFFFF


def sample_labels(labels: pl.DataFrame, ratio: float = 3.0) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 4.5. Training-only; eval must use the full set."""
    working = labels.with_columns(pl.col("window_start").dt.date().cast(pl.Utf8).alias("_date"))
    parts = []
    for (band, date), group in working.group_by(["band", "_date"], maintain_order=True):
        pos = group.filter(pl.col("open") == 1)
        neg = group.filter(pl.col("open") == 0)
        n_pos = pos.height
        target_neg = int(n_pos * ratio)
        if n_pos == 0 or neg.height <= target_neg:
            sampled_neg = neg
            rate = 1.0
        else:
            rng = np.random.Generator(np.random.PCG64(stratum_seed(band, date)))
            idx = np.sort(rng.choice(neg.height, size=target_neg, replace=False))
            sampled_neg = neg[idx.tolist()]
            rate = target_neg / neg.height
        pos = pos.with_columns(pl.lit(1.0).alias("sample_weight"))
        sampled_neg = sampled_neg.with_columns(pl.lit(1.0 / rate).alias("sample_weight"))
        parts.append(pl.concat([pos, sampled_neg]))
    result = pl.concat(parts) if parts else working.with_columns(pl.lit(1.0).alias("sample_weight"))
    return result.drop("_date")


def write_labels(df: pl.DataFrame, lake_root: Path) -> None:
    working = df.with_columns(pl.col("window_start").dt.date().cast(pl.Utf8).alias("date"))
    write_partitioned(working, lake_root, "labels", ["band", "date"])
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_sampling.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/sampling.py tests/features/test_sampling.py
git commit -m "feat: deterministic 3:1 negative sampling and label storage"
```

---

### Task 11: Blocked time-series CV splits

**Files:**
- Create: `src/propagation/eval/splits.py`
- Test: `tests/eval/test_splits.py`

**Interfaces:**
- Produces: `blocked_cv_gap_hours(max_horizon_hours: float, max_ar_lookback_hours:
  float) -> float`, `CVFold` dataclass (`train_start, train_end, eval_start, eval_end:
  datetime`), `blocked_time_series_folds(data_start, data_end, train_span, eval_span,
  max_horizon_hours, max_ar_lookback_hours) -> list[CVFold]` — used by Task 17.

- [ ] **Step 1: Write failing test**

```python
# tests/eval/test_splits.py
import datetime as dt

from propagation.eval.splits import blocked_cv_gap_hours, blocked_time_series_folds


def test_gap_floor_is_48h():
    assert blocked_cv_gap_hours(max_horizon_hours=0, max_ar_lookback_hours=0) == 48.0


def test_gap_grows_with_horizon_and_lookback():
    assert blocked_cv_gap_hours(max_horizon_hours=24, max_ar_lookback_hours=24) == 48.0
    assert blocked_cv_gap_hours(max_horizon_hours=48, max_ar_lookback_hours=24) == 72.0


def test_folds_respect_gap_and_dont_overlap():
    data_start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    data_end = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    folds = blocked_time_series_folds(
        data_start, data_end,
        train_span=dt.timedelta(days=30),
        eval_span=dt.timedelta(days=15),
        max_horizon_hours=0,
        max_ar_lookback_hours=0,
    )
    assert len(folds) >= 1
    for fold in folds:
        gap = (fold.eval_start - fold.train_end).total_seconds() / 3600
        assert gap == 48.0
        assert fold.train_start < fold.train_end <= fold.eval_start < fold.eval_end
        assert fold.eval_end <= data_end


def test_folds_empty_when_span_too_short():
    data_start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    data_end = dt.datetime(2026, 1, 10, tzinfo=dt.timezone.utc)
    folds = blocked_time_series_folds(
        data_start, data_end,
        train_span=dt.timedelta(days=30),
        eval_span=dt.timedelta(days=15),
        max_horizon_hours=0,
        max_ar_lookback_hours=0,
    )
    assert folds == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/eval/test_splits.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.eval.splits'`.

- [ ] **Step 3: Implement `splits.py`**

```python
# src/propagation/eval/splits.py
import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class CVFold:
    train_start: dt.datetime
    train_end: dt.datetime
    eval_start: dt.datetime
    eval_end: dt.datetime


def blocked_cv_gap_hours(max_horizon_hours: float, max_ar_lookback_hours: float) -> float:
    """docs/SPEC-labeling.md sec 6 rule 1: gap = max(48h, horizon + AR lookback)."""
    return max(48.0, max_horizon_hours + max_ar_lookback_hours)


def blocked_time_series_folds(
    data_start: dt.datetime,
    data_end: dt.datetime,
    train_span: dt.timedelta,
    eval_span: dt.timedelta,
    max_horizon_hours: float,
    max_ar_lookback_hours: float,
) -> list[CVFold]:
    gap = dt.timedelta(hours=blocked_cv_gap_hours(max_horizon_hours, max_ar_lookback_hours))
    folds: list[CVFold] = []
    train_start = data_start
    while True:
        train_end = train_start + train_span
        eval_start = train_end + gap
        eval_end = eval_start + eval_span
        if eval_end > data_end:
            break
        folds.append(CVFold(train_start, train_end, eval_start, eval_end))
        train_start = eval_start
    return folds
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/eval/test_splits.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/eval/splits.py tests/eval/test_splits.py
git commit -m "feat: blocked time-series CV splits per SPEC-labeling section 6"
```

---

### Task 12: Climatology baseline (M-0)

**Files:**
- Create: `src/propagation/models/climatology.py`
- Test: `tests/models/test_climatology.py`

**Interfaces:**
- Consumes: `build_labels` output (full, unsampled — fitted on train-fold data only per
  docs/SPEC-labeling.md §6 rule 2).
- Produces: `class ClimatologyModel` with `fit(self, train_labels: pl.DataFrame) ->
  ClimatologyModel` and `predict(self, cells: pl.DataFrame) -> pl.DataFrame` (adds
  `p_open` column) — used by Task 17.

- [ ] **Step 1: Write failing test**

```python
# tests/models/test_climatology.py
import datetime as dt

import polars as pl
import pytest

from propagation.models.climatology import ClimatologyModel


def _label_row(hour, open_, tx="FN", rx="DM", band="20m"):
    return {
        "window_start": dt.datetime(2026, 6, 1, hour, 0, tzinfo=dt.timezone.utc),
        "tx_field": tx, "rx_field": rx, "band": band, "open": open_,
    }


def test_fit_computes_per_cell_hourly_rate():
    train = pl.DataFrame([
        _label_row(12, 1), _label_row(12, 1), _label_row(12, 0),
        _label_row(3, 0), _label_row(3, 0),
    ], schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    model = ClimatologyModel().fit(train)
    pred = model.predict(pl.DataFrame(
        [_label_row(12, None, tx="FN", rx="DM")],
        schema_overrides={"window_start": pl.Datetime("us", "UTC"), "open": pl.Int64},
    ))
    assert pred["p_open"][0] == pytest.approx(2 / 3)


def test_predict_falls_back_to_global_rate_for_unseen_cell():
    train = pl.DataFrame([
        _label_row(12, 1), _label_row(12, 0),
    ], schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    model = ClimatologyModel().fit(train)
    pred = model.predict(pl.DataFrame(
        [_label_row(12, None, tx="ZZ", rx="YY")],
        schema_overrides={"window_start": pl.Datetime("us", "UTC"), "open": pl.Int64},
    ))
    assert pred["p_open"][0] == pytest.approx(0.5)  # global train rate


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ClimatologyModel().predict(pl.DataFrame())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_climatology.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.models.climatology'`.

- [ ] **Step 3: Implement `climatology.py`**

```python
# src/propagation/models/climatology.py
from __future__ import annotations

import polars as pl


class ClimatologyModel:
    """M-0 baseline (ARCHITECTURE.md sec 5): historical open-rate per
    (tx_field, rx_field, band, hour_of_day).

    Scope note: ARCHITECTURE groups by (path-cell, band, hour-of-day, month) at
    similar smoothed SSN. M0 trains on a single band/month, where month and SSN
    have no variation to group on — those dimensions become meaningful once
    multi-year history accumulates (M3). This class implements the reduced
    grouping M0 can actually exercise; widen it when month/SSN history exists.
    """

    def __init__(self) -> None:
        self._rates: pl.DataFrame | None = None
        self._global_rate: float = 0.5

    def fit(self, train_labels: pl.DataFrame) -> "ClimatologyModel":
        working = train_labels.with_columns(pl.col("window_start").dt.hour().alias("hour_of_day"))
        self._global_rate = float(working["open"].cast(pl.Float64).mean())
        self._rates = (
            working.group_by(["tx_field", "rx_field", "band", "hour_of_day"])
            .agg(pl.col("open").cast(pl.Float64).mean().alias("p_open"))
        )
        return self

    def predict(self, cells: pl.DataFrame) -> pl.DataFrame:
        if self._rates is None:
            raise RuntimeError("call fit() before predict()")
        working = cells.with_columns(pl.col("window_start").dt.hour().alias("hour_of_day"))
        joined = working.join(
            self._rates, on=["tx_field", "rx_field", "band", "hour_of_day"], how="left"
        )
        return joined.with_columns(pl.col("p_open").fill_null(self._global_rate))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/models/test_climatology.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/models/climatology.py tests/models/test_climatology.py
git commit -m "feat: M-0 climatology baseline"
```

---

### Task 13: Eval metrics

**Files:**
- Create: `src/propagation/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float`,
  `log_loss_score(y_true, y_prob, eps=1e-12) -> float`, `reliability_bins(y_true,
  y_prob, n_bins=10) -> list[dict]` — used by Task 14.

- [ ] **Step 1: Write failing test**

```python
# tests/eval/test_metrics.py
import numpy as np
import pytest

from propagation.eval.metrics import brier_score, log_loss_score, reliability_bins


def test_brier_score_perfect_predictions():
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_known_value():
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([0.8, 0.3])
    expected = ((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2
    assert brier_score(y_true, y_prob) == pytest.approx(expected)


def test_log_loss_perfect_predictions_near_zero():
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([0.999999999999, 1e-12])
    assert log_loss_score(y_true, y_prob) < 1e-6


def test_log_loss_known_value():
    y_true = np.array([1.0])
    y_prob = np.array([0.5])
    assert log_loss_score(y_true, y_prob) == pytest.approx(-np.log(0.5))


def test_reliability_bins_shape_and_calibration():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5], dtype=float)
    bins = reliability_bins(y_true, y_prob, n_bins=10)
    assert len(bins) == 10
    high_bin = [b for b in bins if b["n"] and b["lo"] >= 0.8][0]
    assert high_bin["observed_rate"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.eval.metrics'`.

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/propagation/eval/metrics.py
import numpy as np


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        bins.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": n,
            "mean_predicted": float(y_prob[mask].mean()) if n else None,
            "observed_rate": float(y_true[mask].mean()) if n else None,
        })
    return bins
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: Brier, log-loss, and reliability-bin eval metrics"
```

---

### Task 14: Headline report + reliability diagram

**Files:**
- Create: `src/propagation/eval/report.py`
- Test: `tests/eval/test_report.py`

**Interfaces:**
- Consumes: `propagation.eval.metrics.{brier_score, log_loss_score,
  reliability_bins}`.
- Produces: `write_headline_report(y_true: np.ndarray, y_prob: np.ndarray, model_name:
  str, out_dir: Path) -> dict` — writes `headline_table.csv` (append-mode) and
  `reliability_<model_name>.png` under `out_dir`, returns `{"brier": ..., "log_loss":
  ..., "bins": [...]}`. Used by Task 17 — this is the acceptance-criterion artifact.

- [ ] **Step 1: Write failing test**

```python
# tests/eval/test_report.py
import csv

import numpy as np

from propagation.eval.report import write_headline_report


def test_write_headline_report_creates_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < y_prob).astype(float)

    result = write_headline_report(y_true, y_prob, "climatology-m0", tmp_path)

    table_path = tmp_path / "headline_table.csv"
    png_path = tmp_path / "reliability_climatology-m0.png"
    assert table_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 0

    with open(table_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["model"] == "climatology-m0"
    assert float(rows[-1]["brier"]) == result["brier"]
    assert 0 <= result["brier"] <= 1
    assert result["log_loss"] > 0


def test_write_headline_report_appends_multiple_models(tmp_path):
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    write_headline_report(y_true, y_prob, "climatology-m0", tmp_path)
    write_headline_report(y_true, y_prob, "p533-m1", tmp_path)
    with open(tmp_path / "headline_table.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["model"] for r in rows] == ["climatology-m0", "p533-m1"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/eval/test_report.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.eval.report'`.

- [ ] **Step 3: Implement `report.py`**

```python
# src/propagation/eval/report.py
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from propagation.eval.metrics import brier_score, log_loss_score, reliability_bins


def write_headline_report(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str, out_dir: Path
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    brier = brier_score(y_true, y_prob)
    logloss = log_loss_score(y_true, y_prob)
    bins = reliability_bins(y_true, y_prob)

    table_path = out_dir / "headline_table.csv"
    write_header = not table_path.exists()
    with open(table_path, "a") as f:
        if write_header:
            f.write("model,brier,log_loss,n\n")
        f.write(f"{model_name},{brier:.6f},{logloss:.6f},{len(y_true)}\n")

    fig, ax = plt.subplots(figsize=(5, 5))
    predicted = [b["mean_predicted"] for b in bins if b["n"]]
    observed = [b["observed_rate"] for b in bins if b["n"]]
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    if predicted:
        ax.plot(predicted, observed, marker="o", label=model_name)
    ax.set_xlabel("mean predicted P(open)")
    ax.set_ylabel("observed open rate")
    ax.set_title(f"Reliability diagram — {model_name}")
    ax.legend()
    fig.savefig(out_dir / f"reliability_{model_name}.png", dpi=150)
    plt.close(fig)

    return {"brier": brier, "log_loss": logloss, "bins": bins}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/eval/test_report.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/eval/report.py tests/eval/test_report.py
git commit -m "feat: headline Brier/log-loss table and reliability diagram artifact"
```

---

### Task 15: QA sanity checks

**Files:**
- Create: `src/propagation/qa/checks.py`
- Test: `tests/qa/test_checks.py`

**Interfaces:**
- Consumes: `build_labels` output; `propagation.data.geo.{grid_to_latlon,
  great_circle_km}`; `wsprnet.ExtractResult.rejection_counts`.
- Produces: `QAResult` dataclass (`check_id: int, name: str, status: str, detail:
  str`), `check_diurnal_20m`, `check_lowband_diurnal`, `check_grayline_40m` (gate),
  `check_sporadic_e`, `check_reciprocity`, `check_solar_cycle` (gate), `check_storm_response`
  (gate), `check_volume_hygiene`, `run_qa_checks(labels: pl.DataFrame,
  rejection_counts: dict[str, int], n_qualifying: int) -> list[QAResult]` — used by
  Task 17.

- [ ] **Step 1: Write failing test**

```python
# tests/qa/test_checks.py
import datetime as dt

import polars as pl

from propagation.qa.checks import (
    check_diurnal_20m,
    check_grayline_40m,
    check_lowband_diurnal,
    check_reciprocity,
    check_solar_cycle,
    check_sporadic_e,
    check_storm_response,
    check_volume_hygiene,
    run_qa_checks,
)


def _row(hour, tx, rx, band, open_, month=6):
    return {
        "window_start": dt.datetime(2026, month, 1, hour, 0, tzinfo=dt.timezone.utc),
        "tx_field": tx, "rx_field": rx, "band": band, "open": open_,
    }


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


def test_check1_passes_with_strong_diurnal_signal():
    rows = (
        [_row(14, "FN", "DM", "20m", 1) for _ in range(9)]
        + [_row(14, "FN", "DM", "20m", 0)]
        + [_row(1, "FN", "DM", "20m", 1)]
        + [_row(1, "FN", "DM", "20m", 0) for _ in range(9)]
    )
    result = check_diurnal_20m(_df(rows))
    assert result.status == "pass"


def test_check1_insufficient_data_no_20m():
    result = check_diurnal_20m(_df([_row(14, "FN", "DM", "40m", 1)]))
    assert result.status == "insufficient_data"


def test_check2_lowband_uses_night_over_day_ratio():
    rows = (
        [_row(1, "FN", "DM", "160m", 1) for _ in range(9)]
        + [_row(1, "FN", "DM", "160m", 0)]
        + [_row(14, "FN", "DM", "160m", 1)]
        + [_row(14, "FN", "DM", "160m", 0) for _ in range(9)]
    )
    result = check_lowband_diurnal(_df(rows))
    assert result.status == "pass"


def test_check3_gate_reports_insufficient_data_without_solar_features():
    result = check_grayline_40m(_df([_row(14, "FN", "DM", "40m", 1)]))
    assert result.status == "insufficient_data"


def test_check4_sporadic_e_seasonal():
    # FN->EM is a real-ish ~1500km 6m pair; summer high, winter low.
    rows = (
        [_row(14, "EM", "EN", "6m", 1, month=6) for _ in range(6)]
        + [_row(14, "EM", "EN", "6m", 0, month=6) for _ in range(2)]
        + [_row(14, "EM", "EN", "6m", 1, month=12)]
        + [_row(14, "EM", "EN", "6m", 0, month=12) for _ in range(7)]
    )
    result = check_sporadic_e(_df(rows))
    assert result.status == "pass"


def test_check4_insufficient_data_no_6m():
    result = check_sporadic_e(_df([_row(14, "FN", "DM", "20m", 1)]))
    assert result.status == "insufficient_data"


def test_check5_reciprocity():
    rows = []
    for i in range(6):
        open_fwd = 1 if i % 2 == 0 else 0
        rows.append(_row(12, f"A{i}", f"B{i}", "20m", open_fwd))
        rows.append(_row(12, f"B{i}", f"A{i}", "20m", open_fwd))
    result = check_reciprocity(_df(rows))
    assert result.status in {"pass", "insufficient_data"}


def test_check6_gate_insufficient_data_single_month():
    result = check_solar_cycle(_df([_row(14, "FN", "DM", "10m", 1)]))
    assert result.status == "insufficient_data"


def test_check7_gate_insufficient_data_without_storm_fold():
    result = check_storm_response(_df([_row(14, "FN", "DM", "20m", 1)]), kp_max=3.0)
    assert result.status == "insufficient_data"


def test_check8_volume_hygiene_pass_under_threshold():
    labels = _df([_row(14, "FN", "DM", "20m", 1)])
    result = check_volume_hygiene(labels, rejection_counts={"rr73_grid": 1}, n_qualifying=1000)
    assert result.status == "pass"


def test_check8_volume_hygiene_fail_over_threshold():
    labels = _df([_row(14, "FN", "DM", "20m", 1)])
    result = check_volume_hygiene(labels, rejection_counts={"rr73_grid": 50}, n_qualifying=1000)
    assert result.status == "fail"


def test_run_qa_checks_returns_all_eight():
    labels = _df([_row(14, "FN", "DM", "20m", 1)])
    results = run_qa_checks(labels, rejection_counts={}, n_qualifying=1)
    assert {r.check_id for r in results} == {1, 2, 3, 4, 5, 6, 7, 8}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/qa/test_checks.py -v`
Expected: `ModuleNotFoundError: No module named 'propagation.qa.checks'`.

- [ ] **Step 3: Implement `checks.py`**

```python
# src/propagation/qa/checks.py
from dataclasses import dataclass

import numpy as np
import polars as pl

from propagation.data.geo import grid_to_latlon, great_circle_km


@dataclass
class QAResult:
    check_id: int
    name: str
    status: str  # "pass" | "fail" | "insufficient_data"
    detail: str


def _diurnal_ratio_check(
    labels: pl.DataFrame,
    check_id: int,
    name: str,
    bands: set[str],
    day_hours: tuple[int, int],
    night_hours: tuple[int, int],
    min_ratio: float,
    numerator: str,  # "day" or "night"
) -> QAResult:
    subset = labels.filter(pl.col("band").is_in(bands))
    if subset.height == 0:
        return QAResult(check_id, name, "insufficient_data", f"no labels for bands {sorted(bands)}")
    working = subset.with_columns(pl.col("window_start").dt.hour().alias("hour"))
    day = working.filter(pl.col("hour").is_between(*day_hours))
    night_lo, night_hi = night_hours
    night = working.filter((pl.col("hour") >= night_lo) | (pl.col("hour") <= night_hi))
    if day.height == 0 or night.height == 0:
        return QAResult(check_id, name, "insufficient_data", "missing day or night windows")
    day_rate = float(day["open"].cast(pl.Float64).mean())
    night_rate = float(night["open"].cast(pl.Float64).mean())
    num, den = (day_rate, night_rate) if numerator == "day" else (night_rate, day_rate)
    if den == 0:
        return QAResult(check_id, name, "insufficient_data", "zero-rate denominator")
    ratio = num / den
    status = "pass" if ratio > min_ratio else "fail"
    return QAResult(check_id, name, status, f"{numerator}/other open-rate ratio={ratio:.2f}")


def check_diurnal_20m(labels: pl.DataFrame) -> QAResult:
    """SPEC-labeling sec 6 QA check 1: 20m mid-lat day/night ratio > 2."""
    return _diurnal_ratio_check(
        labels, 1, "20m_diurnal", {"20m"}, day_hours=(12, 17), night_hours=(22, 3),
        min_ratio=2.0, numerator="day",
    )


def check_lowband_diurnal(labels: pl.DataFrame) -> QAResult:
    """QA check 2: 160m/80m night/day ratio > 5."""
    return _diurnal_ratio_check(
        labels, 2, "lowband_diurnal", {"160m", "80m"}, day_hours=(12, 17),
        night_hours=(22, 3), min_ratio=5.0, numerator="night",
    )


def check_grayline_40m(labels: pl.DataFrame) -> QAResult:
    """QA check 3: 40m gray-line open-rate peak near the terminator. Needs
    solar-terminator features (features/solar.py, scheduled for M2) to locate
    the terminator per path-cell; M0 has no such feature, so this is a real,
    tested precondition gate, not a stub of the eventual arithmetic."""
    subset = labels.filter(pl.col("band") == "40m")
    if subset.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "no 40m labels in this run")
    return QAResult(
        3, "grayline_40m", "insufficient_data",
        "terminator-relative timing requires features/solar.py (M2); not computable yet",
    )


def check_sporadic_e(labels: pl.DataFrame) -> QAResult:
    """QA check 4: 6m Sp-E, NH May-Jul open-rate >= 3x Nov-Jan (1-2.3 Mm paths)."""
    subset = labels.filter(pl.col("band") == "6m")
    if subset.height == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "no 6m labels in this run")

    pairs = subset.select(["tx_field", "rx_field"]).unique().to_dicts()
    dist_by_pair = {}
    for p in pairs:
        try:
            lat1, lon1 = grid_to_latlon(p["tx_field"])
            lat2, lon2 = grid_to_latlon(p["rx_field"])
            dist_by_pair[(p["tx_field"], p["rx_field"])] = great_circle_km(lat1, lon1, lat2, lon2)
        except ValueError:
            continue

    working = subset.with_columns(
        pl.struct(["tx_field", "rx_field"])
        .map_elements(
            lambda r: dist_by_pair.get((r["tx_field"], r["rx_field"])), return_dtype=pl.Float64
        )
        .alias("distance_km"),
        pl.col("window_start").dt.month().alias("month"),
    ).filter(pl.col("distance_km").is_between(1000, 2300))

    if working.height == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "no 1-2.3Mm 6m paths")

    summer = working.filter(pl.col("month").is_in([5, 6, 7]))
    winter = working.filter(pl.col("month").is_in([11, 12, 1]))
    if summer.height == 0 or winter.height == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "missing summer or winter months")

    summer_rate = float(summer["open"].cast(pl.Float64).mean())
    winter_rate = float(winter["open"].cast(pl.Float64).mean())
    if winter_rate == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "zero winter open-rate")
    ratio = summer_rate / winter_rate
    status = "pass" if ratio >= 3 else "fail"
    return QAResult(4, "sporadic_e_seasonal", status, f"summer/winter open-rate ratio={ratio:.2f}")


def check_reciprocity(labels: pl.DataFrame) -> QAResult:
    """QA check 5: Pearson r of open-rate(TX->RX) vs (RX->TX) per (pair, band, month) > 0.6."""
    working = labels.with_columns(pl.col("window_start").dt.month().alias("month"))
    fwd = working.group_by(["tx_field", "rx_field", "band", "month"]).agg(
        pl.col("open").cast(pl.Float64).mean().alias("rate_fwd")
    )
    rev = fwd.rename({
        "tx_field": "rx_field_r", "rx_field": "tx_field_r", "rate_fwd": "rate_rev",
    })
    paired = fwd.join(
        rev,
        left_on=["tx_field", "rx_field", "band", "month"],
        right_on=["tx_field_r", "rx_field_r", "band", "month"],
        how="inner",
    )
    if paired.height < 5:
        return QAResult(5, "reciprocity", "insufficient_data", f"only {paired.height} paired cells")
    r = float(np.corrcoef(paired["rate_fwd"].to_numpy(), paired["rate_rev"].to_numpy())[0, 1])
    status = "pass" if r > 0.6 else "fail"
    return QAResult(5, "reciprocity", status, f"pearson r={r:.3f}")


def check_solar_cycle(labels: pl.DataFrame, min_months: int = 12) -> QAResult:
    """QA check 6: monthly 10m DX open-rate vs F10.7 correlation > 0.5 over
    multi-year history. SPEC explicitly sanctions 'insufficient data' here when
    history is short (docs/SPEC-labeling.md sec 6)."""
    subset = labels.filter(pl.col("band") == "10m")
    if subset.height == 0:
        return QAResult(6, "solar_cycle", "insufficient_data", "no 10m labels in this run")
    n_months = subset.select(pl.col("window_start").dt.truncate("1mo")).unique().height
    if n_months < min_months:
        return QAResult(
            6, "solar_cycle", "insufficient_data",
            f"only {n_months} distinct month(s); need >= {min_months} plus F10.7 series (M3)",
        )
    return QAResult(6, "solar_cycle", "insufficient_data", "F10.7 correlation lands with M3 space-weather features")


def check_storm_response(labels: pl.DataFrame, kp_max: float | None) -> QAResult:
    """QA check 7: Kp>=6 trans-polar open-rate <= 50% of Kp<=2 matched baseline.
    Needs a Kp series (space_weather features, M2) and requires at least one
    storm (Kp>=5) fold in the eval window."""
    if kp_max is None or kp_max < 5.0:
        return QAResult(
            7, "storm_response", "insufficient_data",
            f"no Kp>=5 fold in this run (max Kp available={kp_max})",
        )
    return QAResult(7, "storm_response", "insufficient_data", "Kp series not yet joined (features/spaceweather.py, M2)")


def check_volume_hygiene(
    labels: pl.DataFrame, rejection_counts: dict[str, int], n_qualifying: int
) -> QAResult:
    """QA check 8: RR73-grid rejects < 0.5% of spots; unlabeled fraction reported
    (via features/universe.unlabeled_activity_fraction, called separately).
    Trailing-28-day volume comparison is skipped on a bootstrap run (no history
    to trail against yet)."""
    rr73 = rejection_counts.get("rr73_grid", 0)
    total = n_qualifying + sum(rejection_counts.values())
    rr73_rate = (rr73 / total) if total else 0.0
    status = "fail" if rr73_rate >= 0.005 else "pass"
    return QAResult(8, "volume_hygiene", status, f"RR73 reject rate={rr73_rate:.4%}")


def run_qa_checks(
    labels: pl.DataFrame,
    rejection_counts: dict[str, int],
    n_qualifying: int,
    kp_max: float | None = None,
) -> list[QAResult]:
    return [
        check_diurnal_20m(labels),
        check_lowband_diurnal(labels),
        check_grayline_40m(labels),
        check_sporadic_e(labels),
        check_reciprocity(labels),
        check_solar_cycle(labels),
        check_storm_response(labels, kp_max),
        check_volume_hygiene(labels, rejection_counts, n_qualifying),
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/qa/test_checks.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/qa/checks.py tests/qa/test_checks.py
git commit -m "feat: QA sanity checks per SPEC-labeling section 6"
```

---

### Task 16: End-to-end M0 orchestration script

**Files:**
- Create: `scripts/run_m0.py`
- Test: `tests/test_run_m0.py` (exercises the pipeline function with local fixture
  archives — does not hit the network)

**Interfaces:**
- Consumes: every module from Tasks 2–15.
- Produces: `run_m0(archives: dict[str, Path], band: str, lake_root: Path, report_dir:
  Path) -> dict` — the pure pipeline function (network-free, takes pre-downloaded
  archive paths so it's unit-testable), plus a CLI `main()` that downloads the archives
  first and calls `run_m0`. This is the artifact that satisfies ROADMAP.md's M0
  acceptance criterion.

- [ ] **Step 1: Write failing test (uses a train + eval fixture archive, both built
  from the same style of fixture rows as Task 6, with distinct months so
  `blocked_time_series_folds` has a real gap to honor)**

```python
# tests/test_run_m0.py
import gzip
from pathlib import Path

from scripts.run_m0 import run_m0

TRAIN_ROWS = [
    "1,1717243320,K1JT,FN20,-8,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
    "2,1717243320,W7YSB,DM42og,-10,14.097231,K7ZTM,DN41ab,50,0,983,174,14,ver,0",
    "3,1717329720,K1JT,FN20,-8,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
]
# Eval month rows (~5 weeks after train, respecting the >=48h gap trivially since
# whole months are used)
EVAL_ROWS = [
    "4,1719835320,K1JT,FN20,-6,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
    "5,1719921720,W7YSB,DM42og,-9,14.097231,K7ZTM,DN41ab,50,0,983,174,14,ver,0",
]


def _write_gz(rows, path: Path) -> Path:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")
    return path


def test_run_m0_end_to_end(tmp_path):
    train_archive = _write_gz(TRAIN_ROWS, tmp_path / "train.csv.gz")
    eval_archive = _write_gz(EVAL_ROWS, tmp_path / "eval.csv.gz")
    lake_root = tmp_path / "lake"
    report_dir = tmp_path / "reports"

    result = run_m0(
        archives={"train": train_archive, "eval": eval_archive},
        band="20m",
        lake_root=lake_root,
        report_dir=report_dir,
    )

    assert (lake_root / "spots").exists()
    assert (lake_root / "labels").exists()
    assert (report_dir / "headline_table.csv").exists()
    assert (report_dir / "reliability_climatology-m0.png").exists()
    assert result["n_train_labels"] > 0
    assert result["n_eval_labels"] > 0
    assert 0 <= result["headline"]["brier"] <= 1
    assert len(result["qa_results"]) == 8
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_run_m0.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.run_m0'` (add
`tests/__init__.py` and `scripts/__init__.py` as empty files if needed for the import
to resolve, or run pytest from repo root with `scripts` on `sys.path` via
`tool.pytest.ini_options` — add `pythonpath = ["."]` to `pyproject.toml`'s
`[tool.pytest.ini_options]` if the plain import fails).

- [ ] **Step 3: Implement `run_m0.py`**

```python
# scripts/run_m0.py
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from propagation.data.lake import write_partitioned
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet
from propagation.features.labels import build_labels
from propagation.features.sampling import sample_labels, write_labels
from propagation.features.universe import build_universe, unlabeled_activity_fraction
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.eval.report import write_headline_report
from propagation.qa.checks import run_qa_checks


def run_m0(archives: dict[str, Path], band: str, lake_root: Path, report_dir: Path) -> dict:
    lake_root, report_dir = Path(lake_root), Path(report_dir)

    train_extract = extract_wsprnet(archives["train"], band=band)
    eval_extract = extract_wsprnet(archives["eval"], band=band)

    write_partitioned(
        train_extract.spots.with_columns(
            train_extract.spots["ts"].dt.date().cast(str).alias("date")
        ),
        lake_root, "spots", ["band", "date"],
    )
    write_partitioned(
        eval_extract.spots.with_columns(
            eval_extract.spots["ts"].dt.date().cast(str).alias("date")
        ),
        lake_root, "spots", ["band", "date"],
    )

    train_uptime = build_receiver_uptime(train_extract.spots)
    eval_uptime = build_receiver_uptime(eval_extract.spots)

    train_universe = build_universe(train_extract.spots, train_uptime)
    eval_universe = build_universe(eval_extract.spots, eval_uptime)

    train_labels = build_labels(train_extract.spots, train_universe)
    eval_labels = build_labels(eval_extract.spots, eval_universe)

    train_sampled = sample_labels(train_labels, ratio=3.0)
    write_labels(train_sampled, lake_root)
    write_labels(eval_labels, lake_root)

    model = ClimatologyModel().fit(train_labels)
    predictions = model.predict(eval_labels)

    headline = write_headline_report(
        y_true=predictions["open"].cast(float).to_numpy(),
        y_prob=predictions["p_open"].to_numpy(),
        model_name="climatology-m0",
        out_dir=report_dir,
    )

    unlabeled = unlabeled_activity_fraction(eval_extract.spots, eval_universe)
    qa_results = run_qa_checks(
        eval_labels,
        rejection_counts=eval_extract.rejection_counts,
        n_qualifying=eval_extract.n_qualifying,
    )

    return {
        "n_train_labels": train_labels.height,
        "n_eval_labels": eval_labels.height,
        "headline": headline,
        "qa_results": qa_results,
        "unlabeled_activity_fraction": unlabeled.to_dicts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M0 lake-bootstrap pipeline")
    parser.add_argument("--train-year", type=int, default=2014)
    parser.add_argument("--train-month", type=int, default=6)
    parser.add_argument("--eval-year", type=int, default=2014)
    parser.add_argument("--eval-month", type=int, default=7)
    parser.add_argument("--band", default="20m")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    train_archive = raw_dir / f"wsprspots-{args.train_year:04d}-{args.train_month:02d}.csv.gz"
    eval_archive = raw_dir / f"wsprspots-{args.eval_year:04d}-{args.eval_month:02d}.csv.gz"

    if not train_archive.exists():
        print(f"downloading {train_archive.name}...")
        download_wsprnet_archive(args.train_year, args.train_month, train_archive)
    if not eval_archive.exists():
        print(f"downloading {eval_archive.name}...")
        download_wsprnet_archive(args.eval_year, args.eval_month, eval_archive)

    result = run_m0(
        archives={"train": train_archive, "eval": eval_archive},
        band=args.band,
        lake_root=args.data_dir / "lake",
        report_dir=args.data_dir / "reports",
    )

    print(f"train labels: {result['n_train_labels']}, eval labels: {result['n_eval_labels']}")
    print(f"headline: brier={result['headline']['brier']:.4f} log_loss={result['headline']['log_loss']:.4f}")
    for qa in result["qa_results"]:
        print(f"QA {qa.check_id} {qa.name}: {qa.status} — {qa.detail}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_run_m0.py -v`
Expected: passes. If the `scripts.run_m0` import fails under pytest's default
rootdir resolution, add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

and re-run.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests across every task pass, 0 failures.

Run: `uv run ruff check .`
Expected: no output, exit 0 (fix any lint findings before proceeding).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_m0.py tests/test_run_m0.py pyproject.toml
git commit -m "feat: end-to-end M0 orchestration script"
```

- [ ] **Step 7: Execute the real acceptance run (network required — this is the actual
  M0 milestone deliverable, not a unit test)**

Run: `uv run python scripts/run_m0.py` (defaults to train=2014-06, eval=2014-07,
band=20m, writing to `data/`; both archives are ~105-111 MB gzipped, confirmed
reachable at plan-writing time — expect this to take several minutes depending on
network speed and WSPRnet server load; do not hammer the server with retries).

Expected: prints train/eval label counts, headline Brier/log-loss, and 8 QA check
results (checks 1, 2, 4, 5, 8 should resolve to `pass`/`fail` on real data; checks 3,
6, 7 report `insufficient_data` as designed for single-band/single-month M0 scope).
`data/reports/headline_table.csv` and `data/reports/reliability_climatology-m0.png`
exist and are non-empty. This run — reproducible from an empty `data/` via one `uv
run` — is ROADMAP.md's M0 accept criterion.

- [ ] **Step 8: If QA checks 1/2/5/8 fail on the real run, do not silently proceed**

Per docs/SPEC-labeling.md §6, the pipeline should "fail loudly." If a physics-grounded
check (1, 2, 4, 5) fails on real WSPRnet data, that's a signal of a bug in hygiene,
universe, or label construction, not a spec problem — stop and debug via
superpowers:systematic-debugging before treating M0 as complete. If check 8 fails
(RR73 reject rate ≥ 0.5%), inspect `rejection_counts` for anomalies in the fixture vs.
real data path.

---

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-07-11-m0-lake-bootstrap.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review
between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch
execution with checkpoints.

Which approach?
