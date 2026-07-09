# M0 — Lake Bootstrap + Climatology Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From an empty `data/` directory, one `scripts/m0.sh` run extracts two months of WSPRnet 20m spots into a Parquet lake, builds receiver-uptime tables and monitor-normalized labels per SPEC-labeling.md, passes the QA gates, and produces a Brier score + reliability diagram for the climatology baseline evaluated on a held-out later month.

**Architecture:** Extractors write raw common-schema Parquet into a hive-layout lake (`data/lake/`); a hygiene stage materializes qualified+deduped spots (`spots_q`); label construction follows SPEC-labeling §1–§4 exactly (receiver-uptime evidence, activity-gated universe, monitor-normalized negatives, deterministic 3:1 sampling); the climatology model is a weighted, Laplace-smoothed open-rate lookup with hierarchical fallback; evaluation uses blocked time-series splits with a 48h gap and reports on the FULL label set.

**Tech Stack:** Python 3.11+, uv, polars, DuckDB, pyarrow, numpy, httpx, matplotlib, scikit-learn, pytest, ruff.

**Normative sources:** `docs/SPEC-labeling.md` (labeling), `ARCHITECTURE.md` §1–§3, §5–§6, `docs/superpowers/plans/2026-07-09-INTERFACES.md` (pinned signatures — use them EXACTLY).

## Global Constraints

- Python **3.11+**, `uv`-managed. `ruff` + `pytest`. License: MIT OR Apache-2.0.
- Layout: `src/propagation/…` (src layout), tests mirror under `tests/`.
- Lake root defaults to `./data/lake` (gitignored); every CLI takes
  `--lake-root` to override. Reports/artifacts to `./reports` (gitignored).
- All timestamps UTC. Polars dtype `pl.Datetime("us", "UTC")`; Parquet
  `timestamp[us, tz=UTC]`.
- 15-min windows aligned to UTC boundaries (minute ∈ {0,15,30,45}); a spot
  belongs to the window containing its `ts` (floor). Cells are directional.
- Supported bands (closed set, order canonical):
  `["160m","80m","60m","40m","30m","20m","17m","15m","12m","10m","6m"]`
- **No cqdx imports, ever.** `extract-cqdx` is optional, env-gated
  (`CQDX_R2_*` vars), behind the same extractor interface as public sources.
- Blocked time-series CV, gap ≥ 48 h; eval always on the FULL label set;
  no random splits. Δ_avail = 5 min availability buffer on all
  autoregressive features (train and serve identically).
- Sampling determinism: negatives 3:1 per (band, UTC date) stratum,
  `numpy.random.Generator(numpy.random.PCG64(seed))` with
  `seed = int.from_bytes(hashlib.sha256(f"{band}|{date_iso}".encode()).digest()[:8], "big") & 0xFFFFFFFF`
  where `date_iso` is `YYYY-MM-DD`.
- Conventional commits (`feat:`, `test:`, `chore:`, `docs:`); each milestone
  executes on its own branch, lands by PR. Main moves only by PR merge.

**M0-specific:** no network in tests (WSPRnet download tested against a local fixture `.csv.gz`); data months are 2026-05 (train) and 2026-06 (eval); band 20m.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/propagation/__init__.py`, `src/propagation/data/__init__.py`, `src/propagation/features/__init__.py`, `src/propagation/models/__init__.py`, `src/propagation/eval/__init__.py`, `tests/__init__.py`
- Modify: `.gitignore`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: installable `propagation` package; `uv run pytest` green; console-script table that every later task's CLI plugs into.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "propagation"
version = "0.1.0"
description = "ML-based HF propagation nowcasting"
requires-python = ">=3.11"
license = "MIT OR Apache-2.0"
dependencies = [
    "duckdb>=1.0",
    "polars>=1.0",
    "pyarrow>=16",
    "numpy>=1.26",
    "httpx>=0.27",
    "matplotlib>=3.8",
    "scikit-learn>=1.4",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.5"]

[project.scripts]
extract-wsprnet = "propagation.data.wsprnet:main"
build-spots-q = "propagation.data.hygiene:main"
build-uptime = "propagation.features.labels:main_uptime"
build-labels = "propagation.features.labels:main_labels"
qa-gates = "propagation.features.qa:main"
eval-report = "propagation.eval.report:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/propagation"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package skeleton and gitignore entries**

Create each `__init__.py` listed above as an empty file. Append to `.gitignore`:

```
data/
reports/
.venv/
__pycache__/
*.egg-info/
```

- [ ] **Step 3: Write smoke test** — `tests/test_smoke.py`:

```python
def test_import():
    import propagation  # noqa: F401
```

- [ ] **Step 4: Sync and run**

Run: `uv sync && uv run pytest -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src tests .gitignore
git commit -m "feat: project scaffolding (uv, src layout, console-script table)"
```

---

### Task 2: schema.py — bands, mode classes, windows, spot schema

**Files:**
- Create: `src/propagation/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces (pinned): `BANDS: list[str]`, `DIGI_MODES: frozenset[str]`, `CW_MODES: frozenset[str]`, `mode_class(mode: str) -> str`, `window_floor(ts: datetime) -> datetime`, `SPOT_SCHEMA: dict[str, pl.DataType]`, `QUALIFIED_EXTRA: dict[str, pl.DataType]`. Every later task keys frames off these.

- [ ] **Step 1: Write the failing tests** — `tests/test_schema.py`:

```python
from datetime import datetime, timezone

import polars as pl
import pytest

from propagation import schema


def test_bands_canonical_order():
    assert schema.BANDS == ["160m", "80m", "60m", "40m", "30m", "20m",
                            "17m", "15m", "12m", "10m", "6m"]


def test_mode_class():
    assert schema.mode_class("FT8") == "digi"
    assert schema.mode_class("wspr") == "digi"
    assert schema.mode_class("FST4W") == "digi"
    assert schema.mode_class("CW") == "cw"
    assert schema.mode_class("RTTY") == "cw"
    assert schema.mode_class("SSB") == "other"
    assert schema.mode_class("") == "other"


def test_window_floor():
    ts = datetime(2026, 5, 1, 12, 47, 33, 123456, tzinfo=timezone.utc)
    assert schema.window_floor(ts) == datetime(2026, 5, 1, 12, 45, tzinfo=timezone.utc)
    exact = datetime(2026, 5, 1, 12, 15, tzinfo=timezone.utc)
    assert schema.window_floor(exact) == exact


def test_window_floor_rejects_naive():
    with pytest.raises(ValueError):
        schema.window_floor(datetime(2026, 5, 1, 12, 0))


def test_spot_schema_exact():
    assert list(schema.SPOT_SCHEMA) == [
        "source", "ts", "band", "mode", "freq_hz", "dx_call", "de_call",
        "dx_grid", "de_grid", "dx_lat", "dx_lon", "de_lat", "de_lon",
        "snr_db", "tx_dbm", "distance_km", "bearing_deg",
    ]
    assert schema.SPOT_SCHEMA["ts"] == pl.Datetime("us", "UTC")
    assert schema.SPOT_SCHEMA["snr_db"] == pl.Int16
    assert schema.SPOT_SCHEMA["freq_hz"] == pl.Int64
    assert list(schema.QUALIFIED_EXTRA) == [
        "mode_class", "dx_field", "de_field", "dx_grid4", "de_grid4", "window_start",
    ]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_schema.py -q`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError` on `propagation.schema`.

- [ ] **Step 3: Implement** — `src/propagation/schema.py`:

```python
"""Canonical constants and frame schemas. Normative source: docs/SPEC-labeling.md."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

BANDS: list[str] = ["160m", "80m", "60m", "40m", "30m", "20m",
                    "17m", "15m", "12m", "10m", "6m"]

DIGI_MODES: frozenset[str] = frozenset(
    {"FT8", "FT4", "WSPR", "FST4", "FST4W", "JS8", "JT65", "JT9", "Q65", "MSK144"})
CW_MODES: frozenset[str] = frozenset({"CW", "RTTY"})


def mode_class(mode: str) -> str:
    m = (mode or "").upper()
    if m in DIGI_MODES:
        return "digi"
    if m in CW_MODES:
        return "cw"
    return "other"


def window_floor(ts: datetime) -> datetime:
    """Floor to the containing 15-minute UTC window (SPEC-labeling notation)."""
    if ts.tzinfo is None:
        raise ValueError("window_floor requires a timezone-aware datetime")
    ts = ts.astimezone(timezone.utc)
    return ts.replace(minute=ts.minute - ts.minute % 15, second=0, microsecond=0)


SPOT_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "band": pl.Utf8,
    "mode": pl.Utf8, "freq_hz": pl.Int64, "dx_call": pl.Utf8, "de_call": pl.Utf8,
    "dx_grid": pl.Utf8, "de_grid": pl.Utf8,
    "dx_lat": pl.Float64, "dx_lon": pl.Float64,
    "de_lat": pl.Float64, "de_lon": pl.Float64,
    "snr_db": pl.Int16, "tx_dbm": pl.Int16,
    "distance_km": pl.Float64, "bearing_deg": pl.Float64,
}

QUALIFIED_EXTRA: dict[str, pl.DataType] = {
    "mode_class": pl.Utf8, "dx_field": pl.Utf8, "de_field": pl.Utf8,
    "dx_grid4": pl.Utf8, "de_grid4": pl.Utf8,
    "window_start": pl.Datetime("us", "UTC"),
}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_schema.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/schema.py tests/test_schema.py
git commit -m "feat: canonical schema module (bands, mode classes, windows, spot schema)"
```

---

### Task 3: Maidenhead + great-circle geometry helpers

Hygiene needs grid↔lat/lon conversion and distances (SPEC §1.2 location rules, §1.5 25 km floor). We create `features/geometry.py` now with exactly the pinned M2 signatures for the functions M0 needs; M2 adds the rest of the module (bearing, midpoint, control points, geomag lat) later.

**Files:**
- Create: `src/propagation/features/geometry.py`
- Test: `tests/features/test_geometry.py` (create `tests/features/__init__.py`)

**Interfaces:**
- Consumes: nothing.
- Produces (pinned in INTERFACES.md §features): `grid_to_latlon(grid: str) -> tuple[float, float]` (center of field or grid4), `haversine_km(lat1, lon1, lat2, lon2) -> float`. M0 extension (not pinned, kept here for M2 reuse): `latlon_to_grid4(lat: float, lon: float) -> str`.

- [ ] **Step 1: Write the failing tests** — `tests/features/test_geometry.py`:

```python
import pytest

from propagation.features import geometry


def test_grid_to_latlon_field_center():
    lat, lon = geometry.grid_to_latlon("EM")
    assert (lat, lon) == (35.0, -90.0)


def test_grid_to_latlon_grid4_center():
    lat, lon = geometry.grid_to_latlon("EM12")
    assert (lat, lon) == (32.5, -97.0)


def test_grid_to_latlon_rejects_garbage():
    with pytest.raises(ValueError):
        geometry.grid_to_latlon("E")
    with pytest.raises(ValueError):
        geometry.grid_to_latlon("EMXY")


def test_latlon_to_grid4_roundtrip():
    assert geometry.latlon_to_grid4(35.0, -90.0) == "EM55"
    assert geometry.latlon_to_grid4(32.5, -97.0) == "EM12"
    assert geometry.latlon_to_grid4(89.99, 179.99) == "RR99"
    assert geometry.latlon_to_grid4(-90.0, -180.0) == "AA00"


def test_haversine_equator_degree():
    # 1 degree of longitude at the equator ~= 111.19 km
    assert geometry.haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.1)


def test_haversine_zero():
    assert geometry.haversine_km(35.0, -90.0, 35.0, -90.0) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_geometry.py -q`
Expected: FAIL — `ModuleNotFoundError: propagation.features.geometry`.

- [ ] **Step 3: Implement** — `src/propagation/features/geometry.py`:

```python
"""Maidenhead and great-circle geometry. M0 subset; M2 extends this module."""
from __future__ import annotations

import math

_EARTH_RADIUS_KM = 6371.0088


def grid_to_latlon(grid: str) -> tuple[float, float]:
    """Center lat/lon of a 2-char Maidenhead field or 4-char grid square."""
    g = grid.strip().upper()
    if len(g) not in (2, 4) or not ("A" <= g[0] <= "R" and "A" <= g[1] <= "R"):
        raise ValueError(f"invalid Maidenhead locator: {grid!r}")
    lon = (ord(g[0]) - ord("A")) * 20.0 - 180.0
    lat = (ord(g[1]) - ord("A")) * 10.0 - 90.0
    if len(g) == 2:
        return (lat + 5.0, lon + 10.0)
    if not (g[2].isdigit() and g[3].isdigit()):
        raise ValueError(f"invalid Maidenhead locator: {grid!r}")
    lon += int(g[2]) * 2.0
    lat += int(g[3]) * 1.0
    return (lat + 0.5, lon + 1.0)


def latlon_to_grid4(lat: float, lon: float) -> str:
    """4-char Maidenhead grid square containing (lat, lon)."""
    lon = min(max(lon, -180.0), 179.999999)
    lat = min(max(lat, -90.0), 89.999999)
    f1 = chr(ord("A") + int((lon + 180.0) // 20))
    f2 = chr(ord("A") + int((lat + 90.0) // 10))
    s1 = str(int(((lon + 180.0) % 20.0) // 2))
    s2 = str(int((lat + 90.0) % 10.0))
    return f"{f1}{f2}{s1}{s2}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_geometry.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/geometry.py tests/features
git commit -m "feat: maidenhead + haversine geometry helpers (M0 subset)"
```

---

### Task 4: data/lake.py — partition layout, writer, DuckDB views

**Files:**
- Create: `src/propagation/data/lake.py`
- Test: `tests/data/test_lake.py` (create `tests/data/__init__.py`)

**Interfaces:**
- Consumes: `schema.SPOT_SCHEMA`.
- Produces (pinned): `class Lake` with `root: Path`, `spots_dir(band, d)`, `spots_q_dir(band, d)`, `uptime_dir(band, d)`, `labels_dir(band, d)`, `space_weather_dir(d)`, `write_parquet(df, dest_dir, name="part-0") -> Path`, `connect() -> duckdb.DuckDBPyConnection` registering views `raw_spots`, `spots` (= spots_q), `receiver_uptime`, `labels`, `space_weather` for datasets that have files. Partition-key columns (band, window/ts dates) live IN the files; directories are layout convention only, so views are plain globs without hive_partitioning (avoids duplicate-column clashes).

- [ ] **Step 1: Write the failing tests** — `tests/data/test_lake.py`:

```python
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from propagation.data.lake import Lake


def test_partition_paths(tmp_path: Path):
    lake = Lake(tmp_path)
    d = date(2026, 5, 1)
    assert lake.spots_dir("20m", d) == tmp_path / "spots" / "band=20m" / "date=2026-05-01"
    assert lake.spots_q_dir("20m", d) == tmp_path / "spots_q" / "band=20m" / "date=2026-05-01"
    assert lake.uptime_dir("20m", d) == tmp_path / "receiver_uptime" / "band=20m" / "date=2026-05-01"
    assert lake.labels_dir("20m", d) == tmp_path / "labels" / "band=20m" / "date=2026-05-01"
    assert lake.space_weather_dir(d) == tmp_path / "space_weather" / "date=2026-05-01"


def test_write_and_view_roundtrip(tmp_path: Path):
    lake = Lake(tmp_path)
    df = pl.DataFrame({
        "band": ["20m", "20m"],
        "ts": [datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
               datetime(2026, 5, 1, 12, 5, tzinfo=timezone.utc)],
        "dx_call": ["K5ARH", "W1AW"],
    })
    out = lake.write_parquet(df, lake.spots_q_dir("20m", date(2026, 5, 1)))
    assert out.exists() and out.name == "part-0.parquet"
    con = lake.connect()
    n = con.execute("SELECT count(*) FROM spots").fetchone()[0]
    assert n == 2


def test_connect_skips_empty_datasets(tmp_path: Path):
    con = Lake(tmp_path).connect()
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "spots" not in tables and "labels" not in tables
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_lake.py -q`
Expected: FAIL — `ModuleNotFoundError: propagation.data.lake`.

- [ ] **Step 3: Implement** — `src/propagation/data/lake.py`:

```python
"""Lake layout (ARCHITECTURE §3.1) and DuckDB view registry."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

_VIEWS: dict[str, str] = {
    "raw_spots": "spots",
    "spots": "spots_q",           # everything downstream reads qualified+deduped
    "receiver_uptime": "receiver_uptime",
    "labels": "labels",
    "space_weather": "space_weather",
}


class Lake:
    def __init__(self, root: Path):
        self.root = Path(root)

    def spots_dir(self, band: str, d: date) -> Path:
        return self.root / "spots" / f"band={band}" / f"date={d.isoformat()}"

    def spots_q_dir(self, band: str, d: date) -> Path:
        return self.root / "spots_q" / f"band={band}" / f"date={d.isoformat()}"

    def uptime_dir(self, band: str, d: date) -> Path:
        return self.root / "receiver_uptime" / f"band={band}" / f"date={d.isoformat()}"

    def labels_dir(self, band: str, d: date) -> Path:
        return self.root / "labels" / f"band={band}" / f"date={d.isoformat()}"

    def space_weather_dir(self, d: date) -> Path:
        return self.root / "space_weather" / f"date={d.isoformat()}"

    def write_parquet(self, df: pl.DataFrame, dest_dir: Path, name: str = "part-0") -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / f"{name}.parquet"
        df.write_parquet(out)
        return out

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        for view, subdir in _VIEWS.items():
            base = self.root / subdir
            if base.exists() and any(base.rglob("*.parquet")):
                glob = str(base / "**" / "*.parquet")
                con.execute(
                    f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{glob}')")
        return con
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/data/test_lake.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/data/lake.py tests/data
git commit -m "feat: lake partition layout, parquet writer, duckdb view registry"
```

---

### Task 5: data/hygiene.py — spot hygiene, dedup, `build-spots-q` CLI

Implements SPEC-labeling §1 exactly. Reject-reason counts are persisted per band/date (`lake/hygiene_stats/`) because QA check 8 needs the RR73 rejection rate.

**Files:**
- Create: `src/propagation/data/hygiene.py`
- Create: `tests/conftest.py` (shared synthetic-spot fixture)
- Test: `tests/data/test_hygiene.py`

**Interfaces:**
- Consumes: `schema.SPOT_SCHEMA/QUALIFIED_EXTRA/BANDS/mode_class`, `geometry.grid_to_latlon/latlon_to_grid4/haversine_km`, `Lake`.
- Produces (pinned): `qualify_spots(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]` (qualified with QUALIFIED_EXTRA columns + lat/lon/distance filled; rejects with `reject_reason` Utf8), `dedup_spots(df: pl.DataFrame) -> pl.DataFrame`, `CALLSIGN_RE: re.Pattern`. Extension: `main()` for `build-spots-q` CLI; `lake/hygiene_stats/band=…/date=…/part-0.parquet` with columns `band, date, reject_reason, n` (`reject_reason = NULL` row counts qualifying spots).
- Reject reasons (closed set, first failing rule wins, SPEC §1 order): `missing_core`, `rr73`, `no_location`, `callsign`, `self_spot`, `too_close`.

- [ ] **Step 1: Write the shared fixture** — `tests/conftest.py`:

```python
from datetime import datetime, timezone

import polars as pl
import pytest

from propagation import schema

BASE = {
    "source": "pskreporter",
    "ts": datetime(2026, 5, 1, 12, 3, tzinfo=timezone.utc),
    "band": "20m", "mode": "FT8", "freq_hz": 14074000,
    "dx_call": "K5ARH", "de_call": "G4ABC",
    "dx_grid": "EM12", "de_grid": "IO91",
    "dx_lat": None, "dx_lon": None, "de_lat": None, "de_lon": None,
    "snr_db": -10, "tx_dbm": None, "distance_km": None, "bearing_deg": None,
}


@pytest.fixture
def make_spots():
    def _make(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame([BASE | r for r in rows], schema=schema.SPOT_SCHEMA)
    return _make
```

- [ ] **Step 2: Write the failing tests** — `tests/data/test_hygiene.py`:

```python
from datetime import datetime, timezone

from propagation.data import hygiene


def test_qualifying_spot_gets_derived_columns(make_spots):
    ok, rej = hygiene.qualify_spots(make_spots([{}]))
    assert rej.height == 0 and ok.height == 1
    row = ok.row(0, named=True)
    assert row["mode_class"] == "digi"
    assert row["dx_field"] == "EM" and row["de_field"] == "IO"
    assert row["dx_grid4"] == "EM12" and row["de_grid4"] == "IO91"
    assert row["window_start"] == datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    assert row["distance_km"] and row["distance_km"] > 6000  # EM12 -> IO91


def test_missing_core_fields_rejected(make_spots):
    ok, rej = hygiene.qualify_spots(make_spots([{"band": None}, {"band": "23cm"},
                                                {"mode": None}]))
    assert ok.height == 0
    assert rej["reject_reason"].to_list() == ["missing_core"] * 3


def test_rr73_blocklisted(make_spots):
    ok, rej = hygiene.qualify_spots(make_spots([{"dx_grid": "RR73"}]))
    assert ok.height == 0 and rej["reject_reason"].to_list() == ["rr73"]


def test_grid_normalization(make_spots):
    # 6-char grid truncates to 4; bare field kept with grid4 NULL; 3-char invalid
    ok, rej = hygiene.qualify_spots(make_spots([
        {"dx_grid": "em12ab"},
        {"dx_grid": "EM"},
        {"dx_grid": "EM1"},
    ]))
    assert ok.height == 2 and rej["reject_reason"].to_list() == ["no_location"]
    assert ok["dx_grid4"].to_list() == ["EM12", None]
    assert ok["dx_field"].to_list() == ["EM", "EM"]


def test_latlon_fallback_derives_field(make_spots):
    ok, rej = hygiene.qualify_spots(make_spots([
        {"dx_grid": None, "dx_lat": 32.5, "dx_lon": -97.0}]))
    assert ok.height == 1
    assert ok["dx_field"].to_list() == ["EM"] and ok["dx_grid4"].to_list() == ["EM12"]


def test_callsign_rules(make_spots):
    ok, rej = hygiene.qualify_spots(make_spots([
        {"dx_call": "K5ARH/MM"},        # maritime mobile
        {"de_call": "G4ABC/AM"},        # aeronautical mobile
        {"dx_call": "QRZ?"},            # implausible
        {"dx_call": "<K5ARH>"},         # hashed-call markers stripped -> ok
        {"dx_call": "EA8/K1ABC"},       # leading prefix not in grammar -> reject
    ]))
    assert rej["reject_reason"].to_list() == ["callsign", "callsign", "callsign", "callsign"]
    assert ok.height == 1 and ok["dx_call"].to_list() == ["<K5ARH>"]


def test_self_spot_and_distance_floor(make_spots):
    ok, rej = hygiene.qualify_spots(make_spots([
        {"de_call": "K5ARH/P"},                          # same base call -> self_spot
        {"dx_grid": "IO91", "de_grid": "IO91"},          # same square -> 0 km
    ]))
    assert ok.height == 0
    assert rej["reject_reason"].to_list() == ["self_spot", "too_close"]


def test_dedup_priority_and_tiebreaks(make_spots):
    df, _ = hygiene.qualify_spots(make_spots([
        {"source": "pskreporter", "snr_db": -5},
        {"source": "wsprnet", "mode": "FT8", "snr_db": -20},   # wins on source
        {"source": "rbn", "snr_db": 0},
    ]))
    out = hygiene.dedup_spots(df)
    assert out.height == 1 and out["source"].to_list() == ["wsprnet"]

    df2, _ = hygiene.qualify_spots(make_spots([
        {"snr_db": -5}, {"snr_db": -1},
        {"snr_db": -1, "ts": datetime(2026, 5, 1, 12, 1, tzinfo=timezone.utc)},
    ]))
    out2 = hygiene.dedup_spots(df2)
    assert out2.height == 1
    r = out2.row(0, named=True)
    assert r["snr_db"] == -1 and r["ts"].minute == 1  # max snr, then min ts


def test_dedup_keeps_distinct_windows(make_spots):
    df, _ = hygiene.qualify_spots(make_spots([
        {}, {"ts": datetime(2026, 5, 1, 12, 20, tzinfo=timezone.utc)}]))
    assert hygiene.dedup_spots(df).height == 2
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/data/test_hygiene.py -q`
Expected: FAIL — `ModuleNotFoundError: propagation.data.hygiene`.

- [ ] **Step 4: Implement** — `src/propagation/data/hygiene.py`:

```python
"""Spot hygiene and cross-source dedup. Normative: docs/SPEC-labeling.md §1."""
from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from propagation import schema
from propagation.data.lake import Lake
from propagation.features.geometry import grid_to_latlon, haversine_km, latlon_to_grid4

CALLSIGN_RE = re.compile(r"^[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z](/[A-Z0-9]{1,4})?$")
_SOURCE_PRIORITY = {"wsprnet": 0, "rbn": 1, "pskreporter": 2, "cqdx": 3}
_SUFFIX_RE = re.compile(r"/[A-Z0-9]{1,4}$")


def _clean_call(call: str | None) -> str | None:
    """Strip hashed-call markers; None if implausible or /MM //AM (SPEC §1.3)."""
    if not call:
        return None
    c = call.strip().upper().removeprefix("<").removesuffix(">")
    if c.endswith(("/MM", "/AM")):
        return None
    return c if CALLSIGN_RE.match(c) else None


def _resolve_loc(grid: str | None, lat: float | None, lon: float | None) -> dict:
    """SPEC §1.2. status in {ok, rr73, none}; field/grid4/lat/lon set when ok."""
    if grid and grid.strip():
        g = grid.strip().upper()[:4]
        if g == "RR73":
            return {"status": "rr73", "field": None, "grid4": None, "lat": None, "lon": None}
        if (len(g) == 4 and "A" <= g[0] <= "R" and "A" <= g[1] <= "R"
                and g[2].isdigit() and g[3].isdigit()):
            la, lo = grid_to_latlon(g)
            return {"status": "ok", "field": g[:2], "grid4": g, "lat": la, "lon": lo}
        if len(g) == 2 and "A" <= g[0] <= "R" and "A" <= g[1] <= "R":
            la, lo = grid_to_latlon(g)
            return {"status": "ok", "field": g, "grid4": None, "lat": la, "lon": lo}
    if lat is not None and lon is not None:
        g4 = latlon_to_grid4(lat, lon)
        return {"status": "ok", "field": g4[:2], "grid4": g4, "lat": lat, "lon": lon}
    return {"status": "none", "field": None, "grid4": None, "lat": None, "lon": None}


def qualify_spots(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split into (qualifying spots + QUALIFIED_EXTRA columns, rejects with reject_reason)."""
    reasons: list[str | None] = []
    rows: list[dict] = []
    for r in df.iter_rows(named=True):
        if (r["ts"] is None or r["mode"] is None or r["band"] not in schema.BANDS
                or not r["dx_call"] or not r["de_call"]):
            reasons.append("missing_core"); rows.append(r); continue
        dx = _resolve_loc(r["dx_grid"], r["dx_lat"], r["dx_lon"])
        de = _resolve_loc(r["de_grid"], r["de_lat"], r["de_lon"])
        if dx["status"] == "rr73" or de["status"] == "rr73":
            reasons.append("rr73"); rows.append(r); continue
        if dx["status"] != "ok" or de["status"] != "ok":
            reasons.append("no_location"); rows.append(r); continue
        dxc, dec = _clean_call(r["dx_call"]), _clean_call(r["de_call"])
        if dxc is None or dec is None:
            reasons.append("callsign"); rows.append(r); continue
        if _SUFFIX_RE.sub("", dxc) == _SUFFIX_RE.sub("", dec):
            reasons.append("self_spot"); rows.append(r); continue
        dist = r["distance_km"]
        if dist is None:
            dist = haversine_km(dx["lat"], dx["lon"], de["lat"], de["lon"])
        if dist < 25.0:
            reasons.append("too_close"); rows.append(r); continue
        reasons.append(None)
        rows.append(r | {
            "dx_lat": dx["lat"], "dx_lon": dx["lon"], "de_lat": de["lat"],
            "de_lon": de["lon"], "distance_km": dist,
            "mode_class": schema.mode_class(r["mode"]),
            "dx_field": dx["field"], "de_field": de["field"],
            "dx_grid4": dx["grid4"], "de_grid4": de["grid4"],
            "window_start": schema.window_floor(r["ts"]),
        })
    full_schema = dict(schema.SPOT_SCHEMA) | dict(schema.QUALIFIED_EXTRA)
    ok = pl.DataFrame([r for r, why in zip(rows, reasons) if why is None],
                      schema=full_schema)
    rej = pl.DataFrame(
        [r | {"reject_reason": why} for r, why in zip(rows, reasons) if why is not None],
        schema=dict(schema.SPOT_SCHEMA) | {"reject_reason": pl.Utf8})
    return ok, rej


def dedup_spots(df: pl.DataFrame) -> pl.DataFrame:
    """SPEC §1 cross-source dedup: key (dx_call, de_call, band, mode, window_start);
    keep by source priority, then highest non-null snr_db, then lowest ts."""
    key = ["dx_call", "de_call", "band", "mode", "window_start"]
    return (
        df.with_columns(
            pl.col("source").replace_strict(_SOURCE_PRIORITY, default=99).alias("_prio"),
            pl.col("snr_db").fill_null(-(2 ** 15)).alias("_snr"))
        .sort(["_prio", "_snr", "ts"], descending=[False, True, False])
        .unique(subset=key, keep="first", maintain_order=True)
        .drop(["_prio", "_snr"])
    )


def build_spots_q(lake: Lake, band: str, start: date, end: date) -> None:
    """Qualify+dedup raw spots per day; write spots_q and hygiene_stats."""
    d = start
    while d <= end:
        src = lake.spots_dir(band, d)
        if src.exists() and any(src.glob("*.parquet")):
            raw = pl.read_parquet(src / "*.parquet")
            ok, rej = qualify_spots(raw)
            ok = dedup_spots(ok)
            lake.write_parquet(ok, lake.spots_q_dir(band, d))
            stats = pl.concat([
                rej.group_by("reject_reason").len(name="n"),
                pl.DataFrame({"reject_reason": [None], "n": [ok.height]},
                             schema={"reject_reason": pl.Utf8, "n": pl.UInt32}),
            ]).with_columns(band=pl.lit(band), date=pl.lit(d.isoformat()))
            lake.write_parquet(
                stats, lake.root / "hygiene_stats" / f"band={band}" / f"date={d.isoformat()}")
        d += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualify + dedup raw spots into spots_q")
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--band", required=True, choices=schema.BANDS)
    ap.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    args = ap.parse_args()
    build_spots_q(Lake(args.lake_root), args.band, args.start, args.end)


if __name__ == "__main__":
    main()
```

Note: `qualify_spots` is a row loop — correct first, fast later. A month of 20m WSPR is a few million rows (minutes of CPU); acceptable for M0, revisit with vectorized polars expressions only if it becomes the bottleneck.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/data/test_hygiene.py -q`
Expected: PASS (9 tests).

- [ ] **Step 6: Add a build_spots_q integration test** — append to `tests/data/test_hygiene.py`:

```python
from datetime import date

import polars as pl

from propagation.data.lake import Lake


def test_build_spots_q_writes_partitions_and_stats(tmp_path, make_spots):
    lake = Lake(tmp_path)
    raw = make_spots([{}, {"dx_grid": "RR73"}])
    lake.write_parquet(raw, lake.spots_dir("20m", date(2026, 5, 1)))
    hygiene.build_spots_q(lake, "20m", date(2026, 5, 1), date(2026, 5, 1))
    q = pl.read_parquet(lake.spots_q_dir("20m", date(2026, 5, 1)) / "*.parquet")
    assert q.height == 1
    stats = pl.read_parquet(
        tmp_path / "hygiene_stats" / "band=20m" / "date=2026-05-01" / "*.parquet")
    assert stats.filter(pl.col("reject_reason") == "rr73")["n"].item() == 1
```

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest tests/data/test_hygiene.py -q`
Expected: PASS (10 tests).

- [ ] **Step 8: Commit**

```bash
git add src/propagation/data/hygiene.py tests/conftest.py tests/data/test_hygiene.py
git commit -m "feat: spot hygiene, cross-source dedup, build-spots-q (SPEC-labeling §1)"
```

---

### Task 6: data/base.py + data/wsprnet.py — extractor interface and WSPRnet extractor

**Files:**
- Create: `src/propagation/data/base.py`, `src/propagation/data/wsprnet.py`
- Test: `tests/data/test_wsprnet.py`

**Interfaces:**
- Consumes: `Lake`, `schema.BANDS/SPOT_SCHEMA`.
- Produces (pinned): `Extractor` protocol (`source: str`; `extract(start: date, end: date, bands: list[str], lake: Lake, cache_dir: Path) -> list[Path]`); `WsprnetExtractor` implementing it; `main()` for `extract-wsprnet` CLI. Helpers: `archive_url(year: int, month: int) -> str`, `download_archive(url: str, dest: Path) -> Path`, `parse_archive(path: Path) -> pl.DataFrame` (raw SPOT_SCHEMA), `band_for_freq_hz(freq_hz: int) -> str | None`.
- WSPRnet monthly archive: `http://wsprnet.org/archive/wsprspots-YYYY-MM.csv.gz`, headerless CSV, columns in order: `spot_id, timestamp (epoch s), reporter, reporter_grid, snr, freq_mhz, call_sign, grid, power_dbm, drift, distance, azimuth, band, version, code`. Mapping: `de_call=reporter, de_grid=reporter_grid, dx_call=call_sign, dx_grid=grid, snr_db=snr, tx_dbm=power_dbm, freq_hz=round(freq_mhz*1e6), distance_km=distance, bearing_deg=azimuth, ts=epoch→UTC, mode="WSPR", source="wsprnet"`; `band` derived from `freq_hz` (ignore the archive's integer band column — our band strings are canonical). Cache the `.gz` in `cache_dir`; never redistribute it.

- [ ] **Step 1: Write the failing tests** — `tests/data/test_wsprnet.py`:

```python
import gzip
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from propagation.data import wsprnet
from propagation.data.lake import Lake


def _epoch(y, mo, d, h, mi):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp())


def _fixture_gz(path: Path) -> Path:
    rows = [
        f"1,{_epoch(2026, 5, 1, 12, 4)},G4ABC,IO91,-21,14.097102,K5ARH,EM12,37,0,7523,45,14,2.3_r4,0",
        f"2,{_epoch(2026, 5, 1, 12, 4)},G4ABC,IO91,-15,7.040102,W1AW,FN31,30,0,5280,52,7,2.3_r4,0",
        f"3,{_epoch(2026, 5, 2, 0, 30)},VK3XYZ,QF22,-28,14.097080,K5ARH,EM12,37,-1,14800,265,14,2.3_r4,0",
    ]
    gz = path / "wsprspots-2026-05.csv.gz"
    gz.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))
    return gz


def test_archive_url():
    assert wsprnet.archive_url(2026, 5) == "http://wsprnet.org/archive/wsprspots-2026-05.csv.gz"


def test_band_for_freq_hz():
    assert wsprnet.band_for_freq_hz(14_097_102) == "20m"
    assert wsprnet.band_for_freq_hz(7_040_102) == "40m"
    assert wsprnet.band_for_freq_hz(1_838_000) == "160m"
    assert wsprnet.band_for_freq_hz(50_293_000) == "6m"
    assert wsprnet.band_for_freq_hz(999) is None


def test_parse_archive_maps_columns(tmp_path):
    df = wsprnet.parse_archive(_fixture_gz(tmp_path))
    assert df.height == 3
    r = df.row(0, named=True)
    assert r["source"] == "wsprnet" and r["mode"] == "WSPR"
    assert r["band"] == "20m" and r["freq_hz"] == 14_097_102
    assert r["dx_call"] == "K5ARH" and r["de_call"] == "G4ABC"
    assert r["dx_grid"] == "EM12" and r["de_grid"] == "IO91"
    assert r["snr_db"] == -21 and r["tx_dbm"] == 37
    assert r["distance_km"] == 7523.0 and r["bearing_deg"] == 45.0
    assert r["ts"] == datetime(2026, 5, 1, 12, 4, tzinfo=timezone.utc)
    assert df.schema["ts"] == pl.Datetime("us", "UTC")


def test_extract_writes_band_date_partitions(tmp_path, monkeypatch):
    gz = _fixture_gz(tmp_path)
    monkeypatch.setattr(wsprnet, "download_archive", lambda url, dest: gz)
    lake = Lake(tmp_path / "lake")
    written = wsprnet.WsprnetExtractor().extract(
        date(2026, 5, 1), date(2026, 5, 31), ["20m"], lake, tmp_path)
    assert len(written) == 2  # 20m rows on 05-01 and 05-02; 40m row filtered out
    df = pl.read_parquet(lake.spots_dir("20m", date(2026, 5, 1)) / "*.parquet")
    assert df.height == 1 and df["dx_call"].to_list() == ["K5ARH"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_wsprnet.py -q`
Expected: FAIL — `ModuleNotFoundError: propagation.data.wsprnet`.

- [ ] **Step 3: Implement** — `src/propagation/data/base.py`:

```python
"""Extractor interface (ARCHITECTURE §3.2). All sources, public and private,
implement this; nothing downstream may care which one produced a row."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

from propagation.data.lake import Lake


@runtime_checkable
class Extractor(Protocol):
    source: str  # 'wsprnet' | 'rbn' | 'pskreporter' | 'cqdx' | 'swpc'

    def extract(self, start: date, end: date, bands: list[str],
                lake: Lake, cache_dir: Path) -> list[Path]:
        """Download/convert; write raw-schema Parquet into the lake; return files written."""
        ...
```

and `src/propagation/data/wsprnet.py`:

```python
"""WSPRnet monthly-archive extractor — the deep-history backbone (ARCHITECTURE §3.2)."""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import polars as pl

from propagation import schema
from propagation.data.lake import Lake

_USER_AGENT = "propagation/0.1 (research pipeline; +https://github.com/HagaleTechnologies/propagation)"
_CSV_COLUMNS = ["spot_id", "epoch", "reporter", "reporter_grid", "snr", "freq_mhz",
                "call_sign", "grid", "power_dbm", "drift", "distance", "azimuth",
                "band_int", "version", "code"]
# (low_hz, high_hz, band) — ITU Region-agnostic outer envelopes
_BAND_EDGES = [
    (1_800_000, 2_000_000, "160m"), (3_500_000, 4_000_000, "80m"),
    (5_200_000, 5_450_000, "60m"), (7_000_000, 7_300_000, "40m"),
    (10_100_000, 10_150_000, "30m"), (14_000_000, 14_350_000, "20m"),
    (18_068_000, 18_168_000, "17m"), (21_000_000, 21_450_000, "15m"),
    (24_890_000, 24_990_000, "12m"), (28_000_000, 29_700_000, "10m"),
    (50_000_000, 54_000_000, "6m"),
]


def band_for_freq_hz(freq_hz: int) -> str | None:
    for lo, hi, band in _BAND_EDGES:
        if lo <= freq_hz <= hi:
            return band
    return None


def archive_url(year: int, month: int) -> str:
    return f"http://wsprnet.org/archive/wsprspots-{year:04d}-{month:02d}.csv.gz"


def download_archive(url: str, dest: Path) -> Path:
    """Cache-once download with polite retry/backoff."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with httpx.stream("GET", url, headers={"User-Agent": _USER_AGENT},
                              timeout=120.0, follow_redirects=True) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                tmp.rename(dest)
                return dest
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def parse_archive(path: Path) -> pl.DataFrame:
    raw = pl.read_csv(path, has_header=False, new_columns=_CSV_COLUMNS,
                      schema_overrides={"freq_mhz": pl.Float64, "power_dbm": pl.Int16,
                                        "snr": pl.Int16, "distance": pl.Float64,
                                        "azimuth": pl.Float64, "epoch": pl.Int64})
    return (
        raw.with_columns(freq_hz=(pl.col("freq_mhz") * 1e6).round(0).cast(pl.Int64))
        .with_columns(
            source=pl.lit("wsprnet"),
            ts=pl.from_epoch("epoch", time_unit="s").dt.replace_time_zone("UTC")
                 .cast(pl.Datetime("us", "UTC")),
            band=pl.col("freq_hz").map_elements(band_for_freq_hz, return_dtype=pl.Utf8),
            mode=pl.lit("WSPR"),
            dx_call=pl.col("call_sign"), de_call=pl.col("reporter"),
            dx_grid=pl.col("grid"), de_grid=pl.col("reporter_grid"),
            dx_lat=pl.lit(None, dtype=pl.Float64), dx_lon=pl.lit(None, dtype=pl.Float64),
            de_lat=pl.lit(None, dtype=pl.Float64), de_lon=pl.lit(None, dtype=pl.Float64),
            snr_db=pl.col("snr"), tx_dbm=pl.col("power_dbm"),
            distance_km=pl.col("distance"), bearing_deg=pl.col("azimuth"))
        .select(list(schema.SPOT_SCHEMA))
        .cast(schema.SPOT_SCHEMA)
    )


class WsprnetExtractor:
    source = "wsprnet"

    def extract(self, start: date, end: date, bands: list[str],
                lake: Lake, cache_dir: Path) -> list[Path]:
        written: list[Path] = []
        months: set[tuple[int, int]] = set()
        d = start
        while d <= end:
            months.add((d.year, d.month))
            d += timedelta(days=1)
        for year, month in sorted(months):
            url = archive_url(year, month)
            gz = download_archive(url, cache_dir / url.rsplit("/", 1)[-1])
            df = parse_archive(gz).filter(
                pl.col("band").is_in(bands)
                & (pl.col("ts").dt.date() >= start) & (pl.col("ts").dt.date() <= end))
            for (band, day), part in df.group_by(
                    ["band", pl.col("ts").dt.date().alias("day")], maintain_order=True):
                written.append(lake.write_parquet(part, lake.spots_dir(band, day)))
        return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract WSPRnet monthly archives into the lake")
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--band", action="append", required=True, choices=schema.BANDS,
                    dest="bands")
    ap.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    args = ap.parse_args()
    files = WsprnetExtractor().extract(args.start, args.end, args.bands,
                                       Lake(args.lake_root), args.cache_dir)
    print(f"wrote {len(files)} parquet files")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/data/test_wsprnet.py -q`
Expected: PASS (4 tests).

Note on `group_by` unpacking: polars yields group keys as tuples — if the installed polars returns `(band, day)` nested differently, adjust the unpack, not the test.

- [ ] **Step 5: Commit**

```bash
git add src/propagation/data/base.py src/propagation/data/wsprnet.py tests/data/test_wsprnet.py
git commit -m "feat: extractor interface + WSPRnet monthly-archive extractor"
```

---

### Task 7: features/labels.py part 1 — snr_ft8eq + receiver-uptime table

**Files:**
- Create: `src/propagation/features/labels.py`
- Test: `tests/features/test_labels_uptime.py`

**Interfaces:**
- Consumes: `schema.*`, `Lake`, spots_q frames (SPOT_SCHEMA + QUALIFIED_EXTRA columns).
- Produces (pinned): `snr_ft8eq(snr_db: int, mode: str, tx_dbm: int | None) -> float | None` (SPEC §4.4); `build_receiver_uptime(spots_q: pl.DataFrame, band: str, d: date) -> pl.DataFrame` (SPEC §3) with columns `window_start, de_call, de_field, de_grid4, band, mode_class, n_evidence_reports Int32, first_evidence_ts, last_evidence_ts`. Caller must pass spots_q covering `[d 00:00 − 30min, d+1 00:00 + 30min)`.
- Key derivation used by the implementation: a spot at `ts` with `w = window_floor(ts)` is evidence for exactly the five window starts `{w−30m, w−15m, w, w+15m, w+30m}` (from `ts ∈ [t0−30m, t0+45m) ⇔ t0 ∈ (ts−45m, ts+30m]` on a 15-min lattice).

- [ ] **Step 1: Write the failing tests** — `tests/features/test_labels_uptime.py`:

```python
from datetime import date, datetime, timezone

import polars as pl
import pytest

from propagation.data.hygiene import qualify_spots
from propagation.features import labels


def test_snr_ft8eq_offsets():
    assert labels.snr_ft8eq(-18, "FT8", None) == -18.0
    assert labels.snr_ft8eq(6, "CW", None) == -1.0           # -7 bandwidth offset
    assert labels.snr_ft8eq(-28, "WSPR", 37) == -15.0        # (50 - 37) = +13 power
    assert labels.snr_ft8eq(10, "SSB", None) is None         # unknown reference


def _spots_q(make_spots, rows):
    ok, rej = qualify_spots(make_spots(rows))
    assert rej.height == 0
    return ok


def test_uptime_five_windows_per_spot(make_spots):
    q = _spots_q(make_spots, [{"ts": datetime(2026, 5, 1, 12, 3, tzinfo=timezone.utc)}])
    up = labels.build_receiver_uptime(q, "20m", date(2026, 5, 1))
    got = sorted(up["window_start"].to_list())
    assert got == [datetime(2026, 5, 1, 11, 30, tzinfo=timezone.utc),
                   datetime(2026, 5, 1, 11, 45, tzinfo=timezone.utc),
                   datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
                   datetime(2026, 5, 1, 12, 15, tzinfo=timezone.utc),
                   datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)]
    r = up.row(0, named=True)
    assert r["de_call"] == "G4ABC" and r["mode_class"] == "digi"
    assert r["de_field"] == "IO" and r["de_grid4"] == "IO91"
    assert r["n_evidence_reports"] == 1


def test_uptime_clipped_to_date(make_spots):
    # spot at 00:05 on 5/2 is evidence for 23:35..? windows -> only 23:45 on 5/1 side
    q = _spots_q(make_spots, [{"ts": datetime(2026, 5, 2, 0, 5, tzinfo=timezone.utc)}])
    up = labels.build_receiver_uptime(q, "20m", date(2026, 5, 1))
    assert up["window_start"].to_list() == [datetime(2026, 5, 1, 23, 45, tzinfo=timezone.utc)]
    assert up["mode_class"].n_unique() == 1


def test_uptime_per_mode_class_and_other_excluded(make_spots):
    ts = datetime(2026, 5, 1, 12, 3, tzinfo=timezone.utc)
    q = _spots_q(make_spots, [
        {"ts": ts, "mode": "CW", "de_call": "W3LPL"},
        {"ts": ts, "mode": "SSB", "de_call": "K3LR"},      # other: never uptime evidence
    ])
    up = labels.build_receiver_uptime(q, "20m", date(2026, 5, 1))
    assert set(up["de_call"].to_list()) == {"W3LPL"}
    assert set(up["mode_class"].to_list()) == {"cw"}


def test_uptime_modal_grid_lexicographic_tiebreak(make_spots):
    ts = datetime(2026, 5, 1, 12, 3, tzinfo=timezone.utc)
    q = _spots_q(make_spots, [
        {"ts": ts, "de_grid": "IO91"},
        {"ts": ts, "de_grid": "IO92", "dx_call": "W1AW"},
    ])
    up = labels.build_receiver_uptime(q, "20m", date(2026, 5, 1))
    assert up.filter(pl.col("window_start") == datetime(2026, 5, 1, 12, 0,
                     tzinfo=timezone.utc))["de_grid4"].item() == "IO91"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_labels_uptime.py -q`
Expected: FAIL — `ModuleNotFoundError: propagation.features.labels`.

- [ ] **Step 3: Implement** — `src/propagation/features/labels.py`:

```python
"""Receiver uptime, labels, sampling. Normative: docs/SPEC-labeling.md §2-§4."""
from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from propagation import schema
from propagation.data.lake import Lake

_BW_OFFSET_DIGI = 0.0   # reported re 2500 Hz
_BW_OFFSET_CW = -7.0    # re 500 Hz; -10*log10(2500/500) ~= -7.0 (SPEC §4.4)


def snr_ft8eq(snr_db: int, mode: str, tx_dbm: int | None) -> float | None:
    mc = schema.mode_class(mode)
    if mc == "digi":
        bw = _BW_OFFSET_DIGI
    elif mc == "cw":
        bw = _BW_OFFSET_CW
    else:
        return None
    pwr = (50.0 - tx_dbm) if tx_dbm is not None else 0.0
    return float(snr_db) + bw + pwr


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    t0 = datetime.combine(d, time(0, 0), tzinfo=timezone.utc)
    return t0, t0 + timedelta(days=1)


def build_receiver_uptime(spots_q: pl.DataFrame, band: str, d: date) -> pl.DataFrame:
    """SPEC §3: provably-monitoring windows per (receiver, band, mode_class)."""
    day_start, day_end = _day_bounds(d)
    ev = spots_q.filter(
        (pl.col("band") == band) & pl.col("mode_class").is_in(["digi", "cw"]))
    if ev.height == 0:
        return _empty_uptime()
    offsets = [timedelta(minutes=m) for m in (-30, -15, 0, 15, 30)]
    ev = (
        ev.with_columns(
            pl.concat_list([(pl.col("window_start") + off) for off in offsets])
            .alias("w"))
        .explode("w")
        .filter((pl.col("w") >= day_start) & (pl.col("w") < day_end))
    )
    grouped = (
        ev.group_by(["w", "de_call", "mode_class"])
        .agg(
            n_evidence_reports=pl.len().cast(pl.Int32),
            first_evidence_ts=pl.col("ts").min(),
            last_evidence_ts=pl.col("ts").max(),
            de_grid4=pl.col("de_grid4").drop_nulls().mode().sort().first(),
            _field_mode=pl.col("de_field").mode().sort().first(),
        )
        .with_columns(
            de_field=pl.when(pl.col("de_grid4").is_not_null())
                       .then(pl.col("de_grid4").str.slice(0, 2))
                       .otherwise(pl.col("_field_mode")),
            band=pl.lit(band))
        .rename({"w": "window_start"})
        .select(["window_start", "de_call", "de_field", "de_grid4", "band",
                 "mode_class", "n_evidence_reports", "first_evidence_ts",
                 "last_evidence_ts"])
        .sort(["window_start", "de_call", "mode_class"])
    )
    return grouped


def _empty_uptime() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "window_start": pl.Datetime("us", "UTC"), "de_call": pl.Utf8,
        "de_field": pl.Utf8, "de_grid4": pl.Utf8, "band": pl.Utf8,
        "mode_class": pl.Utf8, "n_evidence_reports": pl.Int32,
        "first_evidence_ts": pl.Datetime("us", "UTC"),
        "last_evidence_ts": pl.Datetime("us", "UTC")})
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_labels_uptime.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/labels.py tests/features/test_labels_uptime.py
git commit -m "feat: snr_ft8eq normalization + receiver-uptime table (SPEC-labeling §3, §4.4)"
```

---

### Task 8: features/labels.py part 2 — universe, positives, monitor-normalized negatives

**Files:**
- Modify: `src/propagation/features/labels.py` (append functions)
- Test: `tests/features/test_labels_build.py`

**Interfaces:**
- Consumes: Task 7's uptime frame; spots_q.
- Produces (pinned): `build_labels(spots_q: pl.DataFrame, uptime: pl.DataFrame, band: str, d: date) -> pl.DataFrame` with columns `window_start, tx_field, rx_field, band, open Int8, n_spots Int32, n_monitors Int32, n_tx_stations Int32, evidence_tier Utf8, snr_ft8eq_p50 Float32 (nullable), sample_weight Float32 (=1.0 here), split_tag Utf8 (null)`. Extension: `label_stats(spots_q, uptime, labels_df, band, d) -> pl.DataFrame` (columns `band, date, n_pos, n_neg, n_unlabeled_activity, unlabeled_fraction`) — the §6.8 "unlabeled-activity fraction"; a window×(tx_field or rx_field) with one-sided evidence (tx active but no monitor, or monitor but no tx) counts as unlabeled activity.
- SPEC rules encoded: positives = any qualifying deduped spot TX→RX in W (any mode, k=1, no SNR floor); N-eligible = same-mode_class monitor AND tx evidence (digi or cw; `other` never establishes eligibility); negative = N-eligible AND zero TX→RX spots in W in ANY mode; `n_monitors`/`n_tx_stations` summed over eligible mode_classes; `evidence_tier='wspr'` iff any tx evidence spot has mode WSPR else `'spot'`; tx evidence uses the exact window (no padding), any receiver worldwide, same band.

- [ ] **Step 1: Write the failing tests** — `tests/features/test_labels_build.py`:

```python
from datetime import date, datetime, timezone

import polars as pl

from propagation.data.hygiene import qualify_spots
from propagation.features import labels

TS = datetime(2026, 5, 1, 12, 3, tzinfo=timezone.utc)
W = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
D = date(2026, 5, 1)


def _q(make_spots, rows):
    ok, _ = qualify_spots(make_spots(rows))
    return ok


def test_positive_label(make_spots):
    q = _q(make_spots, [{"ts": TS}])  # K5ARH(EM12) -> G4ABC(IO91)
    up = labels.build_receiver_uptime(q, "20m", D)
    lab = labels.build_labels(q, up, "20m", D)
    pos = lab.filter((pl.col("tx_field") == "EM") & (pl.col("rx_field") == "IO")
                     & (pl.col("window_start") == W))
    assert pos["open"].to_list() == [1]
    assert pos["n_spots"].to_list() == [1]
    assert pos["snr_ft8eq_p50"].to_list() == [-10.0]
    assert pos["sample_weight"].to_list() == [1.0]


def test_monitor_normalized_negative(make_spots):
    q = _q(make_spots, [
        # G4ABC in IO monitors 20m digi (decodes VK3XYZ in QF)
        {"ts": TS, "dx_call": "VK3XYZ", "dx_grid": "QF22"},
        # W5XYZ in EM is provably transmitting (spotted by JA1AAA in PM)
        {"ts": TS, "dx_call": "W5XYZ", "dx_grid": "EM13",
         "de_call": "JA1AAA", "de_grid": "PM95"},
    ])
    up = labels.build_receiver_uptime(q, "20m", D)
    lab = labels.build_labels(q, up, "20m", D)
    neg = lab.filter((pl.col("tx_field") == "EM") & (pl.col("rx_field") == "IO")
                     & (pl.col("window_start") == W))
    assert neg["open"].to_list() == [0]
    r = neg.row(0, named=True)
    assert r["n_monitors"] == 1 and r["n_tx_stations"] == 1
    assert r["evidence_tier"] == "spot" and r["n_spots"] == 0


def test_one_sided_evidence_is_unlabeled(make_spots):
    # EM station transmitting, but nobody monitoring in IO -> no EM->IO row
    q = _q(make_spots, [{"ts": TS, "dx_call": "W5XYZ", "dx_grid": "EM13",
                         "de_call": "JA1AAA", "de_grid": "PM95"}])
    up = labels.build_receiver_uptime(q, "20m", D)
    lab = labels.build_labels(q, up, "20m", D)
    assert lab.filter((pl.col("tx_field") == "EM")
                      & (pl.col("rx_field") == "IO")).height == 0


def test_mode_class_must_match_for_eligibility(make_spots):
    q = _q(make_spots, [
        # IO monitor is CW-only
        {"ts": TS, "mode": "CW", "dx_call": "VK3XYZ", "dx_grid": "QF22"},
        # EM transmitter proven on digi only
        {"ts": TS, "mode": "FT8", "dx_call": "W5XYZ", "dx_grid": "EM13",
         "de_call": "JA1AAA", "de_grid": "PM95"},
    ])
    up = labels.build_receiver_uptime(q, "20m", D)
    lab = labels.build_labels(q, up, "20m", D)
    assert lab.filter((pl.col("tx_field") == "EM")
                      & (pl.col("rx_field") == "IO")).height == 0


def test_wspr_evidence_tier(make_spots):
    q = _q(make_spots, [
        {"ts": TS, "dx_call": "VK3XYZ", "dx_grid": "QF22"},
        {"ts": TS, "mode": "WSPR", "tx_dbm": 37, "dx_call": "W5XYZ",
         "dx_grid": "EM13", "de_call": "JA1AAA", "de_grid": "PM95"},
    ])
    up = labels.build_receiver_uptime(q, "20m", D)
    lab = labels.build_labels(q, up, "20m", D)
    neg = lab.filter((pl.col("tx_field") == "EM") & (pl.col("rx_field") == "IO")
                     & (pl.col("window_start") == W))
    assert neg["evidence_tier"].to_list() == ["wspr"]


def test_label_stats_reports_unlabeled_fraction(make_spots):
    q = _q(make_spots, [{"ts": TS, "dx_call": "W5XYZ", "dx_grid": "EM13",
                         "de_call": "JA1AAA", "de_grid": "PM95"}])
    up = labels.build_receiver_uptime(q, "20m", D)
    lab = labels.build_labels(q, up, "20m", D)
    stats = labels.label_stats(q, up, lab, "20m", D)
    r = stats.row(0, named=True)
    assert r["n_unlabeled_activity"] > 0
    assert 0.0 < r["unlabeled_fraction"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_labels_build.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build_labels'`.

- [ ] **Step 3: Implement** — append to `src/propagation/features/labels.py`:

```python
def build_labels(spots_q: pl.DataFrame, uptime: pl.DataFrame,
                 band: str, d: date) -> pl.DataFrame:
    """SPEC §2 (universe) + §4 (labels). spots_q must be post-dedup, band-filtered
    superset covering day d; uptime from build_receiver_uptime for the same day."""
    day_start, day_end = _day_bounds(d)
    day = spots_q.filter((pl.col("band") == band)
                         & (pl.col("window_start") >= day_start)
                         & (pl.col("window_start") < day_end))

    # --- positives (any mode) -------------------------------------------------
    ft8eq = pl.struct(["snr_db", "mode", "tx_dbm"]).map_elements(
        lambda s: snr_ft8eq(s["snr_db"], s["mode"], s["tx_dbm"])
        if s["snr_db"] is not None else None, return_dtype=pl.Float64)
    positives = (
        day.group_by(["window_start", "dx_field", "de_field"])
        .agg(n_spots=pl.len().cast(pl.Int32),
             snr_ft8eq_p50=ft8eq.drop_nulls().median().cast(pl.Float32))
        .rename({"dx_field": "tx_field", "de_field": "rx_field"})
        .with_columns(open=pl.lit(1, dtype=pl.Int8))
    )

    # --- N-eligibility per mode_class ----------------------------------------
    monitors = (
        uptime.group_by(["window_start", "de_field", "mode_class"])
        .agg(n_monitors=pl.col("de_call").n_unique().cast(pl.Int32))
        .rename({"de_field": "rx_field"})
    )
    tx_ev = (
        day.filter(pl.col("mode_class").is_in(["digi", "cw"]))
        .group_by(["window_start", "dx_field", "mode_class"])
        .agg(n_tx_stations=pl.col("dx_call").n_unique().cast(pl.Int32),
             _any_wspr=(pl.col("mode") == "WSPR").any())
        .rename({"dx_field": "tx_field"})
    )
    eligible = (
        monitors.join(tx_ev, on=["window_start", "mode_class"], how="inner")
        .group_by(["window_start", "tx_field", "rx_field"])
        .agg(n_monitors=pl.col("n_monitors").sum().cast(pl.Int32),
             n_tx_stations=pl.col("n_tx_stations").sum().cast(pl.Int32),
             evidence_tier=pl.when(pl.col("_any_wspr").any())
                             .then(pl.lit("wspr")).otherwise(pl.lit("spot")).first())
    )

    # --- negatives = eligible minus any-mode positives ------------------------
    negatives = (
        eligible.join(positives.select(["window_start", "tx_field", "rx_field"]),
                      on=["window_start", "tx_field", "rx_field"], how="anti")
        .with_columns(open=pl.lit(0, dtype=pl.Int8),
                      n_spots=pl.lit(0, dtype=pl.Int32),
                      snr_ft8eq_p50=pl.lit(None, dtype=pl.Float32))
    )
    positives = positives.join(
        eligible, on=["window_start", "tx_field", "rx_field"], how="left"
    ).with_columns(
        n_monitors=pl.col("n_monitors").fill_null(0),
        n_tx_stations=pl.col("n_tx_stations").fill_null(0),
        evidence_tier=pl.col("evidence_tier").fill_null("spot"))

    cols = ["window_start", "tx_field", "rx_field", "open", "n_spots",
            "n_monitors", "n_tx_stations", "evidence_tier", "snr_ft8eq_p50"]
    return (
        pl.concat([positives.select(cols), negatives.select(cols)])
        .with_columns(band=pl.lit(band),
                      sample_weight=pl.lit(1.0, dtype=pl.Float32),
                      split_tag=pl.lit(None, dtype=pl.Utf8))
        .select(["window_start", "tx_field", "rx_field", "band", "open", "n_spots",
                 "n_monitors", "n_tx_stations", "evidence_tier", "snr_ft8eq_p50",
                 "sample_weight", "split_tag"])
        .sort(["window_start", "tx_field", "rx_field"])
    )


def label_stats(spots_q: pl.DataFrame, uptime: pl.DataFrame,
                labels_df: pl.DataFrame, band: str, d: date) -> pl.DataFrame:
    """§6.8 unlabeled-activity fraction: windows x fields with one-sided evidence."""
    day_start, day_end = _day_bounds(d)
    day = spots_q.filter((pl.col("band") == band)
                         & (pl.col("window_start") >= day_start)
                         & (pl.col("window_start") < day_end))
    mon = uptime.select(["window_start", "de_field", "mode_class"]).unique() \
                .rename({"de_field": "field"}).with_columns(side=pl.lit("rx"))
    tx = (day.filter(pl.col("mode_class").is_in(["digi", "cw"]))
          .select(["window_start", "dx_field", "mode_class"]).unique()
          .rename({"dx_field": "field"}).with_columns(side=pl.lit("tx")))
    # activity signals that never made it into any labeled cell
    labeled_tx = labels_df.select(["window_start", pl.col("tx_field").alias("field")]).unique()
    labeled_rx = labels_df.select(["window_start", pl.col("rx_field").alias("field")]).unique()
    unlabeled = (
        pl.concat([tx.join(labeled_tx, on=["window_start", "field"], how="anti"),
                   mon.join(labeled_rx, on=["window_start", "field"], how="anti")])
        .select(["window_start", "field"]).unique()
    )
    n_pos = labels_df.filter(pl.col("open") == 1).height
    n_neg = labels_df.filter(pl.col("open") == 0).height
    n_unlab = unlabeled.height
    denom = n_pos + n_neg + n_unlab
    return pl.DataFrame({
        "band": [band], "date": [d.isoformat()], "n_pos": [n_pos], "n_neg": [n_neg],
        "n_unlabeled_activity": [n_unlab],
        "unlabeled_fraction": [n_unlab / denom if denom else 0.0]})
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_labels_build.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/labels.py tests/features/test_labels_build.py
git commit -m "feat: activity-gated universe, positives, monitor-normalized negatives (SPEC §2, §4)"
```

---

### Task 9: features/labels.py part 3 — deterministic sampling + `build-uptime` / `build-labels` CLIs

**Files:**
- Modify: `src/propagation/features/labels.py` (append)
- Test: `tests/features/test_labels_sampling.py`

**Interfaces:**
- Consumes: Tasks 7–8; `Lake`; hygiene spots_q partitions.
- Produces (pinned): `sample_training(labels: pl.DataFrame, band: str, d: date) -> pl.DataFrame` (SPEC §4.5: all positives; negatives 3:1 per (band, UTC date) stratum, keep-all-if-fewer, `sample_weight = 1/sampling_rate` on kept negatives, PCG64 seed per Global Constraints). Extensions: `main_uptime()` (`build-uptime` CLI) and `main_labels()` (`build-labels` CLI) — both loop dates, read spots_q (uptime reads ±1 day for the 30-min padding), write `lake/receiver_uptime/…` and `lake/labels/…` + `lake/label_stats/…`. Training-time sampling happens at model-fit time (Task 12's report CLI), NOT at label-build time — stored labels are always the full set.

- [ ] **Step 1: Write the failing tests** — `tests/features/test_labels_sampling.py`:

```python
from datetime import date, datetime, timedelta, timezone

import polars as pl

from propagation.features import labels

D = date(2026, 5, 1)


def _labels(n_pos: int, n_neg: int) -> pl.DataFrame:
    w0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n_pos + n_neg):
        rows.append({
            "window_start": w0 + timedelta(minutes=15 * (i % 96)),
            "tx_field": "EM", "rx_field": f"{chr(65 + i % 18)}{chr(65 + i // 18 % 18)}",
            "band": "20m", "open": 1 if i < n_pos else 0, "n_spots": 0,
            "n_monitors": 1, "n_tx_stations": 1, "evidence_tier": "spot",
            "snr_ft8eq_p50": None, "sample_weight": 1.0, "split_tag": None})
    return pl.DataFrame(rows).with_columns(
        pl.col("open").cast(pl.Int8), pl.col("sample_weight").cast(pl.Float32),
        pl.col("snr_ft8eq_p50").cast(pl.Float32), pl.col("split_tag").cast(pl.Utf8))


def test_downsamples_to_three_to_one_with_weights():
    out = labels.sample_training(_labels(10, 100), "20m", D)
    assert out.filter(pl.col("open") == 1).height == 10
    neg = out.filter(pl.col("open") == 0)
    assert neg.height == 30
    assert neg["sample_weight"].to_list() == [100 / 30] * 30
    assert out.filter(pl.col("open") == 1)["sample_weight"].to_list() == [1.0] * 10


def test_keep_all_when_under_ratio():
    out = labels.sample_training(_labels(10, 20), "20m", D)
    assert out.filter(pl.col("open") == 0).height == 20
    assert set(out["sample_weight"].to_list()) == {1.0}


def test_deterministic_across_calls():
    a = labels.sample_training(_labels(5, 200), "20m", D)
    b = labels.sample_training(_labels(5, 200), "20m", D)
    assert a.sort(["window_start", "rx_field"]).equals(b.sort(["window_start", "rx_field"]))


def test_different_band_different_sample():
    la = labels.sample_training(_labels(5, 200), "20m", D)
    lb = labels.sample_training(
        _labels(5, 200).with_columns(band=pl.lit("10m")), "10m", D)
    assert not la.drop("band").sort(["window_start", "rx_field"]).equals(
        lb.drop("band").sort(["window_start", "rx_field"]))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_labels_sampling.py -q`
Expected: FAIL — `AttributeError: ... 'sample_training'`.

- [ ] **Step 3: Implement** — append to `src/propagation/features/labels.py`:

```python
def _stratum_seed(band: str, date_iso: str) -> int:
    digest = hashlib.sha256(f"{band}|{date_iso}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0xFFFFFFFF


def sample_training(labels_df: pl.DataFrame, band: str, d: date) -> pl.DataFrame:
    """SPEC §4.5: 3:1 negative downsampling per (band, UTC date) stratum.
    labels_df must contain exactly one (band, date) stratum (the build loop's unit)."""
    pos = labels_df.filter(pl.col("open") == 1)
    neg = labels_df.filter(pl.col("open") == 0)
    target = 3 * pos.height
    if neg.height <= target:
        return pl.concat([pos, neg])
    rng = np.random.Generator(np.random.PCG64(_stratum_seed(band, d.isoformat())))
    idx = rng.choice(neg.height, size=target, replace=False)
    rate = target / neg.height
    kept = neg[np.sort(idx).tolist()].with_columns(
        sample_weight=pl.lit(1.0 / rate, dtype=pl.Float32))
    return pl.concat([pos, kept])


def _read_spots_q(lake: Lake, band: str, days: list[date]) -> pl.DataFrame:
    frames = [pl.read_parquet(lake.spots_q_dir(band, day) / "*.parquet")
              for day in days if lake.spots_q_dir(band, day).exists()]
    if not frames:
        raise FileNotFoundError(f"no spots_q partitions for {band} on {days}")
    return pl.concat(frames)


def _date_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--band", required=True, choices=schema.BANDS)
    ap.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    return ap


def main_uptime() -> None:
    args = _date_args(argparse.ArgumentParser(
        description="Build receiver-uptime tables (SPEC-labeling §3)")).parse_args()
    lake = Lake(args.lake_root)
    d = args.start
    while d <= args.end:
        # ±1 day so the ±30 min evidence padding at date edges is satisfied
        days = [d - timedelta(days=1), d, d + timedelta(days=1)]
        spots = _read_spots_q(lake, args.band, [x for x in days
                                                if lake.spots_q_dir(args.band, x).exists()] or [d])
        up = build_receiver_uptime(spots, args.band, d)
        lake.write_parquet(up, lake.uptime_dir(args.band, d))
        d += timedelta(days=1)


def main_labels() -> None:
    args = _date_args(argparse.ArgumentParser(
        description="Build labels (SPEC-labeling §2, §4)")).parse_args()
    lake = Lake(args.lake_root)
    d = args.start
    while d <= args.end:
        spots = _read_spots_q(lake, args.band, [d])
        up_dir = lake.uptime_dir(args.band, d)
        up = pl.read_parquet(up_dir / "*.parquet")
        lab = build_labels(spots, up, args.band, d)
        lake.write_parquet(lab, lake.labels_dir(args.band, d))
        stats = label_stats(spots, up, lab, args.band, d)
        lake.write_parquet(
            stats, lake.root / "label_stats" / f"band={args.band}" / f"date={d.isoformat()}")
        print(f"{args.band} {d}: {stats.row(0, named=True)}")
        d += timedelta(days=1)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_labels_sampling.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/propagation/features/labels.py tests/features/test_labels_sampling.py
git commit -m "feat: deterministic 3:1 negative sampling + build-uptime/build-labels CLIs (SPEC §4.5)"
```

---

### Task 10: models/base.py + models/climatology.py

**Files:**
- Create: `src/propagation/models/base.py`, `src/propagation/models/climatology.py`
- Test: `tests/models/test_climatology.py` (create `tests/models/__init__.py`)

**Interfaces:**
- Consumes: label frames (Task 8 schema).
- Produces (pinned): `OpennessModel` protocol (`model_id: str`; `predict_p_open(cells: pl.DataFrame) -> pl.DataFrame` — cells has `window_start, tx_field, rx_field, band`; returns cells + `p_open` Float64 in [0,1], null = abstain). `ClimatologyModel(alpha: float = 5.0)` with `model_id = "climatology"`, `fit(train_labels: pl.DataFrame) -> ClimatologyModel` (weighted by `sample_weight`), `predict_p_open` per protocol.
- Model definition: P(open) = Laplace-smoothed weighted open-rate keyed on `(tx_field, rx_field, band, hour-of-day)` with hierarchical fallback cell+hour → cell → band prior; unseen band ⇒ null (abstain). Month/SSN conditioning (ARCHITECTURE M-0 mentions "similar smoothed SSN") is **deliberately deferred** until multi-year data exists — one train month has a single month and a single SSN regime, so the extra key would be degenerate. Record this in the module docstring.
- Smoothing math: `p_cell = (open_cell + alpha * p_band) / (n_cell + alpha)`; `p_cell_hour = (open_cell_hour + alpha * p_cell) / (n_cell_hour + alpha)` where `open_* = Σ weight·open` and `n_* = Σ weight`.

- [ ] **Step 1: Write the failing tests** — `tests/models/test_climatology.py`:

```python
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from propagation.models.climatology import ClimatologyModel

W0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


def _labels(rows):
    return pl.DataFrame(
        [{"window_start": W0 + timedelta(hours=r.get("h", 12)),
          "tx_field": r.get("tx", "EM"), "rx_field": r.get("rx", "IO"),
          "band": r.get("band", "20m"), "open": r["open"],
          "sample_weight": r.get("w", 1.0)} for r in rows]
    ).with_columns(pl.col("open").cast(pl.Int8), pl.col("sample_weight").cast(pl.Float32))


def _cells(h=12, tx="EM", rx="IO", band="20m"):
    return pl.DataFrame({"window_start": [W0 + timedelta(hours=h)],
                         "tx_field": [tx], "rx_field": [rx], "band": [band]})


def test_seen_cell_hour_smoothed_rate():
    train = _labels([{"open": 1}] * 8 + [{"open": 0}] * 2)   # cell+hour rate 0.8
    m = ClimatologyModel(alpha=0.0).fit(train)
    p = m.predict_p_open(_cells())["p_open"].item()
    assert p == pytest.approx(0.8)


def test_sample_weight_respected():
    # 1 positive w=1, 1 negative w=9 -> weighted rate 0.1
    train = _labels([{"open": 1, "w": 1.0}, {"open": 0, "w": 9.0}])
    m = ClimatologyModel(alpha=0.0).fit(train)
    assert m.predict_p_open(_cells())["p_open"].item() == pytest.approx(0.1)


def test_unseen_hour_falls_back_to_cell_rate():
    train = _labels([{"open": 1, "h": 3}] * 3 + [{"open": 0, "h": 3}])
    m = ClimatologyModel(alpha=0.0).fit(train)
    p = m.predict_p_open(_cells(h=20))["p_open"].item()
    assert p == pytest.approx(0.75)


def test_unseen_cell_falls_back_to_band_prior():
    train = _labels([{"open": 1}, {"open": 0}, {"open": 0}, {"open": 0}])
    m = ClimatologyModel(alpha=0.0).fit(train)
    p = m.predict_p_open(_cells(tx="PM", rx="CN"))["p_open"].item()
    assert p == pytest.approx(0.25)


def test_unseen_band_abstains():
    m = ClimatologyModel().fit(_labels([{"open": 1}]))
    out = m.predict_p_open(_cells(band="10m"))
    assert out["p_open"].to_list() == [None]


def test_smoothing_pulls_toward_prior():
    # band prior 0.5; sparse cell 1/1 open; alpha=5 -> (1 + 5*0.5)/(1+5) ~= 0.583
    train = _labels([{"open": 1, "tx": "AA"}, {"open": 0, "tx": "AA"},
                     {"open": 1, "tx": "EM", "rx": "JN"}])
    m = ClimatologyModel(alpha=5.0).fit(train)
    p = m.predict_p_open(_cells(tx="EM", rx="JN"))["p_open"].item()
    assert 0.5 < p < 0.7
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_climatology.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/propagation/models/base.py`:

```python
"""Shared scoring interface. fit() signatures vary per model and are not pinned."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class OpennessModel(Protocol):
    model_id: str

    def predict_p_open(self, cells: pl.DataFrame) -> pl.DataFrame:
        """cells: window_start, tx_field, rx_field, band.
        Returns cells + column p_open Float64 in [0,1], null = abstain."""
        ...
```

and `src/propagation/models/climatology.py`:

```python
"""M-0 climatology baseline (ARCHITECTURE §5): weighted open-rate lookup keyed on
(tx_field, rx_field, band, hour-of-day), Laplace-smoothed, hierarchical fallback
cell+hour -> cell -> band prior. Month/SSN conditioning deferred until multi-year
data exists (single train month = degenerate month/SSN key)."""
from __future__ import annotations

import polars as pl

_CELL = ["tx_field", "rx_field", "band"]


class ClimatologyModel:
    model_id = "climatology"

    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha
        self._cell_hour: pl.DataFrame | None = None
        self._cell: pl.DataFrame | None = None
        self._band: pl.DataFrame | None = None

    def fit(self, train_labels: pl.DataFrame) -> "ClimatologyModel":
        df = train_labels.with_columns(
            hour=pl.col("window_start").dt.hour().cast(pl.Int8),
            _w=pl.col("sample_weight").cast(pl.Float64),
            _wo=pl.col("sample_weight").cast(pl.Float64) * pl.col("open").cast(pl.Float64))
        band = (df.group_by("band")
                .agg(p_band=(pl.col("_wo").sum() / pl.col("_w").sum())))
        cell = (df.group_by(_CELL)
                .agg(_wo=pl.col("_wo").sum(), _w=pl.col("_w").sum())
                .join(band, on="band")
                .with_columns(p_cell=(pl.col("_wo") + self.alpha * pl.col("p_band"))
                                     / (pl.col("_w") + self.alpha))
                .select(_CELL + ["p_cell"]))
        cell_hour = (df.group_by(_CELL + ["hour"])
                     .agg(_wo=pl.col("_wo").sum(), _w=pl.col("_w").sum())
                     .join(cell, on=_CELL)
                     .with_columns(
                         p_cell_hour=(pl.col("_wo") + self.alpha * pl.col("p_cell"))
                                     / (pl.col("_w") + self.alpha))
                     .select(_CELL + ["hour", "p_cell_hour"]))
        self._band, self._cell, self._cell_hour = band, cell, cell_hour
        return self

    def predict_p_open(self, cells: pl.DataFrame) -> pl.DataFrame:
        if self._band is None:
            raise RuntimeError("fit() first")
        out = (cells.with_columns(hour=pl.col("window_start").dt.hour().cast(pl.Int8))
               .join(self._cell_hour, on=_CELL + ["hour"], how="left")
               .join(self._cell, on=_CELL, how="left")
               .join(self._band, on="band", how="left")
               .with_columns(p_open=pl.coalesce(["p_cell_hour", "p_cell", "p_band"])
                             .cast(pl.Float64))
               .drop(["hour", "p_cell_hour", "p_cell", "p_band"]))
        return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/models/test_climatology.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/models tests/models
git commit -m "feat: OpennessModel protocol + climatology baseline (M-0)"
```

---

### Task 11: eval/splits.py + eval/metrics.py

**Files:**
- Create: `src/propagation/eval/splits.py`, `src/propagation/eval/metrics.py`
- Test: `tests/eval/test_splits.py`, `tests/eval/test_metrics.py` (create `tests/eval/__init__.py`)

**Interfaces:**
- Consumes: nothing internal (numpy/sklearn/polars only).
- Produces (pinned): `GAP_HOURS: int = 48`; `@dataclass(frozen=True) class Fold(train_start, train_end, eval_start, eval_end)`; `blocked_folds(data_start: datetime, data_end: datetime, eval_days: int, gap_hours: int = GAP_HOURS) -> list[Fold]`; `brier(y_true, p_pred, weights=None) -> float`; `log_loss(y_true, p_pred, weights=None) -> float`; `pr_auc(y_true, p_pred, weights=None) -> float`; `reliability_table(y_true, p_pred, n_bins: int = 10) -> pl.DataFrame` (columns `bin_lo, bin_hi, n, mean_pred, frac_open`).
- Fold semantics: eval blocks of `eval_days` tile **backward** from `data_end`; each fold trains on `[data_start, eval_start − gap)`; folds whose train span would be < `eval_days` are dropped; `gap_hours < 48` raises `ValueError` (SPEC §6.1 — widening is allowed, narrowing is a leakage bug).

- [ ] **Step 1: Write the failing tests** — `tests/eval/test_splits.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from propagation.eval.splits import GAP_HOURS, Fold, blocked_folds

T0 = datetime(2026, 5, 1, tzinfo=timezone.utc)
T_END = datetime(2026, 7, 1, tzinfo=timezone.utc)


def test_gap_constant():
    assert GAP_HOURS == 48


def test_m0_single_fold_shape():
    folds = blocked_folds(T0, T_END, eval_days=30)
    assert folds[0] == Fold(
        train_start=T0,
        train_end=datetime(2026, 5, 30, tzinfo=timezone.utc),
        eval_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
        eval_end=T_END)


def test_every_fold_respects_gap():
    folds = blocked_folds(T0, datetime(2026, 9, 1, tzinfo=timezone.utc), eval_days=14)
    assert len(folds) >= 2
    for f in folds:
        assert f.eval_start - f.train_end >= timedelta(hours=GAP_HOURS)
        assert f.train_end > f.train_start
        assert (f.train_end - f.train_start) >= timedelta(days=14)


def test_narrow_gap_rejected():
    with pytest.raises(ValueError):
        blocked_folds(T0, T_END, eval_days=30, gap_hours=24)
```

and `tests/eval/test_metrics.py`:

```python
import numpy as np
import pytest

from propagation.eval import metrics


def test_brier_perfect_and_worst():
    y = np.array([1, 0, 1, 0])
    assert metrics.brier(y, y.astype(float)) == 0.0
    assert metrics.brier(y, 1.0 - y.astype(float)) == 1.0


def test_brier_weighted():
    y = np.array([1, 0])
    p = np.array([0.5, 1.0])
    # unweighted: (0.25 + 1.0)/2 ; weighted 3:1 -> (3*0.25 + 1.0)/4
    assert metrics.brier(y, p) == pytest.approx(0.625)
    assert metrics.brier(y, p, weights=np.array([3.0, 1.0])) == pytest.approx(0.4375)


def test_log_loss_clips_extremes():
    y = np.array([1, 0])
    p = np.array([1.0, 0.0])
    assert np.isfinite(metrics.log_loss(y, p))


def test_pr_auc_orders_models():
    y = np.array([1, 1, 0, 0, 1, 0])
    good = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
    bad = 1.0 - good
    assert metrics.pr_auc(y, good) > metrics.pr_auc(y, bad)


def test_reliability_table_bins():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    p = np.linspace(0.05, 0.95, 10)
    tab = metrics.reliability_table(y, p, n_bins=5)
    assert list(tab.columns) == ["bin_lo", "bin_hi", "n", "mean_pred", "frac_open"]
    assert tab["n"].sum() == 10
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/eval -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/propagation/eval/splits.py`:

```python
"""Blocked time-series CV (ARCHITECTURE §6, SPEC-labeling §6.1). Never random."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

GAP_HOURS: int = 48  # = max horizon (24h) + max AR lookback (24h); widen with either


@dataclass(frozen=True)
class Fold:
    train_start: datetime
    train_end: datetime
    eval_start: datetime
    eval_end: datetime


def blocked_folds(data_start: datetime, data_end: datetime,
                  eval_days: int, gap_hours: int = GAP_HOURS) -> list[Fold]:
    if gap_hours < GAP_HOURS:
        raise ValueError(
            f"gap_hours={gap_hours} < {GAP_HOURS}; narrowing the CV gap is a leakage bug "
            "(SPEC-labeling §6.1)")
    gap = timedelta(hours=gap_hours)
    block = timedelta(days=eval_days)
    folds: list[Fold] = []
    eval_end = data_end
    while True:
        eval_start = eval_end - block
        train_end = eval_start - gap
        if eval_start <= data_start or (train_end - data_start) < block:
            break
        folds.append(Fold(train_start=data_start, train_end=train_end,
                          eval_start=eval_start, eval_end=eval_end))
        eval_end = eval_start
    return folds
```

and `src/propagation/eval/metrics.py`:

```python
"""Proper scoring metrics (ARCHITECTURE §6). weights=None means unweighted;
eval on the full label set is unweighted by construction (sample_weight == 1)."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

_EPS = 1e-15


def _w(y, weights):
    return np.ones_like(np.asarray(y, dtype=float)) if weights is None \
        else np.asarray(weights, dtype=float)


def brier(y_true, p_pred, weights=None) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    w = _w(y, weights)
    return float(np.average((p - y) ** 2, weights=w))


def log_loss(y_true, p_pred, weights=None) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), _EPS, 1 - _EPS)
    w = _w(y, weights)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def pr_auc(y_true, p_pred, weights=None) -> float:
    return float(average_precision_score(y_true, p_pred, sample_weight=weights))


def reliability_table(y_true, p_pred, n_bins: int = 10) -> pl.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1],
                     "n": int(mask.sum()),
                     "mean_pred": float(p[mask].mean()) if mask.any() else None,
                     "frac_open": float(y[mask].mean()) if mask.any() else None})
    return pl.DataFrame(rows)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/eval -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/eval tests/eval
git commit -m "feat: blocked time-series folds (48h gap) + proper scoring metrics"
```

---

### Task 12: eval/report.py — headline table, reliability diagram, `eval-report` CLI

**Files:**
- Create: `src/propagation/eval/report.py`
- Test: `tests/eval/test_report.py`

**Interfaces:**
- Consumes: `Lake.connect()` labels view, `ClimatologyModel`, `sample_training`, `blocked_folds`, `metrics.*`.
- Produces (pinned): `headline_table(predictions: dict[str, pl.DataFrame], eval_labels: pl.DataFrame) -> pl.DataFrame` (predictions: model_id → `predict_p_open` output over eval cells; result columns `model, brier, log_loss, pr_auc, n_scored, coverage`; joins on `window_start, tx_field, rx_field, band`; abstains (null p_open) are excluded from metrics and reported via `coverage = n_scored / n_eval`); `reliability_diagram(model_id: str, y_true, p_pred, out_path: Path) -> Path` (matplotlib PNG of reliability_table plus the diagonal); `main()` (`eval-report` CLI).
- CLI contract: `eval-report --band 20m --train-start 2026-05-01 --train-end 2026-05-30 --eval-start 2026-06-01 --eval-end 2026-07-01 --lake-root data/lake --out reports/` → validates the requested ranges against `blocked_folds` gap rule (train_end + 48h ≤ eval_start, else exit 2), loads FULL labels for eval range (never sampled — SPEC §4.5), builds the training set by applying `sample_training` per (band, date) over the train range, fits `ClimatologyModel`, writes `reports/headline.md` (headline table + per-day unlabeled-activity fraction table read from `lake/label_stats/`) and `reports/reliability_climatology.png`.

- [ ] **Step 1: Write the failing tests** — `tests/eval/test_report.py`:

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from propagation.eval import report

W0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)


def _eval_labels(n=40):
    return pl.DataFrame({
        "window_start": [W0 + timedelta(minutes=15 * i) for i in range(n)],
        "tx_field": ["EM"] * n, "rx_field": ["IO"] * n, "band": ["20m"] * n,
        "open": ([1, 0] * (n // 2)),
        "sample_weight": [1.0] * n,
    }).with_columns(pl.col("open").cast(pl.Int8))


def test_headline_table_scores_each_model():
    ev = _eval_labels()
    perfect = ev.select(["window_start", "tx_field", "rx_field", "band"]).with_columns(
        p_open=ev["open"].cast(pl.Float64))
    coin = ev.select(["window_start", "tx_field", "rx_field", "band"]).with_columns(
        p_open=pl.lit(0.5))
    tab = report.headline_table({"perfect": perfect, "coin": coin}, ev)
    assert set(tab["model"].to_list()) == {"perfect", "coin"}
    row = {r["model"]: r for r in tab.iter_rows(named=True)}
    assert row["perfect"]["brier"] == 0.0
    assert row["coin"]["brier"] == 0.25
    assert row["perfect"]["coverage"] == 1.0


def test_headline_table_excludes_abstains_and_reports_coverage():
    ev = _eval_labels()
    half = ev.select(["window_start", "tx_field", "rx_field", "band"]).with_columns(
        p_open=pl.when(pl.arange(0, ev.height) < ev.height // 2)
                 .then(0.5).otherwise(None).cast(pl.Float64))
    tab = report.headline_table({"half": half}, ev)
    r = tab.row(0, named=True)
    assert r["n_scored"] == ev.height // 2 and r["coverage"] == 0.5


def test_reliability_diagram_writes_png(tmp_path: Path):
    y = np.array([1, 0] * 50)
    p = np.linspace(0.01, 0.99, 100)
    out = report.reliability_diagram("climatology", y, p, tmp_path / "rel.png")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/eval/test_report.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/propagation/eval/report.py`:

```python
"""Headline eval artifact (ARCHITECTURE §6): one table, models x metrics,
eval always on the FULL label set."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from propagation import schema
from propagation.data.lake import Lake
from propagation.eval import metrics
from propagation.eval.splits import GAP_HOURS
from propagation.features.labels import sample_training
from propagation.models.climatology import ClimatologyModel

_KEY = ["window_start", "tx_field", "rx_field", "band"]


def headline_table(predictions: dict[str, pl.DataFrame],
                   eval_labels: pl.DataFrame) -> pl.DataFrame:
    n_eval = eval_labels.height
    rows = []
    for model_id, pred in predictions.items():
        joined = (eval_labels.select(_KEY + ["open"])
                  .join(pred.select(_KEY + ["p_open"]), on=_KEY, how="left")
                  .filter(pl.col("p_open").is_not_null()))
        y = joined["open"].to_numpy()
        p = joined["p_open"].to_numpy()
        rows.append({
            "model": model_id,
            "brier": metrics.brier(y, p),
            "log_loss": metrics.log_loss(y, p),
            "pr_auc": metrics.pr_auc(y, p),
            "n_scored": joined.height,
            "coverage": joined.height / n_eval if n_eval else 0.0,
        })
    return pl.DataFrame(rows).sort("brier")


def reliability_diagram(model_id: str, y_true, p_pred, out_path: Path) -> Path:
    tab = metrics.reliability_table(y_true, p_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ok = tab.filter(pl.col("n") > 0)
    ax.plot(ok["mean_pred"], ok["frac_open"], "o-", label=model_id)
    ax.set_xlabel("predicted P(open)")
    ax.set_ylabel("observed open fraction")
    ax.set_title(f"Reliability — {model_id}")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_labels(lake: Lake, band: str, start: date, end: date) -> pl.DataFrame:
    con = lake.connect()
    return con.execute(
        "SELECT * FROM labels WHERE band = ? AND window_start >= ? AND window_start < ?",
        [band, datetime.combine(start, time(), tzinfo=timezone.utc),
         datetime.combine(end, time(), tzinfo=timezone.utc)]).pl().with_columns(
        pl.col("window_start").cast(pl.Datetime("us", "UTC")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Headline eval report")
    ap.add_argument("--band", required=True, choices=schema.BANDS)
    ap.add_argument("--train-start", type=date.fromisoformat, required=True)
    ap.add_argument("--train-end", type=date.fromisoformat, required=True)
    ap.add_argument("--eval-start", type=date.fromisoformat, required=True)
    ap.add_argument("--eval-end", type=date.fromisoformat, required=True)
    ap.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    if (datetime.combine(args.eval_start, time(), tzinfo=timezone.utc)
            - datetime.combine(args.train_end, time(), tzinfo=timezone.utc)
            < timedelta(hours=GAP_HOURS)):
        print(f"train/eval gap < {GAP_HOURS}h — leakage (SPEC-labeling §6.1)",
              file=sys.stderr)
        sys.exit(2)

    lake = Lake(args.lake_root)
    # training set: per-(band, date) deterministic 3:1 sampling (SPEC §4.5)
    full_train = _load_labels(lake, args.band, args.train_start, args.train_end)
    parts = []
    d = args.train_start
    while d < args.train_end:
        nxt = d + timedelta(days=1)
        stratum = full_train.filter(
            (pl.col("window_start").dt.date() >= d) & (pl.col("window_start").dt.date() < nxt))
        if stratum.height:
            parts.append(sample_training(stratum, args.band, d))
        d = nxt
    train = pl.concat(parts)
    eval_labels = _load_labels(lake, args.band, args.eval_start, args.eval_end)  # FULL set

    model = ClimatologyModel().fit(train)
    pred = model.predict_p_open(eval_labels.select(_KEY))
    tab = headline_table({model.model_id: pred}, eval_labels)

    joined = (eval_labels.select(_KEY + ["open"])
              .join(pred, on=_KEY).filter(pl.col("p_open").is_not_null()))
    png = reliability_diagram(model.model_id, joined["open"].to_numpy(),
                              joined["p_open"].to_numpy(),
                              args.out / f"reliability_{model.model_id}.png")

    stats_glob = lake.root / "label_stats" / f"band={args.band}"
    stats_md = ""
    if stats_glob.exists():
        stats = pl.read_parquet(stats_glob / "**" / "*.parquet").sort("date")
        stats_md = "\n## Unlabeled-activity fraction (per day)\n\n" + \
            "\n".join(f"- {r['date']}: {r['unlabeled_fraction']:.3f} "
                      f"(pos={r['n_pos']}, neg={r['n_neg']}, unlab={r['n_unlabeled_activity']})"
                      for r in stats.iter_rows(named=True))

    args.out.mkdir(parents=True, exist_ok=True)
    md = args.out / "headline.md"
    lines = ["# Headline eval", "",
             f"Band {args.band}; train {args.train_start}..{args.train_end}; "
             f"eval {args.eval_start}..{args.eval_end} (FULL label set)", "",
             "| model | Brier | log-loss | PR-AUC | n_scored | coverage |",
             "|---|---|---|---|---|---|"]
    for r in tab.iter_rows(named=True):
        lines.append(f"| {r['model']} | {r['brier']:.4f} | {r['log_loss']:.4f} "
                     f"| {r['pr_auc']:.4f} | {r['n_scored']} | {r['coverage']:.3f} |")
    md.write_text("\n".join(lines) + stats_md + "\n")
    print(f"wrote {md} and {png}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/eval/test_report.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/eval/report.py tests/eval/test_report.py
git commit -m "feat: headline table, reliability diagram, eval-report CLI"
```

---

### Task 13: features/qa.py — physics-grounded QA gates + `qa-gates` CLI

SPEC-labeling §6.8: eight checks, run before any training, fail loudly. Tri-state results: `PASS`, `FAIL`, `INSUFFICIENT` (per SPEC check 6's precedent: a check whose required data doesn't exist reports "insufficient data" rather than passing silently — in M0 that covers checks 2, 4 (bands not extracted), 6 (needs multi-year F10.7) and 7 (needs Kp, no space_weather yet)). CLI exits non-zero on any FAIL; INSUFFICIENT prints a warning and exits 0 unless `--strict`.

**Files:**
- Create: `src/propagation/features/qa.py`
- Test: `tests/features/test_qa.py`

**Interfaces:**
- Consumes: labels frames, `grid_to_latlon`, `haversine_km`, `lake/hygiene_stats`, `lake/label_stats`, `space_weather` view (absent in M0 → INSUFFICIENT).
- Produces: `@dataclass QAResult(check_id: int, name: str, status: str, detail: str)` with `status ∈ {"PASS","FAIL","INSUFFICIENT"}`; `run_all(lake: Lake) -> list[QAResult]`; `main()` (`qa-gates` CLI: `--band` repeatable, `--lake-root`, `--strict`; prints one line per result). Per-check functions `check_1_diurnal_20m(labels: pl.DataFrame) -> QAResult` … `check_8_volume(lake: Lake, band: str) -> QAResult` so tests hit them directly.
- Shared helper: `_path_geometry(labels)` adds `mid_lat, mid_lon, dist_km, lst_hour` (local solar time = (UTC hour + mid_lon/15) mod 24) from field centers via `grid_to_latlon`. Thresholds copied from the SPEC table verbatim; minimum sample size per compared group: 200 labeled rows, else INSUFFICIENT.

- [ ] **Step 1: Write the failing tests** — `tests/features/test_qa.py`:

```python
from datetime import datetime, timedelta, timezone

import polars as pl

from propagation.features import qa

W0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


def _labels(rows):
    return pl.DataFrame(
        [{"window_start": W0 + timedelta(hours=r["h"]), "tx_field": r.get("tx", "EM"),
          "rx_field": r.get("rx", "IO"), "band": r.get("band", "20m"),
          "open": r["open"]} for r in rows]
    ).with_columns(pl.col("open").cast(pl.Int8))


def _diurnal_labels(day_rate: float, night_rate: float, n=400):
    # EM->IO midpoint ~ -45 lon -> LST = UTC - 3h. Day LST 9-15 => UTC 12-18;
    # night LST 21-03 => UTC 0-6.
    rows = []
    for i in range(n):
        rows.append({"h": 12 + (i % 6), "open": 1 if i < day_rate * n else 0})
        rows.append({"h": (i % 6), "open": 1 if i < night_rate * n else 0})
    return _labels(rows)


def test_check1_passes_on_strong_diurnal():
    res = qa.check_1_diurnal_20m(_diurnal_labels(0.6, 0.1))
    assert res.status == "PASS"


def test_check1_fails_on_flat_diurnal():
    res = qa.check_1_diurnal_20m(_diurnal_labels(0.3, 0.3))
    assert res.status == "FAIL"


def test_check1_insufficient_on_tiny_sample():
    res = qa.check_1_diurnal_20m(_diurnal_labels(0.6, 0.1, n=10))
    assert res.status == "INSUFFICIENT"


def test_check2_insufficient_without_low_bands():
    res = qa.check_2_diurnal_low_bands(_labels([{"h": 1, "open": 1}]))
    assert res.status == "INSUFFICIENT"


def test_check5_reciprocity():
    rows = []
    for i in range(300):
        rows.append({"h": i % 24, "tx": "EM", "rx": "IO", "open": int(i % 3 == 0)})
        rows.append({"h": i % 24, "tx": "IO", "rx": "EM", "open": int(i % 3 == 0)})
        rows.append({"h": i % 24, "tx": "PM", "rx": "CN", "open": int(i % 5 == 0)})
        rows.append({"h": i % 24, "tx": "CN", "rx": "PM", "open": int(i % 5 == 0)})
    res = qa.check_5_reciprocity(_labels(rows))
    assert res.status == "PASS"


def test_run_all_returns_eight(tmp_path):
    from propagation.data.lake import Lake
    results = qa.run_all(Lake(tmp_path))
    assert len(results) == 8
    assert all(r.status == "INSUFFICIENT" for r in results)  # empty lake
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/features/test_qa.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/propagation/features/qa.py`:

```python
"""Physics-grounded QA gates (SPEC-labeling §6.8). Run before any training."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from propagation import schema
from propagation.data.lake import Lake
from propagation.features.geometry import grid_to_latlon, haversine_km

_MIN_GROUP = 200  # labeled rows per compared group


@dataclass(frozen=True)
class QAResult:
    check_id: int
    name: str
    status: str  # PASS | FAIL | INSUFFICIENT
    detail: str


def _path_geometry(labels: pl.DataFrame) -> pl.DataFrame:
    def geom(s: dict) -> dict:
        tla, tlo = grid_to_latlon(s["tx_field"])
        rla, rlo = grid_to_latlon(s["rx_field"])
        return {"mid_lat": (tla + rla) / 2, "mid_lon": (tlo + rlo) / 2,
                "dist_km": haversine_km(tla, tlo, rla, rlo)}
    return labels.with_columns(
        pl.struct(["tx_field", "rx_field"]).map_elements(
            geom, return_dtype=pl.Struct({"mid_lat": pl.Float64, "mid_lon": pl.Float64,
                                          "dist_km": pl.Float64})).alias("_g")
    ).unnest("_g").with_columns(
        lst_hour=((pl.col("window_start").dt.hour()
                   + pl.col("mid_lon") / 15.0) % 24.0))


def _ratio_check(check_id: int, name: str, num: pl.DataFrame, den: pl.DataFrame,
                 threshold: float) -> QAResult:
    if num.height < _MIN_GROUP or den.height < _MIN_GROUP:
        return QAResult(check_id, name, "INSUFFICIENT",
                        f"groups {num.height}/{den.height} < {_MIN_GROUP}")
    a, b = num["open"].mean(), den["open"].mean()
    if b == 0:
        return QAResult(check_id, name, "PASS" if a > 0 else "INSUFFICIENT",
                        f"{a:.3f} vs 0")
    ratio = a / b
    ok = ratio > threshold
    return QAResult(check_id, name, "PASS" if ok else "FAIL",
                    f"ratio {ratio:.2f} (threshold {threshold})")


def check_1_diurnal_20m(labels: pl.DataFrame) -> QAResult:
    g = _path_geometry(labels.filter(pl.col("band") == "20m"))
    g = g.filter(pl.col("mid_lat").abs().is_between(25, 55)
                 & pl.col("dist_km").is_between(3000, 8000))
    day = g.filter(pl.col("lst_hour").is_between(9, 15))
    night = g.filter((pl.col("lst_hour") >= 21) | (pl.col("lst_hour") < 3))
    return _ratio_check(1, "20m diurnal day/night > 2", day, night, 2.0)


def check_2_diurnal_low_bands(labels: pl.DataFrame) -> QAResult:
    g = _path_geometry(labels.filter(pl.col("band").is_in(["160m", "80m"])))
    g = g.filter(pl.col("dist_km") > 2000)
    night = g.filter((pl.col("lst_hour") >= 21) | (pl.col("lst_hour") < 3))
    day = g.filter(pl.col("lst_hour").is_between(9, 15))
    return _ratio_check(2, "160m/80m night/day > 5", night, day, 5.0)


def check_3_grayline_40m(labels: pl.DataFrame) -> QAResult:
    g = _path_geometry(labels.filter(pl.col("band") == "40m"))
    g = g.filter(pl.col("dist_km") > 6000)
    gray = g.filter(pl.col("lst_hour").is_between(5, 7)
                    | pl.col("lst_hour").is_between(17, 19))
    midday = g.filter(pl.col("lst_hour").is_between(11, 13))
    if gray.height < _MIN_GROUP or midday.height < _MIN_GROUP:
        return QAResult(3, "40m gray-line max", "INSUFFICIENT",
                        f"groups {gray.height}/{midday.height} < {_MIN_GROUP}")
    ok = gray["open"].mean() > midday["open"].mean()
    return QAResult(3, "40m gray-line max", "PASS" if ok else "FAIL",
                    f"gray {gray['open'].mean():.3f} vs midday {midday['open'].mean():.3f}")


def check_4_sporadic_e_6m(labels: pl.DataFrame) -> QAResult:
    g = _path_geometry(labels.filter(pl.col("band") == "6m"))
    g = g.filter(pl.col("dist_km").is_between(1000, 2300) & (pl.col("mid_lat") > 0))
    summer = g.filter(pl.col("window_start").dt.month().is_in([5, 6, 7]))
    winter = g.filter(pl.col("window_start").dt.month().is_in([11, 12, 1]))
    return _ratio_check(4, "6m Es May-Jul >= 3x Nov-Jan", summer, winter, 3.0)


def check_5_reciprocity(labels: pl.DataFrame) -> QAResult:
    rates = (labels.with_columns(month=pl.col("window_start").dt.strftime("%Y-%m"))
             .group_by(["tx_field", "rx_field", "band", "month"])
             .agg(rate=pl.col("open").mean(), n=pl.len())
             .filter(pl.col("n") >= 20))
    fwd = rates.rename({"rate": "rate_fwd"})
    rev = rates.rename({"tx_field": "rx_field", "rx_field": "tx_field",
                        "rate": "rate_rev"}).select(
        ["tx_field", "rx_field", "band", "month", "rate_rev"])
    both = fwd.join(rev, on=["tx_field", "rx_field", "band", "month"])
    # each unordered pair appears twice (A->B and B->A rows); r is unaffected
    if both.height < 10:
        return QAResult(5, "reciprocity r > 0.6", "INSUFFICIENT",
                        f"{both.height} pair-months < 10")
    r = float(np.corrcoef(both["rate_fwd"].to_numpy(), both["rate_rev"].to_numpy())[0, 1])
    return QAResult(5, "reciprocity r > 0.6", "PASS" if r > 0.6 else "FAIL",
                    f"r = {r:.3f}")


def check_6_solar_cycle(lake: Lake) -> QAResult:
    return QAResult(6, "10m open-rate vs F10.7 corr > 0.5", "INSUFFICIENT",
                    "needs multi-year history + F10.7 (M2 space weather)")


def check_7_storm_response(lake: Lake) -> QAResult:
    base = lake.root / "space_weather"
    if not (base.exists() and any(base.rglob("*.parquet"))):
        return QAResult(7, "Kp>=6 trans-polar suppression", "INSUFFICIENT",
                        "no space_weather data in lake (arrives in M2)")
    return QAResult(7, "Kp>=6 trans-polar suppression", "INSUFFICIENT",
                    "storm check implemented in M2 alongside Kp features")


def check_8_volume(lake: Lake, band: str) -> QAResult:
    stats_dir = lake.root / "label_stats" / f"band={band}"
    hyg_dir = lake.root / "hygiene_stats" / f"band={band}"
    if not (stats_dir.exists() and hyg_dir.exists()):
        return QAResult(8, "volume/hygiene", "INSUFFICIENT", "stats partitions missing")
    stats = pl.read_parquet(stats_dir / "**" / "*.parquet").sort("date").with_columns(
        total=pl.col("n_pos") + pl.col("n_neg"))
    med = stats["total"].tail(28).median()
    bad_days = stats.filter((pl.col("total") > 5 * med) | (pl.col("total") < med / 5))
    hyg = pl.read_parquet(hyg_dir / "**" / "*.parquet")
    total_spots = hyg["n"].sum()
    rr73 = hyg.filter(pl.col("reject_reason") == "rr73")["n"].sum() or 0
    rr73_frac = rr73 / total_spots if total_spots else 0.0
    problems = []
    if bad_days.height:
        problems.append(f"{bad_days.height} days outside 5x of median {med}")
    if rr73_frac >= 0.005:
        problems.append(f"RR73 rejects {rr73_frac:.4f} >= 0.5%")
    if problems:
        return QAResult(8, "volume/hygiene", "FAIL", "; ".join(problems))
    return QAResult(8, "volume/hygiene", "PASS",
                    f"median/day {med}, RR73 {rr73_frac:.4f}")


def run_all(lake: Lake, bands: list[str] | None = None) -> list[QAResult]:
    bands = bands or ["20m"]
    base = lake.root / "labels"
    if base.exists() and any(base.rglob("*.parquet")):
        labels = pl.read_parquet(base / "**" / "*.parquet")
    else:
        labels = pl.DataFrame(schema={"window_start": pl.Datetime("us", "UTC"),
                                      "tx_field": pl.Utf8, "rx_field": pl.Utf8,
                                      "band": pl.Utf8, "open": pl.Int8})
    return [
        check_1_diurnal_20m(labels),
        check_2_diurnal_low_bands(labels),
        check_3_grayline_40m(labels),
        check_4_sporadic_e_6m(labels),
        check_5_reciprocity(labels),
        check_6_solar_cycle(lake),
        check_7_storm_response(lake),
        check_8_volume(lake, bands[0]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SPEC-labeling §6.8 QA gates")
    ap.add_argument("--band", action="append", choices=schema.BANDS, dest="bands")
    ap.add_argument("--lake-root", type=Path, default=Path("data/lake"))
    ap.add_argument("--strict", action="store_true",
                    help="INSUFFICIENT also fails the gate")
    args = ap.parse_args()
    results = run_all(Lake(args.lake_root), args.bands)
    worst = 0
    for r in results:
        print(f"[{r.status:>12}] check {r.check_id}: {r.name} — {r.detail}")
        if r.status == "FAIL" or (args.strict and r.status == "INSUFFICIENT"):
            worst = 1
    sys.exit(worst)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/features/test_qa.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/qa.py tests/features/test_qa.py
git commit -m "feat: physics-grounded QA gates + qa-gates CLI (SPEC-labeling §6.8)"
```

---

### Task 14: end-to-end driver + docs

**Files:**
- Create: `scripts/m0.sh`
- Modify: `CLAUDE.md` (Status stanza: design phase → M0 implemented), `ROADMAP.md` only if acceptance deviates (it shouldn't)
- Test: manual acceptance run (this is the milestone gate, not a pytest)

**Interfaces:**
- Consumes: every CLI above.
- Produces: `scripts/m0.sh` — the ROADMAP acceptance run: empty `data/` → headline report.

- [ ] **Step 1: Write** `scripts/m0.sh`:

```bash
#!/usr/bin/env bash
# M0 acceptance: end-to-end reproducibility from empty data/ (ROADMAP.md).
set -euo pipefail
cd "$(dirname "$0")/.."

LAKE="${LAKE_ROOT:-data/lake}"
CACHE="${CACHE_DIR:-data/cache}"
BAND=20m

uv run extract-wsprnet --start 2026-05-01 --end 2026-06-30 --band $BAND \
  --lake-root "$LAKE" --cache-dir "$CACHE"
uv run build-spots-q --start 2026-05-01 --end 2026-06-30 --band $BAND --lake-root "$LAKE"
uv run build-uptime  --start 2026-05-01 --end 2026-06-30 --band $BAND --lake-root "$LAKE"
uv run build-labels  --start 2026-05-01 --end 2026-06-30 --band $BAND --lake-root "$LAKE"
uv run qa-gates --band $BAND --lake-root "$LAKE"
uv run eval-report --band $BAND \
  --train-start 2026-05-01 --train-end 2026-05-30 \
  --eval-start 2026-06-01 --eval-end 2026-07-01 \
  --lake-root "$LAKE" --out reports/

echo "M0 complete: see reports/headline.md and reports/reliability_climatology.png"
```

Then: `chmod +x scripts/m0.sh`.

- [ ] **Step 2: Run the acceptance test**

Run: `rm -rf data/ reports/ && ./scripts/m0.sh`
Expected: downloads two WSPRnet monthly archives (~100–300 MB total, cached under `data/cache/`), builds the lake, QA gates print 8 lines (check 1 PASS required; 2/4/6/7 INSUFFICIENT expected in M0; 3/5/8 must not FAIL), and `reports/headline.md` + `reports/reliability_climatology.png` exist with a real Brier number for `climatology`.

If check 1 FAILs on real data, STOP — the labeling pipeline is broken (that check is the physics smoke test); debug before proceeding, do not relax the threshold.

- [ ] **Step 3: Run the full test suite + lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green.

- [ ] **Step 4: Update CLAUDE.md Status stanza**

Replace the `## Status` body with:

```markdown
M0 complete: WSPRnet 20m lake, receiver-uptime tables, monitor-normalized
labels, QA gates, climatology baseline + headline report (`scripts/m0.sh`
reproduces from empty `data/`). Next: M1 in ROADMAP.md (vendored P.533
baseline).
```

- [ ] **Step 5: Commit**

```bash
git add scripts/m0.sh CLAUDE.md
git commit -m "feat: M0 end-to-end driver + status update"
```

---

## Self-Review (performed while writing this plan)

**Spec coverage.** ROADMAP M0 criteria → tasks: extractor (T6), Parquet+DuckDB views (T4), receiver-uptime (T7), monitor-normalized labels + unlabeled fraction (T8), deterministic sampling (T9), climatology on held-out month (T10, T12), `uv run` reproducibility (T14), Brier + reliability diagram from eval/report.py (T12). SPEC-labeling §1 (T5), §2 (T8), §3 (T7), §4.1–4.3 (T8), §4.4 (T7), §4.5 (T9), §6.1 gap (T11, T12 CLI guard), §6.8 (T13). Gaps deliberately deferred and documented in-task: month/SSN conditioning (T10), storm/solar-cycle QA checks (T13 → M2), vectorizing the hygiene row loop.

**Placeholder scan.** No TBDs; every code step shows complete code; QA checks 6/7 return explicit INSUFFICIENT results rather than stub logic — that is their correct M0 behavior per SPEC §6.8, not an omission.

**Type consistency.** `spots_q` frames = SPOT_SCHEMA + QUALIFIED_EXTRA everywhere (T5 produces, T7/T8 consume); label columns match SPEC §4.5 storage list exactly (T8 produces, T9/T10/T12/T13 consume); `predict_p_open` in/out (T10) matches what `headline_table` joins on (T12); fold gap constant shared T11→T12. `sample_training` is called per-(band, date) stratum by T12's CLI, matching its single-stratum contract.

**Known judgment calls (flagged for the executor).** (1) RR73 rejects the whole spot, not just the grid — SPEC wording "reject the literal grid" + §6.8 counts rejections per spot; (2) prefix-form callsigns (`EA8/K1ABC`) fail the SPEC regex by design — counted under `callsign` rejects; (3) `blocked_folds` tiles eval blocks backward from data_end (deterministic, matches the M0 one-fold usage); (4) uptime CLI reads ±1 day of spots_q so date-edge windows get their full ±30 min evidence.







