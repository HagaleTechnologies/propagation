# Shared interface contract for M0–M4 implementation plans

Status: pinned 2026-07-09. Every milestone plan in this directory builds
against these exact names, paths, and types. A plan may extend this contract
(new functions, new columns) but may not rename or retype anything here
without updating this file and every plan that consumes the changed item.

Normative sources (this file only *pins names*, it never overrides them):
`ARCHITECTURE.md`, `docs/SPEC-labeling.md`, `docs/SPEC-contract-notes.md`,
`contracts/prediction-surface.v1.schema.json`, `docs/REVIEW-FINDINGS.md`.

---

## Global constraints (apply to every task in every plan)

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

## pyproject (created in M0, extended later)

- name `propagation`, `requires-python = ">=3.11"`.
- M0 deps: `duckdb`, `polars`, `pyarrow`, `numpy`, `httpx`, `matplotlib`,
  `scikit-learn`. Dev: `pytest`, `ruff`.
- Later milestones add via `uv add`: M2 `lightgbm`, `astral`; M3 `paho-mqtt`,
  `boto3` (extra `cqdx`); M4 `jsonschema`.
- Console scripts land under `[project.scripts]` as each milestone adds them,
  e.g. `extract-wsprnet = "propagation.data.wsprnet:main"`.

## Module map (who owns what)

| path | owner | contents |
|---|---|---|
| `src/propagation/schema.py` | M0 | `BANDS`, `SPOT_SCHEMA`, `mode_class()`, window helpers |
| `src/propagation/data/base.py` | M0 | extractor interface |
| `src/propagation/data/wsprnet.py` | M0 | WSPRnet archive extractor |
| `src/propagation/data/hygiene.py` | M0 | spot hygiene + dedup (SPEC §1) |
| `src/propagation/data/lake.py` | M0 | partition paths, writers, DuckDB views |
| `src/propagation/features/labels.py` | M0 | uptime, labels, sampling (SPEC §2–4) |
| `src/propagation/features/qa.py` | M0 | QA gates (SPEC §6.8 checks 1–8) |
| `src/propagation/models/base.py` | M0 | `OpennessModel` protocol |
| `src/propagation/models/climatology.py` | M0 | M-0 baseline |
| `src/propagation/eval/splits.py` | M0 | blocked folds, gap enforcement |
| `src/propagation/eval/metrics.py` | M0 | Brier, log-loss, PR-AUC, reliability |
| `src/propagation/eval/report.py` | M0 | headline table + reliability diagram |
| `baselines/p533/` | M1 | vendored ITURHFProp + build + provenance |
| `src/propagation/models/p533_baseline.py` | M1 | CLI wrapper + scorer |
| `src/propagation/data/swpc.py` | M2 | SWPC space-weather extractor |
| `src/propagation/features/geometry.py` | M2 | grid→latlon, haversine, bearing, midpoint, control points, geomag lat |
| `src/propagation/features/solar.py` | M2 | solar zenith, daylight fraction, gray-line |
| `src/propagation/features/spaceweather.py` | M2 | as-of joins onto window grid |
| `src/propagation/features/history.py` | M2 | autoregressive features (Δ_avail respected) |
| `src/propagation/features/matrix.py` | M2 | assemble full feature matrix |
| `src/propagation/models/gbt.py` | M2 | LightGBM per horizon + calibration |
| `src/propagation/data/rbn.py` | M3 | RBN daily-archive extractor |
| `src/propagation/data/pskreporter.py` | M3 | live MQTT accumulator |
| `src/propagation/data/cqdx_r2.py` | M3 | optional private extractor |
| `src/propagation/serving/score.py` | M4 | batch inference → surface JSON |
| `src/propagation/serving/scoreboard.py` | M4 | live honesty check |
| `src/propagation/serving/registry.py` | M4 | model artifact registry |

## Pinned signatures

### schema.py (M0)

```python
BANDS: list[str]                       # canonical order above
DIGI_MODES: frozenset[str]             # FT8 FT4 WSPR FST4 FST4W JS8 JT65 JT9 Q65 MSK144
CW_MODES: frozenset[str]               # CW RTTY

def mode_class(mode: str) -> str       # 'digi' | 'cw' | 'other'; case-insensitive
def window_floor(ts: datetime) -> datetime   # floor to 15-min UTC boundary

SPOT_SCHEMA: dict[str, pl.DataType]    # raw common spot schema, exactly:
# source Utf8 | ts Datetime(us,UTC) | band Utf8 | mode Utf8 | freq_hz Int64
# dx_call Utf8 | de_call Utf8 | dx_grid Utf8 | de_grid Utf8
# dx_lat Float64 | dx_lon Float64 | de_lat Float64 | de_lon Float64
# snr_db Int16 | tx_dbm Int16 | distance_km Float64 | bearing_deg Float64

QUALIFIED_EXTRA: dict[str, pl.DataType]  # columns hygiene adds:
# mode_class Utf8 | dx_field Utf8(2) | de_field Utf8(2)
# dx_grid4 Utf8 nullable | de_grid4 Utf8 nullable | window_start Datetime(us,UTC)
```

### data/base.py (M0)

```python
class Extractor(Protocol):
    source: str                        # 'wsprnet'|'rbn'|'pskreporter'|'cqdx'|'swpc'
    def extract(self, start: date, end: date, bands: list[str],
                lake: "Lake", cache_dir: Path) -> list[Path]:
        """Download/convert; write raw-schema Parquet into the lake; return files written."""
```

Each extractor module also exposes `main() -> None` (argparse CLI:
`--start --end --band ... --lake-root --cache-dir`).

### data/hygiene.py (M0)

```python
def qualify_spots(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]
    # (qualifying spots with QUALIFIED_EXTRA columns, rejects with reject_reason Utf8)
def dedup_spots(df: pl.DataFrame) -> pl.DataFrame
    # SPEC §1: key (dx_call, de_call, band, mode, window_start);
    # priority wsprnet>rbn>pskreporter>cqdx, then max snr_db, then min ts
CALLSIGN_RE: re.Pattern   # SPEC §1.3 regex, applied after stripping <\ >
```

### data/lake.py (M0)

```python
class Lake:
    def __init__(self, root: Path): ...
    root: Path
    def spots_dir(self, band: str, d: date) -> Path        # spots/band=…/date=…/
    def spots_q_dir(self, band: str, d: date) -> Path      # spots_q/… (qualified+deduped)
    def uptime_dir(self, band: str, d: date) -> Path       # receiver_uptime/…
    def labels_dir(self, band: str, d: date) -> Path       # labels/…
    def space_weather_dir(self, d: date) -> Path           # space_weather/date=…/
    def write_parquet(self, df: pl.DataFrame, dest_dir: Path,
                      name: str = "part-0") -> Path
    def connect(self) -> duckdb.DuckDBPyConnection
        # registers views over whatever partitions exist:
        #   raw_spots, spots (= spots_q), receiver_uptime, labels, space_weather
```

Everything downstream of hygiene reads the `spots` view / `spots_q`
partitions — never `raw_spots`.

### features/labels.py (M0)

```python
def build_receiver_uptime(spots_q: pl.DataFrame, band: str, d: date) -> pl.DataFrame
    # SPEC §3 exactly; needs spots_q covering [d 00:00 − 30min, d+1 00:00 + 30min)
    # columns: window_start, de_call, de_field, de_grid4, band, mode_class,
    #          n_evidence_reports Int32, first_evidence_ts, last_evidence_ts
def build_labels(spots_q: pl.DataFrame, uptime: pl.DataFrame,
                 band: str, d: date) -> pl.DataFrame
    # SPEC §2+§4; columns: window_start, tx_field, rx_field, band, open Int8,
    # n_spots Int32, n_monitors Int32, n_tx_stations Int32, evidence_tier Utf8,
    # snr_ft8eq_p50 Float32 nullable, sample_weight Float32, split_tag Utf8 nullable
    # (sample_weight = 1.0 here; sampling assigns real weights)
def sample_training(labels: pl.DataFrame, band: str, d: date) -> pl.DataFrame
    # SPEC §4.5: 3:1 stratum sampling, deterministic seed (see Global constraints);
    # sets sample_weight = 1/sampling_rate on kept negatives
def snr_ft8eq(snr_db: int, mode: str, tx_dbm: int | None) -> float | None
    # SPEC §4.4 offsets
```

### models/base.py (M0)

```python
class OpennessModel(Protocol):
    model_id: str
    def predict_p_open(self, cells: pl.DataFrame) -> pl.DataFrame:
        """cells: window_start, tx_field, rx_field, band.
        Returns cells + column p_open Float64 in [0,1], null = abstain."""
```

`fit` signatures vary per model and are NOT part of the protocol
(climatology fits on labels; GBT fits on labels+features; P.533 has no fit).
Every model documents its own constructor/fit in its plan.

### eval/ (M0)

```python
# splits.py
GAP_HOURS: int = 48    # widen if max horizon + max AR lookback grows (SPEC §6.1)
@dataclass(frozen=True)
class Fold:
    train_start: datetime; train_end: datetime
    eval_start: datetime;  eval_end: datetime
def blocked_folds(data_start: datetime, data_end: datetime,
                  eval_days: int, gap_hours: int = GAP_HOURS) -> list[Fold]

# metrics.py — all take numpy arrays; weights optional (None = unweighted)
def brier(y_true, p_pred, weights=None) -> float
def log_loss(y_true, p_pred, weights=None) -> float
def pr_auc(y_true, p_pred, weights=None) -> float
def reliability_table(y_true, p_pred, n_bins: int = 10) -> pl.DataFrame
    # columns: bin_lo, bin_hi, n, mean_pred, frac_open

# report.py
def headline_table(predictions: dict[str, pl.DataFrame],
                   eval_labels: pl.DataFrame) -> pl.DataFrame
    # predictions: model_id -> output of predict_p_open over eval cells
    # rows = models, cols = brier/log_loss/pr_auc; eval on FULL label set
def reliability_diagram(model_id: str, y_true, p_pred, out_path: Path) -> Path
def main() -> None    # CLI `eval-report`; writes reports/headline.md + PNG(s)
```

### models/p533_baseline.py (M1)

```python
def p533_score(tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
               band: str, month: int, hour: int, ssn: float) -> P533Result
@dataclass(frozen=True)
class P533Result:
    reliability_pct: float   # BCR, 0–100
    snr_db: float            # median SNR re mode reference bandwidth
class P533Model:             # implements OpennessModel via midpoint-to-midpoint
    model_id = "p533"
    def __init__(self, ssn_by_month: dict[str, float]): ...   # "YYYY-MM" -> SSN
    def predict_p_open(self, cells: pl.DataFrame) -> pl.DataFrame
```

Binary: `baselines/p533/build.sh` produces `baselines/p533/bin/iturhfprop`;
provenance recorded in `baselines/p533/PROVENANCE.md`.

### features/ (M2)

```python
# geometry.py
def grid_to_latlon(grid: str) -> tuple[float, float]     # center of field/grid4
def haversine_km(lat1, lon1, lat2, lon2) -> float
def initial_bearing_deg(lat1, lon1, lat2, lon2) -> float
def midpoint_latlon(lat1, lon1, lat2, lon2) -> tuple[float, float]
def control_points(lat1, lon1, lat2, lon2) -> list[tuple[float, float]]
    # P.533 convention: 1000 km from each terminus along the great circle
def geomag_lat(lat: float, lon: float) -> float          # centered-dipole approx

# solar.py
def solar_zenith_deg(lat: float, lon: float, ts: datetime) -> float
def daylight_fraction(points: list[tuple[float, float]], ts: datetime) -> float
def minutes_since_terminator(lat: float, lon: float, ts: datetime) -> float

# spaceweather.py
def load_sw_asof(lake: Lake, start: datetime, end: datetime) -> pl.DataFrame
    # 15-min grid; estimated (as-of-available) series only; columns include
    # kp_est, kp_est_lag{3,6,12,24,48}h, f107, f107_sm27, xray_flux, sw_speed, sw_bz

# history.py
DELTA_AVAIL_MIN: int = 5
def ar_features(lake: Lake, cells: pl.DataFrame, pred_time_col: str) -> pl.DataFrame
    # trailing 15m/1h/3h/24h counts + median snr_ft8eq for cell, reverse cell,
    # adjacent cells, adjacent bands, same-hour-yesterday, band-global activity;
    # every window ends at pred_time − DELTA_AVAIL_MIN

# matrix.py
def feature_matrix(lake: Lake, labels: pl.DataFrame, horizon_s: int) -> pl.DataFrame
    # pred_time = window_start − horizon; joins geometry+solar+sw+AR features
FEATURE_COLUMNS: list[str]   # canonical ordered list, single source of truth
```

### models/gbt.py (M2)

```python
class GBTModel:                # implements OpennessModel
    model_id: str              # f"gbt-h{horizon_s}"
    def __init__(self, horizon_s: int): ...
    def fit(self, labels: pl.DataFrame, features: pl.DataFrame) -> "GBTModel"
        # binary logloss objective; MUST pass sample_weight; isotonic
        # calibration on a train-fold tail, also weighted (SPEC §4.5)
    def predict_p_open(self, cells: pl.DataFrame) -> pl.DataFrame
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> "GBTModel"
```

### serving/ (M4)

```python
# registry.py
@dataclass(frozen=True)
class ModelRecord:
    model_id: str; version: str; trained_through: date; path: Path
def latest(registry_dir: Path, model_id: str) -> ModelRecord

# score.py
def build_surface(models: dict[int, "OpennessModel"],   # horizon_s -> model
                  cells: pl.DataFrame, valid_from: datetime,
                  record: ModelRecord) -> dict          # contract v1 document
def main() -> None   # CLI `serve-score`: score → validate vs schema → write JSON
# columnar encoding, p_open_pct int 0–100, null = abstain, confidence tier
# from train-fold label density — per contracts/prediction-surface.v1.schema.json

# scoreboard.py
def score_yesterday(lake: Lake, surfaces_dir: Path) -> pl.DataFrame
    # joins archived surfaces vs realized labels; Brier per band/horizon
```

## Cross-milestone dependency summary

- M1 consumes: `Lake`, `labels` view, `eval/*`, `OpennessModel`, `BANDS`.
- M2 consumes: all of M0, plus M1's `P533Model` row in the headline table.
- M3 consumes: M0's `Extractor`/`Lake`/hygiene/labels/QA unchanged (new
  sources, all bands); M2's matrix/GBT retrained at all horizons.
- M4 consumes: M2/M3 `GBTModel.load`, `feature_matrix`, contract schema.
