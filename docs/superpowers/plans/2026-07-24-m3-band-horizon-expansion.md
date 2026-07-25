# M3 Band/Horizon Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the M2 pipeline (currently proven on 20m+10m at h=0 only) to all 11 HF bands and horizons {0,1h,3h,6h,12h,24h}, per ARCHITECTURE.md §5's "one model per horizon, shared across bands, band as a feature."

**Architecture:** Split feature columns into "future-computable" (time, solar, geometry — anchored at the target `window_start`, unchanged) and "as-of-now" (space weather, AR spot history — re-anchored at `prediction_time = window_start − horizon_hours`). Add `band_ordinal` as a shared feature so one GBTModel per horizon trains across all bands. Batch multi-band WSPRnet extraction into one archive pass. Implement the three QA checks that were stubbed pending M2's solar/space-weather features.

**Tech Stack:** Python 3.11+/uv, Polars, LightGBM, pytest — matches the existing repo stack, no new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-24-m3-band-horizon-expansion-design.md` — read it before starting; every task below implements one numbered section of it.
- Definition of done for this plan: pipeline capability + unit tests + one smoke run beyond M2's proven 20m/10m@h=0. The full 11-band×6-horizon historical production run is explicitly out of scope (spec §7).
- Bands: all 11 in `propagation.features.history.BAND_ORDER` (renamed from `_BAND_ORDER` in Task 1) — `["160m","80m","60m","40m","30m","20m","17m","15m","12m","10m","6m"]`.
- Band groups for reporting: low={160m,80m,60m,40m}, mid={30m,20m,17m,15m}, high={12m,10m,6m}.
- Run the full test suite (`uv run pytest`) before declaring any task done — never just the new test file.
- Conventional commits with scope, e.g. `feat(features):`, `feat(qa):`, `feat(scripts):` — this repo's existing commit style (see `git log --oneline`).

---

### Task 1: Promote `BAND_ORDER` and add the `band_ordinal` feature

**Files:**
- Modify: `src/propagation/features/history.py:50,82,85,86,87,270` (rename `_BAND_ORDER` → `BAND_ORDER`)
- Modify: `src/propagation/features/matrix.py` (add `add_band_feature`, extend `FEATURE_COLUMNS`, wire into `build_feature_matrix`)
- Test: `tests/features/test_matrix.py`

**Interfaces:**
- Produces: `propagation.features.history.BAND_ORDER: list[str]` (public, was `_BAND_ORDER`); `propagation.features.matrix.add_band_feature(labels: pl.DataFrame) -> pl.DataFrame` (adds `band_ordinal: Int64`); `"band_ordinal"` added to `FEATURE_COLUMNS`.

- [ ] **Step 1: Rename `_BAND_ORDER` to `BAND_ORDER` in history.py**

In `src/propagation/features/history.py`, replace every occurrence of `_BAND_ORDER` with `BAND_ORDER` (6 occurrences: the definition at line 50, and usages in `_adjacent_bands` and the `band_map` comprehension in `add_history_features`). No other changes in this file.

- [ ] **Step 2: Run the existing test suite to confirm the rename is behavior-preserving**

Run: `uv run pytest tests/features/test_history.py tests/test_leakage.py -v`
Expected: PASS (all previously-passing tests still pass; this is a pure rename)

- [ ] **Step 3: Write the failing test for `add_band_feature`**

Add to `tests/features/test_matrix.py` (extend the existing import line to include `add_band_feature`):

```python
from propagation.features.matrix import (
    FEATURE_COLUMNS,
    add_band_feature,
    add_time_features,
    build_feature_matrix,
)


def test_add_band_feature_is_ordinal_and_monotonic_in_band_order():
    from propagation.features.history import BAND_ORDER
    labels = pl.DataFrame({"band": BAND_ORDER})
    out = add_band_feature(labels)
    assert out["band_ordinal"].to_list() == list(range(len(BAND_ORDER)))


def test_feature_columns_includes_band_ordinal_not_raw_band():
    assert "band_ordinal" in FEATURE_COLUMNS
    assert "band" not in FEATURE_COLUMNS
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/features/test_matrix.py -v -k band`
Expected: FAIL with `ImportError: cannot import name 'add_band_feature'`

- [ ] **Step 5: Implement `add_band_feature` and extend `FEATURE_COLUMNS`**

In `src/propagation/features/matrix.py`, change the import block (line 19-22) to:

```python
from propagation.features.geometry import add_geometry_features
from propagation.features.history import BAND_ORDER, add_history_features
from propagation.features.solar import add_solar_features
from propagation.features.spaceweather import add_spaceweather_features
```

Add, right after the `_TIME_COLS` definition (after line 24):

```python
_BAND_COLS = ["band_ordinal"]
_BAND_ORDINAL = {band: i for i, band in enumerate(BAND_ORDER)}
```

Change the `FEATURE_COLUMNS` line (was line 46) to:

```python
FEATURE_COLUMNS = _TIME_COLS + _BAND_COLS + _GEOMETRY_COLS + _SOLAR_COLS + _SPACEWEATHER_COLS + _HISTORY_COLS
```

Add a new function, after `add_time_features`:

```python
def add_band_feature(labels: pl.DataFrame) -> pl.DataFrame:
    return labels.with_columns(
        pl.col("band").replace_strict(_BAND_ORDINAL, return_dtype=pl.Int64).alias("band_ordinal")
    )
```

Change `build_feature_matrix` (was lines 59-69) to call it:

```python
def build_feature_matrix(labels: pl.DataFrame, full_history: pl.DataFrame, omni: pl.DataFrame) -> pl.DataFrame:
    """`labels` are the rows to build features FOR; `full_history` is the
    complete, unsampled label set for the same period (history features
    need other cells' activity, not just the rows being scored);
    `omni` is `propagation.data.spaceweather.fetch_omni2_range`'s output."""
    out = add_time_features(labels)
    out = add_geometry_features(out)
    out = add_solar_features(out)
    out = add_band_feature(out)
    out = add_spaceweather_features(out, omni)
    out = add_history_features(full_history, out)
    return out
```

(The `horizon_hours` parameter is added to this same function in Task 4 — don't add it yet here, to keep this task's diff focused on the band feature alone.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/features/test_matrix.py tests/test_leakage.py -v`
Expected: PASS — including the pre-existing `test_build_feature_matrix_produces_every_declared_column` (it iterates `FEATURE_COLUMNS`, so `band_ordinal` is covered automatically) and `test_no_label_or_definitive_kp_columns_in_feature_columns` (the forbidden set contains `"band"`, not `"band_ordinal"`, so no collision).

- [ ] **Step 7: Commit**

```bash
git add src/propagation/features/history.py src/propagation/features/matrix.py tests/features/test_matrix.py
git commit -m "feat(features): promote BAND_ORDER and add band_ordinal feature

Shares one GBTModel across all bands per ARCHITECTURE.md sec 5 (M3
band/horizon expansion sub-project)."
```

---

### Task 2: Thread `horizon_hours` through `add_spaceweather_features`

**Files:**
- Modify: `src/propagation/features/spaceweather.py`
- Test: `tests/features/test_spaceweather.py`

**Interfaces:**
- Consumes: none new.
- Produces: `add_spaceweather_features(labels: pl.DataFrame, omni: pl.DataFrame, horizon_hours: float = 0.0) -> pl.DataFrame` — as-of reference point becomes `window_start - horizon_hours`; the returned frame's `window_start` column is unchanged from the input.

- [ ] **Step 1: Write the failing test**

Add to `tests/features/test_spaceweather.py`:

```python
def test_horizon_hours_shifts_the_asof_reference_point():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    omni = _omni(start, 72, kp_fn=lambda i: float(i), f107_fn=lambda i: 100.0)
    labels = pl.DataFrame({
        "window_start": [start + timedelta(hours=48)],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out_h0 = add_spaceweather_features(labels, omni, horizon_hours=0.0)
    out_h6 = add_spaceweather_features(labels, omni, horizon_hours=6.0)
    assert out_h0["kp_now"][0] == pytest.approx(48.0)
    assert out_h6["kp_now"][0] == pytest.approx(42.0)
    assert out_h6["kp_lag3h"][0] == pytest.approx(39.0)
    # the target's real window_start must be unchanged on the output
    assert out_h0["window_start"][0] == out_h6["window_start"][0] == start + timedelta(hours=48)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/features/test_spaceweather.py -v -k horizon`
Expected: FAIL with `TypeError: add_spaceweather_features() got an unexpected keyword argument 'horizon_hours'`

- [ ] **Step 3: Implement the horizon shift**

In `src/propagation/features/spaceweather.py`, change the function signature and the `labels_sorted` line (was lines 15-21):

```python
def add_spaceweather_features(
    labels: pl.DataFrame, omni: pl.DataFrame, horizon_hours: float = 0.0
) -> pl.DataFrame:
    """`omni` is `propagation.data.spaceweather.fetch_omni2_range`'s output
    (hourly, columns time/kp/f107/bz_gsm/solar_wind_speed/dst). All features
    are as-of `window_start - horizon_hours` (the prediction time) via
    backward asof joins -- the most recent OMNI hour AT OR BEFORE prediction
    time, never a future one. `horizon_hours=0` (default) reproduces M2's
    original as-of-window_start behavior exactly. The returned frame's
    `window_start` column is restored to the caller's original (unshifted)
    values before returning, since downstream code joins feature output back
    onto the label matrix by the label's own real window_start."""
    omni = omni.sort("time")
    shift = pl.duration(hours=horizon_hours)
    labels_sorted = labels.sort("window_start").with_columns(
        (pl.col("window_start") - shift).alias("window_start")
    )
```

Leave the rest of the function body (the `_asof_lag` closure, `kp_now`/`kp_lag*`, `other_now`, `f107_daily`, `f107_smoothed` computations, and the `pl.concat(...)` call) exactly as-is — they all reference `labels_sorted.window_start`, which now already holds the shifted prediction time.

Change the final `return` (was line 88) to restore the real window_start before returning:

```python
    out = pl.concat(
        [labels_sorted, kp_now, kp_lag3h, kp_lag6h, kp_lag12h, kp_lag24h, kp_lag48h,
         other_now.rename({"bz_gsm": "bz_gsm_now", "solar_wind_speed": "solar_wind_speed_now",
                            "dst": "dst_now"}),
         f107_daily, f107_smoothed],
        how="horizontal_extend",
    )
    return out.with_columns((pl.col("window_start") + shift).alias("window_start"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/features/test_spaceweather.py -v`
Expected: PASS — including all three pre-existing tests (they call the function without `horizon_hours`, exercising the `0.0` default, so their assertions are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/spaceweather.py tests/features/test_spaceweather.py
git commit -m "feat(features): thread horizon_hours through add_spaceweather_features

As-of-now features re-anchor at prediction_time = window_start -
horizon_hours instead of window_start, per docs/superpowers/specs/
2026-07-24-m3-band-horizon-expansion-design.md sec 2."
```

---

### Task 3: Thread `horizon_hours` through `add_history_features`

**Files:**
- Modify: `src/propagation/features/history.py`
- Test: `tests/features/test_history.py`

**Interfaces:**
- Consumes: none new.
- Produces: `add_history_features(full_history: pl.DataFrame, target_rows: pl.DataFrame, horizon_hours: float = 0.0) -> pl.DataFrame` — AR rolling windows re-anchor at `prediction_time = window_start - horizon_hours`; the returned frame's `window_start` column is unchanged from `target_rows`.

- [ ] **Step 1: Write the failing test**

Add to `tests/features/test_history.py`:

```python
def test_horizon_hours_shifts_history_anchor_to_prediction_time():
    # Source row at 01:30. At horizon=0 (anchor=target's own window_start
    # 02:00), the buffer-adjusted trailing 1h window is [01:00,01:40] --
    # 01:30 counts. At horizon=1h (anchor=prediction_time=01:00), the same
    # window becomes [00:00,00:40] -- 01:30 is now in the FUTURE relative to
    # the anchor and must not count.
    history = _frame([_row(1, 30, "FN", "DM", "20m", 1, 10.0)])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    assert add_history_features(history, target)["same_cell_n_1h"][0] == 1
    out = add_history_features(history, target, horizon_hours=1.0)
    assert out["same_cell_n_1h"][0] == 0
    # window_start on the output must remain the target's real (unshifted) time
    assert out["window_start"][0] == datetime(2026, 6, 1, 2, 0, tzinfo=timezone.utc)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/features/test_history.py -v -k horizon`
Expected: FAIL with `TypeError: add_history_features() got an unexpected keyword argument 'horizon_hours'`

- [ ] **Step 3: Implement the horizon shift**

In `src/propagation/features/history.py`, change the `add_history_features` signature and add the shift at the top and bottom (function currently spans lines 233-326):

```python
def add_history_features(
    full_history: pl.DataFrame, target_rows: pl.DataFrame, horizon_hours: float = 0.0
) -> pl.DataFrame:
    """`horizon_hours=0` (default) reproduces M2's original as-of-
    window_start behavior exactly. For horizon_hours > 0, every AR rolling
    window re-anchors at prediction_time = window_start - horizon_hours by
    temporarily overwriting `target_rows.window_start` with prediction_time
    before running the (otherwise unchanged) relation logic below, then
    restoring the real window_start on the output before returning -- this
    keeps `history_narrow` (the real spot-activity timestamps being rolled
    over) untouched while only the target anchor's own timestamp shifts,
    matching docs/SPEC-labeling.md's "horizon is a training-time join
    offset" framing."""
    shift = pl.duration(hours=horizon_hours)
    target_rows = target_rows.with_columns((pl.col("window_start") - shift).alias("window_start"))

    if full_history.height == 0:
        full_history = pl.DataFrame(
            schema={"window_start": pl.Datetime("us", "UTC"), "tx_field": pl.Utf8, "rx_field": pl.Utf8,
                    "band": pl.Utf8, "n_spots": pl.Int64, "snr_ft8eq_p50": pl.Float64, "open": pl.Int64},
        )
```

(This replaces the original opening `if full_history.height == 0:` block with the shift prepended before it — the rest of the `if full_history.height == 0:` body is unchanged.)

Every other line in the function body already refers to `target_rows` by that name (not a new variable), so no further body changes are needed — the shift is transparent to `history_narrow`, the `same_cell`/`reverse_path`/`adjacent_band`/`adjacent_cell`/`band_wide`/`same_hour_yesterday` blocks, and `out = target_rows.join(...)`.

Change the final `return out` (was line 326) to restore the real window_start:

```python
    return out.with_columns((pl.col("window_start") + shift).alias("window_start"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/features/test_history.py -v`
Expected: PASS — all pre-existing tests call the function without `horizon_hours` (default `0.0`, `shift` is a zero duration, round-trip is a no-op).

- [ ] **Step 5: Commit**

```bash
git add src/propagation/features/history.py tests/features/test_history.py
git commit -m "feat(features): thread horizon_hours through add_history_features

AR spot-history windows re-anchor at prediction_time = window_start -
horizon_hours instead of window_start, per docs/superpowers/specs/
2026-07-24-m3-band-horizon-expansion-design.md sec 2."
```

---

### Task 4: Thread `horizon_hours` through `build_feature_matrix` and extend the leakage audit

**Files:**
- Modify: `src/propagation/features/matrix.py`
- Modify: `tests/test_leakage.py`
- Test: `tests/features/test_matrix.py`

**Interfaces:**
- Consumes: `add_spaceweather_features(..., horizon_hours)` (Task 2), `add_history_features(..., horizon_hours)` (Task 3).
- Produces: `build_feature_matrix(labels: pl.DataFrame, full_history: pl.DataFrame, omni: pl.DataFrame, horizon_hours: float = 0.0) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing test**

Add to `tests/features/test_matrix.py`:

```python
def test_build_feature_matrix_forwards_horizon_hours_to_asof_features():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(72)]
    omni = pl.DataFrame({
        "time": times, "kp": [float(i) for i in range(72)], "f107": [100.0] * 72,
        "bz_gsm": [1.0] * 72, "solar_wind_speed": [400.0] * 72, "dst": [-10.0] * 72,
    }, schema_overrides={"time": pl.Datetime("us", "UTC")})
    ts = start + timedelta(hours=48)
    labels = pl.DataFrame({
        "window_start": [ts], "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
        "open": [1], "n_spots": [3], "snr_ft8eq_p50": [10.0],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out_h0 = build_feature_matrix(labels, full_history=labels, omni=omni, horizon_hours=0.0)
    out_h6 = build_feature_matrix(labels, full_history=labels, omni=omni, horizon_hours=6.0)
    assert out_h0["kp_now"][0] == pytest.approx(48.0)
    assert out_h6["kp_now"][0] == pytest.approx(42.0)
    # target-time features (window_start, time-of-day) are horizon-invariant
    assert out_h0["window_start"][0] == out_h6["window_start"][0] == ts
    assert out_h0["hour_sin"][0] == out_h6["hour_sin"][0]
```

Add `from datetime import timedelta` to the file's existing `from datetime import datetime, timezone` import line if not already present.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/features/test_matrix.py -v -k forwards_horizon`
Expected: FAIL with `TypeError: build_feature_matrix() got an unexpected keyword argument 'horizon_hours'`

- [ ] **Step 3: Implement**

In `src/propagation/features/matrix.py`, change `build_feature_matrix` (written in Task 1) to:

```python
def build_feature_matrix(
    labels: pl.DataFrame, full_history: pl.DataFrame, omni: pl.DataFrame, horizon_hours: float = 0.0
) -> pl.DataFrame:
    """`labels` are the rows to build features FOR; `full_history` is the
    complete, unsampled label set for the same period (history features
    need other cells' activity, not just the rows being scored);
    `omni` is `propagation.data.spaceweather.fetch_omni2_range`'s output.
    `horizon_hours` (default 0) shifts only the as-of-now feature builders
    (space weather, AR history) to prediction_time = window_start -
    horizon_hours; time/geometry/solar features stay anchored at the target
    window_start since they're knowable in advance (docs/superpowers/specs/
    2026-07-24-m3-band-horizon-expansion-design.md sec 2)."""
    out = add_time_features(labels)
    out = add_geometry_features(out)
    out = add_solar_features(out)
    out = add_band_feature(out)
    out = add_spaceweather_features(out, omni, horizon_hours=horizon_hours)
    out = add_history_features(full_history, out, horizon_hours=horizon_hours)
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/features/test_matrix.py -v`
Expected: PASS

- [ ] **Step 5: Write the leakage-audit extension (failing first)**

Add to `tests/test_leakage.py`:

```python
def test_asof_features_never_see_data_after_prediction_time_for_nonzero_horizon():
    history = _frame([_row(20, 0, n=5, snr=20.0)])  # source at 20:00
    target = _frame([_row(22, 0, n=0, snr=None)])    # target at 22:00
    out_h0 = add_history_features(history, target, horizon_hours=0.0)
    out_h6 = add_history_features(history, target, horizon_hours=6.0)
    assert out_h0["same_cell_n_24h"][0] == 1  # legitimately visible: prediction_time=22:00
    assert out_h6["same_cell_n_24h"][0] == 0  # NOT visible: prediction_time=16:00, source is after it
```

This test will already PASS immediately after Task 3's implementation (no new production code needed) — it's a regression-lock, not a new failing-test-first cycle. Run it to confirm:

Run: `uv run pytest tests/test_leakage.py -v -k nonzero_horizon`
Expected: PASS

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests, no regressions)

- [ ] **Step 7: Commit**

```bash
git add src/propagation/features/matrix.py tests/features/test_matrix.py tests/test_leakage.py
git commit -m "feat(features): thread horizon_hours through build_feature_matrix

Completes the horizon mechanic (spec sec 2): as-of-now features anchor at
prediction_time, future-computable features stay at window_start. Extends
the leakage audit to cover the horizon-shifted case."
```

---

### Task 5: Batched multi-band WSPRnet extraction

**Files:**
- Modify: `src/propagation/data/wsprnet.py`
- Test: `tests/data/test_wsprnet.py`

**Interfaces:**
- Consumes: `parse_wsprnet_row`, `ExtractResult`, `SPOT_SCHEMA`, `dedup_spots`, `is_qualifying_spot` (all pre-existing in this file/module).
- Produces: `extract_wsprnet_bands(archive_path: Path, bands: list[str], chunk_size: int = 200_000) -> dict[str, ExtractResult]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/data/test_wsprnet.py` (extend the existing import to include `extract_wsprnet_bands`):

```python
from propagation.data.wsprnet import (
    WSPR_BAND_CODE_TO_BAND,
    extract_wsprnet,
    extract_wsprnet_bands,
    parse_wsprnet_row,
)


def test_extract_wsprnet_bands_matches_separate_single_band_calls(gz_fixture):
    bands = ["20m", "30m"]
    combined = extract_wsprnet_bands(gz_fixture, bands=bands)
    assert set(combined) == set(bands)
    for b in bands:
        single = extract_wsprnet(gz_fixture, band=b)
        assert combined[b].spots.equals(single.spots)
        assert combined[b].n_parsed == single.n_parsed
        assert combined[b].n_qualifying == single.n_qualifying
        assert combined[b].rejection_counts == single.rejection_counts
    # n_lines_read counts every line in the archive regardless of band (same
    # semantics as extract_wsprnet's own n_lines_read) -- shared across bands.
    assert combined["20m"].n_lines_read == combined["30m"].n_lines_read == 4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/data/test_wsprnet.py -v -k extract_wsprnet_bands`
Expected: FAIL with `ImportError: cannot import name 'extract_wsprnet_bands'`

- [ ] **Step 3: Implement**

In `src/propagation/data/wsprnet.py`, add this function after `extract_wsprnet` (after line 155):

```python
def extract_wsprnet_bands(
    archive_path: Path, bands: list[str], chunk_size: int = 200_000
) -> dict[str, ExtractResult]:
    """Single-pass variant of extract_wsprnet for multiple bands: a full
    month's archive covers every band, so extracting N bands via N separate
    extract_wsprnet calls re-decompresses and re-scans the same file N
    times. This scans the archive once, bucketing qualifying rows per
    requested band, using the same chunked-flush memory bound as
    extract_wsprnet (see that function's docstring for the OOM history)."""
    bands_set = set(bands)
    rejection_counts: dict[str, dict[str, int]] = {b: {} for b in bands}
    n_lines_read = 0
    n_parsed: dict[str, int] = {b: 0 for b in bands}

    with tempfile.TemporaryDirectory(prefix="wsprnet-extract-") as td_name:
        td = Path(td_name)
        rows: dict[str, list[dict]] = {b: [] for b in bands}
        chunk_paths: dict[str, list[Path]] = {b: [] for b in bands}

        def _flush(band: str) -> None:
            if rows[band]:
                chunk_path = td / f"{band}-chunk-{len(chunk_paths[band]):06d}.parquet"
                pl.DataFrame(
                    rows[band], schema_overrides={"ts": pl.Datetime("us", "UTC")}
                ).write_parquet(chunk_path)
                chunk_paths[band].append(chunk_path)
                rows[band].clear()

        with gzip.open(archive_path, "rt", encoding="ascii", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                n_lines_read += 1
                parsed = parse_wsprnet_row(line)
                if parsed is None or parsed["band"] not in bands_set:
                    continue
                band = parsed["band"]
                n_parsed[band] += 1
                parsed["ts"] = dt.datetime.fromtimestamp(parsed["ts"], tz=dt.timezone.utc)
                ok, reason = is_qualifying_spot(parsed)
                if not ok:
                    rejection_counts[band][reason] = rejection_counts[band].get(reason, 0) + 1
                    continue
                rows[band].append(parsed)
                if len(rows[band]) >= chunk_size:
                    _flush(band)
        for b in bands:
            _flush(b)

        results: dict[str, ExtractResult] = {}
        for b in bands:
            if not chunk_paths[b]:
                spots = pl.DataFrame(schema=SPOT_SCHEMA)
            else:
                spots = pl.concat([pl.read_parquet(p) for p in chunk_paths[b]], how="vertical_relaxed")
            for col in SPOT_SCHEMA:
                if col not in spots.columns:
                    spots = spots.with_columns(pl.lit(None).alias(col))
            spots = dedup_spots(spots)
            results[b] = ExtractResult(
                spots=spots, n_lines_read=n_lines_read, n_parsed=n_parsed[b],
                n_qualifying=spots.height, rejection_counts=rejection_counts[b],
            )
        return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/data/test_wsprnet.py -v`
Expected: PASS (all tests, including the new one and all pre-existing single-band tests)

- [ ] **Step 5: Commit**

```bash
git add src/propagation/data/wsprnet.py tests/data/test_wsprnet.py
git commit -m "feat(data): add extract_wsprnet_bands for single-pass multi-band extraction

Avoids re-scanning the same month's archive once per band when training a
model shared across all 11 HF bands (spec sec 3)."
```

---

### Task 6: QA check 3 (`grayline_40m`) — real implementation

**Files:**
- Modify: `src/propagation/qa/checks.py`
- Test: `tests/qa/test_checks.py`

**Interfaces:**
- Consumes: `midpoint_hours_since_terminator`, `midpoint_solar_zenith` columns (from `propagation.features.solar.add_solar_features`) when present on the input frame.
- Produces: `check_grayline_40m(labels: pl.DataFrame) -> QAResult` — same signature, real pass/fail logic when the solar columns are present, `insufficient_data` gate otherwise (unchanged from today).

- [ ] **Step 1: Write the failing tests**

Add to `tests/qa/test_checks.py`:

```python
def _grayline_row(hour, terminator_hrs, zenith, open_, month=6):
    return {
        "window_start": dt.datetime(2026, month, 1, hour, 0, tzinfo=dt.timezone.utc),
        "tx_field": "FN", "rx_field": "PM", "band": "40m", "open": open_,
        "midpoint_hours_since_terminator": terminator_hrs,
        "midpoint_solar_zenith": zenith,
    }


def test_check3_grayline_real_computation_pass():
    # FN-PM is ~10894km apart (>6Mm DX threshold). Gray-line rows (near
    # terminator) open more often than midday rows.
    rows = (
        [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=1) for _ in range(8)]
        + [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=0) for _ in range(2)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=1) for _ in range(2)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=0) for _ in range(8)]
    )
    result = check_grayline_40m(_df(rows))
    assert result.status == "pass"


def test_check3_grayline_real_computation_fail():
    rows = (
        [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=1) for _ in range(2)]
        + [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=0) for _ in range(8)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=1) for _ in range(8)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=0) for _ in range(2)]
    )
    result = check_grayline_40m(_df(rows))
    assert result.status == "fail"
```

(The existing `test_check3_gate_reports_insufficient_data_without_solar_features` in this file already locks in the gate behavior for rows lacking the solar columns — no change needed there.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/qa/test_checks.py -v -k check3_grayline_real`
Expected: FAIL — both new tests get `"insufficient_data"` back (current stub) instead of `"pass"`/`"fail"`.

- [ ] **Step 3: Implement**

In `src/propagation/qa/checks.py`, replace `check_grayline_40m` (lines 140-151) with:

```python
def check_grayline_40m(labels: pl.DataFrame) -> QAResult:
    """QA check 3 (docs/SPEC-labeling.md sec 6): 40m gray-line open-rate
    local max within +/-1h of the midpoint terminator vs midday, DX paths
    > 6 Mm. Requires features/solar.py's midpoint_hours_since_terminator
    and midpoint_solar_zenith already joined onto `labels`; without them
    this stays a gate (M0/M1 runs have no solar features to check against)."""
    if "midpoint_hours_since_terminator" not in labels.columns or "midpoint_solar_zenith" not in labels.columns:
        return QAResult(
            3, "grayline_40m", "insufficient_data",
            "terminator-relative timing requires features/solar.py joined onto labels",
        )
    subset = labels.filter(pl.col("band") == "40m")
    if subset.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "no 40m labels in this run")

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
        .alias("distance_km")
    ).filter(pl.col("distance_km") > 6000)
    if working.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "no >6Mm 40m paths")

    terminator = working.filter(pl.col("midpoint_hours_since_terminator").abs() <= 1.0)
    midday = working.filter(pl.col("midpoint_solar_zenith") <= 30.0)
    if terminator.height == 0 or midday.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "missing gray-line or midday windows")

    terminator_rate = float(terminator["open"].cast(pl.Float64).mean())
    midday_rate = float(midday["open"].cast(pl.Float64).mean())
    status = "pass" if terminator_rate > midday_rate else "fail"
    return QAResult(
        3, "grayline_40m", status,
        f"gray-line open-rate={terminator_rate:.2f} vs midday open-rate={midday_rate:.2f}",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/qa/test_checks.py -v`
Expected: PASS (all tests, including the untouched gate test and `test_run_qa_checks_returns_all_eight`)

- [ ] **Step 5: Commit**

```bash
git add src/propagation/qa/checks.py tests/qa/test_checks.py
git commit -m "feat(qa): implement check 3 (grayline_40m) for real

M2 shipped features/solar.py's terminator/zenith columns this check was
stubbed pending; wire them in (spec sec 4)."
```

---

### Task 7: QA check 6 (`solar_cycle`) — real implementation

**Files:**
- Modify: `src/propagation/qa/checks.py`
- Test: `tests/qa/test_checks.py`

**Interfaces:**
- Consumes: `f107_daily` column (from `propagation.features.spaceweather.add_spaceweather_features`) when present.
- Produces: `check_solar_cycle(labels: pl.DataFrame, min_months: int = 12) -> QAResult` — same signature, real correlation logic when `f107_daily` is present and enough months are covered, `insufficient_data` gate otherwise.

- [ ] **Step 1: Write the failing tests**

Add to `tests/qa/test_checks.py`:

```python
def _solar_cycle_row(month_idx, f107, open_):
    return {
        "window_start": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=30 * month_idx),
        "tx_field": "FN", "rx_field": "PM", "band": "10m", "open": open_,
        "f107_daily": f107,
    }


def test_check6_solar_cycle_real_computation_pass():
    # 12 months, f107 rising linearly; monthly open-rate rises in lockstep
    # (m/11 out of 11 rows/month) -- near-perfect positive correlation.
    rows = []
    for m in range(12):
        f107 = 70.0 + m * 10.0
        n_open = m
        rows += [_solar_cycle_row(m, f107, open_=1) for _ in range(n_open)]
        rows += [_solar_cycle_row(m, f107, open_=0) for _ in range(11 - n_open)]
    result = check_solar_cycle(_df(rows))
    assert result.status == "pass"


def test_check6_solar_cycle_real_computation_fail():
    # Same f107 ramp, but open-rate is ANTI-correlated with it.
    rows = []
    for m in range(12):
        f107 = 70.0 + m * 10.0
        n_open = 11 - m
        rows += [_solar_cycle_row(m, f107, open_=1) for _ in range(n_open)]
        rows += [_solar_cycle_row(m, f107, open_=0) for _ in range(11 - n_open)]
    result = check_solar_cycle(_df(rows))
    assert result.status == "fail"
```

(The existing `test_check6_gate_insufficient_data_single_month` continues to assert `"insufficient_data"` — its single-row fixture has no `f107_daily` column and is also below the DX-distance threshold, so it still gates.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/qa/test_checks.py -v -k check6_solar_cycle_real`
Expected: FAIL — both new tests get `"insufficient_data"` back (current stub).

- [ ] **Step 3: Implement**

In `src/propagation/qa/checks.py`, replace `check_solar_cycle` (lines 218-231) with:

```python
def check_solar_cycle(labels: pl.DataFrame, min_months: int = 12) -> QAResult:
    """QA check 6 (docs/SPEC-labeling.md sec 6): monthly 10m DX (>6 Mm)
    open-rate vs F10.7 correlation > 0.5 over multi-year history. Requires
    features/spaceweather.py's f107_daily already joined onto `labels`."""
    subset = labels.filter(pl.col("band") == "10m")
    if subset.height == 0:
        return QAResult(6, "solar_cycle", "insufficient_data", "no 10m labels in this run")

    pairs = subset.select(["tx_field", "rx_field"]).unique().to_dicts()
    dist_by_pair = {}
    for p in pairs:
        try:
            lat1, lon1 = grid_to_latlon(p["tx_field"])
            lat2, lon2 = grid_to_latlon(p["rx_field"])
            dist_by_pair[(p["tx_field"], p["rx_field"])] = great_circle_km(lat1, lon1, lat2, lon2)
        except ValueError:
            continue
    subset = subset.with_columns(
        pl.struct(["tx_field", "rx_field"])
        .map_elements(
            lambda r: dist_by_pair.get((r["tx_field"], r["rx_field"])), return_dtype=pl.Float64
        )
        .alias("distance_km")
    ).filter(pl.col("distance_km") > 6000)
    if subset.height == 0:
        return QAResult(6, "solar_cycle", "insufficient_data", "no DX (>6Mm) 10m paths")

    n_months = subset.select(pl.col("window_start").dt.truncate("1mo")).unique().height
    if n_months < min_months:
        return QAResult(
            6, "solar_cycle", "insufficient_data",
            f"only {n_months} distinct month(s); need >= {min_months}",
        )
    if "f107_daily" not in subset.columns:
        return QAResult(
            6, "solar_cycle", "insufficient_data",
            "F10.7 series requires features/spaceweather.py joined onto labels",
        )

    monthly = subset.with_columns(
        pl.col("window_start").dt.truncate("1mo").alias("month")
    ).group_by("month").agg(
        pl.col("open").cast(pl.Float64).mean().alias("open_rate"),
        pl.col("f107_daily").mean().alias("f107_mean"),
    )
    r = float(np.corrcoef(monthly["open_rate"].to_numpy(), monthly["f107_mean"].to_numpy())[0, 1])
    status = "pass" if r > 0.5 else "fail"
    return QAResult(6, "solar_cycle", status, f"monthly open-rate/F10.7 pearson r={r:.3f}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/qa/test_checks.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/propagation/qa/checks.py tests/qa/test_checks.py
git commit -m "feat(qa): implement check 6 (solar_cycle) for real

M2 shipped features/spaceweather.py's f107_daily this check was stubbed
pending; wire it in (spec sec 4)."
```

---

### Task 8: QA check 7 (`storm_response`) — real implementation

**Files:**
- Modify: `src/propagation/qa/checks.py`
- Test: `tests/qa/test_checks.py`

**Interfaces:**
- Consumes: `kp_now` column (from `add_spaceweather_features`) and `midpoint_geomag_lat` column (from `propagation.features.geometry.add_geometry_features`) when both present.
- Produces: `check_storm_response(labels: pl.DataFrame, kp_max: float | None) -> QAResult` — same signature, real matched-baseline logic when both columns are present and `kp_max >= 5.0`, `insufficient_data` gate otherwise.

- [ ] **Step 1: Write the failing tests**

Add to `tests/qa/test_checks.py`:

```python
def _storm_row(hour, month, kp, open_, geomag_lat=75.0):
    return {
        "window_start": dt.datetime(2026, month, 1, hour, 0, tzinfo=dt.timezone.utc),
        "tx_field": "FN", "rx_field": "DM", "band": "20m", "open": open_,
        "kp_now": kp, "midpoint_geomag_lat": geomag_lat,
    }


def test_check7_storm_response_real_computation_pass():
    # Same (band=20m, hour=12, month=6) bucket in both regimes: storm
    # open-rate 1/10, quiet open-rate 8/10 -> ratio 0.125 <= 0.5.
    rows = (
        [_storm_row(12, 6, kp=7.0, open_=0) for _ in range(9)]
        + [_storm_row(12, 6, kp=7.0, open_=1)]
        + [_storm_row(12, 6, kp=1.0, open_=1) for _ in range(8)]
        + [_storm_row(12, 6, kp=1.0, open_=0) for _ in range(2)]
    )
    result = check_storm_response(_df(rows), kp_max=7.0)
    assert result.status == "pass"


def test_check7_storm_response_real_computation_fail():
    # storm open-rate 9/10, quiet open-rate 8/10 -> ratio 1.125 > 0.5.
    rows = (
        [_storm_row(12, 6, kp=7.0, open_=1) for _ in range(9)]
        + [_storm_row(12, 6, kp=7.0, open_=0)]
        + [_storm_row(12, 6, kp=1.0, open_=1) for _ in range(8)]
        + [_storm_row(12, 6, kp=1.0, open_=0) for _ in range(2)]
    )
    result = check_storm_response(_df(rows), kp_max=7.0)
    assert result.status == "fail"


def test_check7_insufficient_data_without_spaceweather_columns():
    # kp_max clears the >=5 gate, but the frame still lacks kp_now/
    # midpoint_geomag_lat -- must still gate, not crash.
    result = check_storm_response(_df([_row(14, "FN", "DM", "20m", 1)]), kp_max=6.0)
    assert result.status == "insufficient_data"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/qa/test_checks.py -v -k check7_storm_response_real`
Expected: FAIL — both new pass/fail tests get `"insufficient_data"` back (current stub); `test_check7_insufficient_data_without_spaceweather_columns` already passes trivially (not a meaningful failing-first case, but run it alongside to confirm it still passes after Step 3).

- [ ] **Step 3: Implement**

In `src/propagation/qa/checks.py`, replace `check_storm_response` (lines 234-243) with:

```python
def check_storm_response(labels: pl.DataFrame, kp_max: float | None) -> QAResult:
    """QA check 7 (docs/SPEC-labeling.md sec 6): Kp>=6 trans-polar paths
    (|midpoint geomag lat| > 60 deg) open-rate <= 50% of Kp<=2 matched
    (band, hour, month) baseline. Requires features/spaceweather.py's
    kp_now and features/geometry.py's midpoint_geomag_lat already joined
    onto `labels`."""
    if kp_max is None or kp_max < 5.0:
        return QAResult(
            7, "storm_response", "insufficient_data",
            f"no Kp>=5 fold in this run (max Kp available={kp_max})",
        )
    if "kp_now" not in labels.columns or "midpoint_geomag_lat" not in labels.columns:
        return QAResult(
            7, "storm_response", "insufficient_data",
            "space-weather/geometry features not joined onto labels",
        )

    trans_polar = labels.filter(pl.col("midpoint_geomag_lat").abs() > 60.0)
    if trans_polar.height == 0:
        return QAResult(7, "storm_response", "insufficient_data", "no trans-polar (|geomag lat|>60) paths")

    bucketed = trans_polar.with_columns(
        pl.col("window_start").dt.hour().alias("hour"),
        pl.col("window_start").dt.month().alias("month"),
    )
    storm = bucketed.filter(pl.col("kp_now") >= 6.0)
    quiet = bucketed.filter(pl.col("kp_now") <= 2.0)
    if storm.height == 0 or quiet.height == 0:
        return QAResult(
            7, "storm_response", "insufficient_data", "no storm or quiet Kp rows among trans-polar paths"
        )

    storm_rates = storm.group_by(["band", "hour", "month"]).agg(
        pl.col("open").cast(pl.Float64).mean().alias("storm_rate")
    )
    quiet_rates = quiet.group_by(["band", "hour", "month"]).agg(
        pl.col("open").cast(pl.Float64).mean().alias("quiet_rate")
    )
    matched = storm_rates.join(quiet_rates, on=["band", "hour", "month"], how="inner").filter(
        pl.col("quiet_rate") > 0
    )
    if matched.height == 0:
        return QAResult(
            7, "storm_response", "insufficient_data",
            "no matched (band,hour,month) buckets with both regimes",
        )

    ratio = float((matched["storm_rate"] / matched["quiet_rate"]).mean())
    status = "pass" if ratio <= 0.5 else "fail"
    return QAResult(7, "storm_response", status, f"mean matched-bucket storm/quiet open-rate ratio={ratio:.3f}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/qa/test_checks.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full QA + leakage + feature suite once more**

Run: `uv run pytest tests/qa tests/features tests/test_leakage.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/propagation/qa/checks.py tests/qa/test_checks.py
git commit -m "feat(qa): implement check 7 (storm_response) for real

M2 shipped kp_now and M1 shipped midpoint_geomag_lat, both of which this
check was stubbed pending; wire them in (spec sec 4)."
```

---

### Task 9: `scripts/eval_m3.py` — band-group × horizon headline driver

**Files:**
- Create: `scripts/eval_m3.py`
- Test: `tests/scripts/test_eval_m3.py`

**Interfaces:**
- Consumes: `extract_wsprnet_bands` (Task 5), `build_feature_matrix(..., horizon_hours=...)` (Task 4), `BAND_ORDER` (Task 1), `blocked_cv_gap_hours`, `write_headline_report`, `ClimatologyModel`, `P533Model`/`ssn_by_month`, `GBTModel` (all pre-existing).
- Produces: `BAND_GROUPS: dict[str, set[str]]`, `_band_group(band: str) -> str`, `enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours, max_ar_lookback_hours=24.0) -> None`, `write_band_group_reports(models, labels, horizon_hours, out_dir) -> dict`, `main()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_eval_m3.py`:

```python
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_m3 import (  # noqa: E402
    BAND_GROUPS,
    _band_group,
    enforce_blocked_cv_gap,
    write_band_group_reports,
)


def test_band_group_maps_every_band_to_exactly_one_group():
    from propagation.features.history import BAND_ORDER
    seen = set()
    for band in BAND_ORDER:
        group = _band_group(band)
        assert group in BAND_GROUPS
        assert band in BAND_GROUPS[group]
        seen.add(band)
    assert seen == set(BAND_ORDER)


def test_band_group_rejects_unknown_band():
    with pytest.raises(ValueError, match="not in any BAND_GROUPS"):
        _band_group("999m")


def _labels_at(*hours_from_epoch):
    epoch = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    ts = [epoch + timedelta(hours=h) for h in hours_from_epoch]
    return pl.DataFrame({"window_start": ts}, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


def test_enforce_blocked_cv_gap_scales_with_max_horizon():
    # max_horizon_hours=24 -> required gap = max(48, 24+24) = 48 (floor);
    # max_horizon_hours=30 -> required gap = max(48, 30+24) = 54, so an
    # exactly-48h gap that passed at horizon<=24 must now fail.
    train_labels = _labels_at(0)
    eval_labels = _labels_at(48)
    enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours=24.0)  # must not raise
    with pytest.raises(ValueError, match="blocked-CV gap violation"):
        enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours=30.0)


class ConstantModel:
    def __init__(self, p):
        self._p = p

    def predict(self, labels):
        return labels.with_columns(pl.lit(self._p).cast(pl.Float64).alias("p_open"))


def test_write_band_group_reports_writes_one_table_per_group(tmp_path):
    ts = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
    labels = pl.DataFrame({
        "window_start": [ts, ts, ts],
        "tx_field": ["EM", "EM", "EM"], "rx_field": ["PM", "PM", "PM"],
        "band": ["20m", "160m", "6m"], "open": [1, 0, 1],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    models = {"climatology": ConstantModel(0.7), "p533": ConstantModel(0.6), "gbt": ConstantModel(0.5)}
    results = write_band_group_reports(models, labels, horizon_hours=0.0, out_dir=tmp_path)
    assert set(results) == {"low", "mid", "high"}
    for group in ("low", "mid", "high"):
        table = (tmp_path / group / "h0" / "headline_table.csv").read_text()
        assert table.count("\n") == 4  # header + 3 model rows
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_eval_m3.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_m3'`

- [ ] **Step 3: Implement**

Create `scripts/eval_m3.py`:

```python
"""M3 band/horizon expansion driver: trains one GBTModel per requested
horizon, shared across every requested band (band_ordinal is a feature),
and writes climatology/P.533/GBT headline reports sliced by band group
(low/mid/high HF) per ARCHITECTURE.md sec 6 and docs/superpowers/specs/
2026-07-24-m3-band-horizon-expansion-design.md.

Usage:
    uv run python scripts/eval_m3.py --bands 20m 17m 15m 40m --horizons 0 6 \
        --train-months 2024-01 2024-02 2024-03 --eval-months 2024-05 \
        --data-dir data

Storm/quiet slicing and the full 11-band x 6-horizon historical sweep are
out of scope for this script (spec sec 7) -- this produces one headline
table per (band group, horizon) for whatever bands/horizons are requested.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from propagation.data.spaceweather import fetch_omni2_range
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet_bands
from propagation.eval.report import write_headline_report
from propagation.eval.splits import blocked_cv_gap_hours
from propagation.features.labels import build_labels
from propagation.features.matrix import build_feature_matrix
from propagation.features.universe import build_universe
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.models.gbt import GBTModel
from propagation.models.p533 import P533Model, ssn_by_month

BAND_GROUPS: dict[str, set[str]] = {
    "low": {"160m", "80m", "60m", "40m"},
    "mid": {"30m", "20m", "17m", "15m"},
    "high": {"12m", "10m", "6m"},
}
_MAX_AR_LOOKBACK_HOURS = 24.0


def _band_group(band: str) -> str:
    for group, bands in BAND_GROUPS.items():
        if band in bands:
            return group
    raise ValueError(f"band {band!r} not in any BAND_GROUPS entry")


def _build_labels_for_month_all_bands(archive: Path, bands: list[str]) -> pl.DataFrame:
    extracts = extract_wsprnet_bands(archive, bands=bands)
    per_band = []
    for extract in extracts.values():
        uptime = build_receiver_uptime(extract.spots)
        universe = build_universe(extract.spots, uptime)
        per_band.append(build_labels(extract.spots, universe))
    return pl.concat(per_band, how="vertical_relaxed")


def _build_labels_for_months(archives: dict[str, Path], bands: list[str]) -> pl.DataFrame:
    return pl.concat(
        [_build_labels_for_month_all_bands(a, bands) for a in archives.values()], how="vertical_relaxed"
    )


def enforce_blocked_cv_gap(
    train_labels: pl.DataFrame,
    eval_labels: pl.DataFrame,
    max_horizon_hours: float,
    max_ar_lookback_hours: float = _MAX_AR_LOOKBACK_HOURS,
) -> None:
    """Same rule as scripts/eval_m2.py's function of the same name
    (docs/SPEC-labeling.md sec 6 rule 1), generalized: M3 trains one model
    per horizon in --horizons, so the gap must be computed against the
    LARGEST horizon requested, not a fixed 3h."""
    train_end = train_labels["window_start"].max()
    eval_start = eval_labels["window_start"].min()
    required_gap_hours = blocked_cv_gap_hours(max_horizon_hours, max_ar_lookback_hours)
    actual_gap_hours = (eval_start - train_end).total_seconds() / 3600.0
    if actual_gap_hours < required_gap_hours:
        raise ValueError(
            f"blocked-CV gap violation: eval window starts only {actual_gap_hours:.2f}h "
            f"after train window ends, but docs/SPEC-labeling.md sec 6 rule 1 requires "
            f">= {required_gap_hours:.2f}h (blocked_cv_gap_hours(max_horizon_hours="
            f"{max_horizon_hours}, max_ar_lookback_hours={max_ar_lookback_hours})) to "
            f"avoid train/eval leakage. Choose train/eval months with a sufficient gap."
        )


def write_band_group_reports(
    models: dict[str, object], labels: pl.DataFrame, horizon_hours: float, out_dir: Path,
) -> dict[str, dict[str, dict]]:
    """Same pattern as scripts/eval_m2.py::write_three_model_slice_reports,
    sliced by BAND_GROUPS instead of storm/quiet. `labels` must carry `open`
    and `band`. Writes <out_dir>/<group>/h<N>/headline_table.csv."""
    out_dir = Path(out_dir)
    labeled = labels.with_columns(
        pl.col("band").map_elements(_band_group, return_dtype=pl.Utf8).alias("band_group")
    )
    results: dict[str, dict[str, dict]] = {}
    for group in BAND_GROUPS:
        sl = labeled.filter(pl.col("band_group") == group)
        results[group] = {}
        if sl.height == 0:
            continue
        group_dir = out_dir / group / f"h{int(horizon_hours)}"
        for model_name, model in models.items():
            pred = model.predict(sl).drop_nulls("p_open")
            if pred.height == 0:
                print(f"{model_name} abstained on all {sl.height} rows in group {group!r} h={horizon_hours} — skipping")
                continue
            results[group][model_name] = write_headline_report(
                y_true=pred["open"].cast(pl.Float64).to_numpy(),
                y_prob=pred["p_open"].to_numpy(),
                model_name=model_name,
                out_dir=group_dir,
            )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 band/horizon expansion headline eval")
    ap.add_argument("--bands", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", type=float, required=True, help="hours, e.g. 0 1 3 6 12 24")
    ap.add_argument("--train-months", nargs="+", required=True, help="YYYY-MM")
    ap.add_argument("--eval-months", nargs="+", required=True, help="YYYY-MM, held-out")
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

    train_labels = _build_labels_for_months(train_archives, args.bands)
    eval_labels = _build_labels_for_months(eval_archives, args.bands)

    max_horizon = max(args.horizons)
    enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours=max_horizon)

    cache_dir = args.data_dir / "cache"
    all_years = sorted({int(ym.split("-")[0]) for ym in list(args.train_months) + list(args.eval_months)})
    omni = fetch_omni2_range(all_years[0], all_years[-1], cache_dir=cache_dir)
    eval_month_keys = list(args.eval_months)

    out_dir = args.data_dir / "reports" / "m3"
    for horizon_hours in args.horizons:
        train_matrix = build_feature_matrix(
            train_labels, full_history=train_labels, omni=omni, horizon_hours=horizon_hours
        ).with_columns(pl.lit(1.0).alias("sample_weight"))
        eval_matrix = build_feature_matrix(
            eval_labels, full_history=eval_labels, omni=omni, horizon_hours=horizon_hours
        )

        models = {
            "climatology": ClimatologyModel().fit(train_labels),
            "p533": P533Model(ssn_by_month=ssn_by_month(eval_month_keys, cache_dir)),
            "gbt": GBTModel().fit(train_matrix),
        }
        results = write_band_group_reports(models, eval_matrix, horizon_hours, out_dir)
        print(f"h={horizon_hours}h: {results}")

    print(f"wrote {out_dir}/{{low,mid,high}}/h<N>/headline_table.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_eval_m3.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests, no regressions across the whole repo)

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_m3.py tests/scripts/test_eval_m3.py
git commit -m "feat(scripts): add eval_m3.py band-group x horizon headline driver

Generalizes eval_m2.py to an arbitrary band list x horizon list, per
docs/superpowers/specs/2026-07-24-m3-band-horizon-expansion-design.md."
```

---

### Task 10: Smoke run against real WSPRnet data

**Files:** none (verification-only task; no code changes)

**Interfaces:** none new — exercises Tasks 1-9's combined output end-to-end.

- [ ] **Step 1: Run the smoke command**

This deliberately stays small (2 bands including one M2 never trained on, 2 horizons including one non-zero, 1 train + 1 eval month) to bound P.533's known per-row subprocess cost (docs/DECISIONS/0005-m2-acceptance-result.md: "a single band+3-month acceptance run takes several hours ... dominated by P.533's per-row ITURHFProp subprocess calls") while still proving every new code path fires against real data. The full 11-band x 6-horizon sweep is out of scope (spec sec 7) — do not widen this command's scope beyond what's below without checking in first, since band count multiplies both extraction and P.533 cost roughly linearly.

Run:
```bash
uv run python scripts/eval_m3.py \
  --bands 20m 40m --horizons 0 6 \
  --train-months 2024-02 --eval-months 2024-05 \
  --data-dir data
```

Expected: exits 0; prints `h=0.0h: {...}` and `h=6.0h: {...}` result dicts (each containing `low`/`mid` band-group keys with `climatology`/`p533`/`gbt` brier/log_loss numbers — `high` will be empty/absent since neither 20m nor 40m is in the `high` group, matching `write_band_group_reports`' "skip empty groups" behavior); prints the final `wrote data/reports/m3/...` line.

If this exceeds roughly 30 minutes of wall-clock, it's acceptable to stop, rerun with `--bands 40m --horizons 6` alone (single band, single non-zero horizon — still proves the multi-band-capable code path end-to-end since `extract_wsprnet_bands`/`build_feature_matrix(horizon_hours=...)` run either way) and note the reduced scope when reporting results — do not silently widen or narrow further without flagging it.

- [ ] **Step 2: Verify the QA checks fired for real (not permanently stubbed)**

The smoke run doesn't call `run_qa_checks` directly (that's `scripts/run_m0.py`'s job, unchanged by this plan). Confirm Tasks 6-8 work against this real data with a short ad hoc check:

```bash
uv run python3 -c "
from pathlib import Path
from propagation.data.wsprnet import extract_wsprnet_bands
from propagation.features.uptime import build_receiver_uptime
from propagation.features.universe import build_universe
from propagation.features.labels import build_labels
from propagation.features.geometry import add_geometry_features
from propagation.features.solar import add_solar_features
from propagation.data.spaceweather import fetch_omni2_range
from propagation.features.spaceweather import add_spaceweather_features
from propagation.qa.checks import run_qa_checks

archive = Path('data/raw/wsprspots-2024-05.csv.gz')
extracts = extract_wsprnet_bands(archive, bands=['40m'])
extract = extracts['40m']
uptime = build_receiver_uptime(extract.spots)
universe = build_universe(extract.spots, uptime)
labels = build_labels(extract.spots, universe)
labels = add_geometry_features(labels)
labels = add_solar_features(labels)
omni = fetch_omni2_range(2024, 2024, cache_dir=Path('data/cache'))
labels = add_spaceweather_features(labels, omni)
results = run_qa_checks(labels, rejection_counts=extract.rejection_counts, n_qualifying=extract.n_qualifying, kp_max=8.0)
for r in results:
    print(r.check_id, r.name, r.status, r.detail)
"
```

Expected: check 3 (`grayline_40m`) reports `pass`, `fail`, or a *data-driven* `insufficient_data` reason (e.g. "no >6Mm 40m paths" if this month/band genuinely has none) — NOT the old static `"terminator-relative timing requires features/solar.py (M2); not computable yet"` message, since the solar columns are now present. Check 7 (`storm_response`) similarly should not report the old static `"Kp series not yet joined"` message.

- [ ] **Step 3: Report results**

Summarize in the session (not a new doc): exit status, wall-clock time, the printed per-horizon result dicts, and the check 3/7 statuses from Step 2. If anything failed, treat it as a bug to fix (return to the relevant earlier task) rather than silently downgrading this smoke run's scope.

No commit for this task — it's verification, not a code change. If Step 2 surfaces a real bug, fix it under the task whose code owns the bug, with its own test-first cycle and commit.
