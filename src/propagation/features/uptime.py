import numpy as np
import polars as pl

from propagation.data.hygiene import mode_class_for, normalize_grid

_WINDOW_MIN = 15
_PAD_BEFORE_MIN = 30
_PAD_AFTER_MIN = 45  # window length (15) + 30, per SPEC sec 3

_GROUP_COLS = ["window_start", "de_call", "band", "mode_class"]

_EMPTY_SCHEMA = {
    "window_start": pl.Datetime("us", "UTC"),
    "de_call": pl.Utf8,
    "de_field": pl.Utf8,
    "de_grid4": pl.Utf8,
    "band": pl.Utf8,
    "mode_class": pl.Utf8,
    "n_evidence_reports": pl.Int32,
    "first_evidence_ts": pl.Datetime("us", "UTC"),
    "last_evidence_ts": pl.Datetime("us", "UTC"),
}


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


def _modal_location(exploded: pl.DataFrame) -> pl.DataFrame:
    """Per (window_start, de_call, band, mode_class) group, the receiver location
    is the modal de_grid4 among grid4-precision (4-char) reports if any exist in
    the group, else the modal field among field-precision (2-char) reports.
    Ties are broken lexicographically (docs/SPEC-labeling.md sec 3). Groups with
    no usable location (no non-null `_grid_norm` at all) get a null `_modal_grid`
    and are dropped by the caller ("receivers with no usable location contribute
    nothing").

    polars' `Series.mode()` does not guarantee a deterministic (let alone
    lexicographic) order among tied values, so the tie-break is implemented
    explicitly here via an exact-count aggregation + stable sort.
    """
    valid = exploded.filter(pl.col("_grid_norm").is_not_null())
    if valid.height == 0:
        return exploded.select(_GROUP_COLS).unique().with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("_modal_grid")
        )

    # Prefer grid4-precision reports over field-precision ones: within each
    # group, restrict to whichever precision tier is present at its finest
    # granularity (grid4 if any, else field).
    valid = valid.with_columns(
        pl.when(pl.col("_grid_norm").str.len_chars() == 4).then(0).otherwise(1).alias("_prio")
    )
    valid = valid.with_columns(
        pl.col("_prio").min().over(_GROUP_COLS).alias("_min_prio")
    ).filter(pl.col("_prio") == pl.col("_min_prio"))

    counts = valid.group_by([*_GROUP_COLS, "_grid_norm"]).agg(pl.len().alias("_cnt"))
    return (
        counts.sort(["_cnt", "_grid_norm"], descending=[True, False])
        .group_by(_GROUP_COLS, maintain_order=True)
        .agg(pl.col("_grid_norm").first().alias("_modal_grid"))
    )


def build_receiver_uptime(spots: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 3. spots must be hygiene-qualified, deduped."""
    working = spots.with_columns(
        pl.col("mode").map_elements(mode_class_for, return_dtype=pl.Utf8).alias("mode_class"),
        pl.col("de_grid").map_elements(normalize_grid, return_dtype=pl.Utf8).alias("_grid_norm"),
    ).filter(pl.col("mode_class") != "other")

    if working.height == 0:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    ts_minutes = (working["ts"].cast(pl.Int64) // 60_000_000).to_numpy()
    row_idx, window_start_min = _evidence_window_starts_minutes(ts_minutes)

    # row_idx holds ~5x the input row count (each spot pads into multiple
    # overlapping 15-min windows) -- exploding the FULL `working` frame
    # (every spots column, most unused below) at that multiplier is what
    # actually drove a 2026-07-20 OOM incident: measured ~22GB for one
    # month's worth of real 2024 WSPRnet volume from this gather alone.
    # Narrowing to only the columns this function and _modal_location
    # actually read before exploding cuts what gets multiplied 5x.
    # (Converting row_idx to a Python list of boxed ints before indexing,
    # the original `.tolist()`, was a smaller contributor to the same
    # incident; polars accepts a numpy int array directly.)
    narrow = working.select("ts", "de_call", "band", "mode_class", "_grid_norm")
    exploded = narrow[row_idx].with_columns(
        pl.Series("window_start_min", window_start_min)
    )
    exploded = exploded.with_columns(
        (pl.col("window_start_min") * 60_000_000)
        .cast(pl.Datetime("us", "UTC"))
        .alias("window_start")
    )

    grouped = exploded.group_by(_GROUP_COLS).agg(
        pl.len().cast(pl.Int32).alias("n_evidence_reports"),
        pl.col("ts").min().alias("first_evidence_ts"),
        pl.col("ts").max().alias("last_evidence_ts"),
    )

    modal = _modal_location(exploded)
    grouped = grouped.join(modal, on=_GROUP_COLS, how="left")

    return (
        grouped.filter(pl.col("_modal_grid").is_not_null())
        .with_columns(
            pl.col("_modal_grid").str.slice(0, 2).alias("de_field"),
            pl.when(pl.col("_modal_grid").str.len_chars() == 4)
            .then(pl.col("_modal_grid"))
            .otherwise(None)
            .alias("de_grid4"),
        )
        .drop("_modal_grid")
        .select(
            "window_start", "de_call", "de_field", "de_grid4", "band",
            "mode_class", "n_evidence_reports", "first_evidence_ts", "last_evidence_ts",
        )
    )
