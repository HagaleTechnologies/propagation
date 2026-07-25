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
the open buffer zone (T-buffer, T), leaving exactly [T-L, T-buffer] --
closed at T-buffer, per the <= above) rather than a single shifted-anchor
rolling sum, since polars' rolling_sum_by is self-referential on one time
column and can't offset the window's own anchor point independently of its
span.

Note: polars' rolling_sum_by only evaluates a window AT rows that already
exist in the frame being rolled over. A naive implementation that computes
the rolling sums over `full_history` alone and then joins the result onto
`target_rows` by exact window_start match is wrong in general, since a
target row's window_start essentially never coincides with one of the
source rows' own window_start values. Instead, each target row's
(key_cols, window_start) combination is concatenated into the history as a
zero-weight "anchor" row, so the rolling window is evaluated exactly at the
target's own timestamp using only the real history rows' contributions
(the anchor itself never contributes, since it carries n_spots=0 and is
excluded from its own window by closed="left" anyway); the anchor rows are
then filtered back out at the end.

Also note: AVAIL_BUFFER_MIN (20m) is longer than the "15m" lookback, so for
that lookback the "full minus buffer" subtraction can go negative (there is
a real sliver of data -- rows landing in [T-20m, T-15m) -- included in the
buffer-zone sum but not in the (shorter) full-window sum). Structurally
nothing can ever be both within 15 minutes of T and past the 20-minute
availability buffer, so the correct value for that lookback is always 0;
the count is clamped at a lower bound of 0 to enforce this rather than
leaking a negative count into the feature matrix.
"""
from __future__ import annotations

import polars as pl

AVAIL_BUFFER_MIN = 20  # 15-min window duration + Δ_avail=5min

_LOOKBACKS = {"15m": "15m", "1h": "1h", "3h": "3h", "24h": "24h"}
BAND_ORDER = ["160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m", "12m", "10m", "6m"]


def field_neighbors(field: str) -> list[str]:
    """The up-to-8 Maidenhead-field neighbors of a 2-char field. Longitude
    wraps at the +/-180 seam (A<->R); latitude does not wrap (poles).

    Symmetric: B in field_neighbors(A) iff A in field_neighbors(B). Each of
    the 8 (dlon, dlat) offsets has an exact inverse (-dlon, -dlat) also in
    that set, and validity only depends on the *destination* latitude being
    in [0, 17] -- so if A's shift by (dlon, dlat) lands validly on B, B's
    shift by (-dlon, -dlat) lands validly back on A. `add_history_features`
    relies on this: adjacent_cell looks up each target's own neighbors
    directly, equivalent to (and cheaper than) the original "which history
    rows have this target as one of their neighbors" formulation.
    """
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
    i = BAND_ORDER.index(band)
    out = []
    if i > 0:
        out.append(BAND_ORDER[i - 1])
    if i < len(BAND_ORDER) - 1:
        out.append(BAND_ORDER[i + 1])
    return out


def _rolling_raw(
    history: pl.DataFrame, anchor_source: pl.DataFrame, key_cols: list[str], prefix: str
) -> pl.DataFrame:
    """Core of `_rolling_n_and_snr`, stopping short of the final SNR divide:
    returns {prefix}_n_{suffix} (a count, safe to sum across relations) and
    RAW, un-divided {prefix}_num_{suffix}/{prefix}_den_{suffix} (weighted-SNR
    numerator/denominator). Splitting this out lets adjacent_band/
    adjacent_cell compute several relations' contributions separately (see
    module docstring below) and combine them correctly by summing numerator
    and denominator before a single final division -- summing already-
    divided per-relation averages would be wrong.

    `anchor_source` need not be pre-deduplicated: only its distinct
    (window_start, key_cols) combinations matter, exactly as `targets` did
    in the original single-relation `_rolling_n_and_snr`.
    """
    h = (
        history.with_columns(
            pl.when(pl.col("snr_ft8eq_p50").is_not_null())
            .then(pl.col("n_spots"))
            .otherwise(0)
            .alias("_snr_weight"),
            (pl.col("n_spots") * pl.col("snr_ft8eq_p50").fill_null(0.0)).alias("_snr_weighted"),
        )
        .select(["window_start", *key_cols, "n_spots", "_snr_weight", "_snr_weighted"])
        .with_columns(pl.lit(False).alias("_is_anchor"))
    )

    anchors = anchor_source.select(["window_start", *key_cols]).unique().with_columns(
        pl.lit(0, dtype=pl.Int64).alias("n_spots"),
        pl.lit(0, dtype=pl.Int64).alias("_snr_weight"),
        pl.lit(0.0).alias("_snr_weighted"),
        pl.lit(True).alias("_is_anchor"),
    )

    combined = pl.concat([h, anchors], how="vertical_relaxed").sort([*key_cols, "window_start"])

    buffer_str = f"{AVAIL_BUFFER_MIN}m"
    exprs = []
    out_names = []
    for suffix, window in _LOOKBACKS.items():
        n_full = pl.col("n_spots").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        n_buf = pl.col("n_spots").rolling_sum_by("window_start", window_size=buffer_str, closed="none").over(key_cols)
        w_full = pl.col("_snr_weighted").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        w_buf = pl.col("_snr_weighted").rolling_sum_by("window_start", window_size=buffer_str, closed="none").over(key_cols)
        d_full = pl.col("_snr_weight").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        d_buf = pl.col("_snr_weight").rolling_sum_by("window_start", window_size=buffer_str, closed="none").over(key_cols)

        n_name = f"{prefix}_n_{suffix}"
        num_name = f"{prefix}_num_{suffix}"
        den_name = f"{prefix}_den_{suffix}"
        # Clamped at 0: for lookbacks shorter than AVAIL_BUFFER_MIN (the
        # "15m" suffix), full - buffer can go negative even though the
        # structurally-correct answer is always 0 (see module docstring).
        n_expr = (n_full - n_buf).clip(lower_bound=0).alias(n_name)
        num_expr = (w_full - w_buf).alias(num_name)
        den_expr = (d_full - d_buf).alias(den_name)
        exprs += [n_expr, num_expr, den_expr]
        out_names += [n_name, num_name, den_name]

    return (
        combined.with_columns(exprs)
        .filter(pl.col("_is_anchor"))
        .select(["window_start", *key_cols, *out_names])
    )


def _finalize_snr(raw: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Divides `_rolling_raw`'s raw numerator/denominator into the final
    {prefix}_snr_{suffix} columns and drops the raw ones, matching the shape
    `_rolling_n_and_snr` returns."""
    exprs = []
    keep = [c for c in raw.columns if not (c.startswith(f"{prefix}_num_") or c.startswith(f"{prefix}_den_"))]
    for suffix in _LOOKBACKS:
        num = pl.col(f"{prefix}_num_{suffix}")
        den = pl.col(f"{prefix}_den_{suffix}")
        exprs.append(pl.when(den > 0).then(num / den).otherwise(None).alias(f"{prefix}_snr_{suffix}"))
    return raw.with_columns(exprs).select([*keep, *(f"{prefix}_snr_{suffix}" for suffix in _LOOKBACKS)])


def _rolling_n_and_snr(
    history: pl.DataFrame, targets: pl.DataFrame, key_cols: list[str], prefix: str
) -> pl.DataFrame:
    """For every distinct (key_cols, window_start) combination in `targets`,
    computes trailing count + availability-buffer-adjusted weighted-mean SNR
    for each lookback in _LOOKBACKS, using only `history` rows that are
    already available (source.window_start <= target.window_start -
    AVAIL_BUFFER_MIN). Returns one row per distinct target combination:
    window_start, key_cols, and the {prefix}_n_{suffix} / {prefix}_snr_{suffix}
    columns.

    Implementation: the target combinations are concatenated into the
    (sorted) history as zero-weight anchor rows so that polars'
    rolling_sum_by (which only evaluates at rows present in the frame) can
    be evaluated exactly at each target's own timestamp; the anchors are
    filtered back out at the end.
    """
    return _finalize_snr(_rolling_raw(history, targets, key_cols, prefix), prefix)


def _relation_via_expanded_anchors(
    history: pl.DataFrame,
    expanded_anchor_source: pl.DataFrame,
    varying_col: str,
    orig_col: str,
    key_cols: list[str],
    prefix: str,
) -> pl.DataFrame:
    """Shared machinery for adjacent_band/adjacent_cell: each target row maps
    to several (2 or up to 8) *other* keys to aggregate over and sum -- e.g.
    adjacent_cell sums same-cell-style stats across a target rx_field's up
    to 8 Maidenhead neighbors. The original implementation exploded
    `history` itself (2x/8x) to cover every relation copy in one combined
    rolling pass; at real 2024+ WSPRnet volume that multiplies real spot
    payload data by up to 8x and was the direct cause of a 2026-07-20 OOM
    incident (~28GB+ for one month's adjacent_cell alone).

    This instead explodes only `expanded_anchor_source` -- a narrow,
    payload-free frame (window_start, key_cols, plus `orig_col` holding each
    row's *original* target value for whichever of `key_cols` varies across
    relation copies -- `band` for adjacent_band, `rx_field` for
    adjacent_cell) -- and relies on `_rolling_raw`'s own `.unique()` anchor
    construction to collapse it back down to the true distinct (window_start,
    key_cols) cardinality (bounded by real path/hour combinations, orders of
    magnitude below a raw 2x/8x row count) *before* it ever touches
    `history`, which stays at its original, unexploded size.

    `varying_col` (e.g. "band") holds each relation copy's *other* key value
    in `expanded_anchor_source` (matching `history`'s column of that name);
    `orig_col` (e.g. "_orig_band") holds the target row's own value, used to
    attribute each relation copy back and sum raw n_/num_/den_ across them
    (NOT the finalized ratio -- summing already-divided per-relation SNR
    averages would be wrong) before a single final division.
    """
    raw = _rolling_raw(history, expanded_anchor_source.select("window_start", *key_cols), key_cols, prefix)
    attributed = expanded_anchor_source.join(raw, on=["window_start", *key_cols], how="left")
    group_key_cols = [c for c in key_cols if c != varying_col]
    sum_exprs = [pl.col(c).sum().alias(c) for c in raw.columns if c not in ("window_start", *key_cols)]
    combined = attributed.group_by(["window_start", orig_col, *group_key_cols]).agg(sum_exprs)
    return _finalize_snr(combined.rename({orig_col: varying_col}), prefix)


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

    # _rolling_n_and_snr only ever reads window_start/key_cols/n_spots/
    # snr_ft8eq_p50 from `history` (it narrows to exactly that internally);
    # adjacent_band and adjacent_cell join-expand `full_history` up to 2x
    # and 8x respectively before that internal narrowing ever runs, so the
    # unused columns (open, n_monitors, n_tx_stations, evidence_tier) were
    # getting multiplied right along with everything else. Narrowing here,
    # before any of the relation-specific joins, was part of fixing a
    # 2026-07-20 OOM incident (measured ~28GB+ for one month's adjacent_band
    # + adjacent_cell on real 2024 WSPRnet volume with the wide frame).
    history_narrow = full_history.select(
        "window_start", "tx_field", "rx_field", "band", "n_spots", "snr_ft8eq_p50"
    )

    same_cell = _rolling_n_and_snr(history_narrow, target_rows, ["tx_field", "rx_field", "band"], "same_cell")
    out = target_rows.join(same_cell, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"same_cell_n_{suffix}").fill_null(0))

    reverse_hist = history_narrow.rename({"tx_field": "rx_field", "rx_field": "tx_field"})
    reverse = _rolling_n_and_snr(reverse_hist, target_rows, ["tx_field", "rx_field", "band"], "reverse_path")
    out = out.join(reverse, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"reverse_path_n_{suffix}").fill_null(0))

    # adjacent band: for a target on band B, sum same-cell-style stats over
    # B's up to 2 adjacent bands (a target on 20m looks at history on 17m
    # and 15m). Expand the (small, payload-free) target keys per adjacent
    # band -- not `history_narrow` (see _relation_via_expanded_anchors) --
    # keeping each expanded row's own original band in `_orig_band`.
    band_map = pl.DataFrame(
        [(band, adj) for band in BAND_ORDER for adj in _adjacent_bands(band)],
        schema=["_orig_band", "band"], orient="row",
    )
    adj_band_anchor_source = (
        target_rows.select("window_start", "tx_field", "rx_field", pl.col("band").alias("_orig_band"))
        .join(band_map, on="_orig_band")
    )
    adj_band = _relation_via_expanded_anchors(
        history_narrow, adj_band_anchor_source, "band", "_orig_band",
        ["tx_field", "rx_field", "band"], "adjacent_band",
    )
    out = out.join(adj_band, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"adjacent_band_n_{suffix}").fill_null(0))

    # adjacent cell: rx_field's 8 Maidenhead neighbors, same tx_field + band.
    # Same anchor-side-expansion pattern: expand target keys per neighbor of
    # the target's OWN rx_field (field_neighbors is a symmetric relation, so
    # this is equivalent to the original "expand history per neighbor of its
    # rx_field" -- see field_neighbors' module docstring), not `history_narrow`.
    target_rx_values = target_rows.select("rx_field").unique()["rx_field"].to_list()
    neighbor_map_rows = []
    for rx in target_rx_values:
        for nb in field_neighbors(rx):
            neighbor_map_rows.append((rx, nb))
    neighbor_map = pl.DataFrame(
        neighbor_map_rows, schema={"_orig_rx": pl.Utf8, "rx_field": pl.Utf8}, orient="row"
    )
    adj_cell_anchor_source = (
        target_rows.select("window_start", "tx_field", pl.col("rx_field").alias("_orig_rx"), "band")
        .join(neighbor_map, on="_orig_rx")
    )
    adj_cell = _relation_via_expanded_anchors(
        history_narrow, adj_cell_anchor_source, "rx_field", "_orig_rx",
        ["tx_field", "rx_field", "band"], "adjacent_cell",
    )
    out = out.join(adj_cell, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"adjacent_cell_n_{suffix}").fill_null(0))

    # band-wide: all cells, same band -- group by band + window_start only.
    band_wide_src = history_narrow.group_by(["band", "window_start"]).agg(
        pl.col("n_spots").sum(), pl.col("snr_ft8eq_p50").mean().alias("snr_ft8eq_p50"),
    )
    band_wide = _rolling_n_and_snr(band_wide_src, target_rows, ["band"], "band_wide")
    out = out.join(band_wide, on=["window_start", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"band_wide_n_{suffix}").fill_null(0))

    # same-hour-yesterday: point lookup, not an aggregate.
    yesterday_src = full_history.select(
        (pl.col("window_start") + pl.duration(hours=24)).alias("window_start"),
        "tx_field", "rx_field", "band", pl.col("open").alias("same_hour_yesterday_open"),
    )
    out = out.join(yesterday_src, on=["window_start", "tx_field", "rx_field", "band"], how="left")

    return out.with_columns((pl.col("window_start") + shift).alias("window_start"))
