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

    anchors = targets.select(["window_start", *key_cols]).unique().with_columns(
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
        n_buf = pl.col("n_spots").rolling_sum_by("window_start", window_size=buffer_str, closed="left").over(key_cols)
        w_full = pl.col("_snr_weighted").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        w_buf = pl.col("_snr_weighted").rolling_sum_by("window_start", window_size=buffer_str, closed="left").over(key_cols)
        d_full = pl.col("_snr_weight").rolling_sum_by("window_start", window_size=window, closed="left").over(key_cols)
        d_buf = pl.col("_snr_weight").rolling_sum_by("window_start", window_size=buffer_str, closed="left").over(key_cols)

        n_name = f"{prefix}_n_{suffix}"
        snr_name = f"{prefix}_snr_{suffix}"
        # Clamped at 0: for lookbacks shorter than AVAIL_BUFFER_MIN (the
        # "15m" suffix), full - buffer can go negative even though the
        # structurally-correct answer is always 0 (see module docstring).
        n_expr = (n_full - n_buf).clip(lower_bound=0).alias(n_name)
        denom = d_full - d_buf
        snr_expr = pl.when(denom > 0).then((w_full - w_buf) / denom).otherwise(None).alias(snr_name)
        exprs += [n_expr, snr_expr]
        out_names += [n_name, snr_name]

    return (
        combined.with_columns(exprs)
        .filter(pl.col("_is_anchor"))
        .select(["window_start", *key_cols, *out_names])
    )


def add_history_features(full_history: pl.DataFrame, target_rows: pl.DataFrame) -> pl.DataFrame:
    if full_history.height == 0:
        full_history = pl.DataFrame(
            schema={"window_start": pl.Datetime("us", "UTC"), "tx_field": pl.Utf8, "rx_field": pl.Utf8,
                    "band": pl.Utf8, "n_spots": pl.Int64, "snr_ft8eq_p50": pl.Float64, "open": pl.Int64},
        )

    same_cell = _rolling_n_and_snr(full_history, target_rows, ["tx_field", "rx_field", "band"], "same_cell")
    out = target_rows.join(same_cell, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"same_cell_n_{suffix}").fill_null(0))

    reverse_hist = full_history.rename({"tx_field": "rx_field", "rx_field": "tx_field"})
    reverse = _rolling_n_and_snr(reverse_hist, target_rows, ["tx_field", "rx_field", "band"], "reverse_path")
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
    adj_band = _rolling_n_and_snr(expanded, target_rows, ["tx_field", "rx_field", "band"], "adjacent_band")
    out = out.join(adj_band, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"adjacent_band_n_{suffix}").fill_null(0))

    # adjacent cell: rx_field's 8 Maidenhead neighbors, same tx_field + band.
    all_rx = full_history.select("rx_field").unique()["rx_field"].to_list()
    neighbor_map_rows = []
    for rx in all_rx:
        for nb in field_neighbors(rx):
            neighbor_map_rows.append((rx, nb))
    neighbor_map = pl.DataFrame(
        neighbor_map_rows, schema={"_src_rx": pl.Utf8, "rx_field": pl.Utf8}, orient="row"
    )
    expanded_cell = full_history.rename({"rx_field": "_src_rx"}).join(neighbor_map, on="_src_rx").drop("_src_rx")
    adj_cell = _rolling_n_and_snr(expanded_cell, target_rows, ["tx_field", "rx_field", "band"], "adjacent_cell")
    out = out.join(adj_cell, on=["window_start", "tx_field", "rx_field", "band"], how="left")
    for suffix in _LOOKBACKS:
        out = out.with_columns(pl.col(f"adjacent_cell_n_{suffix}").fill_null(0))

    # band-wide: all cells, same band -- group by band + window_start only.
    band_wide_src = full_history.group_by(["band", "window_start"]).agg(
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

    return out
