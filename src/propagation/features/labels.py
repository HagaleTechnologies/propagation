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
