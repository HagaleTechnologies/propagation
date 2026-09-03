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


def normalize_spot_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add any SPOT_SCHEMA column missing from `df` as an all-null column,
    then reorder to SPOT_SCHEMA's canonical column order.

    Every extractor (wsprnet.py, rbn.py, pskreporter.py) builds its
    DataFrame from a list of dicts that never carry dx_field/de_field
    (those are derived later, downstream in features/universe.py), then
    pads the missing SPOT_SCHEMA columns onto whatever the dict-derived
    order happened to be -- which puts dx_field/de_field at the END, not in
    their SPOT_SCHEMA-declared middle position. `pl.concat(...,
    how="vertical_relaxed")` matches columns positionally, not by name, so
    two frames that both nominally satisfy SPOT_SCHEMA but were built via
    different paths (e.g. one from `pl.DataFrame(schema=SPOT_SCHEMA)` when
    there were zero qualifying rows, one padded-from-dicts when there were)
    can still fail to concat. Confirmed live: PRO-9's soak test crashed the
    whole accumulator process on exactly this (`write_hourly_parquet`'s
    on-disk-merge path, second flush to an hour whose first flush had no
    qualifying rows) -- see docs/DECISIONS/ for the incident."""
    missing = [col for col in SPOT_SCHEMA if col not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None).alias(col) for col in missing])
    return df.select(list(SPOT_SCHEMA.keys()))
