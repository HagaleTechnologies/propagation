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
