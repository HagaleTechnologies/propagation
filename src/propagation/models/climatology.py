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
