"""Assembles the full M2 feature matrix from every features/ module, plus
time features (ARCHITECTURE.md sec 4 item 3) and mode-normalized SNR
(reusing propagation.features.labels.snr_ft8eq -- already computed into
labels.snr_ft8eq_p50 upstream by build_labels(), included here as-is, never
reimplemented, per this plan's Global Constraints).
"""
from __future__ import annotations

import math

import polars as pl

from propagation.features.geometry import add_geometry_features
from propagation.features.history import add_history_features
from propagation.features.solar import add_solar_features
from propagation.features.spaceweather import add_spaceweather_features

_TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]
_GEOMETRY_COLS = [
    "distance_km", "bearing_deg", "midpoint_lat", "midpoint_lon",
    "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
    "tx_geomag_lat", "rx_geomag_lat", "midpoint_geomag_lat",
]
_SOLAR_COLS = [
    "midpoint_solar_zenith", "tx_control_solar_zenith", "rx_control_solar_zenith",
    "path_daylight_fraction", "midpoint_hours_since_terminator",
]
_SPACEWEATHER_COLS = [
    "kp_now", "kp_lag3h", "kp_lag6h", "kp_lag12h", "kp_lag24h", "kp_lag48h",
    "f107_daily", "f107_smoothed_27d", "bz_gsm_now", "solar_wind_speed_now", "dst_now",
]
_HISTORY_RELATIONS = ["same_cell", "reverse_path", "adjacent_band", "adjacent_cell", "band_wide"]
_HISTORY_LOOKBACKS = ["15m", "1h", "3h", "24h"]
_HISTORY_COLS = [
    f"{rel}_{stat}_{lb}"
    for rel in _HISTORY_RELATIONS
    for stat in ("n", "snr")
    for lb in _HISTORY_LOOKBACKS
] + ["same_hour_yesterday_open"]
_MODE_NORM_COLS = ["snr_ft8eq_p50"]

FEATURE_COLUMNS = _TIME_COLS + _GEOMETRY_COLS + _SOLAR_COLS + _SPACEWEATHER_COLS + _HISTORY_COLS + _MODE_NORM_COLS


def add_time_features(labels: pl.DataFrame) -> pl.DataFrame:
    return labels.with_columns(
        (2 * math.pi * pl.col("window_start").dt.hour() / 24).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("window_start").dt.hour() / 24).cos().alias("hour_cos"),
        (2 * math.pi * pl.col("window_start").dt.ordinal_day() / 365).sin().alias("doy_sin"),
        (2 * math.pi * pl.col("window_start").dt.ordinal_day() / 365).cos().alias("doy_cos"),
        pl.col("window_start").dt.month().alias("month"),
    )


def build_feature_matrix(labels: pl.DataFrame, full_history: pl.DataFrame, omni: pl.DataFrame) -> pl.DataFrame:
    """`labels` are the rows to build features FOR; `full_history` is the
    complete, unsampled label set for the same period (history features
    need other cells' activity, not just the rows being scored);
    `omni` is `propagation.data.spaceweather.fetch_omni2_range`'s output."""
    out = add_time_features(labels)
    out = add_geometry_features(out)
    out = add_solar_features(out)
    out = add_spaceweather_features(out, omni)
    out = add_history_features(full_history, out)
    return out
