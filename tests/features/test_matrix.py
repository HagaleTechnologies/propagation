from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from propagation.features.matrix import (
    FEATURE_COLUMNS,
    add_band_feature,
    add_time_features,
    build_feature_matrix,
)


def test_add_time_features_sin_cos_pairs_are_unit_circle():
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_time_features(labels)
    assert out["hour_sin"][0] ** 2 + out["hour_cos"][0] ** 2 == pytest.approx(1.0)
    assert out["doy_sin"][0] ** 2 + out["doy_cos"][0] ** 2 == pytest.approx(1.0)
    assert out["month"][0] == 6


def test_build_feature_matrix_produces_every_declared_column(tmp_path):
    from propagation.data.spaceweather import _parse_omni2
    ts = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    labels = pl.DataFrame({
        "window_start": [ts], "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
        "open": [1], "n_spots": [3], "snr_ft8eq_p50": [10.0],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    omni_text = (
        "2026 166  0 2461 51 52  33   6   4.8   4.3  -6.5  84.6   0.4   4.3  -0.5   4.3   0.6   0.1"
        "   1.8   1.3   0.4   1.2  103492.   6.2  399.  -0.4  -4.4 0.014  1.74    4395.   0.1    3."
        "   0.5   0.1 0.001  -0.24   2.60  10.3  7 124     4   25 999999.99 99999.99 99999.99"
        "     0.15     0.07     0.04 -1   3 154.3   0.6   -15    10  5.7"
    )
    omni = _parse_omni2(omni_text, year=2026)
    out = build_feature_matrix(labels, full_history=labels, omni=omni)
    for col in FEATURE_COLUMNS:
        assert col in out.columns, col
    assert out.height == 1


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


def test_build_feature_matrix_forwards_horizon_hours_to_history_features():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    omni = pl.DataFrame({
        "time": [start], "kp": [3.0], "f107": [100.0],
        "bz_gsm": [1.0], "solar_wind_speed": [400.0], "dst": [-10.0],
    }, schema_overrides={"time": pl.Datetime("us", "UTC")})
    ts = start + timedelta(hours=48)
    source_ts = ts - timedelta(hours=2)
    full_history = pl.DataFrame({
        "window_start": [source_ts],
        "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
        "n_spots": [5], "snr_ft8eq_p50": [10.0], "open": [1],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    labels = pl.DataFrame({
        "window_start": [ts], "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
        "open": [1], "n_spots": [3], "snr_ft8eq_p50": [10.0],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out_h0 = build_feature_matrix(labels, full_history=full_history, omni=omni, horizon_hours=0.0)
    out_h6 = build_feature_matrix(labels, full_history=full_history, omni=omni, horizon_hours=6.0)
    # source row is 2h before the target's own window_start: visible in the
    # 24h trailing window at h=0 (prediction_time=ts), but NOT at h=6
    # (prediction_time=ts-6h, which is BEFORE the source row) -- proves
    # build_feature_matrix's add_history_features(..., horizon_hours=...)
    # call site actually forwards the parameter, not just add_spaceweather_features's.
    assert out_h0["same_cell_n_24h"][0] == 5
    assert out_h6["same_cell_n_24h"][0] == 0


def test_add_band_feature_is_ordinal_and_monotonic_in_band_order():
    from propagation.features.history import BAND_ORDER
    labels = pl.DataFrame({"band": BAND_ORDER})
    out = add_band_feature(labels)
    assert out["band_ordinal"].to_list() == list(range(len(BAND_ORDER)))


def test_feature_columns_includes_band_ordinal_not_raw_band():
    assert "band_ordinal" in FEATURE_COLUMNS
    assert "band" not in FEATURE_COLUMNS
