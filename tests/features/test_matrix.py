from datetime import datetime, timezone

import polars as pl
import pytest

from propagation.features.matrix import FEATURE_COLUMNS, add_time_features, build_feature_matrix


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
