from datetime import datetime, timezone

import polars as pl
import pytest

from propagation.features.solar import add_solar_features, solar_zenith_deg


def test_solar_zenith_near_zero_at_equator_equinox_local_noon():
    when = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    assert solar_zenith_deg(0.0, 0.0, when) == pytest.approx(0.0, abs=3.0)


def test_solar_zenith_near_180_at_equator_equinox_local_midnight():
    when = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    assert solar_zenith_deg(0.0, 0.0, when) == pytest.approx(180.0, abs=3.0)


def test_solar_zenith_matches_obliquity_at_solstice_noon():
    # 42N, local solar noon (lon=0), summer solstice -> zenith ~= 42 - 23.44
    when = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    assert solar_zenith_deg(42.0, 0.0, when) == pytest.approx(42.0 - 23.44, abs=1.0)


def test_add_solar_features_requires_geometry_columns():
    from propagation.features.geometry import add_geometry_features
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 6, 21, 18, 0, tzinfo=timezone.utc)],
        "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    with_geo = add_geometry_features(labels)
    out = add_solar_features(with_geo)
    for col in ("tx_solar_zenith", "rx_solar_zenith", "midpoint_solar_zenith",
                "tx_control_solar_zenith", "rx_control_solar_zenith",
                "path_daylight_fraction", "midpoint_hours_since_terminator"):
        assert col in out.columns, col
    assert 0.0 <= out["path_daylight_fraction"][0] <= 1.0
