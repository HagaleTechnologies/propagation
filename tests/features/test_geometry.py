import polars as pl
import pytest

from propagation.features.geometry import (
    add_geometry_features, bearing_deg, control_points, geomag_lat,
)


def test_control_points_1000km_from_each_terminus_on_long_path():
    # FN (Boston, ~42N -71W) to DM (LA area, ~34N -118W): ~4159 km, well over 2000km
    tx_control, rx_control = control_points(42.0, -71.0, 34.0, -118.0)
    from propagation.data.geo import great_circle_km
    d_tx = great_circle_km(42.0, -71.0, *tx_control)
    d_rx = great_circle_km(34.0, -118.0, *rx_control)
    assert d_tx == pytest.approx(1000.0, abs=0.1)
    assert d_rx == pytest.approx(1000.0, abs=0.1)


def test_control_points_degenerate_to_midpoint_under_2000km():
    # FN to EM (Atlanta, ~33N -84W): ~1519 km, under 2000km
    tx_control, rx_control = control_points(42.0, -71.0, 33.0, -84.0)
    assert tx_control == pytest.approx(rx_control)


def test_bearing_deg_due_east_on_equator():
    # from (0,0) to (0,10): due east -> bearing 90
    assert bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=0.5)


def test_bearing_deg_due_north():
    assert bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0, abs=0.5)


def test_geomag_lat_at_the_dipole_pole_is_90():
    from propagation.features.geometry import GEOMAG_POLE_LAT, GEOMAG_POLE_LON
    assert geomag_lat(GEOMAG_POLE_LAT, GEOMAG_POLE_LON) == pytest.approx(90.0, abs=1e-6)


def test_geomag_lat_at_geographic_north_pole_equals_pole_lat():
    from propagation.features.geometry import GEOMAG_POLE_LAT
    assert geomag_lat(90.0, 0.0) == pytest.approx(GEOMAG_POLE_LAT, abs=1e-6)


def test_add_geometry_features_adds_expected_columns():
    labels = pl.DataFrame({
        "window_start": [1], "tx_field": ["FN"], "rx_field": ["DM"], "band": ["20m"],
    })
    out = add_geometry_features(labels)
    for col in ("distance_km", "bearing_deg", "midpoint_lat", "midpoint_lon",
                "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
                "tx_geomag_lat", "rx_geomag_lat", "midpoint_geomag_lat"):
        assert col in out.columns, col
    assert out["distance_km"][0] > 0
