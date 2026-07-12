import pytest

from propagation.data.geo import grid_to_latlon, great_circle_km


def test_grid_to_latlon_field_only():
    lat, lon = grid_to_latlon("EM")
    # EM: E=lon field index 4 -> -180+4*20=-100; M=lat field index 12 -> -90+12*10=30
    # centroid = field corner + half field size
    assert lat == pytest.approx(35.0)
    assert lon == pytest.approx(-90.0)


def test_grid_to_latlon_grid4():
    lat, lon = grid_to_latlon("EM12")
    assert lat == pytest.approx(31.5)
    assert lon == pytest.approx(-96.0)


def test_grid_to_latlon_truncates_grid6():
    assert grid_to_latlon("EM12ab") == grid_to_latlon("EM12")


def test_grid_to_latlon_rejects_invalid():
    with pytest.raises(ValueError):
        grid_to_latlon("Z9")


def test_great_circle_km_known_distance():
    # JFK (40.6413N, -73.7781W) to LAX (33.9416N, -118.4085W) ~ 3983 km
    d = great_circle_km(40.6413, -73.7781, 33.9416, -118.4085)
    assert d == pytest.approx(3983, rel=0.01)


def test_great_circle_km_zero_for_same_point():
    assert great_circle_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)
