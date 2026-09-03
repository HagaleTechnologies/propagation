import pytest

from propagation.data.geo import grid_to_latlon, great_circle_km, latlon_to_grid


def test_grid_to_latlon_field_only():
    lat, lon = grid_to_latlon("EM")
    # EM: E=lon field index 4 -> -180+4*20=-100; M=lat field index 12 -> -90+12*10=30
    # centroid = field corner + half field size
    assert lat == pytest.approx(35.0)
    assert lon == pytest.approx(-90.0)


def test_grid_to_latlon_grid4():
    lat, lon = grid_to_latlon("EM12")
    assert lat == pytest.approx(32.5)
    assert lon == pytest.approx(-97.0)


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


def test_latlon_to_grid_matches_grid4_centroid():
    # Inverse of test_grid_to_latlon_grid4: EM12's own centroid round-trips.
    assert latlon_to_grid(32.5, -97.0) == "EM12"


def test_latlon_to_grid_field_only():
    assert latlon_to_grid(35.0, -90.0, precision=2) == "EM"


def test_latlon_to_grid_known_reference_point():
    # Newington, CT (ARRL HQ), well-known real-world locator FN31pr.
    assert latlon_to_grid(41.7, -72.7) == "FN31"


def test_latlon_to_grid_handles_pole_and_dateline_edges():
    # lat=90/lon=180 must not overflow past field letter 'R' (18 fields of
    # 10deg/20deg each cover exactly -90..90 / -180..180).
    assert latlon_to_grid(90.0, 180.0) == "RR99"
    assert latlon_to_grid(-90.0, -180.0) == "AA00"


def test_latlon_to_grid_rejects_invalid_precision():
    with pytest.raises(ValueError):
        latlon_to_grid(0.0, 0.0, precision=3)
