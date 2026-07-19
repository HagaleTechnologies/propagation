"""Path geometry features (ARCHITECTURE.md sec 4 item 1): great-circle
distance/bearing, midpoint, P.533-convention control points (1000km from
each terminus, degenerating to the midpoint under 2000km — mirrors P.533's
own convention per ARCHITECTURE.md sec 4 item 1), and centered-dipole
geomagnetic latitude (auroral-oval proximity proxy).
"""
from __future__ import annotations

import math

import polars as pl

from propagation.data.geo import grid_to_latlon, great_circle_km

_EARTH_RADIUS_KM = 6371.0088

# Centered-dipole geomagnetic north pole, IGRF-13 epoch 2020.0. Revisit when
# a newer IGRF generation is published (IGRF-14 expected ~2025); the pole
# drifts slowly (~ tenths of a degree per year) so this is not urgent.
GEOMAG_POLE_LAT = 80.65
GEOMAG_POLE_LON = -72.68


def _intermediate_point(lat1: float, lon1: float, lat2: float, lon2: float, f: float) -> tuple[float, float]:
    """Point at fraction f (0=start, 1=end) along the great circle from
    (lat1,lon1) to (lat2,lon2). Standard spherical interpolation formula."""
    d = great_circle_km(lat1, lon1, lat2, lon2)
    delta = d / _EARTH_RADIUS_KM
    if delta == 0:
        return lat1, lon1
    p1, l1 = math.radians(lat1), math.radians(lon1)
    p2, l2 = math.radians(lat2), math.radians(lon2)
    a = math.sin((1 - f) * delta) / math.sin(delta)
    b = math.sin(f * delta) / math.sin(delta)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    lat_i = math.atan2(z, math.sqrt(x * x + y * y))
    lon_i = math.atan2(y, x)
    return math.degrees(lat_i), math.degrees(lon_i)


def control_points(
    tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """P.533-convention control points: 1000km from each terminus along the
    great circle, degenerating to the midpoint for paths under 2000km."""
    dist = great_circle_km(tx_lat, tx_lon, rx_lat, rx_lon)
    if dist <= 2000.0:
        f_tx = f_rx = 0.5
    else:
        f_tx = 1000.0 / dist
        f_rx = 1.0 - 1000.0 / dist
    tx_cp = _intermediate_point(tx_lat, tx_lon, rx_lat, rx_lon, f_tx)
    rx_cp = _intermediate_point(tx_lat, tx_lon, rx_lat, rx_lon, f_rx)
    return tx_cp, rx_cp


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees, 0-360."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def geomag_lat(lat: float, lon: float) -> float:
    """Centered-dipole approximation of geomagnetic latitude."""
    la, lo = math.radians(lat), math.radians(lon)
    pla, plo = math.radians(GEOMAG_POLE_LAT), math.radians(GEOMAG_POLE_LON)
    s = math.sin(la) * math.sin(pla) + math.cos(la) * math.cos(pla) * math.cos(lo - plo)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def add_geometry_features(labels: pl.DataFrame) -> pl.DataFrame:
    """Adds path-geometry columns to a labels-shaped frame (must have
    tx_field, rx_field). Computed per unique (tx_field, rx_field) pair,
    since geometry is static — not per row — then joined back."""
    pairs = labels.select("tx_field", "rx_field").unique()
    rows = []
    for tx_field, rx_field in pairs.iter_rows():
        tx_lat, tx_lon = grid_to_latlon(tx_field)
        rx_lat, rx_lon = grid_to_latlon(rx_field)
        dist = great_circle_km(tx_lat, tx_lon, rx_lat, rx_lon)
        brg = bearing_deg(tx_lat, tx_lon, rx_lat, rx_lon)
        mid_lat, mid_lon = _intermediate_point(tx_lat, tx_lon, rx_lat, rx_lon, 0.5)
        tx_cp, rx_cp = control_points(tx_lat, tx_lon, rx_lat, rx_lon)
        rows.append((
            tx_field, rx_field, dist, brg, mid_lat, mid_lon,
            tx_cp[0], tx_cp[1], rx_cp[0], rx_cp[1],
            geomag_lat(tx_lat, tx_lon), geomag_lat(rx_lat, rx_lon), geomag_lat(mid_lat, mid_lon),
        ))
    geo = pl.DataFrame(
        rows,
        schema=["tx_field", "rx_field", "distance_km", "bearing_deg", "midpoint_lat", "midpoint_lon",
                "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
                "tx_geomag_lat", "rx_geomag_lat", "midpoint_geomag_lat"],
        orient="row",
    )
    return labels.join(geo, on=["tx_field", "rx_field"], how="left")
