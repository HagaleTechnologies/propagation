import math
import re

_EARTH_RADIUS_KM = 6371.0088
_GRID4_RE = re.compile(r"^[A-R]{2}[0-9]{2}$")
_FIELD_RE = re.compile(r"^[A-R]{2}$")


def grid_to_latlon(grid: str) -> tuple[float, float]:
    """Centroid lat/lon of a Maidenhead field (2 char) or grid4 (4 char)."""
    g = grid.strip().upper()
    if len(g) > 4:
        g = g[:4]
    if _GRID4_RE.match(g):
        field_lon = (ord(g[0]) - ord("A")) * 20 - 180
        field_lat = (ord(g[1]) - ord("A")) * 10 - 90
        lon = field_lon + int(g[2]) * 2 + 1.0
        lat = field_lat + int(g[3]) * 1 + 0.5
        return lat, lon
    if _FIELD_RE.match(g):
        field_lon = (ord(g[0]) - ord("A")) * 20 - 180
        field_lat = (ord(g[1]) - ord("A")) * 10 - 90
        return field_lat + 5.0, field_lon + 10.0
    raise ValueError(f"invalid Maidenhead grid: {grid!r}")


def latlon_to_grid(lat: float, lon: float, precision: int = 4) -> str:
    """Inverse of grid_to_latlon: standard Maidenhead locator for a point.
    `precision` is 2 (field only) or 4 (field + square); anything else is
    rejected since this repo has no use for finer (subsquare) precision."""
    if precision not in (2, 4):
        raise ValueError(f"precision must be 2 or 4, got {precision!r}")
    # Clamp just inside the top edge so lon=180/lat=90 land in the last
    # field/square instead of wrapping to the first (360 % 20 == 0).
    lon_frac = min(max(lon, -180.0), 180.0) + 180.0
    lat_frac = min(max(lat, -90.0), 90.0) + 90.0
    lon_frac = min(lon_frac, 359.999999)
    lat_frac = min(lat_frac, 179.999999)
    field_lon = int(lon_frac / 20)
    field_lat = int(lat_frac / 10)
    grid = chr(ord("A") + field_lon) + chr(ord("A") + field_lat)
    if precision == 2:
        return grid
    square_lon = int((lon_frac % 20) / 2)
    square_lat = int(lat_frac % 10)
    return grid + str(square_lon) + str(square_lat)


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))
