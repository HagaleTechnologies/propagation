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
        lat = field_lat + int(g[2]) * 1 + 0.5
        lon = field_lon + int(g[3]) * 2
        return lat, lon
    if _FIELD_RE.match(g):
        field_lon = (ord(g[0]) - ord("A")) * 20 - 180
        field_lat = (ord(g[1]) - ord("A")) * 10 - 90
        return field_lat + 5.0, field_lon + 10.0
    raise ValueError(f"invalid Maidenhead grid: {grid!r}")


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))
