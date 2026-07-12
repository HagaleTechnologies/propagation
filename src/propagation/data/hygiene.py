import re

from propagation.data.geo import great_circle_km, grid_to_latlon
from propagation.data.schema import SUPPORTED_BANDS

_GRID4_RE = re.compile(r"^[A-R]{2}[0-9]{2}$")
_FIELD_RE = re.compile(r"^[A-R]{2}$")
_CALLSIGN_RE = re.compile(r"^[A-Z0-9]{1,3}[0-9][A-Z0-9]{0,3}[A-Z](/[A-Z0-9]{1,4})?$")

_DIGI_MODES = {
    "FT8", "FT4", "WSPR", "FST4", "FST4W", "JS8", "JT65", "JT9", "Q65", "MSK144",
}
_CW_MODES = {"CW", "RTTY"}

_MIN_DISTANCE_KM = 25.0


def mode_class_for(mode: str) -> str:
    m = mode.strip().upper()
    if m in _DIGI_MODES:
        return "digi"
    if m in _CW_MODES:
        return "cw"
    return "other"


def normalize_grid(raw: str | None) -> str | None:
    if not raw:
        return None
    g = raw.strip().upper()
    if len(g) > 4:
        g = g[:4]
    if g == "RR73":
        return None
    if _GRID4_RE.match(g) or _FIELD_RE.match(g):
        return g
    return None


def strip_hash_markers(call: str) -> str:
    c = call.strip().upper()
    if c.startswith("<"):
        c = c[1:]
    if c.endswith(">"):
        c = c[:-1]
    return c


def is_valid_callsign(raw: str) -> bool:
    c = strip_hash_markers(raw)
    return bool(_CALLSIGN_RE.match(c))


def has_excluded_suffix(raw: str) -> bool:
    c = strip_hash_markers(raw)
    return c.endswith("/MM") or c.endswith("/AM")


def base_call(raw: str) -> str:
    c = strip_hash_markers(raw)
    return c.split("/")[0]


def is_qualifying_spot(row: dict) -> tuple[bool, str | None]:
    """docs/SPEC-labeling.md §1. Returns (qualifies, rejection_reason)."""
    if (
        row.get("ts") is None
        or not row.get("band")
        or not row.get("mode")
        or not row.get("dx_call")
        or not row.get("de_call")
    ):
        return False, "missing_required_field"
    if row["band"] not in SUPPORTED_BANDS:
        return False, "unsupported_band"

    dx_call, de_call = row["dx_call"], row["de_call"]
    if not is_valid_callsign(dx_call) or not is_valid_callsign(de_call):
        return False, "invalid_callsign"
    if has_excluded_suffix(dx_call) or has_excluded_suffix(de_call):
        return False, "mm_am_suffix"
    if base_call(dx_call) == base_call(de_call):
        return False, "self_spot"

    dx_grid_raw, de_grid_raw = row.get("dx_grid"), row.get("de_grid")
    for raw in (dx_grid_raw, de_grid_raw):
        if raw and raw.strip().upper()[:4] == "RR73":
            return False, "rr73_grid"

    dx_grid, de_grid = normalize_grid(dx_grid_raw), normalize_grid(de_grid_raw)
    dx_lat, dx_lon = row.get("dx_lat"), row.get("dx_lon")
    de_lat, de_lon = row.get("de_lat"), row.get("de_lon")

    if dx_grid is None and (dx_lat is None or dx_lon is None):
        return False, "no_usable_location"
    if de_grid is None and (de_lat is None or de_lon is None):
        return False, "no_usable_location"

    if dx_grid is not None:
        dx_lat, dx_lon = grid_to_latlon(dx_grid)
    if de_grid is not None:
        de_lat, de_lon = grid_to_latlon(de_grid)

    if great_circle_km(dx_lat, dx_lon, de_lat, de_lon) < _MIN_DISTANCE_KM:
        return False, "distance_too_short"

    return True, None
