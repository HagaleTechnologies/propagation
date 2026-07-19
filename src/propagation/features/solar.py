"""Solar geometry features (ARCHITECTURE.md sec 4 item 2): solar zenith
angle, daylight fraction, gray-line proxy. Computable for any future time
(needed for h>0 horizons) via a simplified NOAA-style solar position
formula (Spencer 1971 declination + equation of time) -- deliberately not
using the `astral` package (PR #10's original decision, still sound: this
avoids a dependency for a formula that's ~20 lines and doesn't need
arc-second precision for a propagation-nowcasting feature).
"""
from __future__ import annotations

import math
from datetime import datetime

import polars as pl


def solar_zenith_deg(lat: float, lon: float, when: datetime) -> float:
    """Solar zenith angle (degrees, 0=overhead sun, 180=solar midnight) at
    (lat, lon) at UTC datetime `when`."""
    doy = when.timetuple().tm_yday
    hour_utc = when.hour + when.minute / 60.0 + when.second / 3600.0
    gamma = 2 * math.pi / 365.0 * (doy - 1 + (hour_utc - 12) / 24.0)
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    time_offset = eqtime + 4 * lon
    tst = hour_utc * 60 + when.second / 60.0 + time_offset
    hour_angle_deg = (tst / 4.0) - 180.0
    ha = math.radians(hour_angle_deg)
    la = math.radians(lat)
    cos_zenith = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    return math.degrees(math.acos(cos_zenith))


def _hours_since_terminator(lat: float, lon: float, when) -> float:
    """Signed hours since the solar terminator crossed this point (positive
    = daylight, growing since sunrise; negative = darkness, growing since
    sunset) -- a cheap gray-line proxy. Approximated by scanning zenith
    angle at 15-minute steps back from `when` for the most recent 90-degree
    crossing; capped at +/-12h (a full half-day either side)."""
    import datetime as dt
    step = dt.timedelta(minutes=15)
    prev_zenith = solar_zenith_deg(lat, lon, when)
    daylight_now = prev_zenith < 90.0
    t = when
    for i in range(1, 49):  # up to 12h back, 15-min steps
        t = t - step
        z = solar_zenith_deg(lat, lon, t)
        crossed = (z < 90.0) != daylight_now
        if crossed:
            hours = i * 0.25
            return hours if daylight_now else -hours
    return 12.0 if daylight_now else -12.0


def add_solar_features(labels: pl.DataFrame) -> pl.DataFrame:
    """Requires tx_geomag_lat-style geometry columns to already be present
    (run after features.geometry.add_geometry_features): tx/rx/midpoint/
    control-point lat+lon and window_start. Computed per row (solar position
    depends on the actual prediction time, not just the static path)."""
    rows = []
    for r in labels.select(
        "window_start", "midpoint_lat", "midpoint_lon",
        "tx_control_lat", "tx_control_lon", "rx_control_lat", "rx_control_lon",
    ).iter_rows(named=True):
        # tx/rx zenith use the terminus's own field-center lat/lon; the
        # frame passed in carries midpoint/control points but not the raw
        # tx/rx lat/lon, so callers building the full matrix (Task 7) pass
        # tx_lat/tx_lon/rx_lat/rx_lon through geometry first if tx/rx zenith
        # (as opposed to control-point zenith) is wanted per-row -- for this
        # module, tx/rx zenith are computed at the control points nearest
        # each terminus, which is the physically relevant point for D-layer
        # absorption at that end of the path anyway.
        w = r["window_start"]
        tx_z = solar_zenith_deg(r["tx_control_lat"], r["tx_control_lon"], w)
        rx_z = solar_zenith_deg(r["rx_control_lat"], r["rx_control_lon"], w)
        mid_z = solar_zenith_deg(r["midpoint_lat"], r["midpoint_lon"], w)
        # path_daylight_fraction: crude proxy = fraction of {tx, rx, mid} in daylight
        frac = sum(z < 90.0 for z in (tx_z, rx_z, mid_z)) / 3.0
        hrs = _hours_since_terminator(r["midpoint_lat"], r["midpoint_lon"], w)
        rows.append((tx_z, rx_z, mid_z, tx_z, rx_z, frac, hrs))
    solar = pl.DataFrame(
        rows,
        schema=["tx_solar_zenith", "rx_solar_zenith", "midpoint_solar_zenith",
                "tx_control_solar_zenith", "rx_control_solar_zenith",
                "path_daylight_fraction", "midpoint_hours_since_terminator"],
        orient="row",
    )
    return pl.concat([labels, solar], how="horizontal")
