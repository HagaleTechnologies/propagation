"""Historical space-weather features, sourced from NASA's OMNI2 hourly
archive (spdf.gsfc.nasa.gov) -- verified live for both 2014 and 2024 during
M2 planning. This is the TRAINING-time source; live serving (M4) uses
SWPC's real-time nowcast feeds instead (a rolling window of hours-to-days,
useless for historical training) via a separate fetcher this module does
not implement -- the feature computation in features/spaceweather.py is
shared, only the raw fetch differs per environment.

Definitive vs estimated: unlike propagation.eval.stratify's definitive-Kp
(eval-only per docs/SPEC-labeling.md sec 6), OMNI2's Kp here IS used as a
training feature. For historical months, OMNI2's values have been fixed for
years and reflect no future information relative to a model trained today;
the real train/serve consistency concern is that live serving's nowcast Kp
will be less precise than this historical reprocessed value -- documented
here, not solved by this module (M4's concern).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl

OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"

_FILL = {
    "bz_gsm": 999.9, "solar_wind_speed": 9999.0, "kp": 99, "dst": 99999, "f107": 999.9,
}


def _decode_omni2_kp(raw: int) -> float | None:
    """OMNI2 word 39: tens digit = whole Kp, units digit in {0,3,7} = "o"/"+"/"-"
    thirds (e.g. 33="3+"=3+1/3, 40="4o"=4.0, 57="5+"=5+2/3). Units digit 9 is
    the fill value (no data)."""
    if raw == _FILL["kp"]:
        return None
    tens, units = divmod(raw, 10)
    offset = {0: 0.0, 3: 1 / 3, 7: 2 / 3}.get(units)
    if offset is None:
        return None
    return tens + offset


def _parse_omni2(text: str, year: int) -> pl.DataFrame:
    rows = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        f = ln.split()
        day_of_year = int(f[1])
        hour = int(f[2])
        from datetime import datetime, timedelta, timezone
        time = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1, hours=hour)
        kp_raw = int(f[38])
        f107 = float(f[50])
        bz_gsm = float(f[16])
        speed = float(f[24])
        dst = float(f[40])
        rows.append((
            time,
            _decode_omni2_kp(kp_raw),
            None if f107 == _FILL["f107"] else f107,
            None if bz_gsm == _FILL["bz_gsm"] else bz_gsm,
            None if speed == _FILL["solar_wind_speed"] else speed,
            None if dst == _FILL["dst"] else dst,
        ))
    return pl.DataFrame(
        rows, schema=["time", "kp", "f107", "bz_gsm", "solar_wind_speed", "dst"], orient="row",
    ).with_columns(pl.col("time").cast(pl.Datetime("us", "UTC")))


def fetch_omni2_year(year: int, cache_dir: Path) -> pl.DataFrame:
    cache = Path(cache_dir) / f"omni2_{year}.dat"
    if not cache.exists():
        resp = httpx.get(OMNI2_URL.format(year=year), timeout=60, follow_redirects=True)
        resp.raise_for_status()
        cache_dir_p = Path(cache_dir)
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
    return _parse_omni2(cache.read_text(), year)


def fetch_omni2_range(start_year: int, end_year: int, cache_dir: Path) -> pl.DataFrame:
    """Inclusive of both start_year and end_year."""
    frames = [fetch_omni2_year(y, cache_dir) for y in range(start_year, end_year + 1)]
    return pl.concat(frames)
