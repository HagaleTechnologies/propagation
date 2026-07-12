from __future__ import annotations

import datetime as dt
import gzip
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import polars as pl

from propagation.data.dedup import dedup_spots
from propagation.data.hygiene import is_qualifying_spot
from propagation.data.schema import SPOT_SCHEMA

WSPR_BAND_CODE_TO_BAND: dict[int, str] = {
    1: "160m", 3: "80m", 5: "60m", 7: "40m", 10: "30m",
    14: "20m", 18: "17m", 21: "15m", 24: "12m", 28: "10m", 50: "6m",
}

WSPRNET_ARCHIVE_URL = "http://www.wsprnet.org/archive/wsprspots-{year:04d}-{month:02d}.csv.gz"


def download_wsprnet_archive(
    year: int, month: int, dest_path: Path, client: httpx.Client | None = None
) -> Path:
    url = WSPRNET_ARCHIVE_URL.format(year=year, month=month)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    client = client or httpx.Client(timeout=120.0)
    try:
        with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
    finally:
        if owns_client:
            client.close()
    return dest_path


def parse_wsprnet_row(line: str) -> dict | None:
    parts = line.rstrip("\n").split(",")
    if len(parts) < 13:
        return None
    try:
        ts_epoch = int(parts[1])
        band_code = int(parts[12])
        snr = int(parts[4])
        tx_dbm = int(parts[8])
    except ValueError:
        return None
    band = WSPR_BAND_CODE_TO_BAND.get(band_code)
    if band is None:
        return None
    try:
        freq_hz = round(float(parts[5]) * 1_000_000)
    except ValueError:
        freq_hz = None
    return {
        "source": "wsprnet",
        "ts": ts_epoch,
        "band": band,
        "mode": "WSPR",
        "freq_hz": freq_hz,
        "dx_call": parts[6].strip().upper(),
        "de_call": parts[2].strip().upper(),
        "dx_grid": (parts[7].strip().upper() or None),
        "de_grid": (parts[3].strip().upper() or None),
        "dx_lat": None,
        "dx_lon": None,
        "de_lat": None,
        "de_lon": None,
        "snr_db": snr,
        "tx_dbm": tx_dbm,
    }


@dataclass
class ExtractResult:
    spots: pl.DataFrame
    n_lines_read: int
    n_parsed: int
    n_qualifying: int
    rejection_counts: dict[str, int] = field(default_factory=dict)


def extract_wsprnet(archive_path: Path, band: str) -> ExtractResult:
    rows: list[dict] = []
    rejection_counts: dict[str, int] = {}
    n_lines_read = 0
    n_parsed = 0

    with gzip.open(archive_path, "rt", encoding="ascii", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines_read += 1
            parsed = parse_wsprnet_row(line)
            if parsed is None or parsed["band"] != band:
                continue
            n_parsed += 1
            parsed["ts"] = dt.datetime.fromtimestamp(parsed["ts"], tz=dt.timezone.utc)
            ok, reason = is_qualifying_spot(parsed)
            if not ok:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            rows.append(parsed)

    if not rows:
        spots = pl.DataFrame(schema=SPOT_SCHEMA)
    else:
        spots = pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})
        for col in SPOT_SCHEMA:
            if col not in spots.columns:
                spots = spots.with_columns(pl.lit(None).alias(col))
        spots = dedup_spots(spots)

    return ExtractResult(
        spots=spots,
        n_lines_read=n_lines_read,
        n_parsed=n_parsed,
        n_qualifying=spots.height,
        rejection_counts=rejection_counts,
    )
