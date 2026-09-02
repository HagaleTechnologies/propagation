"""RBN (Reverse Beacon Network) extractor -- CW/RTTY ground truth (PRO-8).

Format verified empirically against a real archive (2024-01-15) since RBN's
own raw-data page (reversebeacon.net/raw_data/) documents the download URL
but not the column layout:

    callsign,de_pfx,de_cont,freq,band,dx,dx_pfx,dx_cont,mode,db,date,speed,tx_mode

`callsign` is the skimmer (receiver, our `de_call`); `dx` is the spotted
station (our `dx_call`). `band` is already in this repo's band-name form
(e.g. "40m") -- no code-to-band mapping needed, unlike WSPRnet. `mode` is
RBN's spot-type field (e.g. "CQ"), not the RF mode; `tx_mode` (CW/RTTY/PSK…)
is what feeds the common schema's `mode` column. `freq` is kHz; `db` is SNR.

**Location gap**: unlike WSPRnet/PSKReporter, RBN's history archive carries
no Maidenhead grid for either station -- only DXCC prefix + continent. This
extractor resolves `{dx,de}_{lat,lon}` via a DXCC callsign lookup (`dxentity`,
BSD-3-Clause; its underlying cty.plist data is fetched from
country-files.com at first use and cached ~/.local/cty for a week -- a
network dependency this extractor inherits, separate from this repo's own
license). That resolution is **country-centroid, not station-level** --
materially coarser than a 4-char Maidenhead grid (thousands of km vs. tens of
km). Every RBN row's distance_km/bearing_deg/midpoint_geomag_lat feature is
therefore lower-precision than the WSPRnet baseline. This is a real quality
gap worth weighing before claiming "M2's bar still clears with RBN
included" (PRO-8's second acceptance scenario) -- it may not, or may need
per-source uncertainty weighting downstream, which does not exist yet.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import polars as pl

from propagation.data.dedup import dedup_spots
from propagation.data.hygiene import is_qualifying_spot
from propagation.data.schema import SPOT_SCHEMA

RBN_ARCHIVE_URL = "https://data.reversebeacon.net/rbn_history/{date:%Y%m%d}.zip"

LocationResolver = Callable[[str], tuple[float, float] | None]

_SKIMMER_SUFFIX_RE = re.compile(r"-\d+$")


def strip_skimmer_suffix(call: str) -> str:
    """RBN skimmer callsigns carry a `-N` receiver-instance suffix
    (multiple SDRs at one station, e.g. `KM3T-1`/`KM3T-2`/`KM3T-3`) that
    hygiene.is_valid_callsign's amateur-callsign regex rejects outright.
    Stripping it collapses siblings to one physical station -- the right
    granularity for a location-based feature, and the only way most RBN
    skimmer rows survive is_qualifying_spot at all."""
    return _SKIMMER_SUFFIX_RE.sub("", call)


def download_rbn_archive(
    date: dt.date, dest_path: Path, client: httpx.Client | None = None
) -> Path:
    url = RBN_ARCHIVE_URL.format(date=date)
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


def _dxentity_resolver() -> LocationResolver:
    """Lazily-constructed default resolver -- only touches the network/cache
    (country-files.com cty.plist, dxentity's own ~/.local/cty cache) if a
    caller actually needs one, so importing/testing this module never does."""
    from DXEntity import DXCC

    dxcc = DXCC()

    def _resolve(call: str) -> tuple[float, float] | None:
        try:
            rec = dxcc.lookup(call)
        except KeyError:
            return None
        return rec.latitude, rec.longitude

    return _resolve


def parse_rbn_row(row: dict, resolve_location: LocationResolver) -> dict | None:
    try:
        ts = dt.datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        freq_khz = float(row["freq"])
        snr = int(row["db"])
    except (KeyError, ValueError):
        return None
    band = row.get("band")
    mode = row.get("tx_mode")
    dx_call = strip_skimmer_suffix((row.get("dx") or "").strip().upper())
    de_call = strip_skimmer_suffix((row.get("callsign") or "").strip().upper())
    if not band or not mode or not dx_call or not de_call:
        return None

    dx_loc = resolve_location(dx_call)
    de_loc = resolve_location(de_call)

    return {
        "source": "rbn",
        "ts": ts,
        "band": band,
        "mode": mode.strip().upper(),
        "freq_hz": round(freq_khz * 1000),
        "dx_call": dx_call,
        "de_call": de_call,
        "dx_grid": None,
        "de_grid": None,
        "dx_lat": dx_loc[0] if dx_loc else None,
        "dx_lon": dx_loc[1] if dx_loc else None,
        "de_lat": de_loc[0] if de_loc else None,
        "de_lon": de_loc[1] if de_loc else None,
        "snr_db": snr,
        "tx_dbm": None,
    }


@dataclass
class ExtractResult:
    spots: pl.DataFrame
    n_lines_read: int
    n_parsed: int
    n_qualifying: int
    rejection_counts: dict[str, int] = field(default_factory=dict)


def extract_rbn(
    archive_path: Path,
    band: str,
    resolve_location: LocationResolver | None = None,
    chunk_size: int = 200_000,
) -> ExtractResult:
    """Same chunked-flush memory bound as extract_wsprnet (see that module's
    docstring for the OOM history this pattern avoids) -- RBN daily files are
    smaller than a WSPRnet monthly archive, but contest-day volume (300k+
    spots in a day per reversebeacon.net's own docs) still warrants it.
    `resolve_location` defaults to a DXCC prefix lookup (network on first
    call); inject a fake for tests."""
    resolve_location = resolve_location or _dxentity_resolver()
    rejection_counts: dict[str, int] = {}
    n_lines_read = 0
    n_parsed = 0

    with zipfile.ZipFile(archive_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"{archive_path}: no .csv member found in RBN archive")
        raw_text = zf.read(names[0]).decode("ascii", errors="replace")

    with tempfile.TemporaryDirectory(prefix="rbn-extract-") as td_name:
        td = Path(td_name)
        rows: list[dict] = []
        chunk_paths: list[Path] = []

        def _flush() -> None:
            if rows:
                chunk_path = td / f"chunk-{len(chunk_paths):06d}.parquet"
                pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")}).write_parquet(chunk_path)
                chunk_paths.append(chunk_path)
                rows.clear()

        reader = csv.DictReader(io.StringIO(raw_text))
        for raw_row in reader:
            n_lines_read += 1
            if raw_row.get("band") != band:
                continue
            parsed = parse_rbn_row(raw_row, resolve_location)
            if parsed is None:
                continue
            n_parsed += 1
            ok, reason = is_qualifying_spot(parsed)
            if not ok:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            rows.append(parsed)
            if len(rows) >= chunk_size:
                _flush()
        _flush()

        if not chunk_paths:
            spots = pl.DataFrame(schema=SPOT_SCHEMA)
        else:
            spots = pl.concat([pl.read_parquet(p) for p in chunk_paths], how="vertical_relaxed")
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
