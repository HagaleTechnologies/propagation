"""cqdx R2 archive row parsing (PRO-10).

cqdx's private Cloudflare R2 bucket (`cqdx-archive`) stores mirrored spot
reports under `raw/YYYY-MM-DD/HH/{ulid}.ndjson.gz` -- one gzip'd NDJSON
object per hour, each line an enriched camelCase spot record (dispensa
Q-0062, merged answer; recorded in ARCHITECTURE.md sec 3.2 and
thoughts/shared/research/2026-10-07-propagation-codebase-overview.md:175).

This module maps those records onto the shared SPOT_SCHEMA with
`source='cqdx'` (archive PROVENANCE, not payload origin -- a PSKReporter
payload mirrored by cqdx is still a cqdx-sourced row for provenance
purposes, and dedup_spots gives it the lowest priority).

Only confirmed AUTOMATIC reports are admitted: payload `source` in
{"pskreporter", "rbn"}. Tony's direction on PRO-10 (2026-09-03):
`source:'user'` rows are filtered out when sensor-equivalent data is
wanted, and DX-cluster records' provenance does not guarantee an automatic
report, so they are excluded too. (Note this filter is on the PAYLOAD's
source; the emitted row's `source` column is always 'cqdx'.)

The private archive is an optional accelerant, never a dependency: nothing
else in this package imports this module's R2 transport (which lives in
scripts/extract_cqdx.py behind the `cqdx-r2` extra); only these pure
parsing helpers are importable without boto3.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import math
from dataclasses import dataclass, field
from typing import Iterator

import polars as pl

from propagation.data.geo import grid_to_latlon, latlon_to_grid
from propagation.data.hygiene import is_qualifying_spot, normalize_grid
from propagation.data.schema import SPOT_SCHEMA

# Payload `source` values admitted as sensor-equivalent automatic reports.
AUTOMATIC_PAYLOAD_SOURCES: frozenset[str] = frozenset({"pskreporter", "rbn"})


@dataclass
class ArchiveDayParseResult:
    """Rows plus per-reason counts for one archive object (or day).

    Rejection counts never log payloads -- the archive is private data
    adjacent to PII-bearing logbooks (Q-0062), so counts only, no rows.
    """

    rows: list[dict] = field(default_factory=list)
    rejection_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_qualifying(self) -> int:
        return len(self.rows)


def _valid_latlon(lat: object, lon: object) -> tuple[float, float] | None:
    """None unless (lat, lon) is a finite, in-range coordinate pair."""
    if lat is None or lon is None:
        return None
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        return None
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return None
    return lat_f, lon_f


def parse_archive_payload(payload: dict) -> tuple[dict | None, str | None]:
    """Map one enriched camelCase archive record to a SPOT_SCHEMA row dict
    (source='cqdx'). Returns (row, None) or (None, rejection_reason).

    Order of checks mirrors the other extractors: structural parse first
    (timestamp/frequency), then payload-source admission, then
    is_qualifying_spot on the ORIGINAL grid values (so a grid that fails
    hygiene is rejected even when coordinates could have papered over it),
    then grid normalization / derivation.
    """
    if not isinstance(payload, dict):
        return None, "malformed_record"

    payload_source = str(payload.get("source") or "").strip().lower()
    if payload_source not in AUTOMATIC_PAYLOAD_SOURCES:
        return None, "non_automatic_source"

    raw_ts = payload.get("timestamp")
    try:
        ts_epoch = int(raw_ts)
    except (TypeError, ValueError):
        return None, "invalid_timestamp"
    try:
        ts = dt.datetime.fromtimestamp(ts_epoch, tz=dt.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None, "invalid_timestamp"

    raw_freq = payload.get("frequency")
    try:
        freq_hz: int | None = int(raw_freq)
    except (TypeError, ValueError):
        freq_hz = None

    band = (payload.get("band") or None)
    mode = (payload.get("mode") or None)
    if band is not None:
        band = str(band).strip().lower() or None
    if mode is not None:
        mode = str(mode).strip().upper() or None

    dx_call = (payload.get("dxCall") or "").strip().upper() or None
    de_call = (payload.get("deCall") or "").strip().upper() or None

    def _grid(v: object) -> str | None:
        return str(v).strip().upper() or None if v is not None else None

    dx_grid_raw = _grid(payload.get("dxGrid"))
    de_grid_raw = _grid(payload.get("deGrid"))
    dx_ll = _valid_latlon(payload.get("dxLat"), payload.get("dxLon"))
    de_ll = _valid_latlon(payload.get("deLat"), payload.get("deLon"))

    raw_sn = payload.get("snr")
    try:
        snr: int | None = int(raw_sn) if raw_sn is not None else None
    except (TypeError, ValueError):
        snr = None

    row = {
        "source": "cqdx",
        "ts": ts,
        "band": band,
        "mode": mode,
        "freq_hz": freq_hz,
        "dx_call": dx_call,
        "de_call": de_call,
        "dx_grid": dx_grid_raw,
        "de_grid": de_grid_raw,
        "dx_field": None,
        "de_field": None,
        "dx_lat": dx_ll[0] if dx_ll else None,
        "dx_lon": dx_ll[1] if dx_ll else None,
        "de_lat": de_ll[0] if de_ll else None,
        "de_lon": de_ll[1] if de_ll else None,
        "snr_db": snr,
        "tx_dbm": None,
    }

    ok, reason = is_qualifying_spot(row)
    if not ok:
        return None, reason

    dx_grid = normalize_grid(dx_grid_raw)
    de_grid = normalize_grid(de_grid_raw)
    if dx_grid is not None:
        row["dx_grid"] = dx_grid
        row["dx_lat"], row["dx_lon"] = grid_to_latlon(dx_grid)
    elif dx_ll is not None:
        row["dx_grid"] = latlon_to_grid(dx_ll[0], dx_ll[1], precision=4)
    if de_grid is not None:
        row["de_grid"] = de_grid
        row["de_lat"], row["de_lon"] = grid_to_latlon(de_grid)
    elif de_ll is not None:
        row["de_grid"] = latlon_to_grid(de_ll[0], de_ll[1], precision=4)

    return row, None


def parse_archive_object(data: bytes) -> ArchiveDayParseResult:
    """Decode one `raw/.../{ulid}.ndjson.gz` object body: gzip NDJSON,
    decoded one NDJSON line at a time. Malformed JSON lines
    and non-admitted payload sources count as rejections; a gzip integrity
    failure raises (the caller aborts the whole extraction -- a truncated
    object must never publish a silently-partial day)."""
    result = ArchiveDayParseResult()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        for line in gz:  # decoded one NDJSON line at a time, never buffered whole
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                result.rejection_counts["malformed_json"] = (
                    result.rejection_counts.get("malformed_json", 0) + 1
                )
                continue
            row, reason = parse_archive_payload(payload)
            if row is None:
                result.rejection_counts[reason or "unknown"] = (
                    result.rejection_counts.get(reason or "unknown", 0) + 1
                )
                continue
            result.rows.append(row)
    return result


def archive_rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    """Exact-type SPOT_SCHEMA frame from parsed archive rows (empty frame
    with the declared schema when no rows survived)."""
    if not rows:
        return pl.DataFrame(schema=SPOT_SCHEMA)
    return pl.DataFrame(rows, schema=SPOT_SCHEMA)


def iter_requested_days(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    """Inclusive start, exclusive end, in UTC archive partition dates."""
    day = start
    while day < end:
        yield day
        day += dt.timedelta(days=1)
