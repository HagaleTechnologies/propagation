from __future__ import annotations

import datetime as dt
import gzip
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import polars as pl

from propagation.data.dedup import dedup_spots
from propagation.data.hygiene import is_qualifying_spot
from propagation.data.schema import SPOT_SCHEMA, normalize_spot_columns

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


def extract_wsprnet(archive_path: Path, band: str, chunk_size: int = 200_000) -> ExtractResult:
    # A full month's archive covers every band and holds tens of millions of
    # lines at real (2024+) WSPRnet volume; accumulating one Python dict per
    # qualifying row for the whole file before ever building a DataFrame
    # (the original approach) grows a many-GB list of Python objects -- the
    # actual cause of a 2026-07-20 OOM incident that hung the whole machine.
    # Flushing each chunk_size batch straight to a parquet file (rather than
    # keeping a growing list of in-memory DataFrame chunks) bounds resident
    # memory to ~one chunk for the entire scan: keeping all chunks resident
    # still re-accumulates the whole month's data in RAM over time, and
    # CPython's allocator doesn't reliably return freed dict/string memory
    # to the OS across hundreds of alloc/free cycles, so that approach's
    # RSS kept climbing for the whole scan and still triggered a second,
    # separate OOM near-miss even though no single chunk was ever large.
    rejection_counts: dict[str, int] = {}
    n_lines_read = 0
    n_parsed = 0

    with tempfile.TemporaryDirectory(prefix="wsprnet-extract-") as td_name:
        td = Path(td_name)
        rows: list[dict] = []
        chunk_paths: list[Path] = []

        def _flush() -> None:
            if rows:
                chunk_path = td / f"chunk-{len(chunk_paths):06d}.parquet"
                pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")}).write_parquet(chunk_path)
                chunk_paths.append(chunk_path)
                rows.clear()

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
                if len(rows) >= chunk_size:
                    _flush()
        _flush()

        if not chunk_paths:
            spots = pl.DataFrame(schema=SPOT_SCHEMA)
        else:
            spots = pl.concat([pl.read_parquet(p) for p in chunk_paths], how="vertical_relaxed")
        spots = normalize_spot_columns(spots)
        spots = dedup_spots(spots)

    return ExtractResult(
        spots=spots,
        n_lines_read=n_lines_read,
        n_parsed=n_parsed,
        n_qualifying=spots.height,
        rejection_counts=rejection_counts,
    )


def extract_wsprnet_bands(
    archive_path: Path, bands: list[str], chunk_size: int = 200_000
) -> dict[str, ExtractResult]:
    """Single-pass variant of extract_wsprnet for multiple bands: a full
    month's archive covers every band, so extracting N bands via N separate
    extract_wsprnet calls re-decompresses and re-scans the same file N
    times. This scans the archive once, bucketing qualifying rows per
    requested band, using the same chunked-flush memory bound as
    extract_wsprnet (see that function's docstring for the OOM history)."""
    bands_set = set(bands)
    rejection_counts: dict[str, dict[str, int]] = {b: {} for b in bands}
    n_lines_read = 0
    n_parsed: dict[str, int] = {b: 0 for b in bands}

    with tempfile.TemporaryDirectory(prefix="wsprnet-extract-") as td_name:
        td = Path(td_name)
        rows: dict[str, list[dict]] = {b: [] for b in bands}
        chunk_paths: dict[str, list[Path]] = {b: [] for b in bands}

        def _flush(band: str) -> None:
            if rows[band]:
                chunk_path = td / f"{band}-chunk-{len(chunk_paths[band]):06d}.parquet"
                pl.DataFrame(
                    rows[band], schema_overrides={"ts": pl.Datetime("us", "UTC")}
                ).write_parquet(chunk_path)
                chunk_paths[band].append(chunk_path)
                rows[band].clear()

        with gzip.open(archive_path, "rt", encoding="ascii", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                n_lines_read += 1
                parsed = parse_wsprnet_row(line)
                if parsed is None or parsed["band"] not in bands_set:
                    continue
                band = parsed["band"]
                n_parsed[band] += 1
                parsed["ts"] = dt.datetime.fromtimestamp(parsed["ts"], tz=dt.timezone.utc)
                ok, reason = is_qualifying_spot(parsed)
                if not ok:
                    rejection_counts[band][reason] = rejection_counts[band].get(reason, 0) + 1
                    continue
                rows[band].append(parsed)
                if len(rows[band]) >= chunk_size:
                    _flush(band)
        for b in bands:
            _flush(b)

        results: dict[str, ExtractResult] = {}
        for b in bands:
            if not chunk_paths[b]:
                spots = pl.DataFrame(schema=SPOT_SCHEMA)
            else:
                spots = pl.concat([pl.read_parquet(p) for p in chunk_paths[b]], how="vertical_relaxed")
            spots = normalize_spot_columns(spots)
            spots = dedup_spots(spots)
            results[b] = ExtractResult(
                spots=spots, n_lines_read=n_lines_read, n_parsed=n_parsed[b],
                n_qualifying=spots.height, rejection_counts=rejection_counts[b],
            )
        return results
