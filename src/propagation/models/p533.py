"""ITU-R P.533 baseline: wraps the vendored ITURHFProp binary (baselines/p533).

Scores field-center -> field-center paths (ARCHITECTURE.md sec 5 M-1,
midpoint-to-midpoint), reusing propagation.data.geo.grid_to_latlon for the
field-center math. Standalone: no cqdx dependency (README boundaries).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import polars as pl

from propagation.data.geo import grid_to_latlon
from propagation.models.p533_build import binary_path, repo_root

# Primary digital-activity (FT8) dial frequency per band, MHz — the
# frequency at which most label-generating spots occur, hence the honest
# frequency to score P.533 at.
BAND_FREQ_MHZ: dict[str, float] = {
    "160m": 1.840, "80m": 3.573, "60m": 5.357, "40m": 7.074,
    "30m": 10.136, "20m": 14.074, "17m": 18.100, "15m": 21.074,
    "12m": 24.915, "10m": 28.074, "6m": 50.313,
}

# ITURHFProp hour convention: 1..24 UTC in the input card (verify against
# vendored README — see plan "Execution-time verification list" #2).
_HOUR_OFFSET = 1

# BCR = P(SNR >= SNRr). SNRr = -21 dB in BW 2500 Hz is the FT8 decode
# threshold, matching the labels' "open for the most sensitive active mode"
# semantics (docs/SPEC-labeling.md sec 4.4). Path.txpower is dB(1 kW), per
# the vendored upstream's own docs (baselines/p533/upstream/README.md:86 and
# the `txpower` field comment in
# baselines/p533/upstream/P533/Src/P533/P533.h) — NOT linear kW. -10.0
# dB(kW) = 100 W, matching the snr_ft8eq 100 W power reference used in
# propagation.features.labels (pwr_offset relative to 50 dBm).
_TXPOWER = "-10.0"


@dataclass(frozen=True)
class P533Result:
    reliability_pct: float  # BCR, 0-100
    snr_db: float           # median SNR re the card's BW


def render_input_card(
    tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
    month: int, hour_utc: int, ssn: float, freq_mhz: float, data_dir: Path,
) -> str:
    if not 1 <= month <= 12:
        raise ValueError(f"month {month}")
    if not 0 <= hour_utc <= 23:
        raise ValueError(f"hour_utc {hour_utc}")
    rx = f"LL.lat {rx_lat:.4f}\nLL.lng {rx_lon:.4f}\n" \
         f"LR.lat {rx_lat:.4f}\nLR.lng {rx_lon:.4f}\n" \
         f"UL.lat {rx_lat:.4f}\nUL.lng {rx_lon:.4f}\n" \
         f"UR.lat {rx_lat:.4f}\nUR.lng {rx_lon:.4f}\n"
    return (
        f'PathName "propagation-m1"\n'
        f'PathTXName "tx"\n'
        f"Path.L_tx.lat {tx_lat:.4f}\n"
        f"Path.L_tx.lng {tx_lon:.4f}\n"
        f'PathRXName "rx"\n'
        f"Path.L_rx.lat {rx_lat:.4f}\n"
        f"Path.L_rx.lng {rx_lon:.4f}\n"
        f'TXAntFilePath "ISOTROPIC"\n'
        f'RXAntFilePath "ISOTROPIC"\n'
        f"Path.year 2026\n"                      # inert: SSN drives the model
        f"Path.month {month}\n"
        f"Path.hour {hour_utc + _HOUR_OFFSET}\n"
        f"Path.SSN {ssn:.0f}\n"
        f"Path.frequency {freq_mhz:.4f}\n"
        f"Path.txpower {_TXPOWER}\n"
        f"Path.BW 2500.0\n"
        f"Path.SNRr -21.0\n"
        f'Path.SorL "SHORTPATH"\n'
        f'Path.ManMadeNoise "RURAL"\n'
        f'RptFilFormat "RPT_BCR | RPT_SNR"\n'
        f"{rx}"
        f'DataFilePath "{data_dir}/"\n'
    )


# Real ITURHFProp reports do not have a CSV header row above the data row.
# Instead, a "Data Format" section documents each column's 1-based position
# as a line like `Column 34: SNR - Median signal-to-noise ratio (dB)`, and
# the values appear later as a single comma-separated data row inside a
# "Calculated Parameters" section. Match the column *name token* exactly
# (not startswith) since e.g. `SNRXXp` (col 37) also starts with "SNR" but
# is a distinct column from `SNR` (col 34).
_COLUMN_DEF_RE = re.compile(r"^Column\s+(\d+):\s*(\S+)\s*-", re.MULTILINE)


def parse_report(text: str) -> P533Result:
    """Parse an ITURHFProp report: read the `Column NN: NAME - ...` lines in
    the Data Format section to locate the BCR and SNR columns by name (not
    position), then read the first data row in the Calculated Parameters
    section. Column order/count is not assumed."""
    columns: dict[str, int] = {}
    for m in _COLUMN_DEF_RE.finditer(text):
        columns[m.group(2).upper()] = int(m.group(1)) - 1  # 0-based

    bcr_idx = columns.get("BCR")
    snr_idx = columns.get("SNR")
    if bcr_idx is None or snr_idx is None:
        raise ValueError("no header row naming BCR and SNR columns found")

    in_data = False
    for ln in text.splitlines():
        stripped = ln.strip()
        if not in_data:
            if "Calculated Parameters" in stripped and "End" not in stripped:
                in_data = True
            continue
        if not stripped or "Calculated Parameters" in stripped:
            continue
        vals = [v.strip() for v in stripped.split(",")]
        if max(bcr_idx, snr_idx) >= len(vals):
            continue
        try:
            return P533Result(
                reliability_pct=float(vals[bcr_idx]),
                snr_db=float(vals[snr_idx]),
            )
        except ValueError:
            continue
    raise ValueError("BCR/SNR header found but no parseable data row")


def _data_dir() -> Path:
    # Coefficient files ship inside the vendored tree; locate the directory
    # containing them (recorded in baselines/p533/PROVENANCE.md).
    return repo_root() / "baselines" / "p533" / "upstream" / "ITURHFProp" / "Data"


def p533_score(
    tx_lat: float, tx_lon: float, rx_lat: float, rx_lon: float,
    band: str, month: int, hour: int, ssn: float,
) -> P533Result:
    """Run ITURHFProp for one path/hour/month/SSN. `hour` is 0-23 UTC.

    ITU-R P.533 is an HF model: the vendored engine's own ValidatePath.c
    enforces `1.0 <= Path.frequency <= 30.0` (MHz) and returns a clean error
    for anything outside that range -- except it doesn't actually reach that
    check cleanly for every input; on this vendored build, an out-of-range
    frequency (e.g. 6m's 50.313 MHz, which BAND_FREQ_MHZ carries because
    other bands need it, not because P.533 supports 6m) crashes the binary
    with SIGSEGV instead of a reported error. Validate client-side so a band
    outside P.533's physical domain fails as an abstain (ValueError, caught
    by P533Model), not a crash.
    """
    freq = BAND_FREQ_MHZ[band]
    if not 1.0 <= freq <= 30.0:
        raise ValueError(
            f"band {band!r} ({freq} MHz) is outside ITU-R P.533's valid "
            "frequency range (1.0-30.0 MHz, HF only); the vendored binary "
            "segfaults rather than erroring cleanly on out-of-range input"
        )
    card = render_input_card(
        tx_lat=tx_lat, tx_lon=tx_lon, rx_lat=rx_lat, rx_lon=rx_lon,
        month=month, hour_utc=hour, ssn=ssn, freq_mhz=freq, data_dir=_data_dir(),
    )
    # This is the core scoring primitive for the whole M1 P.533 baseline and
    # will be called thousands of times in batch eval jobs (ROADMAP.md /
    # ARCHITECTURE.md) — the temp dir must not leak, so it is cleaned up on
    # every exit path via the context manager.
    with tempfile.TemporaryDirectory(prefix="p533-") as td_name:
        td = Path(td_name)
        in_path = td / "in.txt"
        out_path = td / "out.txt"
        in_path.write_text(card)
        env = dict(os.environ)
        bin_dir = str(binary_path().parent)
        # shared libs live next to the binary (build.sh); cover both loaders
        env["LD_LIBRARY_PATH"] = bin_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = bin_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
        proc = subprocess.run(
            [str(binary_path()), str(in_path), str(out_path)],
            capture_output=True, text=True, timeout=60, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"iturhfprop exit {proc.returncode}: {proc.stderr[:500]}"
            )
        return parse_report(out_path.read_text())


SWPC_SOLAR_CYCLE_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
)


def _fetch_solar_cycle(cache_dir: Path) -> list[dict]:
    cache = cache_dir / "swpc_solar_cycle.json"
    if cache.exists():
        return json.loads(cache.read_text())
    resp = httpx.get(SWPC_SOLAR_CYCLE_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def ssn_by_month(months: list[str], cache_dir: Path) -> dict[str, float]:
    """Monthly SSN for 'YYYY-MM' keys: smoothed where published (< 0 means
    not yet available in the SWPC series), observed otherwise."""
    rows = {r["time-tag"]: r for r in _fetch_solar_cycle(cache_dir)}
    out: dict[str, float] = {}
    for m in months:
        if m not in rows:
            raise KeyError(f"no SWPC SSN entry for {m}")
        r = rows[m]
        smoothed = r.get("smoothed_ssn", -1.0)
        out[m] = float(smoothed) if smoothed is not None and smoothed >= 0 else float(r["ssn"])
    return out


_CACHE_SCHEMA = {
    "tx_field": pl.Utf8, "rx_field": pl.Utf8, "band": pl.Utf8,
    "month": pl.Int64, "hour": pl.Int64, "ssn_round": pl.Int64,
    "p_open": pl.Float64,
}


def _load_p533_cache(cache_path: Path) -> dict[tuple, float | None]:
    if not cache_path.exists():
        return {}
    df = pl.read_parquet(cache_path)
    return {
        (r["tx_field"], r["rx_field"], r["band"], r["month"], r["hour"], r["ssn_round"]): r["p_open"]
        for r in df.iter_rows(named=True)
    }


def _save_p533_cache(cache_path: Path, memo: dict[tuple, float | None]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"tx_field": k[0], "rx_field": k[1], "band": k[2], "month": k[3], "hour": k[4],
         "ssn_round": k[5], "p_open": v}
        for k, v in memo.items()
    ]
    pl.DataFrame(rows, schema=_CACHE_SCHEMA).write_parquet(cache_path)


class P533Model:
    """Predictor over ITURHFProp: field-center -> field-center,
    p_open = BCR/100. Deterministic per (path, band, month, hour, SSN), so
    results are memoized on that key; SSN is rounded to an integer for the
    memo key (finer variation is far below P.533's fidelity). Matches
    ClimatologyModel's predict(cells) -> cells+p_open shape by convention,
    not by a shared protocol class.

    `predict` fans unique (path, band, month, hour, SSN) keys out across a
    thread pool -- each one is a ~50ms subprocess call to the vendored
    binary, and real WSPRnet-scale eval sets carry hundreds of thousands of
    unique keys per month, so a serial loop is a many-hour bottleneck.
    subprocess.run releases the GIL while the child runs, so threads (not
    processes) already parallelize the wall-clock-dominant part.

    `cache_path`, if given, persists the key->p_open memo (including
    engine-failure abstains, since those are just as deterministic per key)
    to a parquet file on disk, loaded at construction and rewritten after
    any new keys are computed. This lets a later `P533Model` instance --
    another horizon in the same eval_m3.py sweep, or a rerun in a later
    session -- reuse scores instead of re-invoking the subprocess for keys
    already seen. Single-writer only: concurrent processes sharing the same
    cache_path will race on the final write, so don't point two parallel
    eval runs at the same cache file."""

    def __init__(
        self, ssn_by_month: dict[str, float], max_workers: int | None = None,
        cache_path: Path | None = None,
    ):
        self._ssn = ssn_by_month          # "YYYY-MM" -> SSN
        self._cache_path = cache_path
        self._memo: dict[tuple, float | None] = (
            _load_p533_cache(cache_path) if cache_path else {}
        )
        self._max_workers = max_workers or os.cpu_count() or 8

    def _resolve_key(self, tx_field: str, rx_field: str, band: str,
                      month_key: str, month: int, hour: int
                      ) -> tuple[tuple, float] | None:
        ssn = self._ssn.get(month_key)
        if ssn is None:
            return None                   # abstain: no SSN for that month
        return (tx_field, rx_field, band, month, hour, round(ssn)), ssn

    @staticmethod
    def _compute(key: tuple, tx_field: str, rx_field: str, band: str,
                 month: int, hour: int, ssn: float) -> tuple[tuple, float | None]:
        tx_lat, tx_lon = grid_to_latlon(tx_field)
        rx_lat, rx_lon = grid_to_latlon(rx_field)
        try:
            res = p533_score(tx_lat, tx_lon, rx_lat, rx_lon, band, month, hour, ssn)
            return key, res.reliability_pct / 100.0
        except (RuntimeError, ValueError):
            return key, None              # abstain on engine failure

    def predict(self, labels: pl.DataFrame) -> pl.DataFrame:
        row_keys: list[tuple | None] = []
        pending: dict[tuple, tuple] = {}
        for ws, tx, rx, band in labels.select(
            "window_start", "tx_field", "rx_field", "band"
        ).iter_rows():
            month_key = f"{ws.year:04d}-{ws.month:02d}"
            resolved = self._resolve_key(tx, rx, band, month_key, ws.month, ws.hour)
            if resolved is None:
                row_keys.append(None)
                continue
            key, ssn = resolved
            row_keys.append(key)
            if key not in self._memo and key not in pending:
                pending[key] = (tx, rx, band, ws.month, ws.hour, ssn)

        if pending:
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = [
                    pool.submit(self._compute, key, *args)
                    for key, args in pending.items()
                ]
                for fut in as_completed(futures):
                    key, value = fut.result()
                    self._memo[key] = value
            if self._cache_path is not None:
                _save_p533_cache(self._cache_path, self._memo)

        p = [self._memo[key] if key is not None else None for key in row_keys]
        return labels.with_columns(pl.Series("p_open", p, dtype=pl.Float64))
