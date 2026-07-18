"""ITU-R P.533 baseline: wraps the vendored ITURHFProp binary (baselines/p533).

Scores field-center -> field-center paths (ARCHITECTURE.md sec 5 M-1,
midpoint-to-midpoint), reusing propagation.data.geo.grid_to_latlon for the
field-center math. Standalone: no cqdx dependency (README boundaries).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
# semantics (docs/SPEC-labeling.md sec 4.4). txpower 100 W matches the
# snr_ft8eq 100 W power reference used in propagation.features.labels
# (pwr_offset relative to 50 dBm). Units of Path.txpower: verify (kW vs
# dBkW), see verification list #3.
_TXPOWER = "0.1"


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
    """Run ITURHFProp for one path/hour/month/SSN. `hour` is 0-23 UTC."""
    freq = BAND_FREQ_MHZ[band]
    card = render_input_card(
        tx_lat=tx_lat, tx_lon=tx_lon, rx_lat=rx_lat, rx_lon=rx_lon,
        month=month, hour_utc=hour, ssn=ssn, freq_mhz=freq, data_dir=_data_dir(),
    )
    # NB: intentionally not `with tempfile.TemporaryDirectory()` — that would
    # delete the directory (including the rendered input card) before this
    # function returns to the caller, on every abrupt-completion path
    # (including `return`), which is exactly the mid-flight state a caller
    # (or a test double for subprocess.run) may need to inspect for
    # debugging. The card/report are a few KB of text; leaving them under
    # the OS temp dir for normal OS-level cleanup is cheap and aids
    # reproducing a failing ITURHFProp run.
    td = Path(tempfile.mkdtemp(prefix="p533-"))
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
