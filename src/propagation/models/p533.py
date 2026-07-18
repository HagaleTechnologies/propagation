"""ITU-R P.533 baseline: wraps the vendored ITURHFProp binary (baselines/p533).

Scores field-center -> field-center paths (ARCHITECTURE.md sec 5 M-1,
midpoint-to-midpoint), reusing propagation.data.geo.grid_to_latlon for the
field-center math. Standalone: no cqdx dependency (README boundaries).
"""
from __future__ import annotations

# Primary digital-activity (FT8) dial frequency per band, MHz — the
# frequency at which most label-generating spots occur, hence the honest
# frequency to score P.533 at.
BAND_FREQ_MHZ: dict[str, float] = {
    "160m": 1.840, "80m": 3.573, "60m": 5.357, "40m": 7.074,
    "30m": 10.136, "20m": 14.074, "17m": 18.100, "15m": 21.074,
    "12m": 24.915, "10m": 28.074, "6m": 50.313,
}

from dataclasses import dataclass
from pathlib import Path

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


def parse_report(text: str) -> P533Result:
    """Parse an ITURHFProp report: find the CSV header naming BCR and SNR
    columns, read the first data row. Column order is not assumed."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        cols = [c.strip() for c in ln.split(",")]
        bcr_idx = next((j for j, c in enumerate(cols) if c.upper().startswith("BCR")), None)
        snr_idx = next((j for j, c in enumerate(cols) if c.upper().startswith("SNR")), None)
        if bcr_idx is None or snr_idx is None:
            continue
        for data_ln in lines[i + 1:]:
            vals = [v.strip() for v in data_ln.split(",")]
            if len(vals) != len(cols):
                continue
            try:
                return P533Result(
                    reliability_pct=float(vals[bcr_idx]),
                    snr_db=float(vals[snr_idx]),
                )
            except ValueError:
                continue
        raise ValueError("BCR/SNR header found but no parseable data row")
    raise ValueError("no header row naming BCR and SNR columns found")
