"""Produce our half of the private P.533 agreement spot-check.

Scores ~100 deterministic pseudo-random paths and writes a CSV; the
comparison against cqdx's propagation-sidecar output happens OUTSIDE this
repo (open/closed boundary — README "Licensing & boundaries"). Not CI.

Usage: uv run python scripts/p533_crosscheck.py --n 100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from propagation.models.p533 import BAND_FREQ_MHZ, p533_score

_SEED = int.from_bytes(hashlib.sha256(b"p533-crosscheck-v1").digest()[:8], "big") & 0xFFFFFFFF


def sample_paths(n: int) -> list[tuple]:
    """Deterministic sample over (lat/lon pairs, band, month, hour, ssn)."""
    rng = np.random.Generator(np.random.PCG64(_SEED))
    bands = sorted(BAND_FREQ_MHZ)
    out = []
    while len(out) < n:
        row = (
            round(float(rng.uniform(-60, 60)), 2),    # tx_lat: populated latitudes
            round(float(rng.uniform(-180, 180)), 2),
            round(float(rng.uniform(-60, 60)), 2),
            round(float(rng.uniform(-180, 180)), 2),
            bands[int(rng.integers(len(bands)))],
            int(rng.integers(1, 13)),
            int(rng.integers(0, 24)),
            float(rng.integers(5, 200)),
        )
        if row not in out:
            out.append(row)
    return out


def run(n: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tx_lat", "tx_lon", "rx_lat", "rx_lon", "band",
                    "month", "hour", "ssn", "reliability_pct", "snr_db"])
        for tx_lat, tx_lon, rx_lat, rx_lon, band, month, hour, ssn in sample_paths(n):
            r = p533_score(tx_lat=tx_lat, tx_lon=tx_lon, rx_lat=rx_lat,
                           rx_lon=rx_lon, band=band, month=month,
                           hour=hour, ssn=ssn)
            w.writerow([tx_lat, tx_lon, rx_lat, rx_lon, band, month, hour,
                        ssn, r.reliability_pct, r.snr_db])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("reports/p533-crosscheck.csv"))
    args = ap.parse_args()
    run(args.n, args.out)
    print(f"wrote {args.out} — compare privately against the cqdx sidecar")


if __name__ == "__main__":
    main()
