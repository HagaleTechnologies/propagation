"""PRO-9 entrypoint: run the live PSKReporter MQTT accumulator forever,
writing hourly Parquet to <out-dir>/pskreporter-YYYY-MM-DDTHH.parquet.

Long-running by design (ARCHITECTURE.md §3.2: PSKReporter has no public bulk
archive, so this accumulates going forward). Meant to run under a process
supervisor (launchd first, matching the M4 serving/score.py plan in
ARCHITECTURE.md §7) so a crash (not just a dropped MQTT connection, which
PSKReporterAccumulator already survives on its own) gets restarted.

Usage:
    uv run python scripts/run_pskreporter_accumulator.py --band 20m --out-dir data/raw/pskreporter
    uv run python scripts/run_pskreporter_accumulator.py --out-dir data/raw/pskreporter  # all bands
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from propagation.data.pskreporter import PSKReporterAccumulator


def main() -> None:
    ap = argparse.ArgumentParser(description="PRO-9 live PSKReporter MQTT accumulator")
    ap.add_argument("--band", default="+", help="band to subscribe to server-side, e.g. 20m; default all bands")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    accumulator = PSKReporterAccumulator(out_dir=args.out_dir, band=args.band)
    accumulator.run_forever()


if __name__ == "__main__":
    main()
