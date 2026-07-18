"""M1 acceptance artifact: climatology + P.533 headline reports with a
storm/quiet split.

Usage:
    uv run python scripts/eval_m1.py --band 20m \
        --train-year 2014 --train-month 6 --eval-year 2014 --eval-month 8 \
        --data-dir data

Eval is on the FULL label set (never downsampled), matching
scripts/run_m0.py's train_labels/eval_labels — NOT the lake's `labels`
table, whose split=train partition is downsampled 3:1
(docs/SPEC-labeling.md sec 4.5) and would bias a fresh ClimatologyModel fit.
Storm slice per docs/SPEC-labeling.md sec 6 leakage rule 5, via definitive
Kp (eval-only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet
from propagation.eval.report import write_headline_report
from propagation.eval.stratify import fetch_definitive_kp, tag_storm_windows
from propagation.features.labels import build_labels
from propagation.features.universe import build_universe
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.models.p533 import P533Model, ssn_by_month


def _build_labels(archives: dict[str, Path], band: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Re-derives full, unsampled train/eval labels the same way
    scripts/run_m0.py does, so a freshly-fit ClimatologyModel here matches
    M0's real numbers exactly."""
    train_extract = extract_wsprnet(archives["train"], band=band)
    eval_extract = extract_wsprnet(archives["eval"], band=band)
    train_uptime = build_receiver_uptime(train_extract.spots)
    eval_uptime = build_receiver_uptime(eval_extract.spots)
    train_universe = build_universe(train_extract.spots, train_uptime)
    eval_universe = build_universe(eval_extract.spots, eval_uptime)
    train_labels = build_labels(train_extract.spots, train_universe)
    eval_labels = build_labels(eval_extract.spots, eval_universe)
    return train_labels, eval_labels


def write_slice_reports(
    models: dict[str, object], labels: pl.DataFrame, out_dir: Path
) -> dict[str, dict[str, dict]]:
    """labels must carry `open` and `is_storm`. `models` maps model_name ->
    object with `.predict(df) -> df_with_p_open`. Writes
    <out_dir>/<slice>/headline_table.csv (+ reliability pngs) per (slice,
    model) via propagation.eval.report.write_headline_report, dropping
    abstained (null p_open) rows before scoring. Returns
    {slice: {model_name: result_dict}}, omitting a model from a slice's dict
    if it abstained on every row in that slice."""
    out_dir = Path(out_dir)
    slices = {
        "overall": labels,
        "storm": labels.filter(pl.col("is_storm")),
        "quiet": labels.filter(~pl.col("is_storm")),
    }
    results: dict[str, dict[str, dict]] = {}
    for slice_name, sl in slices.items():
        results[slice_name] = {}
        slice_dir = out_dir / slice_name
        for model_name, model in models.items():
            pred = model.predict(sl).drop_nulls("p_open")
            if pred.height == 0:
                print(f"{model_name} abstained on all {sl.height} rows in slice {slice_name!r} — skipping")
                continue
            results[slice_name][model_name] = write_headline_report(
                y_true=pred["open"].cast(pl.Float64).to_numpy(),
                y_prob=pred["p_open"].to_numpy(),
                model_name=model_name,
                out_dir=slice_dir,
            )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 P.533-vs-climatology headline eval")
    ap.add_argument("--train-year", type=int, default=2014)
    ap.add_argument("--train-month", type=int, default=6)
    ap.add_argument("--eval-year", type=int, default=2014)
    ap.add_argument("--eval-month", type=int, default=8)
    ap.add_argument("--band", default="20m")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    raw_dir = args.data_dir / "raw"
    train_archive = raw_dir / f"wsprspots-{args.train_year:04d}-{args.train_month:02d}.csv.gz"
    eval_archive = raw_dir / f"wsprspots-{args.eval_year:04d}-{args.eval_month:02d}.csv.gz"
    if not train_archive.exists():
        print(f"downloading {train_archive.name}...")
        download_wsprnet_archive(args.train_year, args.train_month, train_archive)
    if not eval_archive.exists():
        print(f"downloading {eval_archive.name}...")
        download_wsprnet_archive(args.eval_year, args.eval_month, eval_archive)

    train_labels, eval_labels = _build_labels(
        {"train": train_archive, "eval": eval_archive}, args.band
    )

    cache_dir = args.data_dir / "cache"
    eval_month_key = f"{args.eval_year:04d}-{args.eval_month:02d}"
    kp = fetch_definitive_kp(cache_dir)
    eval_tagged = tag_storm_windows(eval_labels, kp)

    models = {
        "climatology": ClimatologyModel().fit(train_labels),
        "p533": P533Model(ssn_by_month=ssn_by_month([eval_month_key], cache_dir)),
    }

    out_dir = args.data_dir / "reports" / "m1"
    results = write_slice_reports(models, eval_tagged, out_dir)
    for slice_name, per_model in results.items():
        n_storm = int(eval_tagged.filter(pl.col("is_storm")).height) if slice_name == "storm" else None
        print(f"{slice_name}: {per_model}" + (f" (n_storm_rows={n_storm})" if n_storm is not None else ""))
    print(f"wrote {out_dir}/{{overall,storm,quiet}}/headline_table.csv")


if __name__ == "__main__":
    main()
