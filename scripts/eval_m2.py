"""M2 acceptance artifact: climatology + P.533 + LightGBM headline reports,
storm/quiet slices, blocked time-series CV.

Usage:
    uv run python scripts/eval_m2.py --band 20m \
        --train-months 2024-01 2024-02 2024-03 --eval-months 2024-05 \
        --data-dir data

Re-derives full, unsampled labels across every requested month the same
way scripts/eval_m1.py's _build_labels does (the lake's train partition is
downsampled 3:1; a fresh ClimatologyModel fit needs the full set).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from propagation.data.spaceweather import fetch_omni2_range
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet
from propagation.eval.report import write_headline_report
from propagation.eval.stratify import fetch_definitive_kp, tag_storm_windows
from propagation.features.matrix import build_feature_matrix
from propagation.features.universe import build_universe
from propagation.features.labels import build_labels
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.models.gbt import GBTModel
from propagation.models.p533 import P533Model, ssn_by_month


def _build_labels_for_month(archive: Path, band: str) -> pl.DataFrame:
    extract = extract_wsprnet(archive, band=band)
    uptime = build_receiver_uptime(extract.spots)
    universe = build_universe(extract.spots, uptime)
    return build_labels(extract.spots, universe)


def _build_labels_for_months(archives: dict[str, Path], band: str) -> pl.DataFrame:
    return pl.concat([_build_labels_for_month(a, band) for a in archives.values()])


def write_three_model_slice_reports(
    models: dict[str, object], labels: pl.DataFrame, out_dir: Path,
) -> dict[str, dict[str, dict]]:
    """Same pattern as scripts/eval_m1.py::write_slice_reports, extended to
    three models. `labels` must carry `open` and `is_storm`."""
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
    ap = argparse.ArgumentParser(description="M2 GBT-vs-P.533-vs-climatology headline eval")
    ap.add_argument("--band", default="20m")
    ap.add_argument("--train-months", nargs="+", required=True, help="YYYY-MM, e.g. 2024-01 2024-02 2024-03")
    ap.add_argument("--eval-months", nargs="+", required=True, help="YYYY-MM, held-out, e.g. 2024-05")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    raw_dir = args.data_dir / "raw"

    def _archive_path(ym: str) -> Path:
        y, m = ym.split("-")
        return raw_dir / f"wsprspots-{y}-{m}.csv.gz"

    train_archives, eval_archives = {}, {}
    for ym in args.train_months:
        p = _archive_path(ym)
        if not p.exists():
            y, m = ym.split("-")
            print(f"downloading {p.name}...")
            download_wsprnet_archive(int(y), int(m), p)
        train_archives[ym] = p
    for ym in args.eval_months:
        p = _archive_path(ym)
        if not p.exists():
            y, m = ym.split("-")
            print(f"downloading {p.name}...")
            download_wsprnet_archive(int(y), int(m), p)
        eval_archives[ym] = p

    train_labels = _build_labels_for_months(train_archives, args.band)
    eval_labels = _build_labels_for_months(eval_archives, args.band)

    cache_dir = args.data_dir / "cache"
    all_years = sorted({int(ym.split("-")[0]) for ym in list(args.train_months) + list(args.eval_months)})
    omni = fetch_omni2_range(all_years[0], all_years[-1], cache_dir=cache_dir)

    train_matrix = build_feature_matrix(train_labels, full_history=train_labels, omni=omni)
    eval_matrix = build_feature_matrix(eval_labels, full_history=eval_labels, omni=omni)
    train_matrix = train_matrix.with_columns(pl.lit(1.0).alias("sample_weight"))

    eval_month_keys = list(args.eval_months)
    kp = fetch_definitive_kp(cache_dir)
    eval_tagged = tag_storm_windows(eval_matrix, kp)

    models = {
        "climatology": ClimatologyModel().fit(train_labels),
        "p533": P533Model(ssn_by_month=ssn_by_month(eval_month_keys, cache_dir)),
        "gbt": GBTModel().fit(train_matrix),
    }

    out_dir = args.data_dir / "reports" / "m2"
    results = write_three_model_slice_reports(models, eval_tagged, out_dir)
    for slice_name, per_model in results.items():
        print(f"{slice_name}: {per_model}")
    print(f"wrote {out_dir}/{{overall,storm,quiet}}/headline_table.csv")


if __name__ == "__main__":
    main()
