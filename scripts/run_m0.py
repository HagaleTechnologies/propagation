from __future__ import annotations

import argparse
from pathlib import Path

from propagation.data.lake import write_partitioned
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet
from propagation.eval.report import write_headline_report
from propagation.features.labels import build_labels
from propagation.features.sampling import sample_labels, write_labels
from propagation.features.universe import build_universe, unlabeled_activity_fraction
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.qa.checks import run_qa_checks


def run_m0(archives: dict[str, Path], band: str, lake_root: Path, report_dir: Path) -> dict:
    lake_root, report_dir = Path(lake_root), Path(report_dir)

    train_extract = extract_wsprnet(archives["train"], band=band)
    eval_extract = extract_wsprnet(archives["eval"], band=band)

    write_partitioned(
        train_extract.spots.with_columns(
            train_extract.spots["ts"].dt.date().cast(str).alias("date")
        ),
        lake_root, "spots", ["band", "date"],
    )
    write_partitioned(
        eval_extract.spots.with_columns(
            eval_extract.spots["ts"].dt.date().cast(str).alias("date")
        ),
        lake_root, "spots", ["band", "date"],
    )

    train_uptime = build_receiver_uptime(train_extract.spots)
    eval_uptime = build_receiver_uptime(eval_extract.spots)

    train_universe = build_universe(train_extract.spots, train_uptime)
    eval_universe = build_universe(eval_extract.spots, eval_uptime)

    train_labels = build_labels(train_extract.spots, train_universe)
    eval_labels = build_labels(eval_extract.spots, eval_universe)

    train_sampled = sample_labels(train_labels, ratio=3.0)
    write_labels(train_sampled, lake_root)
    write_labels(eval_labels, lake_root)

    model = ClimatologyModel().fit(train_labels)
    predictions = model.predict(eval_labels)

    headline = write_headline_report(
        y_true=predictions["open"].cast(float).to_numpy(),
        y_prob=predictions["p_open"].to_numpy(),
        model_name="climatology-m0",
        out_dir=report_dir,
    )

    unlabeled = unlabeled_activity_fraction(eval_extract.spots, eval_universe)
    qa_results = run_qa_checks(
        eval_labels,
        rejection_counts=eval_extract.rejection_counts,
        n_qualifying=eval_extract.n_qualifying,
    )

    return {
        "n_train_labels": train_labels.height,
        "n_eval_labels": eval_labels.height,
        "headline": headline,
        "qa_results": qa_results,
        "unlabeled_activity_fraction": unlabeled.to_dicts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M0 lake-bootstrap pipeline")
    parser.add_argument("--train-year", type=int, default=2014)
    parser.add_argument("--train-month", type=int, default=6)
    parser.add_argument("--eval-year", type=int, default=2014)
    parser.add_argument("--eval-month", type=int, default=7)
    parser.add_argument("--band", default="20m")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    train_archive = raw_dir / f"wsprspots-{args.train_year:04d}-{args.train_month:02d}.csv.gz"
    eval_archive = raw_dir / f"wsprspots-{args.eval_year:04d}-{args.eval_month:02d}.csv.gz"

    if not train_archive.exists():
        print(f"downloading {train_archive.name}...")
        download_wsprnet_archive(args.train_year, args.train_month, train_archive)
    if not eval_archive.exists():
        print(f"downloading {eval_archive.name}...")
        download_wsprnet_archive(args.eval_year, args.eval_month, eval_archive)

    result = run_m0(
        archives={"train": train_archive, "eval": eval_archive},
        band=args.band,
        lake_root=args.data_dir / "lake",
        report_dir=args.data_dir / "reports",
    )

    print(f"train labels: {result['n_train_labels']}, eval labels: {result['n_eval_labels']}")
    print(
        f"headline: brier={result['headline']['brier']:.4f} "
        f"log_loss={result['headline']['log_loss']:.4f}"
    )
    for qa in result["qa_results"]:
        print(f"QA {qa.check_id} {qa.name}: {qa.status} — {qa.detail}")


if __name__ == "__main__":
    main()
