"""PRO-11: storm-window case studies. Reuses eval/stratify.py's Kp-based
storm tagging and the same climatology/P.533/GBT models as scripts/eval_m3.py,
but slices each eval month into before/during/after a specific named storm
(rather than a blanket storm/quiet split) and reports Brier per period per
model plus a time-series plot of predicted vs. observed open rate.

Storms are picked from data/cache/gfz_kp.txt (definitive Kp), one per M3 eval
month (2024-05/07/09), so the three case studies span the full range actually
used by M3's headline result (ADR 0006): the May 2024 Gannon storm (Kp up to
9.0), the July 2024 Kp~5.3 minor storm (July had no stronger event), and the
September 2024 Kp~7.3 storm.

Usage:
    uv run python scripts/eval_storm_case_studies.py --band 20m --data-dir data
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from propagation.data.spaceweather import fetch_omni2_range
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet
from propagation.eval.report import write_headline_report
from propagation.eval.splits import blocked_cv_gap_hours
from propagation.features.labels import build_labels
from propagation.features.matrix import build_feature_matrix
from propagation.features.universe import build_universe
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.models.gbt import GBTModel
from propagation.models.p533 import P533Model, ssn_by_month

PAD_HOURS = 24.0


@dataclass(frozen=True)
class Storm:
    name: str
    eval_month: str  # YYYY-MM
    start: datetime  # UTC, inclusive
    end: datetime  # UTC, exclusive


# Windows derived from the definitive-Kp storm blocks (Kp>=5) in each M3 eval
# month; start/end bound the contiguous run of storm blocks (each block is
# 3h), not just a single block.
STORMS = [
    Storm("2024-05-gannon", "2024-05",
          datetime(2024, 5, 10, 15, tzinfo=timezone.utc),
          datetime(2024, 5, 13, 6, tzinfo=timezone.utc)),
    Storm("2024-07-minor", "2024-07",
          datetime(2024, 7, 30, 3, tzinfo=timezone.utc),
          datetime(2024, 7, 30, 6, tzinfo=timezone.utc)),
    Storm("2024-09-g3", "2024-09",
          datetime(2024, 9, 12, 6, tzinfo=timezone.utc),
          datetime(2024, 9, 13, 3, tzinfo=timezone.utc)),
]


def _build_labels_for_month(archive: Path, band: str) -> pl.DataFrame:
    extract = extract_wsprnet(archive, band=band)
    uptime = build_receiver_uptime(extract.spots)
    universe = build_universe(extract.spots, uptime)
    return build_labels(extract.spots, universe)


def _archive_path(data_dir: Path, ym: str) -> Path:
    y, m = ym.split("-")
    p = data_dir / "raw" / f"wsprspots-{y}-{m}.csv.gz"
    if not p.exists():
        print(f"downloading {p.name}...")
        download_wsprnet_archive(int(y), int(m), p)
    return p


def tag_period(labels: pl.DataFrame, storm: Storm) -> pl.DataFrame:
    """Attach a before/during/after label relative to `storm`, using a
    PAD_HOURS window on each side. Rows outside all three windows are
    dropped (`period` is null)."""
    before_start = storm.start - timedelta(hours=PAD_HOURS)
    after_end = storm.end + timedelta(hours=PAD_HOURS)
    return labels.filter(
        (pl.col("window_start") >= before_start) & (pl.col("window_start") < after_end)
    ).with_columns(
        pl.when(pl.col("window_start") < storm.start)
        .then(pl.lit("before"))
        .when(pl.col("window_start") < storm.end)
        .then(pl.lit("during"))
        .otherwise(pl.lit("after"))
        .alias("period")
    )


def write_period_reports(
    models: dict[str, object], tagged: pl.DataFrame, out_dir: Path,
) -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {}
    for period in ("before", "during", "after"):
        sl = tagged.filter(pl.col("period") == period)
        results[period] = {}
        if sl.height == 0:
            continue
        period_dir = out_dir / period
        for model_name, model in models.items():
            pred = model.predict(sl).drop_nulls("p_open")
            if pred.height == 0:
                print(f"{model_name} abstained on all {sl.height} rows in period {period!r} — skipping")
                continue
            results[period][model_name] = write_headline_report(
                y_true=pred["open"].cast(pl.Float64).to_numpy(),
                y_prob=pred["p_open"].to_numpy(),
                model_name=model_name,
                out_dir=period_dir,
            )
    return results


def plot_timeline(
    models: dict[str, object], tagged: pl.DataFrame, storm: Storm, out_path: Path,
) -> None:
    """The 'map' PRO-11 asks for: hourly-binned observed open rate vs. each
    model's mean predicted p_open across before/during/after, with the storm
    window shaded."""
    binned = tagged.with_columns(pl.col("window_start").dt.truncate("1h").alias("hour"))
    fig, ax = plt.subplots(figsize=(10, 4))

    observed = binned.group_by("hour").agg(pl.col("open").cast(pl.Float64).mean().alias("rate")).sort("hour")
    ax.plot(observed["hour"].to_list(), observed["rate"].to_list(), color="black", label="observed", linewidth=2)

    for model_name, model in models.items():
        pred = model.predict(tagged).drop_nulls("p_open").with_columns(
            pl.col("window_start").dt.truncate("1h").alias("hour")
        )
        m_binned = pred.group_by("hour").agg(pl.col("p_open").mean().alias("rate")).sort("hour")
        ax.plot(m_binned["hour"].to_list(), m_binned["rate"].to_list(), label=model_name, alpha=0.8)

    ax.axvspan(storm.start, storm.end, color="red", alpha=0.12, label="storm window")
    ax.set_ylabel("open rate / P(open)")
    ax.set_title(f"{storm.name}: observed vs. predicted, before/during/after")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="PRO-11 storm-window case studies")
    ap.add_argument("--band", default="20m")
    ap.add_argument("--train-months", nargs="+", default=["2024-01", "2024-02", "2024-03"])
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    train_archives = {ym: _archive_path(args.data_dir, ym) for ym in args.train_months}
    train_labels = pl.concat(
        [_build_labels_for_month(p, args.band) for p in train_archives.values()]
    )

    cache_dir = args.data_dir / "cache"

    eval_months = sorted({s.eval_month for s in STORMS})
    eval_labels_by_month: dict[str, pl.DataFrame] = {}
    for ym in eval_months:
        p = _archive_path(args.data_dir, ym)
        labels = _build_labels_for_month(p, args.band)
        max_horizon = 0.0
        required_gap_hours = blocked_cv_gap_hours(max_horizon, 24.0)
        actual_gap_hours = (labels["window_start"].min() - train_labels["window_start"].max()).total_seconds() / 3600.0
        if actual_gap_hours < required_gap_hours:
            raise ValueError(
                f"blocked-CV gap violation for eval month {ym}: {actual_gap_hours:.2f}h "
                f"< required {required_gap_hours:.2f}h"
            )
        eval_labels_by_month[ym] = labels

    all_years = sorted({int(ym.split("-")[0]) for ym in list(args.train_months) + eval_months})
    omni = fetch_omni2_range(all_years[0], all_years[-1], cache_dir=cache_dir)

    train_matrix = build_feature_matrix(
        train_labels, full_history=train_labels, omni=omni, horizon_hours=0.0
    ).with_columns(pl.lit(1.0).alias("sample_weight"))

    p533_model = P533Model(
        ssn_by_month=ssn_by_month(eval_months, cache_dir),
        cache_path=cache_dir / "p533_scores.parquet",
    )
    models = {
        "climatology": ClimatologyModel().fit(train_labels),
        "p533": p533_model,
        "gbt": GBTModel().fit(train_matrix),
    }

    out_root = args.data_dir / "reports" / "storm_case_studies"
    for storm in STORMS:
        eval_labels = eval_labels_by_month[storm.eval_month]
        eval_matrix = build_feature_matrix(
            eval_labels, full_history=eval_labels, omni=omni, horizon_hours=0.0
        )
        tagged = tag_period(eval_matrix, storm)
        if tagged.height == 0:
            print(f"{storm.name}: no rows in before/during/after window — skipping")
            continue
        storm_dir = out_root / storm.name
        results = write_period_reports(models, tagged, storm_dir)
        print(f"{storm.name}: {results}")
        plot_timeline(models, tagged, storm, storm_dir / "timeline.png")

    print(f"wrote {out_root}/<storm>/{{before,during,after}}/headline_table.csv + timeline.png")


if __name__ == "__main__":
    main()
