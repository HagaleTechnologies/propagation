"""M3 band/horizon expansion driver: trains one GBTModel per requested
horizon, shared across every requested band (band_ordinal is a feature),
and writes climatology/P.533/GBT headline reports sliced by band group
(low/mid/high HF) per ARCHITECTURE.md sec 6 and docs/superpowers/specs/
2026-07-24-m3-band-horizon-expansion-design.md.

Usage:
    uv run python scripts/eval_m3.py --bands 20m 17m 15m 40m --horizons 0 6 \
        --train-months 2024-01 2024-02 2024-03 --eval-months 2024-05 \
        --data-dir data

    Pass --include-rbn to merge RBN (CW ground truth, propagation.data.rbn)
    spots in alongside WSPRnet before building labels -- PRO-8's second
    acceptance scenario. Downloads one archive per day in every requested
    train/eval month (best-effort; a day with no RBN archive is skipped, not
    fatal), so expect this to add real wall-clock time and disk beyond the
    monthly WSPRnet archives.

Storm/quiet slicing and the full 11-band x 6-horizon historical sweep are
out of scope for this script (spec sec 7) -- this produces one headline
table per (band group, horizon) for whatever bands/horizons are requested.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
from pathlib import Path

import httpx
import polars as pl

from propagation.data.rbn import LocationResolver, download_rbn_archive, extract_rbn
from propagation.data.schema import SPOT_SCHEMA
from propagation.data.spaceweather import fetch_omni2_range
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet_bands
from propagation.eval.report import write_headline_report
from propagation.eval.splits import blocked_cv_gap_hours
from propagation.features.labels import build_labels
from propagation.features.matrix import build_feature_matrix
from propagation.features.universe import build_universe
from propagation.features.uptime import build_receiver_uptime
from propagation.models.climatology import ClimatologyModel
from propagation.models.gbt import GBTModel
from propagation.models.p533 import P533Model, ssn_by_month

BAND_GROUPS: dict[str, set[str]] = {
    "low": {"160m", "80m", "60m", "40m"},
    "mid": {"30m", "20m", "17m", "15m"},
    "high": {"12m", "10m", "6m"},
}
_MAX_AR_LOOKBACK_HOURS = 24.0


def _band_group(band: str) -> str:
    for group, bands in BAND_GROUPS.items():
        if band in bands:
            return group
    raise ValueError(f"band {band!r} not in any BAND_GROUPS entry")


def _merge_band_spots(spots: pl.DataFrame, extra_spots: pl.DataFrame | None, band: str) -> pl.DataFrame:
    """Fold in `extra_spots` (e.g. RBN, PRO-8) for one band, if any exist for
    it. Split out as its own pure function so the merge logic is testable
    without a real WSPRnet archive fixture."""
    if extra_spots is None or extra_spots.height == 0:
        return spots
    band_extra = extra_spots.filter(pl.col("band") == band)
    if band_extra.height == 0:
        return spots
    return pl.concat([spots, band_extra], how="vertical_relaxed")


def _build_labels_for_month_all_bands(
    archive: Path, bands: list[str], extra_spots: pl.DataFrame | None = None
) -> pl.DataFrame:
    extracts = extract_wsprnet_bands(archive, bands=bands)
    per_band = []
    for band, extract in extracts.items():
        spots = _merge_band_spots(extract.spots, extra_spots, band)
        uptime = build_receiver_uptime(spots)
        universe = build_universe(spots, uptime)
        per_band.append(build_labels(spots, universe))
    return pl.concat(per_band, how="vertical_relaxed")


def _build_labels_for_months(
    archives: dict[str, Path], bands: list[str], extra_spots_by_month: dict[str, pl.DataFrame] | None = None
) -> pl.DataFrame:
    return pl.concat(
        [
            _build_labels_for_month_all_bands(a, bands, (extra_spots_by_month or {}).get(ym))
            for ym, a in archives.items()
        ],
        how="vertical_relaxed",
    )


def _rbn_month_archive_paths(ym: str, raw_dir: Path) -> list[tuple[dt.date, Path]]:
    year, month = (int(x) for x in ym.split("-"))
    n_days = calendar.monthrange(year, month)[1]
    return [
        (dt.date(year, month, day), raw_dir / f"rbn-{year:04d}{month:02d}{day:02d}.zip")
        for day in range(1, n_days + 1)
    ]


def _download_rbn_month(ym: str, raw_dir: Path) -> list[Path]:
    """Best-effort per day: reversebeacon.net not having an archive for a
    given date is a real, plausible gap (not a bug) -- skip that day rather
    than aborting the whole month's extraction over it."""
    paths = []
    for date, p in _rbn_month_archive_paths(ym, raw_dir):
        if not p.exists():
            print(f"downloading {p.name}...")
            try:
                download_rbn_archive(date, p)
            except httpx.HTTPStatusError as e:
                print(f"  skipping {p.name}: {e}")
                continue
        paths.append(p)
    return paths


def _build_rbn_spots_for_month(
    ym: str, bands: list[str], raw_dir: Path, resolve_location: LocationResolver | None = None
) -> pl.DataFrame:
    """PRO-8's second acceptance scenario: RBN spots merged alongside
    WSPRnet's, feeding the same build_universe/build_labels pipeline (see
    _merge_band_spots). One extract_rbn call per (day, band) -- RBN's daily
    archives are small (reversebeacon.net's own docs cite 300k+ spots on a
    contest day, well under WSPRnet's tens-of-millions-per-month volume), so
    unlike extract_wsprnet_bands this doesn't need a single-pass multi-band
    variant to stay memory-safe."""
    frames = []
    for p in _download_rbn_month(ym, raw_dir):
        for band in bands:
            result = extract_rbn(p, band=band, resolve_location=resolve_location)
            if result.spots.height:
                frames.append(result.spots)
    if not frames:
        return pl.DataFrame(schema=SPOT_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed")


def enforce_blocked_cv_gap(
    train_labels: pl.DataFrame,
    eval_labels: pl.DataFrame,
    max_horizon_hours: float,
    max_ar_lookback_hours: float = _MAX_AR_LOOKBACK_HOURS,
) -> None:
    """Same rule as scripts/eval_m2.py's function of the same name
    (docs/SPEC-labeling.md sec 6 rule 1), generalized: M3 trains one model
    per horizon in --horizons, so the gap must be computed against the
    LARGEST horizon requested, not a fixed 3h."""
    train_end = train_labels["window_start"].max()
    eval_start = eval_labels["window_start"].min()
    required_gap_hours = blocked_cv_gap_hours(max_horizon_hours, max_ar_lookback_hours)
    actual_gap_hours = (eval_start - train_end).total_seconds() / 3600.0
    if actual_gap_hours < required_gap_hours:
        raise ValueError(
            f"blocked-CV gap violation: eval window starts only {actual_gap_hours:.2f}h "
            f"after train window ends, but docs/SPEC-labeling.md sec 6 rule 1 requires "
            f">= {required_gap_hours:.2f}h (blocked_cv_gap_hours(max_horizon_hours="
            f"{max_horizon_hours}, max_ar_lookback_hours={max_ar_lookback_hours})) to "
            f"avoid train/eval leakage. Choose train/eval months with a sufficient gap."
        )


def write_band_group_reports(
    models: dict[str, object], labels: pl.DataFrame, horizon_hours: float, out_dir: Path,
) -> dict[str, dict[str, dict]]:
    """Same pattern as scripts/eval_m2.py::write_three_model_slice_reports,
    sliced by BAND_GROUPS instead of storm/quiet. `labels` must carry `open`
    and `band`. Writes <out_dir>/<group>/h<N>/headline_table.csv."""
    out_dir = Path(out_dir)
    labeled = labels.with_columns(
        pl.col("band").map_elements(_band_group, return_dtype=pl.Utf8).alias("band_group")
    )
    results: dict[str, dict[str, dict]] = {}
    for group in BAND_GROUPS:
        sl = labeled.filter(pl.col("band_group") == group)
        results[group] = {}
        if sl.height == 0:
            continue
        group_dir = out_dir / group / f"h{int(horizon_hours)}"
        for model_name, model in models.items():
            pred = model.predict(sl).drop_nulls("p_open")
            if pred.height == 0:
                print(f"{model_name} abstained on all {sl.height} rows in group {group!r} h={horizon_hours} — skipping")
                continue
            results[group][model_name] = write_headline_report(
                y_true=pred["open"].cast(pl.Float64).to_numpy(),
                y_prob=pred["p_open"].to_numpy(),
                model_name=model_name,
                out_dir=group_dir,
            )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 band/horizon expansion headline eval")
    ap.add_argument("--bands", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", type=float, required=True, help="hours, e.g. 0 1 3 6 12 24")
    ap.add_argument("--train-months", nargs="+", required=True, help="YYYY-MM")
    ap.add_argument("--eval-months", nargs="+", required=True, help="YYYY-MM, held-out")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument(
        "--include-rbn", action="store_true",
        help="merge RBN (CW ground truth) spots into WSPRnet before building labels (PRO-8)",
    )
    ap.add_argument(
        "--p533-workers", type=int, default=None,
        help="P533Model's ThreadPoolExecutor size (each worker forks an iturhfprop subprocess); "
        "defaults to os.cpu_count() (P533Model's own default) -- pass a smaller number to reduce "
        "CPU/memory contention on a machine running other concurrent load (see ADR 0006).",
    )
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

    rbn_by_month: dict[str, pl.DataFrame] = {}
    if args.include_rbn:
        for ym in sorted(set(args.train_months) | set(args.eval_months)):
            print(f"extracting RBN spots for {ym}...")
            rbn_by_month[ym] = _build_rbn_spots_for_month(ym, args.bands, raw_dir)
            print(f"  {rbn_by_month[ym].height} qualifying RBN spots for {ym}")

    train_labels = _build_labels_for_months(train_archives, args.bands, rbn_by_month)
    eval_labels = _build_labels_for_months(eval_archives, args.bands, rbn_by_month)

    max_horizon = max(args.horizons)
    enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours=max_horizon)

    cache_dir = args.data_dir / "cache"
    all_years = sorted({int(ym.split("-")[0]) for ym in list(args.train_months) + list(args.eval_months)})
    omni = fetch_omni2_range(all_years[0], all_years[-1], cache_dir=cache_dir)
    eval_month_keys = list(args.eval_months)

    # P533Model's (path, band, month, hour, SSN) key doesn't depend on
    # horizon_hours at all (window_start/tx_field/rx_field/band are
    # horizon-invariant per build_feature_matrix's design -- only the
    # as-of-now feature builders re-anchor). One instance, reused and
    # disk-cached across every horizon in this sweep, so P.533's per-row
    # subprocess cost is paid once per unique key instead of once per
    # (key, horizon) pair.
    p533_model = P533Model(
        ssn_by_month=ssn_by_month(eval_month_keys, cache_dir),
        cache_path=cache_dir / "p533_scores.parquet",
        max_workers=args.p533_workers,
    )

    out_dir = args.data_dir / "reports" / "m3"
    for horizon_hours in args.horizons:
        train_matrix = build_feature_matrix(
            train_labels, full_history=train_labels, omni=omni, horizon_hours=horizon_hours
        ).with_columns(pl.lit(1.0).alias("sample_weight"))
        eval_matrix = build_feature_matrix(
            eval_labels, full_history=eval_labels, omni=omni, horizon_hours=horizon_hours
        )

        models = {
            "climatology": ClimatologyModel().fit(train_labels),
            "p533": p533_model,
            "gbt": GBTModel().fit(train_matrix),
        }
        results = write_band_group_reports(models, eval_matrix, horizon_hours, out_dir)
        print(f"h={horizon_hours}h: {results}")

    print(f"wrote {out_dir}/{{low,mid,high}}/h<N>/headline_table.csv")


if __name__ == "__main__":
    main()
