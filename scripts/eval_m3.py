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

    PRO-10 acceptance mode: pass --acceptance-pskreporter-dir (and optionally
    --cqdx-dir) to run the public M2 gate instead of the sweep. See
    docs/cqdx-backfill.md.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import httpx
import polars as pl

from propagation.data.dedup import dedup_spots
from propagation.data.rbn import LocationResolver, download_rbn_archive, extract_rbn
from propagation.data.schema import SPOT_SCHEMA, normalize_spot_columns
from propagation.data.spaceweather import fetch_omni2_range
from propagation.data.wsprnet import download_wsprnet_archive, extract_wsprnet_bands
from propagation.eval.metrics import brier_score, log_loss_score
from propagation.eval.report import write_headline_report
from propagation.eval.splits import blocked_cv_gap_hours
from propagation.eval.stratify import fetch_definitive_kp, tag_storm_windows
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


# ---------------------------------------------------------------------------
# PRO-10 acceptance mode: public-only M2 gate + optional cqdx Brier comparison.
#
# The public-only arm must clear M2's bar (ADR 0005: strict GBT win over BOTH
# climatology and P.533 on BOTH Brier and log-loss at h=0) using ONLY
# WSPRnet + RBN + PSKReporter. The optional cqdx arm refits on public training
# spots plus surviving private training rows and scores BOTH fits on the SAME
# public evaluation cohort (identical keys, labels, features), so an apparent
# gain cannot come from changing the evaluation population.
# ---------------------------------------------------------------------------

ACCEPTANCE_BANDS: set[str] = {"20m", "10m"}
PUBLIC_SOURCES: tuple[str, ...] = ("wsprnet", "rbn", "pskreporter")
_SOURCE_DISPLAY = {"wsprnet": "WSPRnet", "rbn": "RBN", "pskreporter": "PSKReporter"}
_PSKR_FILE_RE = re.compile(r"^pskreporter-(\d{4})-(\d{2})-(\d{2})T\d{2}\.parquet$")
_CQDX_FILE_RE = re.compile(r"^cqdx-(\d{4})-(\d{2})-(\d{2})\.parquet$")
_KEY_COLS = ["window_start", "tx_field", "rx_field", "band"]


class AcceptanceUsageError(Exception):
    """Invalid arguments/holdout (band, horizon, months, gap, no storm).
    Usage errors exit 2 before any data work or report write."""


class AcceptanceUnverified(Exception):
    """Missing/unusable evidence (sources, months, predictions, private rows).
    Unverified runs exit 2 and write an `unverified` report."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _month_bounds(ym: str) -> tuple[dt.datetime, dt.datetime]:
    """UTC half-open [start, next_month_start) for a YYYY-MM month key."""
    year, month = (int(x) for x in ym.split("-"))
    start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
    if month == 12:
        end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
    else:
        end = dt.datetime(year, month + 1, 1, tzinfo=dt.timezone.utc)
    return start, end


def _validate_months_key(ym: str) -> str:
    try:
        dt.datetime.strptime(ym, "%Y-%m")
    except ValueError as exc:
        raise AcceptanceUsageError(f"invalid month key {ym!r}: expected YYYY-MM") from exc
    return ym


def _normalize_spot_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize a Parquet-loaded spot frame to exact SPOT_SCHEMA types.
    Handles compatible legacy PSKReporter dtypes (ms/ns timestamps, Int32
    freq/snr) without loosening the schema."""
    df = normalize_spot_columns(df)
    ts = pl.col("ts")
    if getattr(df.schema["ts"], "time_zone", None) is None:
        ts = ts.dt.replace_time_zone("UTC")
    return df.with_columns(
        ts.cast(pl.Datetime("us", "UTC")),
        pl.col("freq_hz").cast(pl.Int64),
        pl.col("snr_db").cast(pl.Int16),
        pl.col("tx_dbm").cast(pl.Int16),
        pl.col("band").str.to_lowercase(),
        pl.col("mode").str.to_uppercase(),
    )


def load_pskreporter_month(psk_dir: Path, ym: str, band: str) -> pl.DataFrame:
    """Read the public accumulator's hourly Parquet files for one UTC month
    and band (PRO-10 acceptance mode's public PSKReporter input). Only files
    matching the accumulator's `pskreporter-YYYY-MM-DDTHH.parquet` naming are
    considered; each file must carry the shared schema and contain ONLY
    source='pskreporter' rows -- a mislabeled or contaminated file (e.g. a
    cqdx backfill file copied in) is a hard error, not a silent skip."""
    psk_dir = Path(psk_dir)
    start, end = _month_bounds(ym)
    frames = []
    for path in sorted(psk_dir.glob("pskreporter-*.parquet")):
        if not _PSKR_FILE_RE.match(path.name):
            print(f"skipping non-accumulator file {path.name}", file=sys.stderr)
            continue
        file_ym = f"{path.name[12:16]}-{path.name[17:19]}"
        if file_ym != ym:
            continue
        df = pl.read_parquet(path)
        missing = [c for c in SPOT_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing shared-schema columns {missing}")
        df = _normalize_spot_frame(df)
        foreign = df.filter(pl.col("source") != "pskreporter")
        if foreign.height:
            raise ValueError(
                f"{path.name}: expected only source='pskreporter' rows, found "
                f"{foreign.height} foreign-source row(s) -- refusing mislabeled input"
            )
        frames.append(df)
    month = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame(schema=SPOT_SCHEMA)
    return month.filter(
        (pl.col("ts") >= start) & (pl.col("ts") < end) & (pl.col("band") == band)
    )


def load_cqdx_month(cqdx_dir: Path, ym: str, band: str) -> pl.DataFrame:
    """Read the cqdx backfill's daily Parquet for one UTC month and band.
    Selection is by OBSERVATION timestamp, not archive partition date (a spot
    observed 23:50 late in a month can sit in the next day's archive object),
    so every daily file is read and rows are filtered by ts; cross-archive-day
    duplicates are removed by the caller's per-month dedup."""
    cqdx_dir = Path(cqdx_dir)
    start, end = _month_bounds(ym)
    frames = []
    for path in sorted(cqdx_dir.glob("cqdx-*.parquet")):
        if not _CQDX_FILE_RE.match(path.name):
            print(f"skipping non-backfill file {path.name}", file=sys.stderr)
            continue
        df = pl.read_parquet(path)
        missing = [c for c in SPOT_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing shared-schema columns {missing}")
        df = _normalize_spot_frame(df)
        foreign = df.filter(pl.col("source") != "cqdx")
        if foreign.height:
            raise ValueError(
                f"{path.name}: expected only source='cqdx' rows, found "
                f"{foreign.height} foreign-source row(s) -- refusing mislabeled input"
            )
        frames.append(df)
    month = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame(schema=SPOT_SCHEMA)
    return month.filter(
        (pl.col("ts") >= start) & (pl.col("ts") < end) & (pl.col("band") == band)
    )


def build_acceptance_labels(spots: pl.DataFrame) -> pl.DataFrame:
    """Spot-to-label steps shared by both arms: same receiver-uptime,
    universe and label builders as the default mode."""
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    return build_labels(spots, universe)


def eval_cohort_sha256(eval_labels: pl.DataFrame) -> str:
    """SHA-256 of the evaluation cohort's keys + labels, ordered by key,
    serialized as UTC ISO timestamps and UTF-8 CSV. Two reports with the
    same digest scored the same population."""
    ordered = eval_labels.select(_KEY_COLS + ["open"]).sort(_KEY_COLS)
    lines = [
        f"{ws.strftime('%Y-%m-%dT%H:%M:%SZ')},{tx},{rx},{band},{open_}"
        for ws, tx, rx, band, open_ in zip(
            ordered["window_start"], ordered["tx_field"], ordered["rx_field"],
            ordered["band"], ordered["open"],
        )
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _aligned_predictions(model_name: str, model, cohort: pl.DataFrame) -> pl.DataFrame:
    """Run `model` over the cohort and return one finite p_open in [0,1] per
    cohort key, aligned to the cohort's row order. Any missing, duplicate or
    extra key, null/abstention, or label mismatch makes the run unverified --
    never a gate pass on a subset."""
    pred = model.predict(cohort)
    for col in _KEY_COLS + ["p_open", "open"]:
        if col not in pred.columns:
            raise AcceptanceUnverified(f"{model_name} produced no {col!r} column")
    keys = pred.select(_KEY_COLS)
    if keys.unique().height != pred.height:
        raise AcceptanceUnverified(f"{model_name} produced duplicate evaluation keys")
    cohort_keys = cohort.select(_KEY_COLS)
    extra = pred.join(cohort_keys, on=_KEY_COLS, how="anti")
    missing = cohort_keys.join(keys, on=_KEY_COLS, how="anti")
    if extra.height or missing.height:
        raise AcceptanceUnverified(
            f"{model_name} predictions do not cover the evaluation cohort "
            f"({missing.height} missing, {extra.height} extra keys)"
        )
    bad = pred.filter(
        pl.col("p_open").is_null()
        | ~pl.col("p_open").is_finite()
        | (pl.col("p_open") < 0.0)
        | (pl.col("p_open") > 1.0)
    )
    if bad.height:
        raise AcceptanceUnverified(
            f"{model_name} abstained or produced non-finite probabilities on {bad.height} rows"
        )
    labeled = pred.select(_KEY_COLS + ["open"]).join(
        cohort.select(_KEY_COLS + ["open"]), on=_KEY_COLS, how="inner", suffix="_cohort"
    )
    if not labeled["open"].equals(labeled["open_cohort"]):
        raise AcceptanceUnverified(f"{model_name} predictions carry labels that disagree with the cohort")
    return cohort.select(_KEY_COLS).join(
        pred.select(_KEY_COLS + ["p_open"]), on=_KEY_COLS, how="left", maintain_order="left"
    ).with_columns(pl.col("p_open"))


def _r6(x: float) -> float:
    return round(float(x), 6)


def _score_arm(model_name: str, model, cohort: pl.DataFrame) -> dict:
    aligned = _aligned_predictions(model_name, model, cohort)
    y_true = cohort["open"].cast(pl.Float64).to_numpy()
    y_prob = aligned["p_open"].to_numpy()
    return {
        "brier": brier_score(y_true, y_prob),
        "log_loss": log_loss_score(y_true, y_prob),
        "n": cohort.height,
    }


def _write_acceptance_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(tmp, path)  # atomic: a completed run replaces the old report


def _spot_source_counts(spots: pl.DataFrame) -> dict[str, int]:
    return {s: int((spots["source"] == s).sum()) for s in (*PUBLIC_SOURCES, "cqdx")}


def _dedup_month_spots(frames: list[pl.DataFrame]) -> pl.DataFrame:
    nonempty = [f for f in frames if f.height]
    if not nonempty:
        return pl.DataFrame(schema=SPOT_SCHEMA)
    return dedup_spots(pl.concat(nonempty, how="vertical_relaxed"))


def run_acceptance(
    args,
    *,
    wsprnet_loader,
    rbn_loader,
    pskreporter_loader=None,
    cqdx_loader=None,
    kp_fetch=None,
    omni_fetch=None,
    climatology_factory=None,
    gbt_factory=None,
    p533_factory=None,
    labels_builder=build_acceptance_labels,
    matrix_builder=None,
    out_base: Path | None = None,
    stderr=sys.stderr,
) -> int:
    """Run the PRO-10 acceptance comparison for one band at h=0. Returns the
    process exit code: 0 pass, 1 a valid comparison that missed a threshold,
    2 unverified or invalid holdout. All injectable pieces default to the
    real loaders/models; tests substitute fixtures."""
    band = args.bands[0]
    out_base = Path(out_base) if out_base else Path(args.data_dir) / "reports" / "m3" / "acceptance" / band
    public_report_path = out_base / "public.json"
    augmented_report_path = out_base / "augmented.json"

    def _unverified(reason: str, path: Path) -> int:
        report["status"] = "unverified"
        report["reason"] = reason
        _write_acceptance_report(report, path)
        print(f"UNVERIFIED on {band} at h=0. {reason} Report: {path}")
        return 2

    report: dict = {
        "band": band,
        "horizon_hours": 0.0,
        "train_months": list(args.train_months),
        "eval_months": list(args.eval_months),
        "public_spots": {
            "train": {s: 0 for s in (*PUBLIC_SOURCES, "cqdx")},
            "eval": {s: 0 for s in (*PUBLIC_SOURCES, "cqdx")},
        },
        "eval_labels_by_month": {},
        "eval_cohort_sha256": None,
        "storm": None,
        "scores": {},
        "public_gate": None,
        "cqdx_comparison": None,
        "status": None,
        "reason": None,
    }

    # --- holdout validation (usage errors: exit 2, no report write) ---
    if len(args.bands) != 1 or band not in ACCEPTANCE_BANDS:
        raise AcceptanceUsageError(
            f"acceptance mode requires exactly one band from {sorted(ACCEPTANCE_BANDS)}, got {args.bands}"
        )
    if [float(h) for h in args.horizons] != [0.0]:
        raise AcceptanceUsageError("acceptance mode requires --horizons 0")
    train_months = [_validate_months_key(ym) for ym in args.train_months]
    eval_months = [_validate_months_key(ym) for ym in args.eval_months]
    if len(set(eval_months)) != len(eval_months):
        raise AcceptanceUsageError(f"repeated eval months: {eval_months}")
    if len(eval_months) < 3:
        raise AcceptanceUsageError(
            f"acceptance requires at least 3 distinct held-out months, got {len(eval_months)}"
        )
    if set(train_months) & set(eval_months):
        raise AcceptanceUsageError(
            f"train and eval months overlap: {sorted(set(train_months) & set(eval_months))}"
        )

    # --- assemble public spots per split/month, dedup per band+month ---
    print(f"assembling public spots for {band}...", file=stderr)
    public_month_spots: dict[tuple[str, str], pl.DataFrame] = {}
    for split, months in (("train", train_months), ("eval", eval_months)):
        for ym in months:
            frames = []
            for name, loader in (
                ("wsprnet", wsprnet_loader), ("rbn", rbn_loader), ("pskreporter", pskreporter_loader),
            ):
                try:
                    spots = loader(ym, band)
                except ValueError as exc:
                    return _unverified(f"Unusable public input: {exc} ", public_report_path)
                spots = spots.filter((pl.col("ts") >= _month_bounds(ym)[0]) & (pl.col("ts") < _month_bounds(ym)[1]))
                if spots.height:
                    frames.append(normalize_spot_columns(spots))
            public_month_spots[(split, ym)] = _dedup_month_spots(frames)
            for source in PUBLIC_SOURCES:
                count = int(
                    (public_month_spots[(split, ym)]["source"] == source).sum()
                )
                report["public_spots"][split][source] += count

    for split in ("train", "eval"):
        for source in PUBLIC_SOURCES:
            if report["public_spots"][split][source] == 0:
                split_name = "training" if split == "train" else "evaluation"
                return _unverified(
                    f"Missing qualifying public {_SOURCE_DISPLAY[source]} spots for the {split_name} split.",
                    public_report_path,
                )

    # --- labels (public) ---
    print("building public labels...", file=stderr)
    train_labels_by_month = {
        ym: labels_builder(public_month_spots[("train", ym)]) for ym in train_months
    }
    eval_labels_by_month = {
        ym: labels_builder(public_month_spots[("eval", ym)]) for ym in eval_months
    }
    for ym, labels in eval_labels_by_month.items():
        report["eval_labels_by_month"][ym] = labels.height
        if labels.height == 0:
            return _unverified(
                f"Empty public labeled population for holdout month {ym}.",
                public_report_path,
            )
    train_labels = pl.concat(list(train_labels_by_month.values()), how="vertical_relaxed")
    eval_labels = pl.concat(list(eval_labels_by_month.values()), how="vertical_relaxed")
    try:
        enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours=0.0)
    except ValueError as exc:
        raise AcceptanceUsageError(str(exc)) from exc

    # --- storm coverage (definitive Kp is eval-stratification only) ---
    if kp_fetch is None:
        kp_fetch = fetch_definitive_kp
    kp = kp_fetch(Path(args.data_dir) / "cache")
    tagged = tag_storm_windows(eval_labels, kp).with_columns(
        pl.col("window_start").dt.truncate("3h").alias("block_start")
    )
    storm_rows = tagged.filter(pl.col("is_storm"))
    if storm_rows.height == 0:
        raise AcceptanceUsageError(
            "no evaluation row falls in a definitive Kp>=5 block -- acceptance "
            "requires a storm-bearing holdout"
        )
    rep_block = (
        storm_rows.group_by("block_start")
        .agg(pl.col("kp").first())
        .sort("kp", descending=True)
        .row(0, named=True)
    )
    report["storm"] = {
        "block_start_utc": rep_block["block_start"].isoformat(),
        "kp": float(rep_block["kp"]),
    }
    report["eval_cohort_sha256"] = eval_cohort_sha256(eval_labels)

    # --- features + public fits ---
    if omni_fetch is None:
        omni_fetch = fetch_omni2_range
    all_years = sorted({int(ym.split("-")[0]) for ym in train_months + eval_months})
    omni = omni_fetch(all_years[0], all_years[-1], Path(args.data_dir) / "cache")
    if matrix_builder is None:
        matrix_builder = build_feature_matrix
    train_matrix = matrix_builder(
        train_labels, full_history=train_labels, omni=omni, horizon_hours=0.0
    ).with_columns(pl.lit(1.0).alias("sample_weight"))
    eval_matrix = matrix_builder(
        eval_labels, full_history=eval_labels, omni=omni, horizon_hours=0.0
    )

    print("fitting public models...", file=stderr)
    climatology = (climatology_factory or ClimatologyModel)().fit(train_labels)
    gbt_public = (gbt_factory or GBTModel)().fit(train_matrix)
    p533 = (p533_factory or (lambda: P533Model(
        ssn_by_month=ssn_by_month(list(eval_months), Path(args.data_dir) / "cache"),
        cache_path=Path(args.data_dir) / "cache" / "p533_scores.parquet",
        max_workers=args.p533_workers,
    )))()

    print("scoring public arm...", file=stderr)
    try:
        report["scores"]["climatology"] = _score_arm("climatology", climatology, eval_matrix)
        report["scores"]["p533"] = _score_arm("p533", p533, eval_matrix)
        report["scores"]["gbt_public"] = _score_arm("gbt_public", gbt_public, eval_matrix)
    except AcceptanceUnverified as exc:
        return _unverified(exc.reason, public_report_path)

    clim, p533_s, gbt_s = (
        report["scores"]["climatology"], report["scores"]["p533"], report["scores"]["gbt_public"]
    )
    public_pass = (
        gbt_s["brier"] < clim["brier"] and gbt_s["brier"] < p533_s["brier"]
        and gbt_s["log_loss"] < clim["log_loss"] and gbt_s["log_loss"] < p533_s["log_loss"]
    )
    report["public_gate"] = "pass" if public_pass else "fail"
    if not public_pass:
        report["status"] = "fail"
        report["reason"] = (
            "public GBT failed to strictly beat both climatology and P.533 "
            "on both Brier and log-loss"
        )
        _write_acceptance_report(report, public_report_path)
        print(f"FAIL public M2 gate on {band} at h=0. Report: {public_report_path}")
        return 1
    report["status"] = "pass"
    _write_acceptance_report(report, public_report_path)
    print(f"PASS public M2 gate on {band} at h=0. Report: {public_report_path}")

    if cqdx_loader is None:
        return 0

    # --- optional cqdx arm: training-only augmentation, same public holdout ---
    print("building cqdx-augmented training...", file=stderr)
    aug_month_spots: dict[str, pl.DataFrame] = {}
    for ym in train_months:
        try:
            private = cqdx_loader(ym, band)
        except ValueError as exc:
            return _unverified(f"Unusable cqdx input: {exc} ", augmented_report_path)
        private = private.filter(
            (pl.col("ts") >= _month_bounds(ym)[0]) & (pl.col("ts") < _month_bounds(ym)[1])
        )
        aug_month_spots[ym] = _dedup_month_spots(
            [public_month_spots[("train", ym)], normalize_spot_columns(private)]
        )
    surviving_private = sum(
        int((aug_month_spots[ym]["source"] == "cqdx").sum()) for ym in train_months
    )
    report["public_spots"]["train"]["cqdx"] = surviving_private

    if surviving_private == 0:
        report["cqdx_comparison"] = {
            "training_spots": 0, "brier_delta": None, "status": "unverified",
        }
        return _unverified(
            "No distinct cqdx training spots survived deduplication against the public sources.",
            augmented_report_path,
        )

    train_labels_aug = pl.concat(
        [labels_builder(aug_month_spots[ym]) for ym in train_months], how="vertical_relaxed"
    )
    try:
        enforce_blocked_cv_gap(train_labels_aug, eval_labels, max_horizon_hours=0.0)
    except ValueError as exc:
        raise AcceptanceUsageError(str(exc)) from exc
    train_matrix_aug = matrix_builder(
        train_labels_aug, full_history=train_labels_aug, omni=omni, horizon_hours=0.0
    ).with_columns(pl.lit(1.0).alias("sample_weight"))

    print("fitting cqdx-augmented GBT...", file=stderr)
    gbt_aug = (gbt_factory or GBTModel)().fit(train_matrix_aug)
    print("scoring cqdx-augmented arm on the same public holdout...", file=stderr)
    try:
        report["scores"]["gbt_augmented"] = _score_arm("gbt_augmented", gbt_aug, eval_matrix)
    except AcceptanceUnverified as exc:
        return _unverified(exc.reason, augmented_report_path)

    brier_delta = report["scores"]["gbt_augmented"]["brier"] - gbt_s["brier"]
    comparison_pass = brier_delta <= 0.0  # full precision, pre-rounding
    report["cqdx_comparison"] = {
        "training_spots": surviving_private,
        "brier_delta": _r6(brier_delta),
        "status": "pass" if comparison_pass else "fail",
    }
    if not comparison_pass:
        report["status"] = "fail"
        report["reason"] = (
            "cqdx-augmented Brier regressed against the public-only fit on the "
            "same public evaluation cohort"
        )
        _write_acceptance_report(report, augmented_report_path)
        print(f"FAIL public M2 gate and cqdx Brier comparison on {band} at h=0. Report: {augmented_report_path}")
        return 1

    report["status"] = "pass"
    _write_acceptance_report(report, augmented_report_path)
    print(f"PASS public M2 gate and cqdx Brier comparison on {band} at h=0. Report: {augmented_report_path}")
    return 0


def _acceptance_wsprnet_loader(raw_dir: Path):
    def load(ym: str, band: str) -> pl.DataFrame:
        y, m = ym.split("-")
        p = raw_dir / f"wsprspots-{y}-{m}.csv.gz"
        if not p.exists():
            print(f"downloading {p.name}...")
            download_wsprnet_archive(int(y), int(m), p)
        return extract_wsprnet_bands(p, bands=[band])[band].spots

    return load


def _acceptance_rbn_loader(raw_dir: Path):
    def load(ym: str, band: str) -> pl.DataFrame:
        return _build_rbn_spots_for_month(ym, [band], raw_dir)

    return load


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
        "--acceptance-pskreporter-dir", type=Path, default=None,
        help="PRO-10 acceptance mode: run the public M2 gate (WSPRnet+RBN+PSKReporter "
        "only, one band from {20m,10m} at h=0, >=3 holdout months) instead of the "
        "band-group sweep; RBN is always enabled in this mode",
    )
    ap.add_argument(
        "--cqdx-dir", type=Path, default=None,
        help="PRO-10 acceptance mode only: cqdx backfill output directory "
        "(scripts/extract_cqdx.py) -- adds a second GBT fit trained on public "
        "training spots plus surviving cqdx training rows, scored on the same "
        "public holdout",
    )
    ap.add_argument(
        "--p533-workers", type=int, default=None,
        help="P533Model's ThreadPoolExecutor size (each worker forks an iturhfprop subprocess); "
        "defaults to os.cpu_count() (P533Model's own default) -- pass a smaller number to reduce "
        "CPU/memory contention on a machine running other concurrent load (see ADR 0006).",
    )
    args = ap.parse_args()

    if args.cqdx_dir is not None and args.acceptance_pskreporter_dir is None:
        ap.error("--cqdx-dir requires --acceptance-pskreporter-dir (acceptance mode)")
    if args.acceptance_pskreporter_dir is not None:
        try:
            code = run_acceptance(
                args,
                wsprnet_loader=_acceptance_wsprnet_loader(args.data_dir / "raw"),
                rbn_loader=_acceptance_rbn_loader(args.data_dir / "raw"),
                pskreporter_loader=lambda ym, band: load_pskreporter_month(
                    args.acceptance_pskreporter_dir, ym, band
                ),
                cqdx_loader=(
                    (lambda ym, band: load_cqdx_month(args.cqdx_dir, ym, band))
                    if args.cqdx_dir is not None else None
                ),
            )
        except AcceptanceUsageError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(code)

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
