import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_m3 import (  # noqa: E402
    BAND_GROUPS,
    _band_group,
    _build_rbn_spots_for_month,
    _download_rbn_month,
    _merge_band_spots,
    _rbn_month_archive_paths,
    enforce_blocked_cv_gap,
    write_band_group_reports,
)
from propagation.data.schema import SPOT_SCHEMA  # noqa: E402

RBN_FIXTURE = Path(__file__).parent.parent / "fixtures" / "rbn_sample.csv"

_FAKE_LOCATIONS = {
    "BD8CS": (39.9, 116.4), "BY8DX": (31.2, 121.5), "W1NT": (41.5, -71.3),
    "K9EI": (41.6, -87.0), "TI7W": (10.0, -84.0), "ND7K": (36.0, -112.0),
    "N4KS": (33.0, -84.0), "KM3T": (39.0, -76.0),
}


def _fake_resolver(call: str):
    return _FAKE_LOCATIONS.get(call)


def test_band_group_maps_every_band_to_exactly_one_group():
    from propagation.features.history import BAND_ORDER
    seen = set()
    for band in BAND_ORDER:
        group = _band_group(band)
        assert group in BAND_GROUPS
        assert band in BAND_GROUPS[group]
        seen.add(band)
    assert seen == set(BAND_ORDER)


def test_band_group_rejects_unknown_band():
    with pytest.raises(ValueError, match="not in any BAND_GROUPS"):
        _band_group("999m")


def _labels_at(*hours_from_epoch):
    epoch = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    ts = [epoch + timedelta(hours=h) for h in hours_from_epoch]
    return pl.DataFrame({"window_start": ts}, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


def test_enforce_blocked_cv_gap_scales_with_max_horizon():
    # max_horizon_hours=24 -> required gap = max(48, 24+24) = 48 (floor);
    # max_horizon_hours=30 -> required gap = max(48, 30+24) = 54, so an
    # exactly-48h gap that passed at horizon<=24 must now fail.
    train_labels = _labels_at(0)
    eval_labels = _labels_at(48)
    enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours=24.0)  # must not raise
    with pytest.raises(ValueError, match="blocked-CV gap violation"):
        enforce_blocked_cv_gap(train_labels, eval_labels, max_horizon_hours=30.0)


class ConstantModel:
    def __init__(self, p):
        self._p = p

    def predict(self, labels):
        return labels.with_columns(pl.lit(self._p).cast(pl.Float64).alias("p_open"))


def test_write_band_group_reports_writes_one_table_per_group(tmp_path):
    ts = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
    labels = pl.DataFrame({
        "window_start": [ts, ts, ts],
        "tx_field": ["EM", "EM", "EM"], "rx_field": ["PM", "PM", "PM"],
        "band": ["20m", "160m", "6m"], "open": [1, 0, 1],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    models = {"climatology": ConstantModel(0.7), "p533": ConstantModel(0.6), "gbt": ConstantModel(0.5)}
    results = write_band_group_reports(models, labels, horizon_hours=0.0, out_dir=tmp_path)
    assert set(results) == {"low", "mid", "high"}
    for group in ("low", "mid", "high"):
        table = (tmp_path / group / "h0" / "headline_table.csv").read_text()
        assert table.count("\n") == 4  # header + 3 model rows


def _spot_row(band: str, **overrides) -> dict:
    row = dict.fromkeys(SPOT_SCHEMA)
    row["band"] = band
    row.update(overrides)
    return row


def test_merge_band_spots_concats_matching_band():
    base = pl.DataFrame([_spot_row("20m", source="wsprnet")], schema=SPOT_SCHEMA)
    extra = pl.DataFrame(
        [_spot_row("20m", source="rbn"), _spot_row("40m", source="rbn")], schema=SPOT_SCHEMA
    )
    merged = _merge_band_spots(base, extra, band="20m")
    assert merged.height == 2
    assert set(merged["source"].to_list()) == {"wsprnet", "rbn"}


def test_merge_band_spots_no_matching_band_is_noop():
    base = pl.DataFrame([_spot_row("20m", source="wsprnet")], schema=SPOT_SCHEMA)
    extra = pl.DataFrame([_spot_row("40m", source="rbn")], schema=SPOT_SCHEMA)
    merged = _merge_band_spots(base, extra, band="20m")
    assert merged.height == 1


def test_merge_band_spots_none_or_empty_extra_is_noop():
    base = pl.DataFrame([_spot_row("20m", source="wsprnet")], schema=SPOT_SCHEMA)
    assert _merge_band_spots(base, None, band="20m").height == 1
    assert _merge_band_spots(base, pl.DataFrame(schema=SPOT_SCHEMA), band="20m").height == 1


def test_rbn_month_archive_paths_covers_every_day(tmp_path):
    entries = _rbn_month_archive_paths("2024-02", tmp_path)  # 2024 is a leap year
    assert len(entries) == 29
    assert entries[0] == (date(2024, 2, 1), tmp_path / "rbn-20240201.zip")
    assert entries[-1] == (date(2024, 2, 29), tmp_path / "rbn-20240229.zip")


def test_download_rbn_month_skips_files_that_already_exist(tmp_path, monkeypatch):
    import eval_m3

    def _boom(*a, **kw):
        raise AssertionError("download_rbn_archive should not be called for an existing file")

    monkeypatch.setattr(eval_m3, "download_rbn_archive", _boom)
    for _date, p in _rbn_month_archive_paths("2024-05", tmp_path):
        with zipfile.ZipFile(p, "w"):
            pass
    paths = _download_rbn_month("2024-05", tmp_path)
    assert len(paths) == 31  # May has 31 days
    assert all(p.exists() for p in paths)


def _not_found(*_args, **_kwargs):
    req = httpx.Request("GET", "https://example.invalid")
    resp = httpx.Response(404, request=req)
    raise httpx.HTTPStatusError("not found", request=req, response=resp)


def test_download_rbn_month_skips_a_day_with_no_archive_rather_than_aborting(tmp_path, monkeypatch):
    import eval_m3

    monkeypatch.setattr(eval_m3, "download_rbn_archive", _not_found)
    paths = _download_rbn_month("2024-01", tmp_path)
    assert paths == []  # every day 404s -- skipped, not raised


def test_build_rbn_spots_for_month_aggregates_across_days_and_filters_band(tmp_path, monkeypatch):
    import eval_m3

    for day in (1, 2):
        zip_path = tmp_path / f"rbn-202401{day:02d}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(f"202401{day:02d}.csv", RBN_FIXTURE.read_text())
    monkeypatch.setattr(eval_m3, "download_rbn_archive", _not_found)  # remaining days 404

    spots = _build_rbn_spots_for_month("2024-01", ["20m"], tmp_path, resolve_location=_fake_resolver)
    # 4 qualifying 20m spots per day per test_rbn.py's fixture-derived count, x 2 days
    assert spots.height == 8
    assert set(spots["band"].unique().to_list()) == {"20m"}
    assert set(spots["source"].unique().to_list()) == {"rbn"}


def test_build_rbn_spots_for_month_empty_when_no_archives_available(tmp_path, monkeypatch):
    import eval_m3

    monkeypatch.setattr(eval_m3, "download_rbn_archive", _not_found)
    spots = _build_rbn_spots_for_month("2024-01", ["20m"], tmp_path, resolve_location=_fake_resolver)
    assert spots.height == 0
    assert spots.schema == pl.DataFrame(schema=SPOT_SCHEMA).schema


# ---------------------------------------------------------------------------
# PRO-10 acceptance mode (public M2 gate + optional cqdx comparison)
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import gzip as _gzip  # noqa: E402
import io as _io  # noqa: E402
import json as _json  # noqa: E402

from eval_m3 import (  # noqa: E402
    AcceptanceUsageError,
    build_acceptance_labels,
    eval_cohort_sha256,
    load_cqdx_month,
    load_pskreporter_month,
    run_acceptance,
)

_ACC_TRAIN = ("2026-01",)
_ACC_EVAL = ("2026-03", "2026-04", "2026-05")


def _acc_spot(ts, source, dx="K1BBB", de="W1AAA", dxg="DM79", deg="FN31", band="20m", mode="FT8", snr=-10):
    return {
        "source": source, "ts": ts, "band": band, "mode": mode, "freq_hz": 14074000,
        "dx_call": dx, "de_call": de, "dx_grid": dxg, "de_grid": deg,
        "dx_field": None, "de_field": None, "dx_lat": None, "dx_lon": None,
        "de_lat": None, "de_lon": None, "snr_db": snr, "tx_dbm": None,
    }


def _acc_windows(ym, days=(2, 3)):
    y, m = (int(x) for x in ym.split("-"))
    return [datetime(y, m, day, hour, 5, tzinfo=timezone.utc) for day in days for hour in (0, 4)]


def _acc_month_spots(ym, days=(2, 3)):
    """Three public sources, each with unique transmissions in four 15-min
    windows: two positive cells plus crossed (monitor x evidence) negatives
    per window."""
    rows = []
    for ts in _acc_windows(ym, days):
        rows.append(_acc_spot(ts, "wsprnet", dx="K1BBB", de="W1AAA", dxg="DM79", deg="FN31"))
        rows.append(_acc_spot(ts, "rbn", dx="K4GGG", de="W3FFF", dxg="EM12", deg="EN50"))
        rows.append(_acc_spot(ts, "pskreporter", dx="K7HHH", de="W2EEE", dxg="EM12", deg="FN31"))
    return pl.DataFrame(rows, schema=SPOT_SCHEMA)


def _private_month_spots(ym):
    """cqdx backfill output rows for one month: one unique private training
    transmission, one exact duplicate of a public wsprnet transmission (same
    dedup key, cqdx loses priority), and one private spot whose observation
    falls in a HOLDOUT month (must never reach training)."""
    train_ts = _acc_windows(ym)[0]
    holdout_ts = _acc_windows("2026-03")[0]
    rows = [
        _acc_spot(train_ts, "cqdx", dx="K8JJJ", de="W1AAA", dxg="DM79", deg="FN31"),
        _acc_spot(train_ts, "cqdx", dx="K1BBB", de="W1AAA", dxg="DM79", deg="FN31", snr=-3),
        _acc_spot(holdout_ts, "cqdx", dx="K9KKK", de="W1AAA", dxg="DM79", deg="FN31"),
    ]
    return pl.DataFrame(rows, schema=SPOT_SCHEMA)


def _write_psk_hours(psk_dir, ym):
    psk_dir.mkdir(parents=True, exist_ok=True)
    frame = _acc_month_spots(ym).filter(pl.col("source") == "pskreporter")
    for ts in _acc_windows(ym):
        hour_rows = frame.filter(pl.col("ts").dt.truncate("1h") == ts.replace(minute=0))
        hour_rows.write_parquet(psk_dir / f"pskreporter-{ts:%Y-%m-%dT%H}.parquet")


def _kp_fixture():
    blocks, kps = [], []
    for ym in (*_ACC_TRAIN, *_ACC_EVAL):
        y, m = (int(x) for x in ym.split("-"))
        blocks += [
            datetime(y, m, 2, 0, tzinfo=timezone.utc), datetime(y, m, 2, 3, tzinfo=timezone.utc),
            datetime(y, m, 3, 0, tzinfo=timezone.utc), datetime(y, m, 3, 3, tzinfo=timezone.utc),
        ]
        kps += [6.7, 2.0, 1.0, 1.3]
    return pl.DataFrame(
        {"block_start": blocks, "kp": kps},
        schema_overrides={"block_start": pl.Datetime("us", "UTC")},
    )


def _omni_fixture():
    times = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=d) for d in range(0, 200, 7)]
    return pl.DataFrame(
        {"time": times, "kp": [2.0] * len(times), "f107": [150.0] * len(times),
         "bz_gsm": [0.0] * len(times), "solar_wind_speed": [400.0] * len(times),
         "dst": [-5.0] * len(times)},
        schema_overrides={"time": pl.Datetime("us", "UTC")},
    )


class FakeModel:
    """Constant or per-row prediction stub with fit() recording its input."""

    model_id = "fake"

    def __init__(self, p_open):
        self._p_open = p_open
        self.fit_input = None

    def fit(self, matrix):
        self.fit_input = matrix
        return self

    def predict(self, cohort):
        if callable(self._p_open):
            ps = [self._p_open(row) for row in cohort.iter_rows(named=True)]
        else:
            ps = [self._p_open] * cohort.height
        return cohort.with_columns(pl.Series("p_open", ps, dtype=pl.Float64))


def _identity_matrix(labels, full_history, omni, horizon_hours):
    return labels


def _acc_args(tmp_path, bands=("20m",), horizons=(0.0,), train=_ACC_TRAIN, eval_months=_ACC_EVAL):
    return argparse.Namespace(
        bands=list(bands), horizons=list(horizons), train_months=list(train),
        eval_months=list(eval_months), data_dir=tmp_path / "data", p533_workers=1,
    )


def _public_loaders(spot_frames):
    """Loaders returning pre-built spot frames, one source each."""
    def loader(source):
        return lambda ym, band: spot_frames[ym].filter(
            (pl.col("source") == source) & (pl.col("band") == band)
        )
    return loader("wsprnet"), loader("rbn"), loader("pskreporter")


def _run_acc(args, *, wspr, rbn, psk, kp=None, omni=None, clim=None, gbt=None, p533=None,
             cqdx_loader=None, labels_builder=build_acceptance_labels,
             matrix_builder=_identity_matrix, stderr=None):
    return run_acceptance(
        args,
        wsprnet_loader=wspr,
        rbn_loader=rbn,
        pskreporter_loader=psk,
        cqdx_loader=cqdx_loader,
        kp_fetch=kp or (lambda cache_dir: _kp_fixture()),
        omni_fetch=omni or (lambda y0, y1, cache_dir: _omni_fixture()),
        climatology_factory=clim or (lambda: FakeModel(0.5)),
        gbt_factory=gbt or (lambda: FakeModel(lambda row: float(row["open"]))),
        p533_factory=p533 or (lambda: FakeModel(0.6)),
        labels_builder=labels_builder,
        matrix_builder=matrix_builder,
        stderr=stderr if stderr is not None else _io.StringIO(),
    )


def _spot_frames(months):
    return {ym: _acc_month_spots(ym) for ym in months}


def test_public_acceptance_uses_only_public_sources(tmp_path, monkeypatch):
    # R2 client creation must fail loudly if anything in this path touches it
    import builtins

    real_import = builtins.__import__

    def _no_boto(name, *a, **kw):
        if name == "boto3":
            raise AssertionError("acceptance mode must not import boto3 or contact R2")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_boto)

    psk_dir = tmp_path / "inputs" / "pskreporter"
    for ym in (*_ACC_TRAIN, *_ACC_EVAL):
        _write_psk_hours(psk_dir, ym)
    # private parquet sitting BESIDE the public inputs must be ignored
    cqdx_dir = tmp_path / "inputs" / "beside"
    cqdx_dir.mkdir(parents=True)
    _private_month_spots(_ACC_TRAIN[0]).write_parquet(cqdx_dir / "cqdx-2026-01-02.parquet")

    spot_frames = _spot_frames((*_ACC_TRAIN, *_ACC_EVAL))
    wspr, rbn, _ = _public_loaders(spot_frames)
    seen_sources = []

    def spy_labels(spots):
        seen_sources.extend(spots["source"].unique().to_list())
        return build_acceptance_labels(spots)

    args = _acc_args(tmp_path)
    code = _run_acc(
        args, wspr=wspr, rbn=rbn,
        psk=lambda ym, band: load_pskreporter_month(psk_dir, ym, band),
        labels_builder=spy_labels,
    )
    assert code == 0
    assert set(seen_sources) == {"wsprnet", "rbn", "pskreporter"}

    report = _json.loads((tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m" / "public.json").read_text())
    assert report["status"] == "pass" and report["cqdx_comparison"] is None
    for split in ("train", "eval"):
        assert report["public_spots"][split]["cqdx"] == 0
        for source in ("wsprnet", "rbn", "pskreporter"):
            assert report["public_spots"][split][source] > 0
    assert set(report["eval_labels_by_month"]) == set(_ACC_EVAL)
    assert all(n > 0 for n in report["eval_labels_by_month"].values())
    assert report["eval_cohort_sha256"] == eval_cohort_sha256(
        pl.concat([build_acceptance_labels(spot_frames[ym]) for ym in _ACC_EVAL])
    )
    assert report["storm"]["kp"] == 6.7
    # public GBT (perfect) beats both constant baselines on both metrics
    assert report["scores"]["gbt_public"]["brier"] < report["scores"]["climatology"]["brier"]
    assert report["public_gate"] == "pass"


def test_acceptance_rejects_mislabeled_input(tmp_path):
    psk_dir = tmp_path / "psk"
    for ym in (*_ACC_TRAIN, *_ACC_EVAL):
        _write_psk_hours(psk_dir, ym)
    # a cqdx backfill file renamed into the accumulator's naming, and a
    # pskreporter-named file containing cqdx-source rows
    _private_month_spots(_ACC_TRAIN[0]).write_parquet(psk_dir / "pskreporter-2026-01-02T00.parquet")
    spot_frames = _spot_frames((*_ACC_TRAIN, *_ACC_EVAL))
    wspr, rbn, _ = _public_loaders(spot_frames)
    code = _run_acc(
        _acc_args(tmp_path), wspr=wspr, rbn=rbn,
        psk=lambda ym, band: load_pskreporter_month(psk_dir, ym, band),
    )
    assert code == 2
    report = _json.loads((tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m" / "public.json").read_text())
    assert report["status"] == "unverified"
    assert "refusing mislabeled input" in report["reason"]

    # the cqdx loader rejects public rows just as symmetrically
    mixed = pl.concat(
        [_private_month_spots("2026-01"), _acc_month_spots("2026-01").head(1)], how="vertical_relaxed"
    )
    d = tmp_path / "cqdx"
    d.mkdir()
    mixed.write_parquet(d / "cqdx-2026-01-02.parquet")
    with pytest.raises(ValueError, match="refusing mislabeled input"):
        load_cqdx_month(d, "2026-01", "20m")


def test_public_gate_requires_both_baselines_and_metrics(tmp_path):
    spot_frames = _spot_frames((*_ACC_TRAIN, *_ACC_EVAL))
    wspr, rbn, psk = _public_loaders(spot_frames)
    report_dir = tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m"

    def run(clim_p, gbt_fn, p533_p):
        return _run_acc(
            _acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk,
            clim=lambda: FakeModel(clim_p), gbt=lambda: FakeModel(gbt_fn), p533=lambda: FakeModel(p533_p),
        )

    perfect = lambda row: float(row["open"])  # noqa: E731
    inverted = lambda row: 1.0 - float(row["open"])  # noqa: E731

    # baseline: strict win on both metrics -> pass
    assert run(0.5, perfect, 0.6) == 0
    # tie with climatology on both metrics -> fail (strict inequality required)
    assert run(0.5, lambda row: 0.5, 0.6) == 1
    # loss to climatology on both metrics -> fail
    assert run(0.5, inverted, 0.6) == 1
    # beats climatology, loses to P.533 on both metrics -> fail
    assert run(0.3, lambda row: 0.5, 0.9) == 1

    # loss on log-loss alone: near-perfect brier (two confidently wrong
    # open=0 rows) while log-loss blows past both constants
    class TwoConfidentMistakes(FakeModel):
        def predict(self, cohort):
            ps, wrong_seen = [], 0
            for row in cohort.iter_rows(named=True):
                if row["open"] == 0 and wrong_seen < 2:
                    ps.append(1.0 - 1e-12)  # confidently wrong on a closed path
                    wrong_seen += 1
                elif row["open"] == 0:
                    ps.append(0.2)
                else:
                    ps.append(0.999999)
            return cohort.with_columns(pl.Series("p_open", ps, dtype=pl.Float64))

    code = _run_acc(
        _acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk,
        clim=lambda: FakeModel(0.9), gbt=lambda: TwoConfidentMistakes(None), p533=lambda: FakeModel(0.8),
    )
    assert code == 1
    report = _json.loads((report_dir / "public.json").read_text())
    assert report["public_gate"] == "fail" and report["status"] == "fail"
    assert report["reason"] and "both Brier and log-loss" in report["reason"]
    assert report["scores"]["gbt_public"]["brier"] < report["scores"]["climatology"]["brier"]
    assert report["scores"]["gbt_public"]["brier"] < report["scores"]["p533"]["brier"]
    assert report["scores"]["gbt_public"]["log_loss"] > report["scores"]["climatology"]["log_loss"]


def test_acceptance_rejects_invalid_holdout(tmp_path):
    spot_frames = _spot_frames((*_ACC_TRAIN, *_ACC_EVAL))
    wspr, rbn, psk = _public_loaders(spot_frames)

    def expect_usage(args, frames=None):
        w, r, p = _public_loaders(frames or spot_frames)
        with pytest.raises(AcceptanceUsageError):
            _run_acc(args, wspr=w, rbn=r, psk=p)

    # wrong band / multiple bands / nonzero horizon
    expect_usage(_acc_args(tmp_path, bands=("40m",)))
    expect_usage(_acc_args(tmp_path, bands=("20m", "10m")))
    expect_usage(_acc_args(tmp_path, horizons=(0.0, 3.0)))
    # fewer than three eval months / repeated months / overlapping splits
    expect_usage(_acc_args(tmp_path, eval_months=("2026-03", "2026-04")))
    expect_usage(_acc_args(tmp_path, eval_months=("2026-03", "2026-03", "2026-04")))
    expect_usage(_acc_args(tmp_path, train=("2026-03",), eval_months=_ACC_EVAL))

    # short gap: train ends late on 2026-03-31, eval starts early on
    # 2026-04-01 -- under the required 48h blocked-CV clearance
    tight_frames = {
        "2026-03": _acc_month_spots("2026-03", days=(31,)),
        "2026-04": _acc_month_spots("2026-04", days=(1,)),
        "2026-05": _acc_month_spots("2026-05"),
        "2026-06": _acc_month_spots("2026-06"),
    }
    train_end = build_acceptance_labels(tight_frames["2026-03"])["window_start"].max()
    eval_start = build_acceptance_labels(tight_frames["2026-04"])["window_start"].min()
    assert (eval_start - train_end).total_seconds() / 3600 < 48
    expect_usage(
        _acc_args(tmp_path, train=("2026-03",), eval_months=("2026-04", "2026-05", "2026-06")),
        tight_frames,
    )

    # no storm: kp fixture without any Kp>=5 block -> invalid holdout, exit 2
    quiet_kp = _kp_fixture().with_columns(pl.lit(1.0).alias("kp"))
    with pytest.raises(AcceptanceUsageError, match="storm"):
        _run_acc(_acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk, kp=lambda cd: quiet_kp)

    # empty holdout month: an eval month whose spots produce no labels
    empty_frames = dict(spot_frames)
    empty_frames["2026-04"] = pl.DataFrame(schema=SPOT_SCHEMA)
    w, r, p = _public_loaders(empty_frames)
    code = _run_acc(_acc_args(tmp_path), wspr=w, rbn=r, psk=p)
    assert code == 2
    report = _json.loads((tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m" / "public.json").read_text())
    assert report["status"] == "unverified" and "2026-04" in report["reason"]


def test_cqdx_changes_training_only_and_preserves_public_holdout(tmp_path):
    spot_frames = _spot_frames((*_ACC_TRAIN, *_ACC_EVAL))
    wspr, rbn, psk = _public_loaders(spot_frames)
    cqdx_dir = tmp_path / "cqdx"
    cqdx_dir.mkdir()
    _private_month_spots(_ACC_TRAIN[0]).write_parquet(cqdx_dir / "cqdx-2026-01-02.parquet")

    builder_inputs = []

    def spy_labels(spots):
        builder_inputs.append(spots)
        return build_acceptance_labels(spots)

    class RecordingGBT(FakeModel):
        pass

    public_fit_inputs = []

    def gbt_factory():
        model = RecordingGBT(lambda row: float(row["open"]))
        orig_fit = model.fit

        def fit(matrix):
            public_fit_inputs.append(matrix)
            return orig_fit(matrix)

        model.fit = fit
        return model

    code = _run_acc(
        _acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk,
        cqdx_loader=lambda ym, band: load_cqdx_month(cqdx_dir, ym, band),
        labels_builder=spy_labels, gbt=gbt_factory,
    )
    assert code == 0

    # augmented month spots reached the label builder with ONLY unique rows:
    # the public duplicate survived, the cqdx copy dropped, the holdout-month
    # private spot never loaded
    aug_inputs = [f for f in builder_inputs if (f["source"] == "cqdx").any()]
    assert len(aug_inputs) == len(_ACC_TRAIN)
    for frame in aug_inputs:
        dupes = frame.with_columns(pl.col("ts").dt.truncate("15m").alias("_w")).group_by(
            ["dx_call", "de_call", "band", "mode", "_w"]
        ).len().filter(pl.col("len") > 1)
        assert dupes.height == 0
        assert (frame["source"] == "cqdx").sum() == 1

    report = _json.loads((tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m" / "augmented.json").read_text())
    assert report["status"] == "pass"
    # only K8JJJ survives: the K1BBB duplicate loses to the public copy
    # (source priority) and K9KKK lives in a holdout month, never loaded
    assert report["cqdx_comparison"]["training_spots"] == 1
    public = _json.loads((tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m" / "public.json").read_text())
    assert report["eval_cohort_sha256"] == public["eval_cohort_sha256"]
    assert report["public_spots"]["eval"] == public["public_spots"]["eval"]
    assert report["public_gate"] == "pass"
    assert report["cqdx_comparison"]["brier_delta"] <= 0


def test_augmented_gate_uses_paired_unrounded_brier(tmp_path):
    spot_frames = _spot_frames((*_ACC_TRAIN, *_ACC_EVAL))
    wspr, rbn, psk = _public_loaders(spot_frames)
    cqdx_dir = tmp_path / "cqdx"
    cqdx_dir.mkdir()
    _private_month_spots(_ACC_TRAIN[0]).write_parquet(cqdx_dir / "cqdx-2026-01-02.parquet")
    cqdx = lambda ym, band: load_cqdx_month(cqdx_dir, ym, band)  # noqa: E731
    report_dir = tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m"

    # exact tie on the paired cohort -> pass at zero
    same = lambda row: float(row["open"])  # noqa: E731
    code = _run_acc(_acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk, cqdx_loader=cqdx, gbt=lambda: FakeModel(same))
    assert code == 0
    report = _json.loads((report_dir / "augmented.json").read_text())
    assert report["cqdx_comparison"]["status"] == "pass"
    assert report["cqdx_comparison"]["brier_delta"] == 0.0

    # positive delta hidden by six-decimal rounding: nudge the augmented arm
    # by +1e-9 per row -- full-precision delta is positive, rounded report
    # shows 0.0, and the gate must still fail
    class TinyWorse(FakeModel):
        def predict(self, cohort):
            # one tick off perfect, inside [0,1]: positive full-precision
            # brier delta that six-decimal rounding displays as 0.0
            ps = [
                (1.0 - 1e-9) if row["open"] == 1 else 1e-9
                for row in cohort.iter_rows(named=True)
            ]
            return cohort.with_columns(pl.Series("p_open", ps, dtype=pl.Float64))

    models = iter([FakeModel(same), TinyWorse(None)])
    code = _run_acc(
        _acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk, cqdx_loader=cqdx,
        gbt=lambda: next(models),
    )
    assert code == 1
    report = _json.loads((report_dir / "augmented.json").read_text())
    assert report["cqdx_comparison"]["brier_delta"] == 0.0  # rounding displays 0.0
    assert report["cqdx_comparison"]["status"] == "fail"
    assert report["status"] == "fail"

    # duplicate-only backfill: every private row duplicates a public
    # transmission, so nothing survives dedup -> unverified
    dup_dir = tmp_path / "cqdx-dup"
    dup_dir.mkdir()
    dup_rows = _acc_month_spots(_ACC_TRAIN[0]).with_columns(pl.lit("cqdx").alias("source"))
    dup_rows.write_parquet(dup_dir / "cqdx-2026-01-02.parquet")
    code = _run_acc(
        _acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk,
        cqdx_loader=lambda ym, band: load_cqdx_month(dup_dir, ym, band),
    )
    assert code == 2
    report = _json.loads((report_dir / "augmented.json").read_text())
    assert report["cqdx_comparison"]["status"] == "unverified"
    assert report["cqdx_comparison"]["training_spots"] == 0


def test_acceptance_marks_missing_sources_or_predictions_unverified(tmp_path):
    spot_frames = _spot_frames((*_ACC_TRAIN, *_ACC_EVAL))
    wspr, rbn, psk = _public_loaders(spot_frames)
    report_dir = tmp_path / "data" / "reports" / "m3" / "acceptance" / "20m"

    # missing public source: RBN contributes nothing anywhere -> unverified
    no_rbn = lambda ym, band: pl.DataFrame(schema=SPOT_SCHEMA)  # noqa: E731
    code = _run_acc(_acc_args(tmp_path), wspr=wspr, rbn=no_rbn, psk=psk)
    assert code == 2
    report = _json.loads((report_dir / "public.json").read_text())
    assert report["status"] == "unverified"
    assert "RBN" in report["reason"]

    # missing predictions: a baseline that abstains on one row -> unverified
    class Abstainer(FakeModel):
        def predict(self, cohort):
            ps = [None] + [0.6] * (cohort.height - 1)
            return cohort.with_columns(pl.Series("p_open", ps, dtype=pl.Float64))

    code = _run_acc(_acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk, p533=lambda: Abstainer(None))
    assert code == 2
    report = _json.loads((report_dir / "public.json").read_text())
    assert report["status"] == "unverified" and "p533" in report["reason"]

    # a GBT dropping one evaluation row entirely -> unverified
    class Dropper(FakeModel):
        def predict(self, cohort):
            return cohort.head(cohort.height - 1).with_columns(
                pl.lit(0.5).cast(pl.Float64).alias("p_open")
            )

    code = _run_acc(_acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk, gbt=lambda: Dropper(None))
    assert code == 2
    report = _json.loads((report_dir / "public.json").read_text())
    assert report["status"] == "unverified" and "cover the evaluation cohort" in report["reason"]

    # and the augmented arm abstaining while the public arm passes
    cqdx_dir = tmp_path / "cqdx"
    cqdx_dir.mkdir()
    _private_month_spots(_ACC_TRAIN[0]).write_parquet(cqdx_dir / "cqdx-2026-01-02.parquet")
    perfect = lambda row: float(row["open"])  # noqa: E731
    models = iter([FakeModel(perfect), Abstainer(None)])
    code = _run_acc(
        _acc_args(tmp_path), wspr=wspr, rbn=rbn, psk=psk,
        cqdx_loader=lambda ym, band: load_cqdx_month(cqdx_dir, ym, band),
        gbt=lambda: next(models),
    )
    assert code == 2
    report = _json.loads((report_dir / "augmented.json").read_text())
    assert report["status"] == "unverified" and "gbt_augmented" in report["reason"]


def test_default_eval_preserves_existing_reports_and_does_not_discover_optional_inputs(tmp_path, monkeypatch):
    import eval_m3 as m3

    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    # optional inputs sitting under data/ must never be picked up by default
    psk_dir = data_dir / "raw" / "pskreporter"
    _write_psk_hours(psk_dir, "2026-01")
    (data_dir / "cqdx").mkdir()
    _private_month_spots("2026-01").write_parquet(data_dir / "cqdx" / "cqdx-2026-01-02.parquet")

    def _archive(rows, y, mo):
        text = "".join(
            f"1,{int(ts.timestamp())},{de},{deg},{snr},14.097100,{dx},{dxg},30,0,3000,100,14,0.7_r752,0\n"
            for ts, dx, dxg, de, deg, snr in rows
        )
        path = raw / f"wsprspots-{y:04d}-{mo:02d}.csv.gz"
        path.write_bytes(_gzip.compress(text.encode()))
        return path

    jan = datetime(2026, 1, 15, 0, tzinfo=timezone.utc)
    mar = datetime(2026, 3, 15, 0, tzinfo=timezone.utc)
    _archive([(jan, "K1BBB", "DM79", "W1AAA", "FN31", -10)], 2026, 1)
    _archive([(mar, "K1BBB", "DM79", "W1AAA", "FN31", -12)], 2026, 3)

    for forbidden in ("run_acceptance", "load_pskreporter_month", "load_cqdx_month"):
        monkeypatch.setattr(
            m3, forbidden,
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError(f"{forbidden} must not run in default mode")),
        )

    class FakeP533:
        model_id = "p533"

        def predict(self, cohort):
            return cohort.with_columns(pl.lit(0.6).cast(pl.Float64).alias("p_open"))

    class FakeGBT:
        model_id = "gbt"

        def fit(self, matrix):
            return self

        def predict(self, cohort):
            return cohort.with_columns(pl.lit(0.5).cast(pl.Float64).alias("p_open"))

    monkeypatch.setattr(m3, "P533Model", lambda **kw: FakeP533())
    monkeypatch.setattr(m3, "GBTModel", FakeGBT)
    monkeypatch.setattr(m3, "fetch_omni2_range", lambda y0, y1, cache_dir: _omni_fixture())
    monkeypatch.setattr(m3, "ssn_by_month", lambda months, cache_dir: {})

    argv = [
        "eval_m3.py", "--bands", "20m", "--horizons", "0",
        "--train-months", "2026-01", "--eval-months", "2026-03",
        "--data-dir", str(data_dir),
    ]
    monkeypatch.setattr("sys.argv", argv)
    import contextlib

    out = _io.StringIO()
    with contextlib.redirect_stdout(out):
        m3.main()

    table = data_dir / "reports" / "m3" / "mid" / "h0" / "headline_table.csv"
    assert table.exists()
    assert not (data_dir / "reports" / "m3" / "acceptance").exists()
    assert "wrote" in out.getvalue()
