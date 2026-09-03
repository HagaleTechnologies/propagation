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
