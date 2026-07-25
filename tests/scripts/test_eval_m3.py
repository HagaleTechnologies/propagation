import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_m3 import (  # noqa: E402
    BAND_GROUPS,
    _band_group,
    enforce_blocked_cv_gap,
    write_band_group_reports,
)


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
