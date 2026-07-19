import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_m2 import enforce_blocked_cv_gap, write_three_model_slice_reports  # noqa: E402


class ConstantModel:
    def __init__(self, p):
        self._p = p

    def predict(self, labels):
        return labels.with_columns(pl.lit(self._p).cast(pl.Float64).alias("p_open"))


def test_write_three_model_slice_reports_writes_three_rows(tmp_path):
    ts = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
    labels = pl.DataFrame({
        "window_start": [ts, ts.replace(hour=1)],
        "tx_field": ["EM", "EM"], "rx_field": ["PM", "PM"],
        "band": ["20m", "20m"], "open": [1, 0], "is_storm": [True, False],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    models = {"climatology": ConstantModel(0.7), "p533": ConstantModel(0.6), "gbt": ConstantModel(0.5)}
    results = write_three_model_slice_reports(models, labels, tmp_path)
    assert set(results) == {"overall", "storm", "quiet"}
    for slice_name in results:
        table = (tmp_path / slice_name / "headline_table.csv").read_text()
        assert table.count("\n") == 4  # header + 3 model rows


def _labels_at(*hours_from_epoch):
    epoch = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    ts = [epoch + timedelta(hours=h) for h in hours_from_epoch]
    return pl.DataFrame(
        {"window_start": ts},
        schema_overrides={"window_start": pl.Datetime("us", "UTC")},
    )


def test_enforce_blocked_cv_gap_raises_on_adjacent_months():
    # train ends at hour 0, eval starts 15 minutes later — a real,
    # gap-violating split like adjacent-month CLI args would produce.
    train_labels = _labels_at(-72, -48, -24, 0)
    eval_labels = _labels_at(0.25, 24, 48)
    with pytest.raises(ValueError, match="blocked-CV gap violation"):
        enforce_blocked_cv_gap(train_labels, eval_labels)


def test_enforce_blocked_cv_gap_raises_on_overlapping_months():
    # eval_start before train_end: out-of-order/overlapping months.
    train_labels = _labels_at(0, 24, 48)
    eval_labels = _labels_at(-10, 10)
    with pytest.raises(ValueError, match="blocked-CV gap violation"):
        enforce_blocked_cv_gap(train_labels, eval_labels)


def test_enforce_blocked_cv_gap_passes_when_gap_sufficient():
    # 48h is the required floor for horizon=3h, AR lookback=24h (27 < 48).
    train_labels = _labels_at(-24, 0)
    eval_labels = _labels_at(48, 72)
    enforce_blocked_cv_gap(train_labels, eval_labels)  # must not raise
