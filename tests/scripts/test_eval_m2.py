import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_m2 import write_three_model_slice_reports  # noqa: E402


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
