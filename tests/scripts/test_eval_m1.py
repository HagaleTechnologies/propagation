import gzip
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_m1 import _build_labels, write_slice_reports  # noqa: E402


class ConstantModel:
    def __init__(self, p):
        self._p = p

    def predict(self, labels):
        return labels.with_columns(pl.lit(self._p).cast(pl.Float64).alias("p_open"))


class AbstainingModel:
    """Abstains (null p_open) on every row — exercises the drop_nulls path."""

    def predict(self, labels):
        return labels.with_columns(pl.lit(None).cast(pl.Float64).alias("p_open"))


def _labels():
    ts = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "window_start": [ts, ts.replace(hour=1)],
            "tx_field": ["EM", "EM"], "rx_field": ["PM", "PM"],
            "band": ["20m", "20m"], "open": [1, 0],
            "is_storm": [True, False],
        },
        schema_overrides={"window_start": pl.Datetime("us", "UTC")},
    )


def test_write_slice_reports_writes_three_slices_two_models(tmp_path):
    models = {"climatology": ConstantModel(0.7), "p533": ConstantModel(0.6)}
    results = write_slice_reports(models, _labels(), tmp_path)
    assert set(results) == {"overall", "storm", "quiet"}
    for slice_name, per_model in results.items():
        assert set(per_model) == {"climatology", "p533"}
        assert (tmp_path / slice_name / "headline_table.csv").exists()
    # header + 1 row per model (climatology, p533) = 3 lines
    assert (tmp_path / "storm" / "headline_table.csv").read_text().count("\n") == 3


def test_write_slice_reports_skips_a_model_that_fully_abstains_in_a_slice(tmp_path, capsys):
    models = {"climatology": ConstantModel(0.7), "p533": AbstainingModel()}
    results = write_slice_reports(models, _labels(), tmp_path)
    assert "p533" not in results["overall"]
    assert "climatology" in results["overall"]
    assert "abstained on all" in capsys.readouterr().out


TRAIN_ROWS = [
    "1,1717243320,K1JT,FN20,-8,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
    "2,1717243320,W7YSB,DM42og,-10,14.097231,K7ZTM,DN41ab,50,0,983,174,14,ver,0",
]
EVAL_ROWS = [
    "4,1719835320,K1JT,FN20,-6,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
    "5,1719921720,W7YSB,DM42og,-9,14.097231,K7ZTM,DN41ab,50,0,983,174,14,ver,0",
]


def _write_gz(rows, path: Path) -> Path:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")
    return path


def test_build_labels_returns_unsampled_train_and_eval(tmp_path):
    train_archive = _write_gz(TRAIN_ROWS, tmp_path / "train.csv.gz")
    eval_archive = _write_gz(EVAL_ROWS, tmp_path / "eval.csv.gz")
    train_labels, eval_labels = _build_labels(
        {"train": train_archive, "eval": eval_archive}, band="20m"
    )
    assert train_labels.height > 0
    assert eval_labels.height > 0
    assert "p_open" not in train_labels.columns  # unscored, raw labels
