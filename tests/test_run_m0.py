import gzip
from pathlib import Path

import duckdb
import pytest

from propagation.data.lake import register_view
from scripts.run_m0 import run_m0

TRAIN_ROWS = [
    "1,1717243320,K1JT,FN20,-8,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
    "2,1717243320,W7YSB,DM42og,-10,14.097231,K7ZTM,DN41ab,50,0,983,174,14,ver,0",
    "3,1717329720,K1JT,FN20,-8,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
]
# Eval month rows (~5 weeks after train, respecting the >=48h gap trivially since
# whole months are used)
EVAL_ROWS = [
    "4,1719835320,K1JT,FN20,-6,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
    "5,1719921720,W7YSB,DM42og,-9,14.097231,K7ZTM,DN41ab,50,0,983,174,14,ver,0",
]
# Same station pairs as TRAIN_ROWS, but timestamped only 5h after train's last
# row (1717329720 + 5h = 1717347720) -- well under the 48h leakage-safety floor.
CLOSE_EVAL_ROWS = [
    "6,1717347720,K1JT,FN20,-6,14.097100,W6SZ,DM14ed,50,0,3086,79,14,ver,0",
    "7,1717348020,W7YSB,DM42og,-9,14.097231,K7ZTM,DN41ab,50,0,983,174,14,ver,0",
]


def _write_gz(rows, path: Path) -> Path:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")
    return path


def test_run_m0_end_to_end(tmp_path):
    train_archive = _write_gz(TRAIN_ROWS, tmp_path / "train.csv.gz")
    eval_archive = _write_gz(EVAL_ROWS, tmp_path / "eval.csv.gz")
    lake_root = tmp_path / "lake"
    report_dir = tmp_path / "reports"

    result = run_m0(
        archives={"train": train_archive, "eval": eval_archive},
        band="20m",
        lake_root=lake_root,
        report_dir=report_dir,
    )

    assert (lake_root / "spots").exists()
    assert (lake_root / "labels").exists()
    assert (lake_root / "receiver_uptime").exists()
    assert (report_dir / "headline_table.csv").exists()
    assert (report_dir / "reliability_climatology-m0.png").exists()
    assert result["n_train_labels"] > 0
    assert result["n_eval_labels"] > 0
    assert 0 <= result["headline"]["brier"] <= 1
    assert len(result["qa_results"]) == 8

    # The labels table must have one uniform schema across train and eval
    # partitions: read_parquet(..., hive_partitioning=true) over the whole glob
    # must succeed without union_by_name, and both splits must be queryable.
    con = duckdb.connect(":memory:")
    register_view(con, "labels", str(lake_root / "labels" / "**" / "*.parquet"))
    splits = {row[0] for row in con.execute("SELECT DISTINCT split FROM labels").fetchall()}
    assert splits == {"train", "eval"}


def test_run_m0_rejects_insufficient_train_eval_gap(tmp_path):
    """docs/SPEC-labeling.md sec 6 rule 1: train/eval must be separated by >=48h
    (the current floor, since M0 has no horizon/AR-lookback features). This is
    the enforcement point that guards against a future milestone silently
    reusing an adjacent, leakage-unsafe train/eval split."""
    train_archive = _write_gz(TRAIN_ROWS, tmp_path / "train.csv.gz")
    eval_archive = _write_gz(CLOSE_EVAL_ROWS, tmp_path / "eval_close.csv.gz")

    with pytest.raises(AssertionError, match="leakage-safety floor"):
        run_m0(
            archives={"train": train_archive, "eval": eval_archive},
            band="20m",
            lake_root=tmp_path / "lake",
            report_dir=tmp_path / "reports",
        )
