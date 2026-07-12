import gzip
from pathlib import Path

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
    assert (report_dir / "headline_table.csv").exists()
    assert (report_dir / "reliability_climatology-m0.png").exists()
    assert result["n_train_labels"] > 0
    assert result["n_eval_labels"] > 0
    assert 0 <= result["headline"]["brier"] <= 1
    assert len(result["qa_results"]) == 8
