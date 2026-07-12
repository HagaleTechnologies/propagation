import csv

import numpy as np

from propagation.eval.report import write_headline_report


def test_write_headline_report_creates_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=200)
    y_true = (rng.uniform(0, 1, size=200) < y_prob).astype(float)

    result = write_headline_report(y_true, y_prob, "climatology-m0", tmp_path)

    table_path = tmp_path / "headline_table.csv"
    png_path = tmp_path / "reliability_climatology-m0.png"
    assert table_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 0

    with open(table_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["model"] == "climatology-m0"
    assert float(rows[-1]["brier"]) == result["brier"]
    assert 0 <= result["brier"] <= 1
    assert result["log_loss"] > 0


def test_write_headline_report_appends_multiple_models(tmp_path):
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    write_headline_report(y_true, y_prob, "climatology-m0", tmp_path)
    write_headline_report(y_true, y_prob, "p533-m1", tmp_path)
    with open(tmp_path / "headline_table.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["model"] for r in rows] == ["climatology-m0", "p533-m1"]
