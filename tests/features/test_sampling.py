import datetime as dt
import hashlib

import duckdb
import polars as pl
import pytest

from propagation.data.lake import register_view
from propagation.features.sampling import sample_labels, stratum_seed, write_labels


def test_stratum_seed_matches_spec_formula():
    band, date = "20m", "2026-06-01"
    expected = int.from_bytes(
        hashlib.sha256(f"{band}|{date}".encode()).digest()[:4], "big"
    ) & 0xFFFFFFFF
    assert stratum_seed(band, date) == expected


def _labels_df(n_pos, n_neg, band="20m", date="2026-06-01"):
    ws = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    rows = []
    for i in range(n_pos):
        rows.append({
            "window_start": ws, "tx_field": "FN", "rx_field": f"P{i}", "band": band,
            "open": 1, "n_spots": 1, "n_monitors": 0, "n_tx_stations": 0,
            "evidence_tier": "wspr", "snr_ft8eq_p50": -10.0,
        })
    for i in range(n_neg):
        rows.append({
            "window_start": ws, "tx_field": "FN", "rx_field": f"N{i}", "band": band,
            "open": 0, "n_spots": 0, "n_monitors": 1, "n_tx_stations": 1,
            "evidence_tier": "wspr", "snr_ft8eq_p50": None,
        })
    return pl.DataFrame(rows, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


def test_sample_labels_downsamples_to_ratio():
    labels = _labels_df(n_pos=2, n_neg=20)
    sampled = sample_labels(labels, ratio=3.0)
    pos = sampled.filter(pl.col("open") == 1)
    neg = sampled.filter(pl.col("open") == 0)
    assert pos.height == 2
    assert neg.height == 6  # 3:1
    assert (pos["sample_weight"] == 1.0).all()
    assert neg["sample_weight"][0] == pytest.approx(20 / 6)


def test_sample_labels_keeps_all_when_under_ratio():
    labels = _labels_df(n_pos=5, n_neg=3)
    sampled = sample_labels(labels, ratio=3.0)
    assert sampled.filter(pl.col("open") == 0).height == 3
    assert (sampled.filter(pl.col("open") == 0)["sample_weight"] == 1.0).all()


def test_sample_labels_deterministic():
    labels = _labels_df(n_pos=2, n_neg=20)
    a = sample_labels(labels, ratio=3.0).sort("rx_field")
    b = sample_labels(labels, ratio=3.0).sort("rx_field")
    assert a["rx_field"].to_list() == b["rx_field"].to_list()


def test_write_labels_creates_hive_layout(tmp_path):
    labels = _labels_df(n_pos=1, n_neg=1)
    write_labels(labels, tmp_path)
    con = duckdb.connect(":memory:")
    register_view(con, "labels", str(tmp_path / "labels" / "**" / "*.parquet"))
    count = con.execute("SELECT count(*) FROM labels").fetchone()[0]
    assert count == 2
