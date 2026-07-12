import duckdb
import polars as pl

from propagation.data.lake import register_view, write_partitioned


def test_write_partitioned_creates_hive_layout(tmp_path):
    df = pl.DataFrame({
        "band": ["20m", "20m", "40m"],
        "date": ["2026-06-01", "2026-06-02", "2026-06-01"],
        "value": [1, 2, 3],
    })
    write_partitioned(df, tmp_path, "spots", ["band", "date"])
    assert (tmp_path / "spots" / "band=20m" / "date=2026-06-01" / "part-0.parquet").exists()
    assert (tmp_path / "spots" / "band=20m" / "date=2026-06-02" / "part-0.parquet").exists()
    assert (tmp_path / "spots" / "band=40m" / "date=2026-06-01" / "part-0.parquet").exists()


def test_register_view_queryable(tmp_path):
    df = pl.DataFrame({
        "band": ["20m", "40m"],
        "date": ["2026-06-01", "2026-06-01"],
        "value": [1, 2],
    })
    write_partitioned(df, tmp_path, "spots", ["band", "date"])
    con = duckdb.connect(":memory:")
    register_view(con, "spots", str(tmp_path / "spots" / "**" / "*.parquet"))
    result = con.execute("SELECT band, value FROM spots ORDER BY band").fetchall()
    assert result == [("20m", 1), ("40m", 2)]
