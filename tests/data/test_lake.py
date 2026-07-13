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


def test_register_view_date_partition_type_coercion(tmp_path):
    """Verify that DuckDB infers partition columns from their path representation.

    When a DataFrame with a string-typed date column is partitioned by that column,
    hive_partitioning=true infers the partition column as DuckDB DATE (not VARCHAR).
    """
    import datetime

    df = pl.DataFrame({
        "date": ["2026-06-01", "2026-06-02"],
        "value": [10, 20],
    })
    write_partitioned(df, tmp_path, "events", ["date"])
    con = duckdb.connect(":memory:")
    register_view(con, "events", str(tmp_path / "events" / "**" / "*.parquet"))

    # Query the date partition column and verify it can be used as a DATE type
    result = con.execute(
        "SELECT date, value FROM events WHERE date = '2026-06-01'::DATE ORDER BY value"
    ).fetchall()
    # DuckDB returns datetime.date objects for DATE columns
    assert result == [(datetime.date(2026, 6, 1), 10)], (
        "partition column coerced to DATE; query with DATE literal should work"
    )

    # Verify result_date is a date object representing 2026-06-01
    result_all = con.execute("SELECT date FROM events ORDER BY date").fetchall()
    assert len(result_all) == 2
    # The first result should be June 1 2026
    assert result_all[0][0].isoformat() == "2026-06-01"
    assert result_all[1][0].isoformat() == "2026-06-02"
