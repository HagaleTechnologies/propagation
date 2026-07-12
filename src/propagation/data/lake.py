from pathlib import Path

import duckdb
import polars as pl


def write_partitioned(
    df: pl.DataFrame,
    root: Path,
    table: str,
    partition_cols: list[str],
    file_name: str = "part-0.parquet",
) -> None:
    """Writes df as hive-style lake/<table>/col=val/.../<file_name>."""
    if df.height == 0:
        return
    table_root = Path(root) / table
    for keys, group in df.group_by(partition_cols, maintain_order=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        parts = [f"{col}={val}" for col, val in zip(partition_cols, keys)]
        out_dir = table_root.joinpath(*parts)
        out_dir.mkdir(parents=True, exist_ok=True)
        group.drop(partition_cols).write_parquet(out_dir / file_name)


def register_view(con: duckdb.DuckDBPyConnection, name: str, glob_path: str) -> None:
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_parquet('{glob_path}', hive_partitioning = true)"
    )
