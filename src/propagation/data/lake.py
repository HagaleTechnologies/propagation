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
    """Writes df as hive-style lake/<table>/col=val/.../<file_name>.

    If the target file already exists, picks the next available numbered filename
    (e.g. if file_name="part-0.parquet" and that exists, tries part-0-1.parquet,
    part-0-2.parquet, etc.) to avoid silent overwrites while preserving the first
    write's exact filename.
    """
    if df.height == 0:
        return
    table_root = Path(root) / table
    for keys, group in df.group_by(partition_cols, maintain_order=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        parts = [f"{col}={val}" for col, val in zip(partition_cols, keys)]
        out_dir = table_root.joinpath(*parts)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collision avoidance: find the next available filename if target exists
        target_path = out_dir / file_name
        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            counter = 1
            while True:
                target_path = out_dir / f"{stem}-{counter}{suffix}"
                if not target_path.exists():
                    break
                counter += 1

        group.drop(partition_cols).write_parquet(target_path)


def register_view(con: duckdb.DuckDBPyConnection, name: str, glob_path: str) -> None:
    """Registers a DuckDB view over hive-partitioned Parquet files.

    DuckDB's hive_partitioning=true infers column types from partition path
    segments. For example, a partition segment like date=2026-06-01 will be
    inferred as DuckDB DATE, not VARCHAR, even if the original DataFrame column
    was a string. Callers filtering/joining on partition columns should be aware
    the returned type may differ from what was written.
    """
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_parquet('{glob_path}', hive_partitioning = true)"
    )
