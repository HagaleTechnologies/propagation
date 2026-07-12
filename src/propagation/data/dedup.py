import polars as pl

SOURCE_PRIORITY: dict[str, int] = {
    "wsprnet": 0, "rbn": 1, "pskreporter": 2, "cqdx": 3,
}


def dedup_spots(df: pl.DataFrame) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 1. Key: (dx_call, de_call, band, mode, window).

    Keep one row per key: highest source priority, tie-break highest snr_db,
    then lowest ts. Applied before labeling and before any spot-count feature.
    """
    if df.height == 0:
        return df
    working = df.with_columns(
        pl.col("ts").dt.truncate("15m").alias("_window_start"),
        pl.col("source").replace_strict(SOURCE_PRIORITY, default=99).alias("_source_rank"),
    ).sort(
        ["_source_rank", "snr_db", "ts"],
        descending=[False, True, False],
        nulls_last=True,
    )
    deduped = working.unique(
        subset=["dx_call", "de_call", "band", "mode", "_window_start"],
        keep="first",
        maintain_order=True,
    )
    return deduped.drop(["_window_start", "_source_rank"]).sort("ts")
