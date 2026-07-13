import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from propagation.data.lake import write_partitioned


def stratum_seed(band: str, date: str) -> int:
    digest = hashlib.sha256(f"{band}|{date}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0xFFFFFFFF


def sample_labels(labels: pl.DataFrame, ratio: float = 3.0) -> pl.DataFrame:
    """docs/SPEC-labeling.md sec 4.5. Training-only; eval must use the full set."""
    working = labels.with_columns(pl.col("window_start").dt.date().cast(pl.Utf8).alias("_date"))
    parts = []
    for (band, date), group in working.group_by(["band", "_date"], maintain_order=True):
        pos = group.filter(pl.col("open") == 1)
        neg = group.filter(pl.col("open") == 0)
        n_pos = pos.height
        target_neg = int(n_pos * ratio)
        if n_pos == 0 or neg.height <= target_neg:
            sampled_neg = neg
            rate = 1.0
        else:
            rng = np.random.Generator(np.random.PCG64(stratum_seed(band, date)))
            idx = np.sort(rng.choice(neg.height, size=target_neg, replace=False))
            sampled_neg = neg[idx.tolist()]
            rate = target_neg / neg.height
        pos = pos.with_columns(pl.lit(1.0).alias("sample_weight"))
        sampled_neg = sampled_neg.with_columns(pl.lit(1.0 / rate).alias("sample_weight"))
        parts.append(pl.concat([pos, sampled_neg]))
    result = pl.concat(parts) if parts else working.with_columns(pl.lit(1.0).alias("sample_weight"))
    return result.drop("_date")


def write_labels(df: pl.DataFrame, lake_root: Path) -> None:
    working = df.with_columns(pl.col("window_start").dt.date().cast(pl.Utf8).alias("date"))
    write_partitioned(working, lake_root, "labels", ["band", "date"])
