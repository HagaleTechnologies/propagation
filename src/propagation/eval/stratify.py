"""Eval-only storm stratification from definitive Kp.

Definitive Kp is allowed for EVAL STRATIFICATION ONLY (docs/SPEC-labeling.md
sec 6, leakage rule 5); it must never feed a model feature (models use the
estimated series, M2). Storm definition: a window is storm iff its
containing 3-h block's Kp >= 5. This tags at 15-min-window granularity for
post-hoc eval slicing; SPEC's own storm rule operates at fold/day
granularity for CV purposes, which this module does not implement.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl

GFZ_KP_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_since_1932.txt"


def _parse_gfz(text: str) -> pl.DataFrame:
    """Parse the GFZ Kp/ap archive into (block_start, kp) rows.

    Columns are whitespace-separated: YYYY MM DD hh.h hh._m days days_m Kp ap D.
    Only rows where D == 1 (definitive Kp) are kept; D == 0 (provisional, not
    yet reprocessed) rows are dropped so the result matches the promise made
    by fetch_definitive_kp's name. Malformed lines (too few fields, or a
    non-numeric value in a numeric column) raise a ValueError naming the
    offending line rather than an opaque IndexError/ValueError.
    """
    rows = []
    for ln in text.splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split()
        # YYYY MM DD hh.h hh._m days days_m Kp ap D
        if len(f) < 10:
            raise ValueError(
                "Malformed GFZ Kp archive line: expected 10 whitespace-"
                "separated fields (YYYY MM DD hh.h hh._m days days_m Kp ap D) "
                f"but got {len(f)}: {ln!r}"
            )
        try:
            year, month, day = int(f[0]), int(f[1]), int(f[2])
            hour = int(float(f[3]))
            kp = float(f[7])
            definitive = int(f[9])
        except ValueError as exc:
            raise ValueError(
                "Malformed GFZ Kp archive line: expected numeric fields "
                f"(YYYY MM DD hh.h hh._m days days_m Kp ap D) but got: {ln!r}"
            ) from exc
        if definitive != 1:
            continue
        rows.append(
            (
                f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00",
                kp,
            )
        )
    return pl.DataFrame(rows, schema=["block_start", "kp"], orient="row").with_columns(
        pl.col("block_start").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="us")
        .dt.replace_time_zone("UTC")
    )


def fetch_definitive_kp(cache_dir: Path) -> pl.DataFrame:
    cache = cache_dir / "gfz_kp.txt"
    if not cache.exists():
        resp = httpx.get(GFZ_KP_URL, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
    return _parse_gfz(cache.read_text())


def tag_storm_windows(labels: pl.DataFrame, kp: pl.DataFrame) -> pl.DataFrame:
    """Attach the containing 3-h block's Kp to each 15-min window."""
    return (
        labels.with_columns(
            pl.col("window_start").dt.truncate("3h").alias("block_start")
        )
        .join(kp, on="block_start", how="left")
        .drop("block_start")
        .with_columns((pl.col("kp") >= 5.0).fill_null(False).alias("is_storm"))
    )
