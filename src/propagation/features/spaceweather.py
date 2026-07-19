"""Space-weather features (ARCHITECTURE.md sec 4 item 4): Kp now + lagged
3/6/12/24/48h, F10.7 daily + trailing 27-day mean, solar wind Bz/speed, DST.

Exempt from the blocked-CV horizon+lookback gap sum (docs/SPEC-labeling.md
sec 6 rule 1: "not derived from spots") -- but every value here is computed
strictly as-of `window_start` (asof-backward join, never a centered window),
which is the real leakage safeguard independent of any CV gap. See
tests/test_leakage.py for the dedicated test asserting this.
"""
from __future__ import annotations

import polars as pl


def add_spaceweather_features(labels: pl.DataFrame, omni: pl.DataFrame) -> pl.DataFrame:
    """`omni` is `propagation.data.spaceweather.fetch_omni2_range`'s output
    (hourly, columns time/kp/f107/bz_gsm/solar_wind_speed/dst). All features
    are as-of `window_start` via backward asof joins -- the most recent OMNI
    hour AT OR BEFORE window_start, never a future one."""
    omni = omni.sort("time")
    labels_sorted = labels.sort("window_start")

    # 1h tolerance: hourly OMNI2 data means any covered window_start is at
    # most just under an hour past its backing OMNI row; anything further
    # (e.g. window_start falling outside OMNI's fetched coverage entirely)
    # must yield null rather than silently extrapolating an arbitrarily
    # stale value backward.
    _TOL = "1h"

    def _asof_lag(hours: float, suffix: str) -> pl.DataFrame:
        shifted = labels_sorted.with_columns(
            (pl.col("window_start") - pl.duration(hours=hours)).alias("_lag_time")
        )
        joined = shifted.join_asof(
            omni.select("time", "kp"), left_on="_lag_time", right_on="time",
            strategy="backward", tolerance=_TOL,
        )
        return joined.select(pl.col("kp").alias(f"kp_{suffix}"))

    kp_now = _asof_lag(0, "now")
    kp_lag3h = _asof_lag(3, "lag3h")
    kp_lag6h = _asof_lag(6, "lag6h")
    kp_lag12h = _asof_lag(12, "lag12h")
    kp_lag24h = _asof_lag(24, "lag24h")
    kp_lag48h = _asof_lag(48, "lag48h")

    # Bz/solar-wind-speed/Dst are near-real-time telemetry: an exact-timestamp
    # match at window_start is legitimately "already known" at that instant,
    # so these use the default inclusive (allow_exact_matches=True) backward
    # asof match.
    other_now = labels_sorted.join_asof(
        omni.select("time", "bz_gsm", "solar_wind_speed", "dst"),
        left_on="window_start", right_on="time", strategy="backward", tolerance=_TOL,
    ).select("bz_gsm", "solar_wind_speed", "dst")

    # F10.7 is a once-daily ground measurement (observed near local noon) that
    # OMNI2 backfills across all 24 hourly rows of its UTC day. At the exact
    # top of a day boundary, that day's own reading has not actually been
    # taken yet, so exclude exact-timestamp matches here (unlike the
    # near-real-time fields above) -- this is the leakage-safety distinction
    # documented at module level, applied precisely at the day-boundary edge
    # case.
    f107_daily = labels_sorted.join_asof(
        omni.select("time", "f107"), left_on="window_start", right_on="time",
        strategy="backward", tolerance=_TOL, allow_exact_matches=False,
    ).select(pl.col("f107").alias("f107_daily"))

    # 27-day trailing mean computed on the hourly series directly (equivalent
    # to a daily-mean-of-daily-means since f107 is constant within a day in
    # OMNI2). `closed="left"` already excludes each row's own value from its
    # own trailing aggregate, so no `allow_exact_matches` override is needed
    # here -- the aggregate anchored at an exact-match row is already
    # backward-looking only.
    daily_omni = omni.rolling("time", period="27d", closed="left").agg(
        pl.col("f107").mean().alias("f107_smoothed_27d")
    )
    f107_smoothed = labels_sorted.join_asof(
        daily_omni, left_on="window_start", right_on="time", strategy="backward", tolerance=_TOL,
    ).select("f107_smoothed_27d")

    out = pl.concat(
        [labels_sorted, kp_now, kp_lag3h, kp_lag6h, kp_lag12h, kp_lag24h, kp_lag48h,
         other_now.rename({"bz_gsm": "bz_gsm_now", "solar_wind_speed": "solar_wind_speed_now",
                            "dst": "dst_now"}),
         f107_daily, f107_smoothed],
        how="horizontal_extend",
    )
    return out
