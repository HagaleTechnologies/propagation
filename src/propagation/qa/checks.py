from dataclasses import dataclass

import numpy as np
import polars as pl

from propagation.data.geo import grid_to_latlon, great_circle_km


def _circular_mean_lon(lon1: float, lon2: float) -> float:
    """Average two longitudes in degrees, correctly handling antimeridian wraparound.

    For example, -170 and 170 are 20 degrees apart via the dateline, not 340 apart
    via Greenwich, so their midpoint is near ±180, not 0.
    """
    diff = lon2 - lon1
    if diff > 180:
        lon2 -= 360
    elif diff < -180:
        lon2 += 360
    mid = (lon1 + lon2) / 2.0
    if mid > 180:
        mid -= 360
    elif mid < -180:
        mid += 360
    return mid


@dataclass
class QAResult:
    check_id: int
    name: str
    status: str  # "pass" | "fail" | "insufficient_data"
    detail: str


def _diurnal_ratio_check(
    labels: pl.DataFrame,
    check_id: int,
    name: str,
    bands: set[str],
    day_hours: tuple[int, int],
    night_hours: tuple[int, int],
    min_ratio: float,
    numerator: str,  # "day" or "night"
    distance_km_range: tuple[float, float | None] | None = None,
) -> QAResult:
    subset = labels.filter(pl.col("band").is_in(bands))
    if subset.height == 0:
        return QAResult(check_id, name, "insufficient_data", f"no labels for bands {sorted(bands)}")

    local_time_corrected = distance_km_range is not None
    if distance_km_range is not None:
        lo, hi = distance_km_range
        pairs = subset.select(["tx_field", "rx_field"]).unique().to_dicts()
        dist_by_pair: dict[tuple[str, str], float] = {}
        mid_lon_by_pair: dict[tuple[str, str], float] = {}
        for p in pairs:
            try:
                lat1, lon1 = grid_to_latlon(p["tx_field"])
                lat2, lon2 = grid_to_latlon(p["rx_field"])
                key = (p["tx_field"], p["rx_field"])
                dist_by_pair[key] = great_circle_km(lat1, lon1, lat2, lon2)
                mid_lon_by_pair[key] = _circular_mean_lon(lon1, lon2)
            except ValueError:
                continue

        subset = subset.with_columns(
            pl.struct(["tx_field", "rx_field"])
            .map_elements(
                lambda r: dist_by_pair.get((r["tx_field"], r["rx_field"])), return_dtype=pl.Float64
            )
            .alias("distance_km"),
            pl.struct(["tx_field", "rx_field"])
            .map_elements(
                lambda r: mid_lon_by_pair.get((r["tx_field"], r["rx_field"])), return_dtype=pl.Float64
            )
            .alias("mid_lon"),
        )
        if hi is None:
            subset = subset.filter(pl.col("distance_km") > lo)
        else:
            subset = subset.filter(pl.col("distance_km").is_between(lo, hi))
        if subset.height == 0:
            return QAResult(
                check_id, name, "insufficient_data",
                f"no paths in distance range {distance_km_range} km",
            )

    if local_time_corrected:
        working = subset.with_columns(
            ((pl.col("window_start").dt.hour() + pl.col("mid_lon") / 15.0) % 24.0).alias("hour")
        )
    else:
        working = subset.with_columns(pl.col("window_start").dt.hour().alias("hour"))

    day = working.filter(pl.col("hour").is_between(*day_hours))
    night_lo, night_hi = night_hours
    night = working.filter((pl.col("hour") >= night_lo) | (pl.col("hour") <= night_hi))
    if day.height == 0 or night.height == 0:
        return QAResult(check_id, name, "insufficient_data", "missing day or night windows")
    day_rate = float(day["open"].cast(pl.Float64).mean())
    night_rate = float(night["open"].cast(pl.Float64).mean())
    num, den = (day_rate, night_rate) if numerator == "day" else (night_rate, day_rate)
    if den == 0:
        return QAResult(check_id, name, "insufficient_data", "zero-rate denominator")
    ratio = num / den
    status = "pass" if ratio > min_ratio else "fail"
    if local_time_corrected:
        detail = (
            f"{numerator}/other open-rate ratio={ratio:.2f} (local-time corrected via "
            "path-midpoint longitude; full solar-geometry accuracy lands with M2's "
            "features/solar.py)"
        )
    else:
        detail = f"{numerator}/other open-rate ratio={ratio:.2f}"
    return QAResult(check_id, name, status, detail)


def check_diurnal_20m(labels: pl.DataFrame) -> QAResult:
    """SPEC-labeling sec 6 QA check 1: 20m mid-lat (3-8 Mm paths) day/night
    ratio > 2. Uses local-time correction via path-midpoint longitude since
    WSPRnet is a global network and raw UTC-hour bucketing mixes local
    morning/midnight across regions."""
    return _diurnal_ratio_check(
        labels, 1, "20m_diurnal", {"20m"}, day_hours=(12, 17), night_hours=(22, 3),
        min_ratio=2.0, numerator="day", distance_km_range=(3000, 8000),
    )


def check_lowband_diurnal(labels: pl.DataFrame) -> QAResult:
    """QA check 2: 160m/80m (paths > 2 Mm) night/day ratio > 5. Uses
    local-time correction via path-midpoint longitude (see check_diurnal_20m)."""
    return _diurnal_ratio_check(
        labels, 2, "lowband_diurnal", {"160m", "80m"}, day_hours=(12, 17),
        night_hours=(22, 3), min_ratio=5.0, numerator="night",
        distance_km_range=(2000, None),
    )


def check_grayline_40m(labels: pl.DataFrame) -> QAResult:
    """QA check 3 (docs/SPEC-labeling.md sec 6): 40m gray-line open-rate
    local max within +/-1h of the midpoint terminator vs midday, DX paths
    > 6 Mm. Requires features/solar.py's midpoint_hours_since_terminator
    and midpoint_solar_zenith already joined onto `labels`; without them
    this stays a gate (M0/M1 runs have no solar features to check against)."""
    if "midpoint_hours_since_terminator" not in labels.columns or "midpoint_solar_zenith" not in labels.columns:
        return QAResult(
            3, "grayline_40m", "insufficient_data",
            "terminator-relative timing requires features/solar.py joined onto labels",
        )
    subset = labels.filter(pl.col("band") == "40m")
    if subset.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "no 40m labels in this run")

    pairs = subset.select(["tx_field", "rx_field"]).unique().to_dicts()
    dist_by_pair = {}
    for p in pairs:
        try:
            lat1, lon1 = grid_to_latlon(p["tx_field"])
            lat2, lon2 = grid_to_latlon(p["rx_field"])
            dist_by_pair[(p["tx_field"], p["rx_field"])] = great_circle_km(lat1, lon1, lat2, lon2)
        except ValueError:
            continue
    working = subset.with_columns(
        pl.struct(["tx_field", "rx_field"])
        .map_elements(
            lambda r: dist_by_pair.get((r["tx_field"], r["rx_field"])), return_dtype=pl.Float64
        )
        .alias("distance_km")
    ).filter(pl.col("distance_km") > 6000)
    if working.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "no >6Mm 40m paths")

    terminator = working.filter(pl.col("midpoint_hours_since_terminator").abs() <= 1.0)
    midday = working.filter(pl.col("midpoint_solar_zenith") <= 30.0)
    if terminator.height == 0 or midday.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "missing gray-line or midday windows")

    terminator_rate = float(terminator["open"].cast(pl.Float64).mean())
    midday_rate = float(midday["open"].cast(pl.Float64).mean())
    status = "pass" if terminator_rate > midday_rate else "fail"
    return QAResult(
        3, "grayline_40m", status,
        f"gray-line open-rate={terminator_rate:.2f} vs midday open-rate={midday_rate:.2f}",
    )


def check_sporadic_e(labels: pl.DataFrame) -> QAResult:
    """QA check 4: 6m Sp-E, NH May-Jul open-rate >= 3x Nov-Jan (1-2.3 Mm paths)."""
    subset = labels.filter(pl.col("band") == "6m")
    if subset.height == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "no 6m labels in this run")

    pairs = subset.select(["tx_field", "rx_field"]).unique().to_dicts()
    dist_by_pair = {}
    for p in pairs:
        try:
            lat1, lon1 = grid_to_latlon(p["tx_field"])
            lat2, lon2 = grid_to_latlon(p["rx_field"])
            dist_by_pair[(p["tx_field"], p["rx_field"])] = great_circle_km(lat1, lon1, lat2, lon2)
        except ValueError:
            continue

    working = subset.with_columns(
        pl.struct(["tx_field", "rx_field"])
        .map_elements(
            lambda r: dist_by_pair.get((r["tx_field"], r["rx_field"])), return_dtype=pl.Float64
        )
        .alias("distance_km"),
        pl.col("window_start").dt.month().alias("month"),
    ).filter(pl.col("distance_km").is_between(1000, 2300))

    if working.height == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "no 1-2.3Mm 6m paths")

    summer = working.filter(pl.col("month").is_in([5, 6, 7]))
    winter = working.filter(pl.col("month").is_in([11, 12, 1]))
    if summer.height == 0 or winter.height == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "missing summer or winter months")

    summer_rate = float(summer["open"].cast(pl.Float64).mean())
    winter_rate = float(winter["open"].cast(pl.Float64).mean())
    if winter_rate == 0:
        return QAResult(4, "sporadic_e_seasonal", "insufficient_data", "zero winter open-rate")
    ratio = summer_rate / winter_rate
    status = "pass" if ratio >= 3 else "fail"
    return QAResult(4, "sporadic_e_seasonal", status, f"summer/winter open-rate ratio={ratio:.2f}")


def check_reciprocity(labels: pl.DataFrame) -> QAResult:
    """QA check 5: Pearson r of open-rate(TX->RX) vs (RX->TX) per (pair, band, month) > 0.6."""
    working = labels.with_columns(pl.col("window_start").dt.month().alias("month"))
    fwd = working.group_by(["tx_field", "rx_field", "band", "month"]).agg(
        pl.col("open").cast(pl.Float64).mean().alias("rate_fwd")
    )
    rev = fwd.rename({
        "tx_field": "rx_field_r", "rx_field": "tx_field_r", "rate_fwd": "rate_rev",
    })
    paired = fwd.join(
        rev,
        left_on=["tx_field", "rx_field", "band", "month"],
        right_on=["tx_field_r", "rx_field_r", "band", "month"],
        how="inner",
    )
    if paired.height < 5:
        return QAResult(5, "reciprocity", "insufficient_data", f"only {paired.height} paired cells")
    r = float(np.corrcoef(paired["rate_fwd"].to_numpy(), paired["rate_rev"].to_numpy())[0, 1])
    status = "pass" if r > 0.6 else "fail"
    return QAResult(5, "reciprocity", status, f"pearson r={r:.3f}")


def check_solar_cycle(labels: pl.DataFrame, min_months: int = 12) -> QAResult:
    """QA check 6 (docs/SPEC-labeling.md sec 6): monthly 10m DX (>6 Mm)
    open-rate vs F10.7 correlation > 0.5 over multi-year history. Requires
    features/spaceweather.py's f107_daily already joined onto `labels`."""
    subset = labels.filter(pl.col("band") == "10m")
    if subset.height == 0:
        return QAResult(6, "solar_cycle", "insufficient_data", "no 10m labels in this run")

    pairs = subset.select(["tx_field", "rx_field"]).unique().to_dicts()
    dist_by_pair = {}
    for p in pairs:
        try:
            lat1, lon1 = grid_to_latlon(p["tx_field"])
            lat2, lon2 = grid_to_latlon(p["rx_field"])
            dist_by_pair[(p["tx_field"], p["rx_field"])] = great_circle_km(lat1, lon1, lat2, lon2)
        except ValueError:
            continue
    subset = subset.with_columns(
        pl.struct(["tx_field", "rx_field"])
        .map_elements(
            lambda r: dist_by_pair.get((r["tx_field"], r["rx_field"])), return_dtype=pl.Float64
        )
        .alias("distance_km")
    ).filter(pl.col("distance_km") > 6000)
    if subset.height == 0:
        return QAResult(6, "solar_cycle", "insufficient_data", "no DX (>6Mm) 10m paths")

    n_months = subset.select(pl.col("window_start").dt.truncate("1mo")).unique().height
    if n_months < min_months:
        return QAResult(
            6, "solar_cycle", "insufficient_data",
            f"only {n_months} distinct month(s); need >= {min_months}",
        )
    if "f107_daily" not in subset.columns:
        return QAResult(
            6, "solar_cycle", "insufficient_data",
            "F10.7 series requires features/spaceweather.py joined onto labels",
        )

    monthly = subset.with_columns(
        pl.col("window_start").dt.truncate("1mo").alias("month")
    ).group_by("month").agg(
        pl.col("open").cast(pl.Float64).mean().alias("open_rate"),
        pl.col("f107_daily").mean().alias("f107_mean"),
    )
    r = float(np.corrcoef(monthly["open_rate"].to_numpy(), monthly["f107_mean"].to_numpy())[0, 1])
    status = "pass" if r > 0.5 else "fail"
    return QAResult(6, "solar_cycle", status, f"monthly open-rate/F10.7 pearson r={r:.3f}")


def check_storm_response(labels: pl.DataFrame, kp_max: float | None) -> QAResult:
    """QA check 7: Kp>=6 trans-polar open-rate <= 50% of Kp<=2 matched baseline.
    Needs a Kp series (space_weather features, M2) and requires at least one
    storm (Kp>=5) fold in the eval window."""
    if kp_max is None or kp_max < 5.0:
        return QAResult(
            7, "storm_response", "insufficient_data",
            f"no Kp>=5 fold in this run (max Kp available={kp_max})",
        )
    return QAResult(7, "storm_response", "insufficient_data", "Kp series not yet joined (features/spaceweather.py, M2)")


def check_volume_hygiene(
    labels: pl.DataFrame, rejection_counts: dict[str, int], n_qualifying: int
) -> QAResult:
    """QA check 8: RR73-grid rejects < 0.5% of spots; unlabeled fraction reported
    (via features/universe.unlabeled_activity_fraction, called separately).
    Trailing-28-day volume comparison is skipped on a bootstrap run (no history
    to trail against yet)."""
    rr73 = rejection_counts.get("rr73_grid", 0)
    total = n_qualifying + sum(rejection_counts.values())
    rr73_rate = (rr73 / total) if total else 0.0
    status = "fail" if rr73_rate >= 0.005 else "pass"
    return QAResult(8, "volume_hygiene", status, f"RR73 reject rate={rr73_rate:.4%}")


def run_qa_checks(
    labels: pl.DataFrame,
    rejection_counts: dict[str, int],
    n_qualifying: int,
    kp_max: float | None = None,
) -> list[QAResult]:
    return [
        check_diurnal_20m(labels),
        check_lowband_diurnal(labels),
        check_grayline_40m(labels),
        check_sporadic_e(labels),
        check_reciprocity(labels),
        check_solar_cycle(labels),
        check_storm_response(labels, kp_max),
        check_volume_hygiene(labels, rejection_counts, n_qualifying),
    ]
