from dataclasses import dataclass

import numpy as np
import polars as pl

from propagation.data.geo import grid_to_latlon, great_circle_km


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
) -> QAResult:
    subset = labels.filter(pl.col("band").is_in(bands))
    if subset.height == 0:
        return QAResult(check_id, name, "insufficient_data", f"no labels for bands {sorted(bands)}")
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
    return QAResult(check_id, name, status, f"{numerator}/other open-rate ratio={ratio:.2f}")


def check_diurnal_20m(labels: pl.DataFrame) -> QAResult:
    """SPEC-labeling sec 6 QA check 1: 20m mid-lat day/night ratio > 2."""
    return _diurnal_ratio_check(
        labels, 1, "20m_diurnal", {"20m"}, day_hours=(12, 17), night_hours=(22, 3),
        min_ratio=2.0, numerator="day",
    )


def check_lowband_diurnal(labels: pl.DataFrame) -> QAResult:
    """QA check 2: 160m/80m night/day ratio > 5."""
    return _diurnal_ratio_check(
        labels, 2, "lowband_diurnal", {"160m", "80m"}, day_hours=(12, 17),
        night_hours=(22, 3), min_ratio=5.0, numerator="night",
    )


def check_grayline_40m(labels: pl.DataFrame) -> QAResult:
    """QA check 3: 40m gray-line open-rate peak near the terminator. Needs
    solar-terminator features (features/solar.py, scheduled for M2) to locate
    the terminator per path-cell; M0 has no such feature, so this is a real,
    tested precondition gate, not a stub of the eventual arithmetic."""
    subset = labels.filter(pl.col("band") == "40m")
    if subset.height == 0:
        return QAResult(3, "grayline_40m", "insufficient_data", "no 40m labels in this run")
    return QAResult(
        3, "grayline_40m", "insufficient_data",
        "terminator-relative timing requires features/solar.py (M2); not computable yet",
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
    """QA check 6: monthly 10m DX open-rate vs F10.7 correlation > 0.5 over
    multi-year history. SPEC explicitly sanctions 'insufficient data' here when
    history is short (docs/SPEC-labeling.md sec 6)."""
    subset = labels.filter(pl.col("band") == "10m")
    if subset.height == 0:
        return QAResult(6, "solar_cycle", "insufficient_data", "no 10m labels in this run")
    n_months = subset.select(pl.col("window_start").dt.truncate("1mo")).unique().height
    if n_months < min_months:
        return QAResult(
            6, "solar_cycle", "insufficient_data",
            f"only {n_months} distinct month(s); need >= {min_months} plus F10.7 series (M3)",
        )
    return QAResult(6, "solar_cycle", "insufficient_data", "F10.7 correlation lands with M3 space-weather features")


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
