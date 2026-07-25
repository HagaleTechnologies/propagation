import datetime as dt

import polars as pl

from propagation.qa.checks import (
    _circular_mean_lon,
    check_diurnal_20m,
    check_grayline_40m,
    check_lowband_diurnal,
    check_reciprocity,
    check_solar_cycle,
    check_sporadic_e,
    check_storm_response,
    check_volume_hygiene,
    run_qa_checks,
)


def _row(hour, tx, rx, band, open_, month=6):
    return {
        "window_start": dt.datetime(2026, month, 1, hour, 0, tzinfo=dt.timezone.utc),
        "tx_field": tx, "rx_field": rx, "band": band, "open": open_,
    }


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


# FN-DM: computed via propagation.data.geo.grid_to_latlon/great_circle_km
# (see /Users/thagale/Code/propagation/.claude/worktrees/m0-lake-bootstrap/.superpowers/sdd/task-15-report.md
# for the exact script output): distance=3543.2 km (within check1's 3-8 Mm
# and check2's >2 Mm ranges), midpoint longitude=-90.0 deg, so
# local_hour = (utc_hour + mid_lon/15) % 24 = (utc_hour - 6) % 24.
# local_hour=14 (mid-day) <-> utc_hour=20; local_hour=1 (mid-night) <-> utc_hour=7.
_FN_DM_UTC_DAY = 20
_FN_DM_UTC_NIGHT = 7

# FN-EM: distance=2025.7 km -- below check1's 3000 km floor, so it is used
# below as a check1 (3-8 Mm) out-of-range pair.
# EM-EN: distance=1112.0 km -- below check2's 2000 km floor, used below as a
# check2 (>2 Mm) out-of-range pair.


def test_check1_passes_with_strong_diurnal_signal():
    rows = (
        [_row(_FN_DM_UTC_DAY, "FN", "DM", "20m", 1) for _ in range(9)]
        + [_row(_FN_DM_UTC_DAY, "FN", "DM", "20m", 0)]
        + [_row(_FN_DM_UTC_NIGHT, "FN", "DM", "20m", 1)]
        + [_row(_FN_DM_UTC_NIGHT, "FN", "DM", "20m", 0) for _ in range(9)]
    )
    result = check_diurnal_20m(_df(rows))
    assert result.status == "pass"
    assert "local-time corrected" in result.detail


def test_check1_insufficient_data_no_20m():
    result = check_diurnal_20m(_df([_row(14, "FN", "DM", "40m", 1)]))
    assert result.status == "insufficient_data"


def test_check1_distance_filter_excludes_out_of_range_pairs():
    # FN-EM is 2025.7 km -- outside check1's required 3000-8000 km band, so
    # an all-FN-EM fixture must yield insufficient_data even with a strong
    # apparent diurnal signal in the raw rows.
    rows = (
        [_row(_FN_DM_UTC_DAY, "FN", "EM", "20m", 1) for _ in range(5)]
        + [_row(_FN_DM_UTC_NIGHT, "FN", "EM", "20m", 0) for _ in range(5)]
    )
    result = check_diurnal_20m(_df(rows))
    assert result.status == "insufficient_data"
    assert "distance range" in result.detail

    # Mixing in the out-of-range pair alongside the in-range FN-DM signal
    # must not change the ratio computed from FN-DM alone.
    in_range_rows = (
        [_row(_FN_DM_UTC_DAY, "FN", "DM", "20m", 1) for _ in range(9)]
        + [_row(_FN_DM_UTC_DAY, "FN", "DM", "20m", 0)]
        + [_row(_FN_DM_UTC_NIGHT, "FN", "DM", "20m", 1)]
        + [_row(_FN_DM_UTC_NIGHT, "FN", "DM", "20m", 0) for _ in range(9)]
    )
    result_in_range_only = check_diurnal_20m(_df(in_range_rows))
    result_mixed = check_diurnal_20m(_df(in_range_rows + rows))
    assert result_mixed.status == "pass"
    assert result_mixed.detail == result_in_range_only.detail


def test_check2_lowband_uses_night_over_day_ratio():
    rows = (
        [_row(_FN_DM_UTC_NIGHT, "FN", "DM", "160m", 1) for _ in range(9)]
        + [_row(_FN_DM_UTC_NIGHT, "FN", "DM", "160m", 0)]
        + [_row(_FN_DM_UTC_DAY, "FN", "DM", "160m", 1)]
        + [_row(_FN_DM_UTC_DAY, "FN", "DM", "160m", 0) for _ in range(9)]
    )
    result = check_lowband_diurnal(_df(rows))
    assert result.status == "pass"
    assert "local-time corrected" in result.detail


def test_check2_distance_filter_insufficient_data_for_short_pair():
    # EM-EN is 1112.0 km -- outside check2's required >2000 km band.
    rows = (
        [_row(_FN_DM_UTC_NIGHT, "EM", "EN", "160m", 1) for _ in range(9)]
        + [_row(_FN_DM_UTC_DAY, "EM", "EN", "160m", 0) for _ in range(9)]
    )
    result = check_lowband_diurnal(_df(rows))
    assert result.status == "insufficient_data"
    assert "distance range" in result.detail


def _grayline_row(hour, terminator_hrs, zenith, open_, month=6):
    return {
        "window_start": dt.datetime(2026, month, 1, hour, 0, tzinfo=dt.timezone.utc),
        "tx_field": "FN", "rx_field": "PM", "band": "40m", "open": open_,
        "midpoint_hours_since_terminator": terminator_hrs,
        "midpoint_solar_zenith": zenith,
    }


def test_check3_grayline_real_computation_pass():
    # FN-PM is ~10894km apart (>6Mm DX threshold). Gray-line rows (near
    # terminator) open more often than midday rows.
    rows = (
        [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=1) for _ in range(8)]
        + [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=0) for _ in range(2)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=1) for _ in range(2)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=0) for _ in range(8)]
    )
    result = check_grayline_40m(_df(rows))
    assert result.status == "pass"


def test_check3_grayline_real_computation_fail():
    rows = (
        [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=1) for _ in range(2)]
        + [_grayline_row(6, terminator_hrs=0.5, zenith=88.0, open_=0) for _ in range(8)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=1) for _ in range(8)]
        + [_grayline_row(12, terminator_hrs=5.0, zenith=20.0, open_=0) for _ in range(2)]
    )
    result = check_grayline_40m(_df(rows))
    assert result.status == "fail"


def test_check3_gate_reports_insufficient_data_without_solar_features():
    result = check_grayline_40m(_df([_row(14, "FN", "DM", "40m", 1)]))
    assert result.status == "insufficient_data"


def test_check4_sporadic_e_seasonal():
    # FN->EM is a real-ish ~1500km 6m pair; summer high, winter low.
    rows = (
        [_row(14, "EM", "EN", "6m", 1, month=6) for _ in range(6)]
        + [_row(14, "EM", "EN", "6m", 0, month=6) for _ in range(2)]
        + [_row(14, "EM", "EN", "6m", 1, month=12)]
        + [_row(14, "EM", "EN", "6m", 0, month=12) for _ in range(7)]
    )
    result = check_sporadic_e(_df(rows))
    assert result.status == "pass"


def test_check4_insufficient_data_no_6m():
    result = check_sporadic_e(_df([_row(14, "FN", "DM", "20m", 1)]))
    assert result.status == "insufficient_data"


def test_check5_reciprocity():
    rows = []
    for i in range(6):
        open_fwd = 1 if i % 2 == 0 else 0
        rows.append(_row(12, f"A{i}", f"B{i}", "20m", open_fwd))
        rows.append(_row(12, f"B{i}", f"A{i}", "20m", open_fwd))
    result = check_reciprocity(_df(rows))
    assert result.status in {"pass", "insufficient_data"}


def _solar_cycle_row(month_idx, f107, open_):
    # Create one row per month, with proper calendar month progression
    # month_idx 0 -> Jan 2026, 1 -> Feb 2026, etc.
    year = 2026 + (month_idx // 12)
    month = (month_idx % 12) + 1
    return {
        "window_start": dt.datetime(year, month, 1, tzinfo=dt.timezone.utc),
        "tx_field": "FN", "rx_field": "PM", "band": "10m", "open": open_,
        "f107_daily": f107,
    }


def test_check6_solar_cycle_real_computation_pass():
    # 12 months, f107 rising linearly; monthly open-rate rises in lockstep
    # (m/11 out of 11 rows/month) -- near-perfect positive correlation.
    rows = []
    for m in range(12):
        f107 = 70.0 + m * 10.0
        n_open = m
        rows += [_solar_cycle_row(m, f107, open_=1) for _ in range(n_open)]
        rows += [_solar_cycle_row(m, f107, open_=0) for _ in range(11 - n_open)]
    result = check_solar_cycle(_df(rows))
    assert result.status == "pass"


def test_check6_solar_cycle_real_computation_fail():
    # Same f107 ramp, but open-rate is ANTI-correlated with it.
    rows = []
    for m in range(12):
        f107 = 70.0 + m * 10.0
        n_open = 11 - m
        rows += [_solar_cycle_row(m, f107, open_=1) for _ in range(n_open)]
        rows += [_solar_cycle_row(m, f107, open_=0) for _ in range(11 - n_open)]
    result = check_solar_cycle(_df(rows))
    assert result.status == "fail"


def test_check6_gate_insufficient_data_single_month():
    result = check_solar_cycle(_df([_row(14, "FN", "DM", "10m", 1)]))
    assert result.status == "insufficient_data"


def test_check7_gate_insufficient_data_without_storm_fold():
    result = check_storm_response(_df([_row(14, "FN", "DM", "20m", 1)]), kp_max=3.0)
    assert result.status == "insufficient_data"


def test_check8_volume_hygiene_pass_under_threshold():
    labels = _df([_row(14, "FN", "DM", "20m", 1)])
    result = check_volume_hygiene(labels, rejection_counts={"rr73_grid": 1}, n_qualifying=1000)
    assert result.status == "pass"


def test_check8_volume_hygiene_fail_over_threshold():
    labels = _df([_row(14, "FN", "DM", "20m", 1)])
    result = check_volume_hygiene(labels, rejection_counts={"rr73_grid": 50}, n_qualifying=1000)
    assert result.status == "fail"


def test_run_qa_checks_returns_all_eight():
    labels = _df([_row(14, "FN", "DM", "20m", 1)])
    results = run_qa_checks(labels, rejection_counts={}, n_qualifying=1)
    assert {r.check_id for r in results} == {1, 2, 3, 4, 5, 6, 7, 8}


def test_circular_mean_lon_antimeridian_crossing():
    """Test that _circular_mean_lon correctly handles antimeridian wraparound.

    Pair AJ-RO crosses the antimeridian with:
    - AJ: lon=-170.0 (field centroid)
    - RO: lon=170.0 (field centroid)
    - Distance: 5841.2 km (within check1's 3000-8000 km range)

    The naive average (lon1 + lon2) / 2 = 0.0, which is geographically incorrect.
    The correct midpoint should be near ±180 degrees.
    """
    lon1, lon2 = -170.0, 170.0
    mid_lon = _circular_mean_lon(lon1, lon2)
    # The midpoint should be near ±180 (they're the same location on the sphere)
    # Due to how we normalize, we expect 180 or close to it
    assert abs(abs(mid_lon) - 180.0) < 0.1, f"Expected mid_lon near ±180, got {mid_lon}"
    # Verify it's NOT the naive average
    naive_mid = (lon1 + lon2) / 2.0
    assert naive_mid == 0.0  # The bug case
    assert mid_lon != naive_mid  # The fix distinguishes them


def test_circular_mean_lon_normal_case():
    """Test that _circular_mean_lon still works correctly for normal (non-antimeridian) cases."""
    # Normal case: -90 and 90 should average to 0
    assert _circular_mean_lon(-90.0, 90.0) == 0.0
    # Normal case: -50 and -30 should average to -40
    assert _circular_mean_lon(-50.0, -30.0) == -40.0
    # Normal case: 30 and 50 should average to 40
    assert _circular_mean_lon(30.0, 50.0) == 40.0


def test_check1_antimeridian_crossing_pair():
    """Test that check1 correctly uses antimeridian-aware longitude for local-time correction.

    Uses pair AJ-RO which crosses the antimeridian (lon=-170 to +170).
    With local-time correction via _circular_mean_lon, the path midpoint should be
    near ±180°, giving hour offset of ±12, not 0.
    """
    # AJ-RO distance is 5841.2 km, in range [3000, 8000].
    # Correct mid_lon is ~180 or -180, which gives offset of ±12 hours.
    # UTC hour 14 -> local hour (14 + 180/15) % 24 = (14 + 12) % 24 = 2 (night)
    # UTC hour 2 -> local hour (2 + 180/15) % 24 = (2 + 12) % 24 = 14 (day)
    rows = (
        [_row(2, "AJ", "RO", "20m", 1) for _ in range(9)]
        + [_row(2, "AJ", "RO", "20m", 0)]
        + [_row(14, "AJ", "RO", "20m", 1)]
        + [_row(14, "AJ", "RO", "20m", 0) for _ in range(9)]
    )
    result = check_diurnal_20m(_df(rows))
    assert result.status == "pass", f"Expected pass but got {result.status}: {result.detail}"
    assert "local-time corrected" in result.detail
