import datetime as dt

import polars as pl

from propagation.qa.checks import (
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
