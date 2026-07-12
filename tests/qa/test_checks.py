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


def test_check1_passes_with_strong_diurnal_signal():
    rows = (
        [_row(14, "FN", "DM", "20m", 1) for _ in range(9)]
        + [_row(14, "FN", "DM", "20m", 0)]
        + [_row(1, "FN", "DM", "20m", 1)]
        + [_row(1, "FN", "DM", "20m", 0) for _ in range(9)]
    )
    result = check_diurnal_20m(_df(rows))
    assert result.status == "pass"


def test_check1_insufficient_data_no_20m():
    result = check_diurnal_20m(_df([_row(14, "FN", "DM", "40m", 1)]))
    assert result.status == "insufficient_data"


def test_check2_lowband_uses_night_over_day_ratio():
    rows = (
        [_row(1, "FN", "DM", "160m", 1) for _ in range(9)]
        + [_row(1, "FN", "DM", "160m", 0)]
        + [_row(14, "FN", "DM", "160m", 1)]
        + [_row(14, "FN", "DM", "160m", 0) for _ in range(9)]
    )
    result = check_lowband_diurnal(_df(rows))
    assert result.status == "pass"


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
