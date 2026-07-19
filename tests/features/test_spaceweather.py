from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from propagation.features.spaceweather import add_spaceweather_features


def _omni(start: datetime, n_hours: int, kp_fn, f107_fn):
    times = [start + timedelta(hours=i) for i in range(n_hours)]
    return pl.DataFrame({
        "time": times,
        "kp": [kp_fn(i) for i in range(n_hours)],
        "f107": [f107_fn(i) for i in range(n_hours)],
        "bz_gsm": [1.0] * n_hours,
        "solar_wind_speed": [400.0] * n_hours,
        "dst": [-10.0] * n_hours,
    }, schema_overrides={"time": pl.Datetime("us", "UTC")})


def test_kp_lags_pick_the_right_trailing_hourly_value():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    omni = _omni(start, 72, kp_fn=lambda i: float(i), f107_fn=lambda i: 100.0)
    labels = pl.DataFrame({
        "window_start": [start + timedelta(hours=48)],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    # at t=48h: kp_now uses the most recent OMNI hour AT OR BEFORE t -> kp[48]=48
    assert out["kp_now"][0] == pytest.approx(48.0)
    assert out["kp_lag3h"][0] == pytest.approx(45.0)
    assert out["kp_lag24h"][0] == pytest.approx(24.0)
    assert out["kp_lag48h"][0] == pytest.approx(0.0)


def test_f107_smoothed_27d_is_a_trailing_not_centered_mean():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # f107 ramps up by 1 per day; a CENTERED mean at day 30 would include
    # future days and be biased high relative to a trailing mean.
    omni = _omni(start, 24 * 40, kp_fn=lambda i: 3.0, f107_fn=lambda i: float(i // 24))
    labels = pl.DataFrame({
        "window_start": [start + timedelta(days=30)],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    # trailing 27-day mean of day-values [3..29] (30 - 27 = day 3 through day 29 inclusive-ish)
    # must be strictly less than day 30's own value (since it's an increasing ramp) --
    # a centered window would pull in days >30 and could exceed it.
    assert out["f107_smoothed_27d"][0] < 30.0
    assert out["f107_daily"][0] == pytest.approx(29.0)  # most recent day at/before window_start


def test_missing_omni_coverage_gives_nulls_not_a_crash():
    omni = _omni(datetime(2020, 1, 1, tzinfo=timezone.utc), 10, kp_fn=lambda i: 3.0, f107_fn=lambda i: 100.0)
    labels = pl.DataFrame({
        "window_start": [datetime(2026, 1, 1, tzinfo=timezone.utc)],  # far outside omni's coverage
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    assert out["kp_now"][0] is None
