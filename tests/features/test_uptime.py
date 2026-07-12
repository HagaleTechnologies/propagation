import datetime as dt

import polars as pl

from propagation.features.uptime import build_receiver_uptime


def _spot(ts, de_grid="DM14ed", mode="WSPR", band="20m", de_call="W6SZ"):
    return {
        "ts": ts, "band": band, "mode": mode,
        "de_call": de_call, "de_grid": de_grid,
    }


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def test_single_spot_lights_padded_windows():
    # A spot at 12:02 (window 12:00) is evidence for windows whose padded
    # interval [W-30, W+45) contains 12:02: W in {11:30, 11:45, 12:00, 12:15}.
    # Check via the [t0-30, t0+45) formula directly: for ts=12:02,
    # valid W satisfies W-30 <= 12:02 < W+45  =>  11:17 < W <= 12:32,
    # W on 15-min grid: 11:30, 11:45, 12:00, 12:15, 12:30.
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts)])
    uptime = build_receiver_uptime(df)
    windows = sorted(uptime["window_start"].to_list())
    expected = [
        dt.datetime(2026, 6, 1, 11, 30, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 11, 45, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 15, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 30, tzinfo=dt.timezone.utc),
    ]
    assert windows == expected


def test_spot_on_window_boundary_lights_padded_windows():
    # A spot exactly at 12:00:00 (on a window boundary). Valid W satisfies
    # W - 30 <= 12:00 < W + 45  =>  11:15 < W <= 12:30, W on 15-min grid:
    # 11:30, 11:45, 12:00, 12:15, 12:30. Note W=11:15 is EXCLUDED (W+45=12:00
    # is not > ts=12:00, strict inequality), confirming the padding is
    # asymmetric (30 before / 45 after), not a naive +/-30 window.
    ts = dt.datetime(2026, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts)])
    uptime = build_receiver_uptime(df)
    windows = sorted(uptime["window_start"].to_list())
    expected = [
        dt.datetime(2026, 6, 1, 11, 30, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 11, 45, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 15, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, 12, 30, tzinfo=dt.timezone.utc),
    ]
    assert windows == expected


def test_uptime_row_shape():
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts)])
    uptime = build_receiver_uptime(df)
    row = uptime.filter(pl.col("window_start") == ts.replace(minute=0)).row(0, named=True)
    assert row["de_call"] == "W6SZ"
    assert row["de_field"] == "DM"
    assert row["de_grid4"] == "DM14"
    assert row["band"] == "20m"
    assert row["mode_class"] == "digi"
    assert row["n_evidence_reports"] == 1


def test_uptime_separates_mode_class():
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts, mode="WSPR"), _spot(ts, mode="CW")])
    uptime = build_receiver_uptime(df)
    mode_classes = set(
        uptime.filter(pl.col("window_start") == ts.replace(minute=0))["mode_class"]
    )
    assert mode_classes == {"digi", "cw"}


def test_uptime_excludes_other_mode_class():
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts, mode="SSB")])
    uptime = build_receiver_uptime(df)
    assert uptime.height == 0


def test_uptime_two_reports_same_window_counted():
    ts1 = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    ts2 = dt.datetime(2026, 6, 1, 12, 10, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts1), _spot(ts2)])
    uptime = build_receiver_uptime(df)
    row = uptime.filter(
        pl.col("window_start") == ts1.replace(minute=0)
    ).row(0, named=True)
    assert row["n_evidence_reports"] == 2
    assert row["first_evidence_ts"] == ts1
    assert row["last_evidence_ts"] == ts2


def test_uptime_grid4_tie_broken_lexicographically():
    # Two distinct grid4 values, each reported once (a count tie). SPEC sec 3
    # requires the tie be broken lexicographically, not by insertion/hash
    # order (polars' Series.mode() ordering on ties is not guaranteed
    # deterministic/lexicographic across versions/runs).
    ts1 = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    ts2 = dt.datetime(2026, 6, 1, 12, 5, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts1, de_grid="DM14"), _spot(ts2, de_grid="CM87")])
    uptime = build_receiver_uptime(df)
    row = uptime.filter(pl.col("window_start") == ts1.replace(minute=0)).row(0, named=True)
    assert row["de_grid4"] == "CM87"
    assert row["de_field"] == "CM"


def test_uptime_prefers_grid4_over_field_even_when_field_more_common():
    # SPEC sec 3: "modal de_grid4 (or field, if only fields reported)" -- a
    # receiver location is computed from grid4-precision reports whenever any
    # exist, even if a field-only report is more numerous within the window.
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    rows = [_spot(ts, de_grid="CM87")]
    rows += [_spot(ts, de_grid="DM") for _ in range(5)]
    df = _df(rows)
    uptime = build_receiver_uptime(df)
    row = uptime.filter(pl.col("window_start") == ts.replace(minute=0)).row(0, named=True)
    assert row["de_grid4"] == "CM87"
    assert row["de_field"] == "CM"


def test_uptime_field_only_modal_with_lexicographic_tie():
    ts1 = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    ts2 = dt.datetime(2026, 6, 1, 12, 5, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts1, de_grid="DM"), _spot(ts2, de_grid="CM")])
    uptime = build_receiver_uptime(df)
    row = uptime.filter(pl.col("window_start") == ts1.replace(minute=0)).row(0, named=True)
    assert row["de_field"] == "CM"
    assert row["de_grid4"] is None


def test_uptime_no_usable_location_contributes_nothing():
    # SPEC sec 3: "Receivers with no usable location contribute nothing."
    ts = dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)
    df = _df([_spot(ts, de_grid=None)])
    uptime = build_receiver_uptime(df)
    assert uptime.height == 0
