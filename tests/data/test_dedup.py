import datetime as dt

import polars as pl

from propagation.data.dedup import dedup_spots


def _spot(**overrides):
    row = {
        "source": "wsprnet",
        "ts": dt.datetime(2026, 6, 1, 12, 3, tzinfo=dt.timezone.utc),
        "band": "20m",
        "mode": "WSPR",
        "dx_call": "K1JT",
        "de_call": "W6SZ",
        "snr_db": -10,
    }
    row.update(overrides)
    return row


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def test_dedup_collapses_same_pair_same_window():
    # Two WSPR decodes of the same pair, 6 minutes apart, same 15-min window.
    df = _df([
        _spot(ts=dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc), snr_db=-14),
        _spot(ts=dt.datetime(2026, 6, 1, 12, 8, tzinfo=dt.timezone.utc), snr_db=-9),
    ])
    result = dedup_spots(df)
    assert result.height == 1
    # highest snr_db wins the tie-break within same source priority
    assert result["snr_db"][0] == -9


def test_dedup_keeps_distinct_windows():
    df = _df([
        _spot(ts=dt.datetime(2026, 6, 1, 12, 2, tzinfo=dt.timezone.utc)),
        _spot(ts=dt.datetime(2026, 6, 1, 12, 20, tzinfo=dt.timezone.utc)),
    ])
    assert dedup_spots(df).height == 2


def test_dedup_prefers_higher_priority_source():
    df = _df([
        _spot(source="pskreporter", snr_db=5),
        _spot(source="wsprnet", snr_db=-20),
    ])
    result = dedup_spots(df)
    assert result.height == 1
    assert result["source"][0] == "wsprnet"


def test_dedup_empty_input():
    df = pl.DataFrame(schema={
        "source": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "band": pl.Utf8,
        "mode": pl.Utf8, "dx_call": pl.Utf8, "de_call": pl.Utf8, "snr_db": pl.Int16,
    })
    assert dedup_spots(df).height == 0
