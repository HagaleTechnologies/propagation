import datetime as dt

import polars as pl

from propagation.features.labels import build_labels, snr_ft8eq
from propagation.features.universe import build_universe
from propagation.features.uptime import build_receiver_uptime

W0 = dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.timezone.utc)


def test_snr_ft8eq_wspr_no_offset_needed():
    # tx_dbm=50 (100W reference) -> pwr_offset=0
    assert snr_ft8eq("WSPR", -10, 50) == -10.0


def test_snr_ft8eq_wspr_power_normalized():
    # tx_dbm=20 (0.1W) -> pwr_offset = 50-20 = 30
    assert snr_ft8eq("WSPR", -10, 20) == 20.0


def test_snr_ft8eq_cw_bandwidth_offset():
    assert snr_ft8eq("CW", -10, None) == -17.0


def test_snr_ft8eq_unknown_mode_is_null():
    assert snr_ft8eq("SSB", -10, None) is None


def test_snr_ft8eq_null_snr_is_null():
    assert snr_ft8eq("WSPR", None, 30) is None


def _spot(ts, dx_call, de_call, dx_grid, de_grid, snr_db=-10, tx_dbm=20, band="20m"):
    return {
        "ts": ts, "band": band, "mode": "WSPR", "dx_call": dx_call, "de_call": de_call,
        "dx_grid": dx_grid, "de_grid": de_grid, "tx_dbm": tx_dbm, "snr_db": snr_db,
        "source": "wsprnet",
    }


def _df(rows):
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Datetime("us", "UTC")})


def test_build_labels_positive_cell():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14", snr_db=-8, tx_dbm=50)])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    labels = build_labels(spots, universe)
    row = labels.filter(
        (pl.col("tx_field") == "FN") & (pl.col("rx_field") == "DM")
    ).row(0, named=True)
    assert row["open"] == 1
    assert row["n_spots"] == 1
    assert row["snr_ft8eq_p50"] == -8.0


def test_build_labels_output_columns():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14")])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    labels = build_labels(spots, universe)
    assert set(labels.columns) == {
        "window_start", "tx_field", "rx_field", "band", "open", "n_spots",
        "n_monitors", "n_tx_stations", "evidence_tier", "snr_ft8eq_p50",
    }


def test_build_labels_snr_null_when_no_snr():
    spots = _df([_spot(W0, "K1JT", "W6SZ", "FN20", "DM14", snr_db=None)])
    uptime = build_receiver_uptime(spots)
    universe = build_universe(spots, uptime)
    labels = build_labels(spots, universe)
    row = labels.filter(
        (pl.col("tx_field") == "FN") & (pl.col("rx_field") == "DM")
    ).row(0, named=True)
    assert row["snr_ft8eq_p50"] is None
