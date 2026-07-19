from datetime import datetime, timezone

import polars as pl

from propagation.features.history import add_history_features, field_neighbors


def _row(hour, minute, tx, rx, band, n_spots, snr, open_=1):
    return {
        "window_start": datetime(2026, 6, 1, hour, minute, tzinfo=timezone.utc),
        "tx_field": tx, "rx_field": rx, "band": band,
        "n_spots": n_spots, "snr_ft8eq_p50": snr, "open": open_,
    }


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema_overrides={"window_start": pl.Datetime("us", "UTC")},
    )


def test_field_neighbors_interior_field_has_8():
    assert len(field_neighbors("FN")) == 8
    assert "EM" in field_neighbors("FN")  # SW neighbor


def test_field_neighbors_wraps_longitude_at_seam():
    # field "AA" is the westmost field (lon -180..-160); its west neighbor
    # wraps to "RA" (lon 160..180)
    assert "RA" in field_neighbors("AA")


def test_same_cell_trailing_count_respects_availability_buffer():
    # source rows every 15 min for cell (FN,DM,20m); target row at 02:00
    # asks for trailing 1h -- availability buffer means the source window
    # ending at 01:45 (i.e. window_start=01:45) is NOT yet available at
    # 02:00 (becomes available at 01:45+20min=02:05), so only windows with
    # window_start <= 02:00-20min=01:40 count: 01:00,01:15,01:30 (three).
    history = _frame([
        _row(1, 0, "FN", "DM", "20m", 1, 10.0),
        _row(1, 15, "FN", "DM", "20m", 1, 10.0),
        _row(1, 30, "FN", "DM", "20m", 1, 10.0),
        _row(1, 45, "FN", "DM", "20m", 1, 10.0),  # too recent, buffered out
    ])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["same_cell_n_1h"][0] == 3


def test_reverse_path_swaps_tx_and_rx():
    history = _frame([_row(1, 0, "DM", "FN", "20m", 5, 10.0)])  # reverse of target
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["reverse_path_n_3h"][0] == 5


def test_adjacent_band_looks_at_neighboring_bands_same_cell():
    history = _frame([
        _row(1, 0, "FN", "DM", "17m", 2, 10.0),  # one band up from 20m
        _row(1, 0, "FN", "DM", "15m", 9, 10.0),  # NOT adjacent to 20m (two up)
    ])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["adjacent_band_n_3h"][0] == 2


def test_band_wide_sums_across_all_cells_same_band():
    history = _frame([
        _row(1, 0, "FN", "DM", "20m", 3, 10.0),
        _row(1, 0, "EM", "CN", "20m", 4, 10.0),
        _row(1, 0, "FN", "DM", "40m", 100, 10.0),  # different band, excluded
    ])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["band_wide_n_3h"][0] == 7


def test_same_hour_yesterday_is_a_point_lookup_not_an_aggregate():
    history = _frame([_row(2, 0, "FN", "DM", "20m", 1, 10.0, open_=1)])
    yesterday = history.with_columns(
        (pl.col("window_start") - pl.duration(hours=24)).alias("window_start")
    )
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(yesterday, target)
    assert out["same_hour_yesterday_open"][0] == 1


def test_no_history_gives_zero_count_null_snr():
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(_frame([]), target)
    assert out["same_cell_n_24h"][0] == 0
    assert out["same_cell_snr_24h"][0] is None
