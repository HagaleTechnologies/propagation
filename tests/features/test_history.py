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


def test_adjacent_cell_looks_at_maidenhead_neighbors_same_tx_and_band():
    # Confirm via field_neighbors directly (rather than hardcoding an
    # assumption) that "EM" is a Maidenhead neighbor of the target rx "DM",
    # and that "FN" is NOT -- used as the negative control.
    assert "EM" in field_neighbors("DM")
    assert "FN" not in field_neighbors("DM")
    history = _frame([
        _row(1, 0, "FN", "EM", "20m", 6, 10.0),   # rx=EM, a neighbor of DM: counted
        _row(1, 0, "FN", "FN", "20m", 99, 10.0),  # rx=FN, NOT a neighbor of DM: excluded
    ])
    target = _frame([_row(2, 0, "FN", "DM", "20m", 0, None)])
    out = add_history_features(history, target)
    assert out["adjacent_cell_n_3h"][0] == 6


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


def test_self_referential_call_matches_production_pattern_no_fanout():
    # Task 7's build_feature_matrix calls add_history_features with
    # full_history and target_rows as the literal same table (every other
    # test above uses deliberately separate history/target fixtures). Build
    # one table spanning several hours with two distinct (tx,rx,band) cells,
    # one of which (FN,DM,20m) has closely-spaced rows so there's a real
    # trailing count to hand-verify.
    rows = []
    # Cell A (FN,DM,20m): 9 rows every 15 min, 00:00..02:00, n_spots=1 each.
    for i in range(9):
        hh, mm = divmod(i * 15, 60)
        rows.append(_row(hh, mm, "FN", "DM", "20m", 1, 10.0))
    # Cell B (EM,CN,40m): 3 rows on the hour, 00:00/01:00/02:00, n_spots=2 each.
    for hh in (0, 1, 2):
        rows.append(_row(hh, 0, "EM", "CN", "40m", 2, 10.0))
    table = _frame(rows)
    assert table.height == 12

    out = add_history_features(table, table)

    # (a) no fanout/duplication despite the real rows and their injected
    # zero-weight anchors sharing timestamps: row count in == row count out.
    assert out.height == table.height

    # (b) hand-computed same_cell_n_1h at T=02:00. Availability buffer is
    # 20min, so the trailing 1h window [01:00, 02:00) minus the buffer zone
    # [01:40, 02:00) leaves [01:00, 01:40). For cell A that's the rows at
    # 01:00, 01:15, 01:30 (01:45 falls inside the buffered-out zone) -> 3
    # (n_spots=1 each). For cell B, only the 01:00 row falls in
    # [01:00, 01:40) -> n_spots=2. The row at 02:00 itself (both the real
    # row and its own anchor) is excluded by closed="left", confirming no
    # double-counting from the duplicate timestamp.
    two_am = datetime(2026, 6, 1, 2, 0, tzinfo=timezone.utc)
    cell_a_2am = out.filter(
        (pl.col("tx_field") == "FN") & (pl.col("rx_field") == "DM")
        & (pl.col("band") == "20m") & (pl.col("window_start") == two_am)
    )
    cell_b_2am = out.filter(
        (pl.col("tx_field") == "EM") & (pl.col("rx_field") == "CN")
        & (pl.col("band") == "40m") & (pl.col("window_start") == two_am)
    )
    assert cell_a_2am.height == 1
    assert cell_a_2am["same_cell_n_1h"][0] == 3
    assert cell_b_2am.height == 1
    assert cell_b_2am["same_cell_n_1h"][0] == 2
