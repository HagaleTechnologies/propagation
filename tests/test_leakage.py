"""Leakage audit for M2's feature matrix (docs/SPEC-labeling.md sec 6,
ARCHITECTURE.md sec 6). Seven properties, each a real failure mode a subtler
bug could reintroduce silently:
1. A source spot inside the Δ_avail=20min buffer is excluded from history features.
2. A source spot just outside the buffer IS included (the boundary is exact, not approximate).
3. The blocked-CV gap for M2's real parameters (horizon=3h, AR lookback=24h) is 48h, the floor.
4. Blocked folds actually honor that computed gap end-to-end.
5. No definitive-Kp or any label column ends up in FEATURE_COLUMNS.
6. `snr_ft8eq_p50` -- the anchor row's OWN mode-normalized SNR, computed by
   build_labels() from the very spots that define its `open` value -- must
   never appear in FEATURE_COLUMNS (a live acceptance run found a LightGBM
   model trained on it hitting ~0 Brier by learning its null-iff-open=0
   pattern instead of any real signal).
7. No feature column's null-vs-not-null pattern is a perfect proxy for
   `open`, guarding against the same failure mode recurring under a
   different column name.
Plus an eighth, specific to M2: space-weather features must be strictly
trailing (as-of), never centered, independent of any CV gap consideration.
"""
from datetime import datetime, timedelta, timezone

import polars as pl

from propagation.eval.splits import blocked_cv_gap_hours, blocked_time_series_folds
from propagation.features.history import add_history_features
from propagation.features.matrix import FEATURE_COLUMNS, build_feature_matrix


def _row(hour, minute, n=1, snr=10.0):
    return {
        "window_start": datetime(2026, 6, 1, hour, minute, tzinfo=timezone.utc),
        "tx_field": "FN", "rx_field": "DM", "band": "20m", "n_spots": n, "snr_ft8eq_p50": snr, "open": 1,
    }


def _frame(rows):
    return pl.DataFrame(rows, schema_overrides={"window_start": pl.Datetime("us", "UTC")})


def test_spot_just_inside_availability_buffer_is_excluded():
    # target at 02:00; a source window at 01:41 becomes available at
    # 01:41+20min=02:01, one minute AFTER the target time -> must be excluded.
    history = _frame([_row(1, 41)])
    target = _frame([_row(2, 0, n=0, snr=None)])
    out = add_history_features(history, target)
    assert out["same_cell_n_24h"][0] == 0


def test_spot_just_outside_availability_buffer_is_included():
    # source window at 01:40 becomes available at exactly 02:00 -> included
    # (the buffer boundary is inclusive of "available at or before now").
    history = _frame([_row(1, 40)])
    target = _frame([_row(2, 0, n=0, snr=None)])
    out = add_history_features(history, target)
    assert out["same_cell_n_24h"][0] == 1


def test_asof_features_never_see_data_after_prediction_time_for_nonzero_horizon():
    history = _frame([_row(20, 0, n=5, snr=20.0)])  # source at 20:00
    target = _frame([_row(22, 0, n=0, snr=None)])    # target at 22:00
    out_h0 = add_history_features(history, target, horizon_hours=0.0)
    out_h6 = add_history_features(history, target, horizon_hours=6.0)
    # same_cell_n_24h sums n_spots (5 here), not row count.
    assert out_h0["same_cell_n_24h"][0] == 5  # legitimately visible: prediction_time=22:00
    assert out_h6["same_cell_n_24h"][0] == 0  # NOT visible: prediction_time=16:00, source is after it


def test_m2_blocked_cv_gap_is_the_48h_floor_not_widened():
    # ROADMAP.md M2: horizon up to +3h; ARCHITECTURE.md sec 4 item 5: AR
    # lookback up to 24h. 3 + 24 = 27 < 48 -> floor applies, unchanged from M0/M1.
    gap = blocked_cv_gap_hours(max_horizon_hours=3, max_ar_lookback_hours=24)
    assert gap == 48.0


def test_blocked_folds_honor_the_computed_gap():
    folds = blocked_time_series_folds(
        data_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data_end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        train_span=timedelta(days=30),
        eval_span=timedelta(days=15),
        max_horizon_hours=3, max_ar_lookback_hours=24,
    )
    assert len(folds) > 0
    for fold in folds:
        gap = (fold.eval_start - fold.train_end).total_seconds() / 3600
        assert gap >= 48.0


def test_no_label_or_definitive_kp_columns_in_feature_columns():
    forbidden = {"open", "split", "sample_weight", "window_start", "tx_field", "rx_field", "band",
                 "n_monitors", "n_tx_stations", "evidence_tier"}
    assert forbidden.isdisjoint(set(FEATURE_COLUMNS))
    # definitive Kp (propagation.eval.stratify) is a wholly separate module
    # from the training feature (propagation.features.spaceweather's OMNI2
    # Kp) -- confirm the feature columns don't accidentally name anything
    # that reads as the eval-only series.
    assert not any("definitive" in c for c in FEATURE_COLUMNS)


def test_snr_ft8eq_p50_is_not_a_feature():
    # It's the anchor row's own label-defining value (null iff open=0),
    # not a leading indicator -- see module docstring property 6.
    assert "snr_ft8eq_p50" not in FEATURE_COLUMNS


def test_no_feature_column_perfectly_encodes_open_via_nulls():
    # Rows spaced 48h apart, same cell, so no history relation ever finds a
    # neighbor within any lookback window -- history counts stay uniformly
    # 0 and history SNR averages stay uniformly null, so neither pattern
    # coincidentally lines up with `open` in this small fixture. The only
    # column that legitimately correlates 1:1 with `open` via nulls is
    # snr_ft8eq_p50 itself, already excluded from FEATURE_COLUMNS above.
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    labels = pl.DataFrame({
        "window_start": [start + timedelta(hours=48 * i) for i in range(4)],
        "tx_field": ["FN"] * 4, "rx_field": ["DM"] * 4, "band": ["20m"] * 4,
        "open": [1, 0, 1, 0],
        "n_spots": [3, 0, 2, 0],
        "snr_ft8eq_p50": [10.0, None, 8.0, None],
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})

    omni_times = [start + timedelta(hours=i) for i in range(24 * 10)]
    omni = pl.DataFrame({
        "time": omni_times, "kp": [3.0] * len(omni_times), "f107": [100.0] * len(omni_times),
        "bz_gsm": [1.0] * len(omni_times), "solar_wind_speed": [400.0] * len(omni_times),
        "dst": [-10.0] * len(omni_times),
    }, schema_overrides={"time": pl.Datetime("us", "UTC")})

    out = build_feature_matrix(labels, full_history=labels, omni=omni)
    is_negative = out["open"] == 0
    for col in FEATURE_COLUMNS:
        is_null = out[col].is_null()
        assert not (is_null == is_negative).all(), (
            f"{col!r} perfectly encodes `open` via its null pattern -- likely leakage"
        )


def test_spaceweather_features_are_trailing_not_centered():
    # A 27-day mean computed with a CENTERED window would include rows after
    # window_start; verify by constructing OMNI data where only a future
    # spike would move the smoothed value, and confirming it doesn't.
    from propagation.features.spaceweather import add_spaceweather_features
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(24 * 40)]
    f107 = [10.0] * (24 * 30) + [10000.0] * (24 * 10)  # huge spike starting day 30
    omni = pl.DataFrame({
        "time": times, "kp": [3.0] * len(times), "f107": f107,
        "bz_gsm": [1.0] * len(times), "solar_wind_speed": [400.0] * len(times), "dst": [-10.0] * len(times),
    }, schema_overrides={"time": pl.Datetime("us", "UTC")})
    labels = pl.DataFrame({
        "window_start": [start + timedelta(days=29, hours=23)],  # just before the spike
    }, schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    out = add_spaceweather_features(labels, omni)
    # a trailing 27-day mean at this point sees none of the spike yet
    assert out["f107_smoothed_27d"][0] < 100.0
