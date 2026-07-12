import datetime as dt

from propagation.eval.splits import blocked_cv_gap_hours, blocked_time_series_folds


def test_gap_floor_is_48h():
    assert blocked_cv_gap_hours(max_horizon_hours=0, max_ar_lookback_hours=0) == 48.0


def test_gap_grows_with_horizon_and_lookback():
    assert blocked_cv_gap_hours(max_horizon_hours=24, max_ar_lookback_hours=24) == 48.0
    assert blocked_cv_gap_hours(max_horizon_hours=48, max_ar_lookback_hours=24) == 72.0


def test_folds_respect_gap_and_dont_overlap():
    data_start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    data_end = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    folds = blocked_time_series_folds(
        data_start, data_end,
        train_span=dt.timedelta(days=30),
        eval_span=dt.timedelta(days=15),
        max_horizon_hours=0,
        max_ar_lookback_hours=0,
    )
    assert len(folds) >= 1
    for fold in folds:
        gap = (fold.eval_start - fold.train_end).total_seconds() / 3600
        assert gap == 48.0
        assert fold.train_start < fold.train_end <= fold.eval_start < fold.eval_end
        assert fold.eval_end <= data_end


def test_folds_empty_when_span_too_short():
    data_start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    data_end = dt.datetime(2026, 1, 10, tzinfo=dt.timezone.utc)
    folds = blocked_time_series_folds(
        data_start, data_end,
        train_span=dt.timedelta(days=30),
        eval_span=dt.timedelta(days=15),
        max_horizon_hours=0,
        max_ar_lookback_hours=0,
    )
    assert folds == []
