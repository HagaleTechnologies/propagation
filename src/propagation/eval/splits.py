import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class CVFold:
    train_start: dt.datetime
    train_end: dt.datetime
    eval_start: dt.datetime
    eval_end: dt.datetime


def blocked_cv_gap_hours(max_horizon_hours: float, max_ar_lookback_hours: float) -> float:
    """docs/SPEC-labeling.md sec 6 rule 1: gap = max(48h, horizon + AR lookback)."""
    return max(48.0, max_horizon_hours + max_ar_lookback_hours)


def blocked_time_series_folds(
    data_start: dt.datetime,
    data_end: dt.datetime,
    train_span: dt.timedelta,
    eval_span: dt.timedelta,
    max_horizon_hours: float,
    max_ar_lookback_hours: float,
) -> list[CVFold]:
    gap = dt.timedelta(hours=blocked_cv_gap_hours(max_horizon_hours, max_ar_lookback_hours))
    folds: list[CVFold] = []
    train_start = data_start
    while True:
        train_end = train_start + train_span
        eval_start = train_end + gap
        eval_end = eval_start + eval_span
        if eval_end > data_end:
            break
        folds.append(CVFold(train_start, train_end, eval_start, eval_end))
        train_start = eval_start
    return folds
