import datetime as dt

import polars as pl
import pytest

from propagation.models.climatology import ClimatologyModel


def _label_row(hour, open_, tx="FN", rx="DM", band="20m"):
    return {
        "window_start": dt.datetime(2026, 6, 1, hour, 0, tzinfo=dt.timezone.utc),
        "tx_field": tx, "rx_field": rx, "band": band, "open": open_,
    }


def test_fit_computes_per_cell_hourly_rate():
    train = pl.DataFrame([
        _label_row(12, 1), _label_row(12, 1), _label_row(12, 0),
        _label_row(3, 0), _label_row(3, 0),
    ], schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    model = ClimatologyModel().fit(train)
    pred = model.predict(pl.DataFrame(
        [_label_row(12, None, tx="FN", rx="DM")],
        schema_overrides={"window_start": pl.Datetime("us", "UTC"), "open": pl.Int64},
    ))
    assert pred["p_open"][0] == pytest.approx(2 / 3)


def test_predict_falls_back_to_global_rate_for_unseen_cell():
    train = pl.DataFrame([
        _label_row(12, 1), _label_row(12, 0),
    ], schema_overrides={"window_start": pl.Datetime("us", "UTC")})
    model = ClimatologyModel().fit(train)
    pred = model.predict(pl.DataFrame(
        [_label_row(12, None, tx="ZZ", rx="YY")],
        schema_overrides={"window_start": pl.Datetime("us", "UTC"), "open": pl.Int64},
    ))
    assert pred["p_open"][0] == pytest.approx(0.5)  # global train rate


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ClimatologyModel().predict(pl.DataFrame())
