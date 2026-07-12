import numpy as np
import pytest

from propagation.eval.metrics import brier_score, log_loss_score, reliability_bins


def test_brier_score_perfect_predictions():
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_known_value():
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([0.8, 0.3])
    expected = ((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2
    assert brier_score(y_true, y_prob) == pytest.approx(expected)


def test_log_loss_perfect_predictions_near_zero():
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([0.999999999999, 1e-12])
    assert log_loss_score(y_true, y_prob) < 1e-6


def test_log_loss_known_value():
    y_true = np.array([1.0])
    y_prob = np.array([0.5])
    assert log_loss_score(y_true, y_prob) == pytest.approx(-np.log(0.5))


def test_reliability_bins_shape_and_calibration():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5], dtype=float)
    bins = reliability_bins(y_true, y_prob, n_bins=10)
    assert len(bins) == 10
    high_bin = [b for b in bins if b["n"] and b["lo"] >= 0.8][0]
    assert high_bin["observed_rate"] == pytest.approx(0.5)
