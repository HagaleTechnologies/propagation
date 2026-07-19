"""LightGBM model over the M2 feature matrix (ARCHITECTURE.md sec 5 M-2).
Matches ClimatologyModel/P533Model's shape by convention: .fit(labels_with_
features) -> self, .predict(features) -> features + p_open. No shared base
class. Isotonic calibration is fit on a held-out time-tail slice of the
training data (never the eval set), per docs/SPEC-labeling.md sec 4.5's
requirement that sample_weight feed both the booster and the calibrator.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import polars as pl
from sklearn.isotonic import IsotonicRegression

from propagation.features.matrix import FEATURE_COLUMNS

_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
}
_CALIBRATION_TAIL_FRACTION = 0.15


class GBTModel:
    model_id = "gbt"

    def __init__(self) -> None:
        self._booster: lgb.Booster | None = None
        self._calibrator: IsotonicRegression | None = None

    def fit(self, train_features: pl.DataFrame) -> "GBTModel":
        """`train_features` must carry FEATURE_COLUMNS, `open`, and
        `sample_weight`, already time-sorted is not required (sorted
        internally by window_start if present) -- split the time tail off
        for early stopping + calibration, never touching the eval set."""
        df = train_features
        if "window_start" in df.columns:
            df = df.sort("window_start")
        n = df.height
        n_tail = max(1, int(n * _CALIBRATION_TAIL_FRACTION))
        fit_part, tail_part = df.head(n - n_tail), df.tail(n_tail)

        X_fit = fit_part.select(FEATURE_COLUMNS).to_numpy()
        y_fit = fit_part["open"].cast(float).to_numpy()
        w_fit = fit_part["sample_weight"].to_numpy()
        X_tail = tail_part.select(FEATURE_COLUMNS).to_numpy()
        y_tail = tail_part["open"].cast(float).to_numpy()
        w_tail = tail_part["sample_weight"].to_numpy()

        train_set = lgb.Dataset(X_fit, label=y_fit, weight=w_fit, feature_name=FEATURE_COLUMNS)
        valid_set = lgb.Dataset(X_tail, label=y_tail, weight=w_tail, reference=train_set)
        self._booster = lgb.train(
            _LGB_PARAMS, train_set, num_boost_round=2000, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        raw_tail_pred = self._booster.predict(X_tail, num_iteration=self._booster.best_iteration)
        self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._calibrator.fit(raw_tail_pred, y_tail, sample_weight=w_tail)
        return self

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        if self._booster is None or self._calibrator is None:
            raise RuntimeError("call fit() (or load()) before predict()")
        X = features.select(FEATURE_COLUMNS).to_numpy()
        raw = self._booster.predict(X, num_iteration=self._booster.best_iteration)
        calibrated = self._calibrator.predict(raw)
        return features.with_columns(pl.Series("p_open", calibrated, dtype=pl.Float64))

    def save(self, path: Path) -> None:
        if self._booster is None or self._calibrator is None:
            raise RuntimeError("call fit() before save()")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path / "booster.txt"))
        (path / "calibrator.pkl").write_bytes(pickle.dumps(self._calibrator))
        (path / "meta.json").write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}))

    @classmethod
    def load(cls, path: Path) -> "GBTModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        if meta["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError(
                "feature column drift: saved model was trained on a different "
                "FEATURE_COLUMNS than the current code defines -- retrain, "
                "don't load a stale artifact against changed features."
            )
        model = cls()
        model._booster = lgb.Booster(model_file=str(path / "booster.txt"))
        model._calibrator = pickle.loads((path / "calibrator.pkl").read_bytes())
        return model
