import json

import numpy as np
import polars as pl
import pytest

from propagation.models.gbt import GBTModel


def _synthetic_features(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    from propagation.features.matrix import FEATURE_COLUMNS
    data = {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    # make `open` a real (noisy) function of the first feature so the model
    # has something learnable to fit, not pure noise.
    first_col = FEATURE_COLUMNS[0]
    logit = data[first_col] * 2.0
    p = 1 / (1 + np.exp(-logit))
    data["open"] = (rng.uniform(size=n) < p).astype(int)
    data["sample_weight"] = np.ones(n)
    return pl.DataFrame(data)


def test_fit_predict_roundtrip_beats_a_coinflip():
    train = _synthetic_features(n=3000, seed=1)
    test = _synthetic_features(n=1000, seed=2)
    model = GBTModel().fit(train)
    pred = model.predict(test)
    assert "p_open" in pred.columns
    assert pred.height == test.height
    assert pred["p_open"].is_between(0.0, 1.0).all()
    from propagation.eval.metrics import brier_score
    brier = brier_score(test["open"].cast(float).to_numpy(), pred["p_open"].to_numpy())
    assert brier < 0.25  # a coinflip (p=0.5 always) scores exactly 0.25 on balanced-ish data


def test_save_load_roundtrip(tmp_path):
    train = _synthetic_features(n=1000, seed=3)
    model = GBTModel().fit(train)
    path = tmp_path / "model"
    model.save(path)
    loaded = GBTModel.load(path)
    test = _synthetic_features(n=200, seed=4)
    pred_a = model.predict(test)["p_open"].to_numpy()
    pred_b = loaded.predict(test)["p_open"].to_numpy()
    np.testing.assert_allclose(pred_a, pred_b, rtol=1e-6)


def test_load_rejects_feature_column_drift(tmp_path):
    train = _synthetic_features(n=500, seed=5)
    model = GBTModel().fit(train)
    path = tmp_path / "model"
    model.save(path)
    meta = json.loads((path / "meta.json").read_text())
    meta["feature_columns"] = meta["feature_columns"][:-1]  # simulate drift
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature"):
        GBTModel.load(path)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GBTModel().predict(pl.DataFrame())
