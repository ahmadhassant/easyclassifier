"""Every model a user can select can be saved after training and loaded again
with identical predictions (trained_model.pkl must always work), and a model
that cannot be saved never costs the user the rest of the results."""

import joblib
import numpy as np
import pandas as pd
import pytest

from easyresearch.regression import analysis as ra
from easyresearch.regression.models import build_registry as regressors
from easyresearch.timeseries import ForecastSettings, run_forecast
from easyresearch.timeseries import data as td
from easyresearch.timeseries.models import build_registry as forecasters
from easyresearch.timeseries.signal_models import build_registry as signal_models
from easyclassifier.logbook import LogBook


def _roundtrip(model, path):
    joblib.dump(model, path)
    return joblib.load(path)


@pytest.mark.parametrize("key", list(forecasters()))
def test_every_forecaster_saves_and_reloads(tmp_path, key):
    y = td.prepare_series(td.load_demo(), "CO2_ppm").values[:240]
    model = forecasters()[key].make(24, 6, 12).fit(y)
    loaded = _roundtrip(model, tmp_path / "model.pkl")
    assert np.allclose(loaded.predict(6), model.predict(6))


@pytest.mark.parametrize("key", list(signal_models()))
def test_every_signal_classifier_saves_and_reloads(tmp_path, key):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 40)
    X = pd.DataFrame([np.sin(2 * np.pi * (2 + i % 2) * t) + rng.normal(0, .2, 40) for i in range(40)])
    y = np.arange(40) % 2
    model = signal_models()[key].factory().fit(X, y)
    loaded = _roundtrip(model, tmp_path / "model.pkl")
    assert (loaded.predict(X) == model.predict(X)).all()


@pytest.mark.parametrize("key", [k for k, s in regressors().items() if s.available])
def test_every_regressor_saves_and_reloads(tmp_path, key):
    from easyresearch.regression.data import load_demo
    df = load_demo().head(120)
    prep = ra.prepare(df, "Progression", LogBook())
    model = ra._pipeline_spec(regressors()[key], prep.X, prep.cfg).factory().fit(prep.X, prep.y)
    loaded = _roundtrip(model, tmp_path / "model.pkl")
    assert np.allclose(loaded.predict(prep.X), model.predict(prep.X))


def test_a_model_that_cannot_be_saved_does_not_lose_the_results(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise TypeError("cannot pickle this object")
    monkeypatch.setattr(joblib, "dump", refuse)
    run = run_forecast(td.load_demo(), "CO2_ppm", ForecastSettings(models=["ridge"], horizon=6, figures=False),
                       str(tmp_path), "s", "s", progress=lambda m: None)
    out = run["out"]
    assert any("could not be saved" in w for w in run["warnings"])
    import os
    assert not os.path.exists(os.path.join(out, "trained_model.pkl"))
    for name in ("forecast.csv", "summary.csv", "settings.json", "log.txt"):
        assert os.path.exists(os.path.join(out, name))
