"""Forecasting models, all with default parameters (no tuning).

Every model has the same interface: ``fit(y)`` on the history, then
``predict(h)`` for the next h values after the end of that history.

Families
--------
* Reference rules: naive (repeat the last value) and seasonal naive (repeat
  the value one season ago). They are always included; a model is only
  useful if it beats them.
* Statistical: exponential smoothing with damped trend and seasonality
  (ETS; Hyndman et al.) and the Theta method (Assimakopoulos & Nikolopoulos,
  2000), from statsmodels.
* Machine learning on past values: the last L values predict the next H
  values at once. Windows are expressed relative to their last value and
  divided by the spread of the training series, so models that cannot
  extrapolate (trees, nearest neighbours) still follow trends and levels.
* Deep networks (PyTorch): see ``deep.py``; same windows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from easyclassifier.distances import hassanat_matrix

from ..common.references import REFERENCES
from . import deep

SEED = 0
KNN_K = 5


def window_length(n: int, season: int, horizon: int) -> int:
    """Past values used as input: two seasons (at least 2 horizons, at least
    10), but no more than 100 and no more than a quarter of the series."""
    L = max(2 * season, 2 * horizon, 10)
    return int(max(4, min(L, 100, n // 4)))


# --------------------------------------------------------------------------- #
# Reference rules and statistical models
# --------------------------------------------------------------------------- #

class Naive:
    def fit(self, y):
        self.last_ = float(y[-1])
        return self

    def predict(self, h):
        return np.full(h, self.last_)


class SeasonalNaive:
    def __init__(self, season):
        self.season = season

    def fit(self, y):
        self.cycle_ = np.asarray(y[-self.season:], dtype=float)
        return self

    def predict(self, h):
        return np.array([self.cycle_[i % self.season] for i in range(h)])


class ETS:
    """Additive damped trend; additive seasonality when a season is set."""

    def __init__(self, season):
        self.season = season

    def fit(self, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = self.season > 1 and len(y) >= 2 * self.season
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = ExponentialSmoothing(
                np.asarray(y, dtype=float), trend="add", damped_trend=True,
                seasonal="add" if seasonal else None,
                seasonal_periods=self.season if seasonal else None,
                initialization_method="estimated").fit()
        return self

    def predict(self, h):
        return np.asarray(self.model_.forecast(h), dtype=float)


class Theta:
    def __init__(self, season):
        self.season = season

    def fit(self, y):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        y = np.asarray(y, dtype=float)
        seasonal = self.season > 1 and len(y) >= 2 * self.season
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = ThetaModel(
                y, period=self.season if seasonal else 1,
                deseasonalize=seasonal,
                method="additive" if np.any(y <= 0) else "auto").fit()
        return self

    def predict(self, h):
        return np.asarray(self.model_.forecast(h), dtype=float)


# --------------------------------------------------------------------------- #
# Models on windows of past values
# --------------------------------------------------------------------------- #

def make_windows(y, L, H):
    y = np.asarray(y, dtype=float)
    n = len(y) - L - H + 1
    if n <= 0:
        return np.empty((0, L)), np.empty((0, H))
    X = np.stack([y[i:i + L] for i in range(n)])
    Y = np.stack([y[i + L:i + L + H] for i in range(n)])
    return X, Y


class HassanatKNN:
    """Average future of the k most similar past windows (Hassanat
    distance; the signed form applies to the negative relative values)."""

    def __init__(self, k=KNN_K):
        self.k = k

    def fit(self, X, Y):
        self.X_, self.Y_ = np.asarray(X, float), np.asarray(Y, float)
        return self

    def predict(self, X):
        D = hassanat_matrix(np.asarray(X, float), self.X_)
        k = min(self.k, len(self.X_))
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        return self.Y_[idx].mean(axis=1)


class WindowModel:
    """Learns next-H values from the last L values, relative to the last
    value of each window."""

    MIN_WINDOWS = 10

    def __init__(self, L, H, regressor=None, arch=None):
        self.L, self.H = L, H
        self.regressor, self.arch = regressor, arch

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        X, Y = make_windows(y, self.L, self.H)
        need = 30 if self.arch else self.MIN_WINDOWS
        if len(X) < need:
            raise ValueError(f"needs at least {need} training windows of "
                             f"{self.L}+{self.H} values")
        self.scale_ = float(np.std(np.diff(y))) or float(np.std(y)) or 1.0
        last = X[:, -1:]
        Xr, Yr = (X - last) / self.scale_, (Y - last) / self.scale_
        if self.arch:
            self.net_ = deep.train(self.arch, Xr, Yr)
        else:
            self.model_ = self.regressor().fit(Xr, Yr)
        self.history_ = y[-self.L:]
        return self

    def predict(self, h):
        if h > self.H:
            raise ValueError("horizon longer than the trained horizon")
        w = self.history_
        xr = ((w - w[-1]) / self.scale_)[None, :]
        out = deep.predict(self.net_, xr) if self.arch else \
            self.model_.predict(xr)
        return (w[-1] + np.asarray(out).reshape(-1) * self.scale_)[:h]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass
class ForecasterSpec:
    key: str
    name: str
    family: str                       # reference | statistical | ml | deep
    make: Callable[[int, int, int], object]   # (L, H, season) -> model
    default_selected: bool = False
    needs_season: bool = False
    available: bool = True
    reason: str = ""


# Factories are module-level functions (not lambdas), so a trained model can
# be saved with joblib and loaded again.
def _random_forest():
    return RandomForestRegressor(random_state=SEED)


def _gradient_boosting():
    return MultiOutputRegressor(HistGradientBoostingRegressor(random_state=SEED))


def _window(regressor=None, arch=None):
    return lambda L, H, m: WindowModel(L, H, regressor=regressor, arch=arch)


def build_registry() -> Dict[str, ForecasterSpec]:
    specs = [
        ForecasterSpec("naive", "Naive (last value)", "reference",
                       lambda L, H, m: Naive()),
        ForecasterSpec("seasonal_naive", "Seasonal naive (value one season "
                       "ago)", "reference", lambda L, H, m: SeasonalNaive(m),
                       needs_season=True),
        ForecasterSpec("ets", "Exponential smoothing (ETS)", "statistical",
                       lambda L, H, m: ETS(m), default_selected=True),
        ForecasterSpec("theta", "Theta method", "statistical",
                       lambda L, H, m: Theta(m), default_selected=True),
        ForecasterSpec("ridge", "Linear model on past values (Ridge)", "ml",
                       _window(Ridge), default_selected=True),
        ForecasterSpec("random_forest", "Random Forest on past values", "ml",
                       _window(_random_forest), default_selected=True),
        ForecasterSpec("gradient_boosting", "Gradient Boosting on past "
                       "values", "ml", _window(_gradient_boosting)),
        ForecasterSpec("knn", f"KNN on past values (Hassanat distance, "
                       f"k={KNN_K})", "ml", _window(HassanatKNN)),
        ForecasterSpec("mlp", "Deep: MLP", "deep", _window(arch="mlp")),
        ForecasterSpec("lstm", "Deep: LSTM", "deep", _window(arch="lstm")),
        ForecasterSpec("gru", "Deep: GRU", "deep", _window(arch="gru")),
        ForecasterSpec("tcn", "Deep: temporal convolutional network (TCN)",
                       "deep", _window(arch="tcn")),
        ForecasterSpec("nbeats", "Deep: N-BEATS", "deep",
                       _window(arch="nbeats")),
        ForecasterSpec("transformer", "Deep: Transformer", "deep",
                       _window(arch="transformer")),
    ]
    return {s.key: s for s in specs}


REFERENCE_KEYS = ["naive", "seasonal_naive"]

# Citation texts live in common/references.py (one list for every report).
STAT_CITATIONS = {"ets": REFERENCES["hyndman2002"],
                  "theta": REFERENCES["assimakopoulos2000"]}
STATSMODELS_CITATION = REFERENCES["statsmodels"]


def spec_unusable(spec: ForecasterSpec, season: int) -> Optional[str]:
    if spec.needs_season and season <= 1:
        return "no seasonal pattern assumed for this series"
    return None
