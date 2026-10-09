"""Regression models with their authors' default parameters.

No hyperparameter tuning is done (the EasyClassifier policy): defaults were
chosen by each method's authors to work well across many datasets, while
values tuned on one dataset rarely carry over to another.

Two models (SVR and the neural network) are sensitive to the units of the
predicted value: SVR's default tolerance (epsilon = 0.1) and the network's
default learning rate assume values of roughly unit size. For them the
predicted value is standardised inside the pipeline (learned on training
rows only) and transformed back for every prediction. This is a change of
units, not a tuned parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from easyclassifier.distances import hassanat_matrix

SEED = 0
KNN_K = 5

from ..common.hassanat import HASSANAT_CITATIONS  # noqa: F401  (re-exported)


class HassanatKNNRegressor(BaseEstimator, RegressorMixin):
    """K-nearest neighbours regression with the Hassanat distance.

    The prediction is the average value of the k closest training rows.
    """

    def __init__(self, n_neighbors: int = KNN_K):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        self.n_features_in_ = self.X_.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.n_neighbors, self.X_.shape[0])
        out = np.empty(X.shape[0])
        # Blocks keep the distance matrix small for large datasets.
        for s in range(0, X.shape[0], 512):
            D = hassanat_matrix(X[s:s + 512], self.X_)
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            out[s:s + 512] = self.y_[idx].mean(axis=1)
        return out


def _unit_target(model):
    return TransformedTargetRegressor(regressor=model,
                                      transformer=StandardScaler())


@dataclass
class RegressorSpec:
    key: str
    name: str
    factory: Callable[[], object]
    scaling: Optional[str]          # None, "standard" or "minmax"
    available: bool = True
    reason: str = ""
    default_selected: bool = False


def _optional(key, name, module, factory, scaling=None) -> RegressorSpec:
    try:
        __import__(module)
        return RegressorSpec(key, name, factory, scaling)
    except Exception:  # noqa: BLE001
        return RegressorSpec(key, name, factory, scaling, False,
                             f"install '{module}' to enable")


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(verbosity=0, random_state=SEED)


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(verbose=-1, random_state=SEED)


def build_registry() -> Dict[str, RegressorSpec]:
    specs = [
        RegressorSpec("linear_regression", "Linear Regression",
                      LinearRegression, "standard", default_selected=True),
        RegressorSpec("ridge", "Ridge Regression", Ridge, "standard"),
        RegressorSpec("decision_tree", "Decision Tree",
                      lambda: DecisionTreeRegressor(random_state=SEED), None),
        RegressorSpec("random_forest", "Random Forest",
                      lambda: RandomForestRegressor(random_state=SEED), None,
                      default_selected=True),
        RegressorSpec("gradient_boosting", "Gradient Boosting",
                      lambda: HistGradientBoostingRegressor(random_state=SEED),
                      None, default_selected=True),
        RegressorSpec("svr", "Support Vector Regression (SVR)",
                      lambda: _unit_target(SVR()), "standard"),
        RegressorSpec("knn", f"KNN (Hassanat distance, k={KNN_K})",
                      lambda: HassanatKNNRegressor(KNN_K), "minmax"),
        RegressorSpec("neural_network", "Neural Network (MLP)",
                      lambda: _unit_target(MLPRegressor(max_iter=1000,
                                                        random_state=SEED)),
                      "standard"),
        _optional("xgboost", "XGBoost", "xgboost", _xgb),
        _optional("lightgbm", "LightGBM", "lightgbm", _lgbm),
    ]
    return {s.key: s for s in specs}
