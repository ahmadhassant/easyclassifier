"""Classifier registry (Phase 10 & 23).

Each classifier is registered with a key, a friendly name, and a factory. New
algorithms can be added here without touching the wizard. Optional
dependencies (XGBoost, LightGBM) are detected gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .distances import make_knn


@dataclass
class ClassifierSpec:
    key: str
    name: str
    factory: Callable[[], object]
    help_key: str
    available: bool = True
    reason: str = ""


# Every classifier that uses randomness gets a fixed seed, so the same data
# and EasyClassifier version always give exactly the same results.
SEED = 0


def _svc():
    # probability=True lets us compute ROC AUC / PR curves.
    return SVC(probability=True, random_state=SEED)


def _xgb_factory():
    from xgboost import XGBClassifier
    return XGBClassifier(eval_metric="logloss", verbosity=0,
                         random_state=SEED)


def _lgbm_factory():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(verbose=-1, random_state=SEED)


def _optional(key, name, factory, help_key, module) -> ClassifierSpec:
    try:
        __import__(module)
        return ClassifierSpec(key, name, factory, help_key, True)
    except Exception:  # noqa: BLE001
        return ClassifierSpec(
            key, name, factory, help_key, False,
            reason=f"install '{module}' to enable",
        )


def build_registry() -> Dict[str, ClassifierSpec]:
    specs = [
        ClassifierSpec("decision_tree", "Decision Tree",
                       lambda: DecisionTreeClassifier(random_state=SEED),
                       "decision_tree"),
        ClassifierSpec("random_forest", "Random Forest",
                       lambda: RandomForestClassifier(random_state=SEED),
                       "random_forest"),
        ClassifierSpec("svm", "Support Vector Machine (SVM)",
                       _svc, "svm"),
        ClassifierSpec("logistic_regression", "Logistic Regression",
                       lambda: LogisticRegression(max_iter=1000),
                       "logistic_regression"),
        ClassifierSpec("knn", "K-Nearest Neighbours (KNN)",
                       lambda: make_knn("hassanat", 5), "knn"),
        ClassifierSpec("naive_bayes", "Naive Bayes",
                       GaussianNB, "naive_bayes"),
        _optional("xgboost", "XGBoost", _xgb_factory, "xgboost", "xgboost"),
        _optional("lightgbm", "LightGBM", _lgbm_factory, "lightgbm",
                  "lightgbm"),
        ClassifierSpec("neural_network", "Neural Network (MLP)",
                       lambda: MLPClassifier(max_iter=500,
                                             random_state=SEED),
                       "neural_network"),
    ]
    return {s.key: s for s in specs}


def available_specs(registry: Dict[str, ClassifierSpec]) -> List[ClassifierSpec]:
    return list(registry.values())
