"""Validation, metrics, honest final score and plain-language summaries.

Models are ranked by R² on rows they were not trained on. Ranking by RMSE
would give the same order, because both compare the same squared errors on
the same rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split

import pandas as pd

from ..common.hassanat import mhsp
from ..common.output import fmt  # noqa: F401  (used by figures and report)

METRICS = {
    "r2": "R²",
    "rmse": "RMSE",
    "mae": "MAE",
    "medae": "Median AE",
    "mape": "MAPE",
    "mhsp": "MHSP",
}
LOWER_IS_BETTER = {"rmse", "mae", "medae", "mape"}

VALIDATION = {
    "kfold5": "5-fold cross-validation",
    "kfold10": "10-fold cross-validation",
    "holdout": "single 80/20 validation split",
}
SELECTION = {
    "auto": "Automatic (recommended)",
    "nested": "Nested cross-validation",
    "final_test": "Separate 20% final test",
}
NESTED_MAX_ROWS = 2000      # same rule as EasyClassifier 0.8.1
FINAL_TEST_SIZE = 0.2
SPLIT_SEED = 42


@dataclass
class Result:
    key: str
    name: str
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    rows: np.ndarray                     # positions of the scored rows
    selection_score: float               # R²
    fold_scores: List[float] = field(default_factory=list)
    note: str = ""


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
    }
    # A percentage error only makes sense when no actual value is zero or
    # changes sign.
    if np.all(y_true > 0) or np.all(y_true < 0):
        out["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    out["mhsp"] = mhsp(y_true, y_pred)
    return out


def _make(key, name, y_true, y_pred, rows, fold_scores=None, note=""):
    m = compute_metrics(y_true, y_pred)
    return Result(key, name, m, np.asarray(y_true, dtype=float),
                  np.asarray(y_pred, dtype=float), np.asarray(rows),
                  m["r2"], list(fold_scores or []), note)


def splitter(method: str, n: int):
    wanted = 10 if method == "kfold10" else 5
    return KFold(n_splits=max(2, min(wanted, n)), shuffle=True,
                 random_state=SPLIT_SEED)


def recommend_selection(n_rows: int) -> str:
    return "nested" if n_rows <= NESTED_MAX_ROWS else "final_test"


def split_final_test(n: int):
    idx = np.arange(n)
    return train_test_split(idx, test_size=FINAL_TEST_SIZE,
                            random_state=SPLIT_SEED)


def evaluate(spec, X, y, validation: str, rows=None) -> Result:
    """Fit on training parts only; score on the held-out parts.

    ``spec.factory()`` must return an unfitted pipeline. ``rows`` are the
    positions of X/y in the cleaned data (for the predictions file).
    """
    y = np.asarray(y, dtype=float)
    rows = np.arange(len(y)) if rows is None else np.asarray(rows)
    if validation == "holdout":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                                  random_state=SPLIT_SEED)
        model = spec.factory().fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        return _make(spec.key, spec.name, y[te], pred, rows[te],
                     [float(r2_score(y[te], pred))])
    pred = np.empty(len(y))
    folds = []
    for tr, te in splitter(validation, len(y)).split(X):
        model = spec.factory().fit(X.iloc[tr], y[tr])
        pred[te] = model.predict(X.iloc[te])
        folds.append(float(r2_score(y[te], pred[te])))
    return _make(spec.key, spec.name, y, pred, rows, folds)


def evaluate_on_test(spec, X, y, dev, test) -> Result:
    y = np.asarray(y, dtype=float)
    model = spec.factory().fit(X.iloc[dev], y[dev])
    pred = model.predict(X.iloc[test])
    return _make(spec.key, spec.name, y[test], pred, test,
                 note=f"{spec.name}, trained on {len(dev)} rows, tested "
                      f"once on {len(test)} unseen rows")


def nested_cv(specs, X, y, validation: str, progress=None) -> Result:
    """Nested cross-validation estimate of 'compare, then pick the best'."""
    y = np.asarray(y, dtype=float)
    pred = np.empty(len(y))
    folds, winners = [], []
    for fold, (tr, te) in enumerate(splitter("kfold5", len(y)).split(X), 1):
        best, best_score = None, -np.inf
        for spec in specs:
            try:
                r = evaluate(spec, X.iloc[tr], y[tr], validation)
            except Exception:  # noqa: BLE001
                continue
            if r.selection_score > best_score:
                best, best_score = spec, r.selection_score
        if best is None:
            raise RuntimeError("no model could be trained in an outer fold")
        winners.append(best.name)
        if progress:
            progress(fold, best.name)
        model = best.factory().fit(X.iloc[tr], y[tr])
        pred[te] = model.predict(X.iloc[te])
        folds.append(float(r2_score(y[te], pred[te])))
    counts = {w: winners.count(w) for w in dict.fromkeys(winners)}
    note = "; ".join(f"{w} chosen in {c}/{len(winners)} folds"
                     for w, c in counts.items())
    return _make("nested", "Selected model (nested CV estimate)", y, pred,
                 np.arange(len(y)), folds, note)


# --------------------------------------------------------------------------- #
# Reference: always predicting the average
# --------------------------------------------------------------------------- #

BASELINE_NAME = "Reference: always predict the average"


class _Spec:
    def __init__(self, key, name, factory):
        self.key, self.name, self.factory = key, name, factory


BASELINE_SPEC = _Spec("baseline", BASELINE_NAME,
                      lambda: DummyRegressor(strategy="mean"))


def baseline_like(result: Result, X, y, method: str, validation: str,
                  dev=None, test=None) -> Result:
    """The same evaluation as ``result``, for the predict-the-average rule
    (its average is learned from the training rows of each split)."""
    if method == "final_test":
        r = evaluate_on_test(BASELINE_SPEC, X, y, dev, test)
    elif method == "nested":
        r = evaluate(BASELINE_SPEC, X, y, "kfold5")
    else:
        r = evaluate(BASELINE_SPEC, X, y, validation)
    return replace(r, note="")


# --------------------------------------------------------------------------- #
# Which columns matter
# --------------------------------------------------------------------------- #

MAX_EVAL_ROWS = 500
N_REPEATS = 5


def _perm(model, X_eval, y_eval, seed):
    if len(y_eval) > MAX_EVAL_ROWS:
        idx = np.random.default_rng(seed).choice(len(y_eval), MAX_EVAL_ROWS,
                                                 replace=False)
        X_eval, y_eval = X_eval.iloc[idx], y_eval[idx]
    return permutation_importance(model, X_eval, y_eval, scoring="r2",
                                  n_repeats=N_REPEATS,
                                  random_state=seed).importances


def importance(spec, X, y, validation: str, dev=None, test=None,
               seed: int = 0) -> pd.DataFrame:
    """Drop in R² when a column is shuffled, on rows not used for training
    (Breiman, 2001). Computed for the original columns."""
    y = np.asarray(y, dtype=float)
    parts = []
    if dev is not None and test is not None:
        model = spec.factory().fit(X.iloc[dev], y[dev])
        parts.append(_perm(model, X.iloc[test], y[test], seed))
        how = "untouched final test set"
    elif validation == "holdout":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                                  random_state=SPLIT_SEED)
        model = spec.factory().fit(X.iloc[tr], y[tr])
        parts.append(_perm(model, X.iloc[te], y[te], seed))
        how = "hold-out test part"
    else:
        for i, (tr, te) in enumerate(splitter("kfold5", len(y)).split(X)):
            model = spec.factory().fit(X.iloc[tr], y[tr])
            parts.append(_perm(model, X.iloc[te], y[te], seed + i))
        how = "test part of each cross-validation fold"
    allv = np.concatenate(parts, axis=1)
    table = pd.DataFrame({"column": [str(c) for c in X.columns],
                          "importance": allv.mean(axis=1),
                          "std": allv.std(axis=1)}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
    table.attrs["measured_on"] = how
    return table


# --------------------------------------------------------------------------- #
# Plain-language interpretation
# --------------------------------------------------------------------------- #

def interpret_final(final: Result, baseline: Result, target: str) -> List[str]:
    m, b = final.metrics, baseline.metrics
    lines = []
    if m["r2"] <= 0:
        lines.append(
            f"The predictions are no better than always using the average "
            f"{target} (R² = {m['r2']:.3f}). The columns may not contain the "
            "information needed to predict it, or there are too few rows.")
        return lines
    lines.append(
        f"On rows it had not seen, the model explained {m['r2'] * 100:.0f}% "
        f"of the differences in {target} between rows (R² = {m['r2']:.3f}).")
    lines.append(
        f"Predictions were off by {fmt(m['mae'])} on average (MAE); half of "
        f"them were within {fmt(m['medae'])} of the actual value.")
    if b["rmse"] > 0:
        gain = 1 - m["rmse"] / b["rmse"]
        lines.append(
            f"The typical error (RMSE {fmt(m['rmse'])}) is {gain * 100:.0f}% "
            f"smaller than when always predicting the average "
            f"(RMSE {fmt(b['rmse'])}).")
    if "mape" in m:
        lines.append(f"In relative terms, predictions were off by "
                     f"{m['mape'] * 100:.1f}% on average (MAPE).")
    lines.append(f"The mean Hassanat similarity between predicted and actual "
                 f"values was {m['mhsp']:.1f}% (MHSP; 100% = perfect).")
    return lines


def interpret_comparison(results: List[Result]) -> str:
    if len(results) < 2:
        return ""
    a, b = results[0], results[1]
    diff = a.selection_score - b.selection_score
    spread = float(np.std(a.fold_scores)) if len(a.fold_scores) > 1 else None
    if diff < 0.0005:
        return (f"{a.name} and {b.name} had the same R²; {a.name} was "
                "selected because it is listed first.")
    if spread is None:
        return (f"{a.name} had the highest R²; with a single validation "
                "split the uncertainty of this ranking cannot be judged.")
    if diff < spread:
        return (f"{a.name} had the highest R², but its lead over {b.name} "
                f"({diff:.3f}) is smaller than the variation between test "
                f"folds ({spread:.3f}). The top models perform similarly.")
    return (f"{a.name} had the highest R², ahead of {b.name} by {diff:.3f}, "
            f"which is larger than the variation between test folds "
            f"({spread:.3f}).")
