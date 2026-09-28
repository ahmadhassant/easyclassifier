"""Tests for honest scoring of the selected (best) classifier.

Run with:  python -m pytest tests
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from easyclassifier.evaluation import (
    evaluate,
    evaluate_on_test,
    nested_cv,
    recommend_selection,
    split_final_test,
)
from easyclassifier.models import ClassifierSpec


class Spy(BaseEstimator, ClassifierMixin):
    """Records which rows it was trained on and which it predicted."""

    events = []

    def fit(self, X, y):
        Spy.events.append(("fit", frozenset(X.index)))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        Spy.events.append(("predict", frozenset(X.index)))
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        Spy.events.append(("predict", frozenset(X.index)))
        p = np.zeros((len(X), len(self.classes_)))
        p[:, 0] = 1
        return p


class Guesser(BaseEstimator, ClassifierMixin):
    """A useless classifier: its answer is a pseudo-random function of each
    row's values (fixed by ``seed``), unrelated to the labels. Different rows
    get independent answers, so luck on one set of rows says nothing about
    another."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.w_ = np.random.default_rng(self.seed).normal(size=X.shape[1])
        return self

    def predict(self, X):
        h = np.sin(np.asarray(X, dtype=float) @ self.w_ * 1e4)
        return self.classes_[(h > 0).astype(int)]


def _data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.integers(0, 2, n)
    return X, y


def _assert_no_row_predicted_by_a_model_trained_on_it():
    last_fit = None
    for kind, rows in Spy.events:
        if kind == "fit":
            last_fit = rows
        else:
            assert last_fit is not None
            assert not (rows & last_fit), "a row was predicted by a model " \
                                          "that was trained on it"


@pytest.mark.parametrize("validation", ["holdout", "kfold5", "stratified"])
def test_final_test_rows_are_never_seen_during_comparison(validation):
    X, y = _data()
    dev, test = split_final_test(X, y)
    assert not set(dev) & set(test)
    assert len(test) == pytest.approx(0.2 * len(y), abs=1)
    # Class proportions are kept (stratified).
    assert abs(y[test].mean() - y.mean()) < 0.05

    Spy.events = []
    spec = ClassifierSpec("spy", "Spy", Spy, "")
    evaluate(spec, X.iloc[dev], y[dev], ["0", "1"], validation, [])
    seen = set().union(*(rows for _, rows in Spy.events))
    assert not seen & set(X.index[test]), "test rows leaked into comparison"
    _assert_no_row_predicted_by_a_model_trained_on_it()

    Spy.events = []
    evaluate_on_test(spec, X.iloc[dev], y[dev], X.iloc[test], y[test],
                     ["0", "1"], [])
    (k1, fit_rows), *preds = Spy.events
    assert k1 == "fit" and fit_rows == frozenset(X.index[dev])
    assert preds and all(k == "predict" and rows == frozenset(X.index[test])
                         for k, rows in preds)


def test_nested_cv_never_tests_on_training_rows():
    X, y = _data()
    Spy.events = []
    specs = [ClassifierSpec("s1", "Spy 1", Spy, ""),
             ClassifierSpec("s2", "Spy 2", Spy, "")]
    res = nested_cv(specs, X, y, ["0", "1"], "kfold5", ["accuracy"])
    _assert_no_row_predicted_by_a_model_trained_on_it()
    assert len(res.y_pred) == len(y)          # every row tested exactly once
    assert "folds" in res.note


# 20 classifiers whose answers are meaningless: any "winner" won purely by
# luck, so an honest estimate of it must sit near chance (0.5). Results are
# averaged over several random datasets, since a single one is noisy. Seeds
# are fixed, so the tests give the same result on every run.
GUESSERS = [ClassifierSpec(f"g{i}", f"Guesser {i}",
                           (lambda i=i: Guesser(seed=i)), "")
            for i in range(20)]


def _best_comparison_score(X, y):
    return max(evaluate(s, X, y, ["0", "1"], "kfold5", []).selection_score
               for s in GUESSERS)


def test_nested_cv_removes_selection_optimism():
    best, nested = [], []
    for seed in range(8):
        X, y = _data(n=200, seed=100 + seed)
        best.append(_best_comparison_score(X, y))
        nested.append(nested_cv(GUESSERS, X, y, ["0", "1"], "kfold5",
                                []).selection_score)
    best, nested = np.mean(best), np.mean(nested)
    assert best > 0.55                  # the lucky winner looks good...
    assert nested < best - 0.03         # ...nested CV removes the optimism
    assert abs(nested - 0.5) < 0.04     # and lands near chance


def test_final_test_set_removes_selection_optimism():
    best, final = [], []
    for seed in range(10):
        X, y = _data(n=500, seed=200 + seed)
        dev, test = split_final_test(X, y, seed=seed)
        scores = [evaluate(s, X.iloc[dev], y[dev], ["0", "1"], "kfold5",
                           []).selection_score for s in GUESSERS]
        winner = GUESSERS[int(np.argmax(scores))]
        best.append(max(scores))
        final.append(evaluate_on_test(winner, X.iloc[dev], y[dev],
                                      X.iloc[test], y[test], ["0", "1"],
                                      []).selection_score)
    best, final = np.mean(best), np.mean(final)
    assert best > 0.53                  # the lucky winner looks good...
    assert final < best - 0.02          # ...the untouched test set disagrees
    assert abs(final - 0.5) < 0.04      # and lands near chance


def test_recommendation_depends_on_data_size():
    assert recommend_selection(np.array([0, 1] * 1001)) == "final_test"
    assert recommend_selection(np.array([0, 1] * 1000)) == "nested"
    assert recommend_selection(np.array([0, 1] * 20)) == "nested"
    # Enough rows overall but a rare class -> nested.
    assert recommend_selection(np.array([0] * 2490 + [1] * 10)) == "nested"
