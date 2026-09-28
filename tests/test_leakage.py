"""Tests proving that preprocessing is learned from training data only.

Run with:  python -m pytest tests
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from easyclassifier import preprocessing as pp
from easyclassifier.distances import make_knn
from easyclassifier.evaluation import evaluate, fit_final_model
from easyclassifier.models import ClassifierSpec


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(10, 2, n),
        "b": rng.normal(-5, 3, n),
        "city": rng.choice(["NY", "LA", "SF"], n),
    })
    X.loc[rng.choice(n, 20, replace=False), "a"] = np.nan
    X.loc[rng.choice(n, 10, replace=False), "city"] = np.nan
    y = (X["b"] + rng.normal(0, 1, n) > -5).astype(int).to_numpy()
    return pp.cast_categoricals(X), y


def test_imputer_and_scaler_learn_from_training_rows_only():
    X, y = _data()
    # Make the test part extreme, so any leakage would shift the statistics.
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3,
                                           random_state=1, stratify=y)
    X_te = X_te.copy()
    X_te["a"] = X_te["a"] + 1000

    cfg = pp.PrepConfig(impute="median", scale=True)
    pipe = pp.build_pipeline(X_tr, cfg, DecisionTreeClassifier(), scale=True)
    pipe.fit(X_tr, y_tr)

    num = pipe.named_steps["prep"].named_transformers_["num"]
    imputer, scaler = num.named_steps["impute"], num.named_steps["scale"]
    train_median_a = X_tr["a"].median()
    assert imputer.statistics_[0] == pytest.approx(train_median_a)
    filled = X_tr[["a", "b"]].fillna({"a": train_median_a})
    assert scaler.mean_ == pytest.approx(filled.mean().to_numpy())
    # And the extreme test data changed nothing:
    pipe.predict(X_te)
    assert imputer.statistics_[0] == pytest.approx(train_median_a)


def test_each_cv_fold_learns_its_own_statistics():
    X, y = _data()
    cfg = pp.PrepConfig(impute="mean", scale=True)
    medians = []
    for tr, _ in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        pipe = pp.build_pipeline(X, cfg, DecisionTreeClassifier(), scale=True)
        pipe.fit(X.iloc[tr], y[tr])
        stat = pipe.named_steps["prep"].named_transformers_["num"] \
            .named_steps["impute"].statistics_[0]
        assert stat == pytest.approx(X.iloc[tr]["a"].mean())
        medians.append(stat)
    assert len(set(np.round(medians, 8))) > 1  # folds really differ
    assert all(m != pytest.approx(X["a"].mean()) for m in medians)


def test_unseen_category_in_test_data_does_not_crash():
    X, y = _data()
    cfg = pp.PrepConfig(encoding="onehot")
    pipe = pp.build_pipeline(X, cfg, DecisionTreeClassifier(), scale=False)
    pipe.fit(X, y)
    new = X.head(3).copy()
    new["city"] = ["Paris", "Amman", np.nan]
    assert len(pipe.predict(new)) == 3


@pytest.mark.parametrize("validation", ["holdout", "kfold5", "stratified"])
@pytest.mark.parametrize("distance", ["hassanat", "euclidean"])
def test_evaluate_runs_with_pipeline(validation, distance):
    X, y = _data()
    cfg = pp.PrepConfig()
    spec = ClassifierSpec(
        "knn", "KNN",
        lambda: pp.build_pipeline(X, cfg, make_knn(distance, 5),
                                  scale=distance != "hassanat"),
        "knn")
    res = evaluate(spec, X, y, ["0", "1"], validation, ["accuracy", "roc_auc"])
    assert 0.5 < res.metrics["accuracy"] <= 1.0


def test_saved_model_accepts_new_raw_data():
    X, y = _data()
    cfg = pp.PrepConfig()
    spec = ClassifierSpec(
        "dt", "DT",
        lambda: pp.build_pipeline(X, cfg, DecisionTreeClassifier(), False),
        "decision_tree")
    model = fit_final_model(spec, X, y)
    raw = pd.DataFrame({"a": [np.nan, 12.0], "b": [-3.0, -9.0],
                        "city": ["LA", None]})
    assert len(model.predict(pp.cast_categoricals(raw))) == 2


def test_scaling_rules():
    from easyclassifier.recommend import scaling_for
    assert scaling_for("knn", "hassanat") == "minmax"
    assert scaling_for("knn", "euclidean") == "standard"
    assert scaling_for("svm") == "standard"
    assert scaling_for("random_forest") is None
    assert scaling_for("knn", "hassanat", choice=False) is None
    assert scaling_for("random_forest", choice=True,
                       method="minmax") == "minmax"
    X, y = _data()
    pipe = pp.build_pipeline(X, pp.PrepConfig(), DecisionTreeClassifier(),
                             "minmax").fit(X, y)
    Xt = pipe.named_steps["prep"].transform(X)
    num = Xt[:, :2]
    assert num.min() >= 0 and num.max() <= 1
