"""Tests for figures, colour themes, file formats and interpretations."""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from easyclassifier import figures as figs
from easyclassifier import preprocessing as pp
from easyclassifier.diagnostics import (
    interpret_comparison,
    interpret_learning_curve,
    learning_curve_data,
)
from easyclassifier.evaluation import evaluate
from easyclassifier.importance import honest_importance
from easyclassifier.latex_report import build_tex
from easyclassifier.models import ClassifierSpec


def _data(n_classes=2, n=90, seed=0, missing=True):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n),
                      "city": rng.choice(["NY", "LA", "SF"], n)})
    y = np.digitize(X["a"] + 0.5 * rng.normal(size=n),
                    np.quantile(X["a"], np.linspace(0, 1, n_classes + 1)
                                [1:-1]))
    if missing:
        X.loc[[1, 5, 9], "b"] = np.nan
    names = [f"class {i}" for i in range(n_classes)]
    return pp.cast_categoricals(X), y, names


def _results(X, y, names, validation="kfold5"):
    cfg = pp.PrepConfig()
    specs = [
        ClassifierSpec("lr", "Logistic Regression", lambda: pp.build_pipeline(
            X, cfg, LogisticRegression(max_iter=500), True), ""),
        ClassifierSpec("dt", "Decision Tree", lambda: pp.build_pipeline(
            X, cfg, DecisionTreeClassifier(random_state=0), False), ""),
    ]
    res = [evaluate(s, X, y, names, validation, ["accuracy"]) for s in specs]
    return sorted(res, key=lambda r: -r.selection_score), specs


def test_default_figure_sets():
    assert figs.default_figures(False, False) == [
        "class_distribution", "comparison", "confusion_matrix", "roc_curve",
        "feature_importance"]
    both = figs.default_figures(True, True)
    assert "pr_curve" in both and "missing_values" in both
    assert both == [k for k in figs.FIGURES if k in both]   # report order
    assert set(figs.STANDARD_FIGURES + list(figs.WHEN_NEEDED)
               + figs.OPTIONAL_FIGURES) == set(figs.FIGURES)


def test_fold_scores_are_kept():
    X, y, names = _data()
    res, _ = _results(X, y, names)
    assert all(len(r.fold_scores) == 5 for r in res)
    res_h, _ = _results(X, y, names, "holdout")
    assert all(len(r.fold_scores) == 1 for r in res_h)


@pytest.mark.parametrize("theme", list(figs.THEMES))
@pytest.mark.parametrize("n_classes", [2, 3])
def test_every_figure_in_every_theme(tmp_path, theme, n_classes):
    X, y, names = _data(n_classes)
    res, specs = _results(X, y, names)
    imp = honest_importance(specs[0], X, y, "kfold5")
    curve = learning_curve_data(specs[0], X, y)
    fm = figs.FigureMaker(str(tmp_path), theme=theme, dpi=40,
                          class_names=names)
    fm.class_distribution(y)
    fm.missing_values(X)
    fm.correlation(X, prefer=list(imp["column"]))
    fm.comparison(res, res[0].classifier_name, 0.8, "Final score")
    fm.confusion(res[0])
    fm.roc(res[0])
    fm.pr(res[0])
    fm.importance(imp)
    fm.columns_by_class(X, y, list(imp["column"]))
    fm.learning_curve(curve)
    assert set(fm.files) == set(figs.FIGURES)
    for paths in fm.files.values():
        assert os.path.getsize(paths["png"]) > 1000


def test_vector_formats(tmp_path):
    X, y, names = _data()
    for fmt in ("png+pdf", "png+svg"):
        fm = figs.FigureMaker(str(tmp_path / fmt), formats=fmt, dpi=40,
                              class_names=names)
        paths = fm.class_distribution(y)
        ext = fmt.split("+")[1]
        assert set(paths) == {"png", ext}
        with open(paths[ext], "rb") as fh:
            head = fh.read(100)
        assert head.startswith(b"%PDF") if ext == "pdf" else b"<svg" in head \
            or b"<?xml" in head


def test_figures_that_do_not_apply_are_skipped(tmp_path):
    X, y, names = _data(missing=False)
    fm = figs.FigureMaker(str(tmp_path), dpi=40, class_names=names)
    assert fm.missing_values(X) is None                  # nothing missing
    assert fm.correlation(X[["a", "city"]]) is None       # one numeric col
    res, _ = _results(X, y, names)
    res[0].y_proba = None
    assert fm.roc(res[0]) is None                         # no probabilities


def test_comparison_without_fold_spread(tmp_path):
    X, y, names = _data(n=40)
    res, _ = _results(X, y, names, "loo")
    assert all(r.fold_scores == [] for r in res)
    fm = figs.FigureMaker(str(tmp_path), dpi=40, class_names=names)
    assert fm.comparison(res, res[0].classifier_name)


class _R:
    def __init__(self, name, score, folds):
        self.classifier_name, self.selection_score = name, score
        self.fold_scores = folds


def test_interpret_comparison():
    clear = interpret_comparison([_R("A", 0.90, [0.89, 0.90, 0.91]),
                                  _R("B", 0.70, [0.7])])
    assert "larger than the variation" in clear
    close = interpret_comparison([_R("A", 0.80, [0.6, 0.8, 1.0]),
                                  _R("B", 0.78, [0.8])])
    assert "not clear-cut" in close
    tie = interpret_comparison([_R("A", 0.8, [0.8]), _R("B", 0.8, [0.8]),
                                _R("C", 0.5, [0.5])])
    assert "A and B had the same" in tie and "listed first" in tie
    assert interpret_comparison([_R("A", 0.8, [0.8])]) == ""


def test_interpret_learning_curve():
    base = dict(sizes=np.array([10, 20, 30]), train_mean=np.array(
        [1.0, 0.95, 0.9]), train_std=np.zeros(3), val_std=np.zeros(3))
    rising = interpret_learning_curve({**base, "val_mean": np.array(
        [0.6, 0.7, 0.8])})
    flat = interpret_learning_curve({**base, "val_mean": np.array(
        [0.6, 0.85, 0.855])})
    assert "more data would probably improve" in rising
    assert "levelled off" in flat


def test_report_uses_vector_figures_and_explains_them():
    from test_report import _context
    ctx = _context()
    ctx.figures = {"confusion_matrix": {"png": "figures/confusion_matrix.png",
                                        "pdf": "figures/confusion_matrix.pdf"},
                   "roc_curve": {"png": "figures/roc_curve.png"}}
    ctx.comparison_text = "A and B had the same balanced accuracy."
    doc = build_tex(ctx)
    assert "figures/confusion_matrix.pdf" in doc          # vector preferred
    assert "figures/confusion_matrix.png" not in doc
    assert "figures/roc_curve.png" in doc
    assert "the diagonal holds the correct predictions" in doc.lower()
