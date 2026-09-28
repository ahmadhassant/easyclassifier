"""Tests for the LaTeX/PDF report and the column-importance measure."""

import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from easyclassifier import preprocessing as pp
from easyclassifier.evaluation import evaluate
from easyclassifier.importance import honest_importance
from easyclassifier.latex_report import (
    ReportContext,
    build_tex,
    tex,
    write_and_compile,
)
from easyclassifier.models import ClassifierSpec


def test_tex_escapes_special_characters():
    assert tex("R&D_spend 50% $#{x}~^") == (
        r"R\&D\_spend 50\% \$\#\{x\}\textasciitilde{}\textasciicircum{}")
    assert tex("a<b>c") == r"a\textless{}b\textgreater{}c"
    assert tex("Café") == "Café"          # Latin-1 accents are kept
    assert tex("مدينة") == "?????"         # not typesettable by pdfLaTeX


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n),
                      "city": rng.choice(["A", "B"], n)})
    y = (signal > 0).astype(int)
    return pp.cast_categoricals(X), y


def _spec(X):
    cfg = pp.PrepConfig()
    return ClassifierSpec(
        "dt", "Decision Tree",
        lambda: pp.build_pipeline(X, cfg, DecisionTreeClassifier(
            random_state=0), scale=False), "decision_tree")


@pytest.mark.parametrize("validation", ["kfold5", "holdout"])
def test_importance_finds_the_informative_column(validation):
    X, y = _data()
    table = honest_importance(_spec(X), X, y, validation)
    assert list(table.columns) == ["column", "importance", "std"]
    assert table.iloc[0]["column"] == "signal"
    assert table.iloc[0]["importance"] > 0.3
    assert set(table["column"]) == {"signal", "noise", "city"}  # original
    others = table[table["column"] != "signal"]["importance"]
    assert (others.abs() < 0.1).all()


def test_importance_on_final_test_set():
    X, y = _data()
    idx = np.arange(len(y))
    table = honest_importance(_spec(X), X, y, "kfold5",
                              dev=idx[:160], test=idx[160:])
    assert table.iloc[0]["column"] == "signal"
    assert table.attrs["measured_on"] == "untouched final test set"


def _context(tmp_figs=None, importance=None):
    X, y = _data(60)
    res = evaluate(_spec(X), X, y, ["no", "yes"], "kfold5",
                   ["accuracy", "f1"])
    return ReportContext(
        software_citation="Hassanat, A. (2026). EasyClassifier.",
        version="0.0-test", dataset="R&D data_50%.csv",
        rows_loaded=62, cols_loaded=4, target="Outcome_%",
        target_grouping="", class_counts={"no": 30, "yes": 30},
        rows_used=60, predictor_columns=list(X.columns),
        left_out={"ID": "different in every row (ID or name)"},
        data_notes=["2 duplicate row(s) were removed."],
        impute="median", encoding="auto", scaled={},
        classifiers=["Decision Tree"], knn_text="", hassanat_used=False,
        validation="kfold5", selection=None, n_dev=0, n_test=0,
        best_name="Decision Tree", final=None, results=[res],
        metrics=["accuracy", "f1"], figures=tmp_figs or {},
        importance=importance)


def test_build_tex_contains_methods_and_references():
    doc = build_tex(_context())
    assert doc.startswith(r"\documentclass")
    assert doc.rstrip().endswith(r"\end{document}")
    assert "Methods (ready to adapt for your paper)" in doc
    assert r"R\&D data\_50\%.csv" in doc
    assert "without hyperparameter tuning" in doc
    assert "learned from the training data only" in doc
    assert r"\bibitem{sklearn}" in doc
    assert r"\bibitem{hassanat2014}" not in doc   # KNN/Hassanat not used


@pytest.mark.skipif(not any(shutil.which(e) for e in
                            ("tectonic", "pdflatex", "xelatex", "lualatex")),
                    reason="no LaTeX installed")
def test_report_compiles_to_pdf(tmp_path):
    X, y = _data(60)
    imp = honest_importance(_spec(X), X, y, "kfold5")
    tex_path, pdf_path, msg = write_and_compile(_context(importance=imp),
                                                str(tmp_path))
    assert pdf_path is not None, msg
    assert (tmp_path / "report.pdf").stat().st_size > 10_000
    assert not (tmp_path / "report.aux").exists()      # cleaned up
