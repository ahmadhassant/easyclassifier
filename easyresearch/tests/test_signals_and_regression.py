"""Tests for signal classification and the regression core."""

import numpy as np
import pandas as pd
import pytest

from easyresearch.timeseries import signal_data as sd
from easyresearch.timeseries.classify import (SignalSettings,
                                              run_signal_classification)
from easyresearch.timeseries.signal_models import (DeepClassifier, Rocket,
                                                   features, minmax, znorm)


def test_gaps_filled_within_each_recording_only():
    df = sd.load_demo().head(30).copy()
    df.loc[0, "t5"] = np.nan
    r = sd.prepare(df, "label")
    expected = (df.loc[0, "t4"] + df.loc[0, "t6"]) / 2
    assert r.X.loc[0, "t5"] == pytest.approx(expected)
    assert any("neighbouring" in n for n in r.notes)


def test_ucr_tsv_format(tmp_path):
    rows = [[c] + list(np.sin(np.linspace(0, 6, 40)) * (c + 1))
            for c in (1, 2) for _ in range(5)]
    p = tmp_path / "Toy_TRAIN.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", header=False, index=False)
    df = sd.load_table(str(p))
    r = sd.prepare(df, sd.suggest_label(df))
    assert r.X.shape == (10, 40)


def test_per_recording_scaling_is_independent_of_other_recordings():
    X = np.random.default_rng(0).normal(size=(6, 50))
    for f in (znorm, minmax, features):
        np.testing.assert_allclose(f(X)[:2], f(X[:2]))


def test_rocket_and_deep_are_deterministic():
    df = sd.load_demo().head(60)
    r = sd.prepare(df, "label")
    X, y = r.X, r.labels.to_numpy()
    a = Rocket(n_kernels=500).fit(X, y).predict_proba(X)
    b = Rocket(n_kernels=500).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)
    m1 = DeepClassifier("lstm").fit(X, y).predict_proba(X)
    m2 = DeepClassifier("lstm").fit(X, y).predict_proba(X)
    np.testing.assert_allclose(m1, m2, atol=1e-6)


def test_signal_run_nested_without_deep(tmp_path):
    from sklearn.metrics import balanced_accuracy_score
    r = run_signal_classification(
        sd.load_demo(), "label",
        SignalSettings(models=["knn", "features_rf"]), str(tmp_path),
        "demo", "demo", progress=lambda m: None)
    assert r["method"] == "nested"
    pred = pd.read_csv(f"{r['out']}/predictions.csv")
    assert len(pred) == 300
    assert balanced_accuracy_score(pred.actual, pred.predicted) == \
        pytest.approx(r["shown"].selection_score)
    assert r["reference"].selection_score == pytest.approx(1 / 3)


def test_signal_run_with_deep_uses_final_test(tmp_path):
    r = run_signal_classification(
        sd.load_demo().head(120), "label",
        SignalSettings(models=["knn", "lstm"], figures=False), str(tmp_path),
        "demo", "demo", progress=lambda m: None)
    assert r["method"] == "final_test"
    assert r["dev_rows"] + r["test_rows"] == 120


def test_regression_reports_mhsp(tmp_path):
    from easyresearch.regression import Settings, run_analysis
    from easyresearch.regression.data import load_demo
    r = run_analysis(load_demo(), "Progression",
                     Settings(models=["linear_regression"], figures=False),
                     str(tmp_path), "demo", "demo")
    m = r["shown"].metrics
    assert 0 < m["mhsp"] < 100 and m["r2"] > .4
    tex = open(f"{r['out']}/report.tex", encoding="utf-8").read()
    assert "math12223623" in tex and "10009844" not in tex


def test_figure_choices_in_the_cores(tmp_path):
    """Theme, file format and the figures to draw are honoured; an unknown
    choice is refused before any model is trained."""
    import os
    from easyresearch.regression import Settings, run_analysis
    from easyresearch.regression.data import load_demo
    r = run_analysis(load_demo(), "Progression",
                     Settings(models=["linear_regression", "ridge"], figure_theme="greyscale",
                              figure_format="png+svg",
                              figure_keys=["residuals", "comparison"]),
                     str(tmp_path / "r"), "demo", "demo", progress=lambda m: None)
    assert list(r["figures"]) == ["comparison", "residuals"]       # report order
    assert all(os.path.exists(os.path.join(r["out"], f))
               for files in r["figures"].values() for f in files.values())
    assert all(set(files) == {"png", "svg"} for files in r["figures"].values())
    seen = []
    for settings, message in [
            (Settings(models=["ridge"], figure_theme="neon"), "Unknown figure theme"),
            (Settings(models=["ridge"], figure_format="gif"), "Unknown figure format"),
            (Settings(models=["ridge"], figure_keys=["pie"]), "Unknown figure")]:
        with pytest.raises(ValueError, match=message):
            run_analysis(load_demo(), "Progression", settings, str(tmp_path / "x"),
                         "demo", "demo", progress=seen.append)
    assert seen == [] and not (tmp_path / "x").exists()
    s = run_signal_classification(sd.load_demo().head(90), "label",
                                  SignalSettings(models=["knn"], figure_keys=[]),
                                  str(tmp_path / "s"), "s", "s", progress=lambda m: None)
    assert s["figures"] == {}


def test_command_line_figure_choices(tmp_path):
    from easyresearch import cli
    assert cli.main(["regress", "--demo", "--models", "linear_regression,ridge",
                     "--figures", "residuals", "--figure-format", "png+pdf",
                     "--figure-theme", "greyscale", "--out", str(tmp_path)]) == 0
    out = next((tmp_path / "Results").iterdir())
    assert sorted(p.name for p in (out / "figures").iterdir()) == \
        ["residuals.pdf", "residuals.png"]
    assert cli.main(["regress", "--demo", "--figures", "pie",
                     "--out", str(tmp_path / "x")]) == 1
