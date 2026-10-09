"""Evaluation integrity: nothing learned or chosen may depend on rows that are
used to score the result.

* preprocessing statistics come from training rows only (regression);
* the untouched final test rows play no part in the model comparison: if
  they are replaced by noise, every comparison score stays exactly the same
  (regression and signal classification);
* forecasting: the comparison never uses the final period (see also
  test_timeseries.py), checked here by changing the final period.

Run with:  python -m pytest easyresearch/tests
"""

import numpy as np
import pandas as pd
import pytest
from easyclassifier import preprocessing as pp
from easyclassifier.evaluation import split_final_test
from easyclassifier.logbook import LogBook
from sklearn.impute import SimpleImputer

from easyresearch.regression import Settings, run_analysis
from easyresearch.regression import analysis as ra
from easyresearch.regression import evaluation as rev
from easyresearch.regression.models import build_registry as regressors
from easyresearch.timeseries import ForecastSettings, run_forecast
from easyresearch.timeseries import data as td
from easyresearch.timeseries import evaluation as tev
from easyresearch.timeseries import signal_data as sd
from easyresearch.timeseries.classify import (SignalSettings,
                                              run_signal_classification)


def _regression_data(n, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(5, 2, size=n)
    group = rng.choice(["north", "south", "east"], n)
    y = 3 * x1 + 0.5 * x2 + np.where(group == "north", 2.0, 0.0) \
        + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "region": group, "Yield": y})
    df.loc[rng.choice(n, n // 20, replace=False), "x2"] = np.nan
    return df


def _comparison(run):
    return {r.key: r.metrics for r in run["results"]}


def test_regression_preprocessing_learns_from_training_rows_only():
    df = _regression_data(300)
    prep = ra.prepare(df, "Yield", LogBook())
    X = prep.X
    tr = np.arange(200)
    X_extreme = X.copy()
    X_extreme.loc[X_extreme.index[200:], "x2"] += 1000      # test rows only
    spec = ra._pipeline_spec(regressors()["ridge"], X_extreme, prep.cfg)
    model = spec.factory().fit(X_extreme.iloc[tr], prep.y[tr])
    num = model.named_steps["prep"].named_transformers_["num"]
    imputer = num.named_steps["impute"]
    expected = SimpleImputer(strategy=imputer.strategy).fit(
        X_extreme.iloc[tr][["x1", "x2"]]).statistics_
    assert imputer.statistics_ == pytest.approx(expected)
    full = SimpleImputer(strategy=imputer.strategy).fit(
        X_extreme[["x1", "x2"]]).statistics_
    assert imputer.statistics_[1] != pytest.approx(full[1])   # would differ if leaked


@pytest.mark.parametrize("validation", ["kfold5", "holdout"])
def test_regression_final_test_rows_never_reach_the_comparison(tmp_path,
                                                                validation):
    df = _regression_data(2400, seed=1)        # > 2,000 rows: final test set
    settings = Settings(models=["linear_regression", "decision_tree", "knn"],
                        validation=validation, figures=False)
    first = run_analysis(df, "Yield", settings, str(tmp_path / "a"),
                         "a.csv", "a", progress=lambda m: None)
    assert first["method"] == "final_test"
    _, test = rev.split_final_test(first["rows_used"])
    noisy = df.copy()
    rng = np.random.default_rng(99)
    rows = noisy.index[test]
    noisy.loc[rows, "x1"] = rng.normal(0, 50, len(rows))
    noisy.loc[rows, "Yield"] = rng.normal(0, 50, len(rows))
    second = run_analysis(noisy, "Yield", settings, str(tmp_path / "b"),
                          "b.csv", "b", progress=lambda m: None)
    assert _comparison(second) == _comparison(first)
    assert second["best"].key == first["best"].key
    assert second["final"].metrics["r2"] < first["final"].metrics["r2"] - 0.5


def _recordings(n=120, length=60, seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    rows = []
    for i in range(n):
        cls = ("sine", "square", "saw")[i % 3]
        base = {"sine": np.sin(2 * np.pi * 3 * t),
                "square": np.sign(np.sin(2 * np.pi * 3 * t)),
                "saw": 2 * ((3 * t) % 1) - 1}[cls]
        rows.append([cls] + list(base + rng.normal(0, 0.3, length)))
    return pd.DataFrame(rows, columns=["label"] + [f"t{i}" for i in
                                                  range(1, length + 1)])


def test_signal_final_test_recordings_never_reach_the_comparison(tmp_path):
    df = _recordings()
    settings = SignalSettings(models=["knn", "features_rf"],
                              selection="final_test", figures=False)
    first = run_signal_classification(df, "label", settings,
                                      str(tmp_path / "a"), "a", "a",
                                      progress=lambda m: None)
    recs = sd.prepare(df, "label")
    y, _, _ = pp.encode_target(recs.labels)
    _, test = split_final_test(recs.X, y)
    noisy = df.copy()
    signal = [c for c in df.columns if c != "label"]
    noisy.loc[noisy.index[test], signal] = np.random.default_rng(5).normal(
        0, 3, (len(test), len(signal)))
    second = run_signal_classification(noisy, "label", settings,
                                       str(tmp_path / "b"), "b", "b",
                                       progress=lambda m: None)
    before = {r.classifier_key: r.metrics for r in first["results"]}
    after = {r.classifier_key: r.metrics for r in second["results"]}
    assert after == before
    assert second["final"].metrics["balanced_accuracy"] < \
        first["final"].metrics["balanced_accuracy"]


def test_forecast_comparison_ignores_the_final_period(tmp_path):
    df = td.load_demo()
    settings = ForecastSettings(models=["ridge", "theta"], horizon=6,
                                figures=False)
    first = run_forecast(df, "CO2_ppm", settings, str(tmp_path / "a"), "a",
                         "a", progress=lambda m: None)
    dev_end = first["plan"].dev_end
    series = td.prepare_series(df, "CO2_ppm")
    changed = df.copy()
    final_months = set(series.index[dev_end:].strftime("%Y-%m"))
    mask = changed["month"].isin(final_months)
    changed.loc[mask, "CO2_ppm"] = changed.loc[mask, "CO2_ppm"] + 25.0
    second = run_forecast(changed, "CO2_ppm", settings, str(tmp_path / "b"),
                          "b", "b", progress=lambda m: None)
    assert second["plan"].dev_end == dev_end
    assert {r.key: r.metrics for r in second["results"]} == \
        {r.key: r.metrics for r in first["results"]}
    assert all(o >= dev_end for o in second["plan"].test_origins)
    assert max(second["plan"].dev_origins) + 6 <= dev_end
    assert second["final"].metrics["mae"] != pytest.approx(
        first["final"].metrics["mae"])


def test_forecast_split_leaves_whole_horizons_for_the_final_period():
    from easyresearch.timeseries.forecast import horizon_fits
    from easyresearch.timeseries.models import window_length
    checked = 0
    for n in range(40, 400, 7):
        for season in (1, 4, 7, 12):
            for h in (1, 3, 6, 12, 24):
                if not horizon_fits(n, h, season):
                    continue
                plan = tev.plan_split(n, h, window_length(int(0.6 * n), season, h),
                                      season)
                assert plan.dev_end + len(plan.test_origins) * h == n
                assert all(o + h <= plan.dev_end for o in plan.dev_origins)
                checked += 1
    assert checked > 100


@pytest.mark.parametrize("months", [60, 84, 150])
def test_automatic_horizon_always_fits_the_series(tmp_path, months):
    """The recommended horizon is never one the evaluation would refuse."""
    from easyresearch.timeseries.forecast import choose_horizon
    t = np.arange(months)
    df = pd.DataFrame({"month": pd.date_range("2000-01-01", periods=months,
                                              freq="MS").strftime("%Y-%m"),
                       "sales": 100 + t + 10 * np.sin(2 * np.pi * t / 12)})
    series = td.prepare_series(df, "sales")
    H = choose_horizon(series)
    usual = td.default_horizon(td.prepare_series(df, "sales"))
    assert 1 <= H <= usual
    assert (H < usual) == any("shortened" in note for note in series.notes)
    run = run_forecast(df, "sales", ForecastSettings(models=["ridge"],
                                                     figures=False),
                       str(tmp_path), "s", "s", progress=lambda m: None)
    assert run["horizon"] == H


def test_short_series_rules():
    """30 values (the smallest series accepted) can always be forecast at
    least one step ahead; a shorter series is refused with the length needed."""
    from easyresearch.timeseries.forecast import choose_horizon, horizon_fits
    for season in (1, 4, 7, 12):
        assert horizon_fits(td.MIN_POINTS, 1, season)
    df = pd.DataFrame({"v": np.arange(30.0) + np.sin(np.arange(30))})
    assert choose_horizon(td.prepare_series(df, "v")) >= 1
    with pytest.raises(ValueError, match=r"at least 30"):
        td.prepare_series(pd.DataFrame({"v": np.arange(25.0)}), "v")
    short = td.Series(np.arange(25.0), pd.RangeIndex(25), "v", None,
                      "in row order", 1, None, [])
    with pytest.raises(ValueError, match=r"needs about 30"):
        choose_horizon(short)


def test_short_series_comparison_overlaps_but_never_reaches_the_final_period():
    plan = tev.plan_split(30, 3, 4, 1)
    assert plan.overlapping
    assert all(o + 3 <= plan.dev_end for o in plan.dev_origins)
    assert len(plan.dev_origins) >= 2
    run_df = pd.DataFrame({"v": 20 + np.arange(30.0) * 0.5 + np.sin(np.arange(30))})
    import tempfile
    with tempfile.TemporaryDirectory() as out:
        run = run_forecast(run_df, "v", ForecastSettings(models=["ridge", "theta"], figures=False),
                           out, "s", "s", progress=lambda m: None)
    assert any("overlap" in n for n in run["notes"])


def test_command_line_runs_record_settings_and_versions(tmp_path):
    import json
    from easyresearch import cli
    assert cli.main(["regress", "--demo", "--models", "ridge",
                     "--no-figures", "--out", str(tmp_path)]) == 0
    out = next((tmp_path / "Results").iterdir())
    record = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    assert record["task"] == "regression"
    assert record["column"] == "Progression"
    assert record["used"]["final_evaluation"] == "cross_validation"
    assert "easyresearch:" in (out / "versions.txt").read_text(encoding="utf-8")
