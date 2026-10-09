"""Tests for the time-series core: data checks, honesty in time, measures,
complete runs."""

import numpy as np
import pandas as pd
import pytest

from easyresearch.common.hassanat import hassanat_similarity, mhsp
from easyresearch.timeseries import ForecastSettings, run_forecast
from easyresearch.timeseries import data as td
from easyresearch.timeseries import evaluation as ev
from easyresearch.timeseries.models import build_registry, window_length


# --------------------------------------------------------------------- MHSP

def test_mhsp_values():
    assert mhsp([5, 7], [5, 7]) == pytest.approx(100.0)
    # (1+2)/(1+4) = 0.6 ; (1+10)/(1+10) = 1  -> 80%
    assert mhsp([2, 10], [4, 10]) == pytest.approx(80.0)
    # signed form: min=-2 -> (1-2+2)/(1+3+2) = 1/6
    assert hassanat_similarity(-2, 3) == pytest.approx(1 / 6)
    assert hassanat_similarity(3, -2) == pytest.approx(1 / 6)


# --------------------------------------------------------------- the series

def _frame(dates, values, name="Sales"):
    return pd.DataFrame({"date": dates, name: values})


def test_monthly_gaps_use_only_past_values():
    dates = pd.date_range("2000-01-01", periods=60, freq="MS")
    vals = np.arange(60, dtype=float)
    df = _frame(dates.strftime("%Y-%m-%d"), vals).drop(index=[10, 11])
    s = td.prepare_series(df, "Sales")
    assert s.unit == "M" and s.season == 12 and s.filled == 2
    assert len(s.values) == 60
    assert s.values[10] == 9 and s.values[11] == 9     # last known value


def test_mid_month_dates_and_year_numbers():
    d = pd.date_range("2001-01-15", periods=48, freq="MS") + pd.Timedelta(
        days=14)
    assert td.prepare_series(_frame(d, np.random.rand(48)), "Sales").unit == "M"
    years = pd.DataFrame({"year": range(1950, 2000),
                          "Catch": np.random.rand(50)})
    s = td.prepare_series(years, "Catch")
    assert s.unit == "Y" and s.time_column == "year"


def test_business_days_and_row_order():
    d = pd.bdate_range("2020-01-01", periods=80)
    assert td.prepare_series(_frame(d, np.random.rand(80)), "Sales").unit == "B"
    s = td.prepare_series(pd.DataFrame({"x": np.random.rand(40)}), "x")
    assert s.unit is None and s.season == 1


def test_irregular_dates_and_short_series_refused():
    d = pd.to_datetime(["2020-01-01", "2020-01-04", "2020-03-01"] * 12)
    d = d + pd.to_timedelta(np.arange(36) * 97, unit="D")
    with pytest.raises(ValueError, match="evenly spaced"):
        td.prepare_series(_frame(d, np.random.rand(36)), "Sales")
    with pytest.raises(ValueError, match="at least"):
        td.prepare_series(_frame(pd.date_range("2000", periods=12,
                                               freq="MS"), range(12)),
                          "Sales")


# ------------------------------------------------------------ honest timing

def test_split_never_lets_comparison_see_the_final_period():
    plan = ev.plan_split(300, 12, 24, 12)
    assert all(o + 12 <= plan.dev_end for o in plan.dev_origins)
    assert plan.test_origins[0] == plan.dev_end
    assert plan.dev_origins == sorted(plan.dev_origins)
    with pytest.raises(ValueError, match="too short"):
        ev.plan_split(60, 12, 24, 12)


@pytest.mark.parametrize("key", ["ridge", "random_forest", "knn", "ets",
                                 "theta", "mlp", "lstm", "nbeats"])
def test_forecasts_do_not_depend_on_future_values(key):
    rng = np.random.default_rng(0)
    t = np.arange(240)
    y = 10 + 0.05 * t + np.sin(2 * np.pi * t / 12) + rng.normal(0, .2, 240)
    spec = build_registry()[key]
    L = window_length(150, 12, 6)
    origin = 200
    a = spec.make(L, 6, 12).fit(y[:origin]).predict(6)
    y2 = y.copy()
    y2[origin:] = 999.0                     # change the future completely
    b = spec.make(L, 6, 12).fit(y2[:origin]).predict(6)
    np.testing.assert_allclose(a, b)
    r1 = ev.rolling(spec, lambda: spec.make(L, 6, 12), y, [origin], 6, 12)
    np.testing.assert_allclose(r1.forecast[0], a)


def test_measures_match_definitions():
    actual = np.array([[10., 12., np.nan]])
    fc = np.array([[11., 9., 5.]])
    r = ev.Result("x", "x", [0], actual, fc, np.array([2.0]))
    ev.score(r)
    assert r.metrics["mae"] == pytest.approx(2.0)
    assert r.metrics["rmse"] == pytest.approx(np.sqrt(5))
    assert r.metrics["mase"] == pytest.approx(1.0)
    assert r.metrics["mhsp"] == pytest.approx(mhsp([10, 12], [11, 9]))


# ---------------------------------------------------------------- full runs

def test_demo_forecast_run(tmp_path):
    r = run_forecast(td.load_demo(), "CO2_ppm",
                     ForecastSettings(models=["ets", "ridge", "knn"]),
                     str(tmp_path), td.DEMO_NAME, "co2", td.DEMO_CITATION,
                     progress=lambda m: None)
    H = r["horizon"]
    assert H == 12 and len(r["future"]) == H
    assert np.all(r["lower"] < r["future"]) and np.all(r["future"] < r["upper"])
    assert r["final"].metrics["mase"] < 1          # beats seasonal naive
    assert {x.key for x in r["results"]} >= {"naive", "seasonal_naive"}
    out = r["out"]
    saved = pd.read_csv(f"{out}/final_period_forecasts.csv")
    assert np.mean(np.abs(saved.error)) == pytest.approx(
        r["final"].metrics["mae"])
    assert saved["steps_ahead"].max() == H
    first_test = pd.Timestamp(saved["date" if "date" in saved else
                                    saved.columns[1]].min())
    assert first_test == r["series"].index[r["plan"].dev_end]
    fc = pd.read_csv(f"{out}/forecast.csv")
    assert len(fc) == H and pd.Timestamp(fc.iloc[0, 0]) == pd.Timestamp(
        "2002-01-01")
    for f in ("report.tex", "trained_model.pkl", "citations.txt", "log.txt",
              "results.xlsx", "figures/forecast.png",
              "figures/final_period.png", "figures/decomposition.png"):
        assert (tmp_path / out / f).exists(), f
    tex = open(f"{out}/report.tex", encoding="utf-8").read()
    assert "MHSP" in tex and "Journal of American Science" not in tex
    assert "math12223623" in tex                 # MHSP paper
    assert "ETCEA57049.2022.10009844" in tex     # KNN used the distance
    assert "arXiv:1409.0923" in tex and "big.2018.0175" not in tex
    assert "math12223623" in open(f"{out}/citations.txt",
                                  encoding="utf-8").read()


def test_mhsp_can_choose_the_model(tmp_path):
    r = run_forecast(td.load_demo(), "CO2_ppm",
                     ForecastSettings(models=["ets", "ridge"],
                                      selection_metric="mhsp", figures=False),
                     str(tmp_path), "co2", "co2", progress=lambda m: None)
    scores = [x.metrics["mhsp"] for x in r["results"]]
    assert scores == sorted(scores, reverse=True)
    tex = open(f"{r['out']}/report.tex", encoding="utf-8").read()
    assert "math12223623" in tex and "10009844" not in tex  # no KNN


def test_cli_forecast(tmp_path):
    from easyresearch.cli import main
    assert main(["forecast", "--demo", "--models", "ridge", "--no-figures",
                 "--out", str(tmp_path)]) == 0
