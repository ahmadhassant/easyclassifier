"""Time-ordered evaluation, forecast measures and plain-language summaries.

Nothing is ever shuffled. The series is split once, by time:

    |---------------- development ----------------|---- final period ----|

* Comparison (rolling origin): at several origins near the end of the
  development part, every model is trained on the values before the origin
  and forecasts the next H values. Models are ranked by their errors there.
* Final evaluation: the selected model is evaluated in the same way at
  origins inside the final period, which played no part in the comparison.
  At each origin it is retrained on everything before it (as it would be in
  real use) and forecasts the next H values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..common.hassanat import mhsp
from ..common.output import fmt

METRICS = {
    "mase": "MASE",
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
    "mhsp": "MHSP",
}
LOWER_IS_BETTER = {"mase", "mae", "rmse", "smape"}
SELECTION_METRICS = {"mase": "MASE", "rmse": "RMSE", "mae": "MAE",
                     "mhsp": "MHSP"}

MAX_DEV_ORIGINS = 5
MIN_DEV_ORIGINS = 2
MAX_TEST_ORIGINS = 10
FINAL_SHARE = 0.2


@dataclass
class Plan:
    horizon: int
    window: int
    dev_end: int                       # first index of the final period
    dev_origins: List[int]
    test_origins: List[int]
    overlapping: bool = False          # comparison forecasts overlap (short series)


@dataclass
class Result:
    key: str
    name: str
    origins: List[int]
    actual: np.ndarray                 # (origins, H), NaN beyond the data
    forecast: np.ndarray
    scales: np.ndarray                 # MASE scale per origin
    metrics: Dict[str, float] = field(default_factory=dict)
    fold_scores: List[float] = field(default_factory=list)   # per origin
    note: str = ""


MIN_WINDOWS = 10      # training windows (past values -> next H) at the first origin
MIN_TRAIN = 20        # values before the first comparison origin, at least


def plan_split(n: int, horizon: int, window: int, season: int) -> Plan:
    """Final period: the last ~20% (whole horizons, at least one). Before it,
    up to five comparison origins, one horizon apart, whose forecasts end
    where the final period starts; each origin leaves enough history (ten
    training windows for the models on past values, one season for the
    seasonal rule). When the series is too short for origins a whole horizon
    apart, they are placed closer together, so the comparison forecasts
    overlap (still never reaching into the final period)."""
    n_test = int(np.clip(round(FINAL_SHARE * n / horizon), 1,
                         MAX_TEST_ORIGINS))
    dev_end = n - n_test * horizon
    min_train = max(window + horizon + MIN_WINDOWS - 1, season + horizon,
                    MIN_TRAIN)
    last = dev_end - horizon           # its forecasts end at the final period
    room = last - min_train            # space for further origins before it
    if room < MIN_DEV_ORIGINS - 1:
        raise ValueError(
            f"The series is too short for a forecast horizon of {horizon}: "
            f"an honest evaluation needs about "
            f"{n + MIN_DEV_ORIGINS - 1 - room} values, and {n} are "
            "available. Choose a shorter horizon.")
    k = min(MAX_DEV_ORIGINS, room // horizon + 1)
    if k >= MIN_DEV_ORIGINS:
        step, overlapping = horizon, False
    else:
        k = min(MAX_DEV_ORIGINS, room + 1)
        step, overlapping = max(1, room // (k - 1)), True
    dev = [last - (k - 1 - i) * step for i in range(k)]
    test = [dev_end + j * horizon for j in range(n_test)]
    return Plan(horizon, window, dev_end, dev, test, overlapping)


def mase_scale(history: np.ndarray, season: int) -> float:
    """Mean absolute change from one season (or one step) earlier."""
    m = season if season > 1 and len(history) > season else 1
    d = np.abs(history[m:] - history[:-m])
    s = float(np.mean(d)) if len(d) else 0.0
    return s if s > 0 else 1.0


def rolling(spec, make, y, origins, horizon, season) -> Result:
    actual = np.full((len(origins), horizon), np.nan)
    forecast = np.full((len(origins), horizon), np.nan)
    scales = np.empty(len(origins))
    for i, o in enumerate(origins):
        model = make().fit(y[:o])
        h = min(horizon, len(y) - o)
        forecast[i, :h] = model.predict(horizon)[:h]
        actual[i, :h] = y[o:o + h]
        scales[i] = mase_scale(y[:o], season)
    r = Result(spec.key, spec.name, list(origins), actual, forecast, scales)
    score(r)
    return r


def _values(actual, forecast, scales=None):
    mask = ~np.isnan(actual)
    a, f = actual[mask], forecast[mask]
    s = None if scales is None else np.broadcast_to(
        scales[:, None], actual.shape)[mask]
    return a, f, s


def compute(a, f, s) -> Dict[str, float]:
    e = a - f
    denom = np.abs(a) + np.abs(f)
    smape = np.where(denom > 0, 2 * np.abs(e) / np.where(denom > 0, denom, 1),
                     0.0)
    return {
        "mase": float(np.mean(np.abs(e) / s)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "smape": float(100 * np.mean(smape)),
        "mhsp": mhsp(a, f),
    }


def score(r: Result) -> None:
    r.metrics = compute(*_values(r.actual, r.forecast, r.scales))
    r.fold_scores = [compute(*_values(r.actual[i:i + 1], r.forecast[i:i + 1],
                                      r.scales[i:i + 1]))
                     for i in range(len(r.origins))]


def selection_value(r: Result, metric: str) -> float:
    """Higher is better, whatever the metric."""
    v = r.metrics[metric]
    return -v if metric in LOWER_IS_BETTER else v


def error_by_step(results: List[Result]) -> np.ndarray:
    """Root mean squared error at each step ahead, pooled over results."""
    e = np.concatenate([r.actual - r.forecast for r in results], axis=0)
    return np.sqrt(np.nanmean(e ** 2, axis=0))


Z80 = 1.2816


def interval_coverage(dev: Result, test: Result) -> float:
    """Share of final-period values inside the 80% band built from the
    comparison errors of the same model."""
    sd = error_by_step([dev])
    lo = test.forecast - Z80 * sd
    hi = test.forecast + Z80 * sd
    a = test.actual
    mask = ~np.isnan(a)
    return float(np.mean((a[mask] >= lo[mask]) & (a[mask] <= hi[mask])))


# --------------------------------------------------------------------------- #
# Plain language
# --------------------------------------------------------------------------- #

def interpret(final: Result, naive: Result, seasonal: Result,
              unit_label: str, horizon: int, season: int,
              coverage: float) -> List[str]:
    m = final.metrics
    rule = ("repeating the value one season earlier" if season > 1
            else "repeating the previous value")
    steps = f"{horizon} {unit_label} step{'s' if horizon > 1 else ''}"
    lines = [f"Forecasting {steps} ahead in the final period (not used to "
             f"choose the model), the forecasts were off by {fmt(m['mae'])} "
             f"on average (MAE; RMSE {fmt(m['rmse'])}).",
             f"MASE = {m['mase']:.3f}: "
             + (f"the errors were smaller than those of simply {rule}"
                if m["mase"] < 1 else
                f"the errors were not smaller than those of simply {rule}")
             + " within the known history (MASE below 1 means better than "
             "that rule).",
             f"The mean Hassanat similarity between forecasts and actual "
             f"values was {m['mhsp']:.2f}% (MHSP; 100% = perfect).",
             f"sMAPE = {m['smape']:.2f}% (average error relative to the size "
             "of the values)."]
    refs = [r for r in (seasonal, naive) if r is not None]
    if refs:
        best_ref = min(refs, key=lambda r: r.metrics["mae"])
        if best_ref.metrics["mae"] > 0:
            gain = 1 - m["mae"] / best_ref.metrics["mae"]
            lines.append(
                f"Compared with the {best_ref.name.split(' (')[0].lower()} "
                f"rule, the average error is {abs(gain) * 100:.0f}% "
                + ("smaller." if gain >= 0 else "LARGER - the simple rule "
                   "forecast better in the final period."))
    lines.append(f"The 80% uncertainty band (from the comparison errors) "
                 f"contained {coverage * 100:.0f}% of the actual values in "
                 "the final period" + (" - about as intended." if
                                       0.65 <= coverage <= 0.95 else
                                       (" - wider than needed." if coverage >
                                        0.95 else " - the band is too narrow; "
                                        "treat it with caution.")))
    return lines


def interpret_comparison(results: List[Result], metric: str) -> str:
    if len(results) < 2:
        return ""
    a, b = results[0], results[1]
    label = SELECTION_METRICS[metric]
    va, vb = a.metrics[metric], b.metrics[metric]
    folds = [f[metric] for f in a.fold_scores]
    spread = float(np.std(folds)) if len(folds) > 1 else None
    diff = abs(va - vb)
    if diff < 1e-9:
        return (f"{a.name} and {b.name} had the same {label}; {a.name} was "
                "selected because it is listed first.")
    if spread is not None and diff < spread:
        return (f"{a.name} had the best {label} ({va:.3f}), but its lead over "
                f"{b.name} ({vb:.3f}) is smaller than the variation between "
                f"forecast origins ({spread:.3f}). The top models perform "
                "similarly.")
    return (f"{a.name} had the best {label} ({va:.3f}), clearly ahead of "
            f"{b.name} ({vb:.3f}).")
