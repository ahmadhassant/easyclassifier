"""Forecasting figures in EasyClassifier's themes (colour-blind safe by
default, 300 dpi)."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from easyclassifier.figures import FigureMaker  # noqa: E402

from .evaluation import SELECTION_METRICS, Z80  # noqa: E402

FIGURES = {
    "forecast": "The series and the forecast",
    "final_period": "Forecasts in the final period",
    "comparison": "Model comparison",
    "error_by_step": "Error by steps ahead",
    "decomposition": "Trend and seasonal pattern",
}

CAPTIONS = {
    "forecast": "Recent history, the final period used for the honest "
                "evaluation (shaded) and the forecast beyond the last value "
                "with its 80% uncertainty band, made by the selected model "
                "retrained on all values.",
    "final_period": "The final period, which was not used to choose the "
                    "model. Each coloured segment is one forecast made from "
                    "the values before it; the black line is what actually "
                    "happened.",
    "comparison": "Error of each model at the comparison origins (dots: one "
                  "forecast origin each; bar: overall). The two reference "
                  "rules are drawn in grey italics. The diamond, if shown, is "
                  "the selected model's score in the final period.",
    "error_by_step": "How the typical error (RMSE) grows with the number of "
                     "steps ahead, for the selected model and the seasonal "
                     "(or plain) naive rule, in the comparison and the final "
                     "period.",
    "decomposition": "The series split into a slowly changing trend, a "
                     "repeating seasonal pattern and the remaining "
                     "irregular part (STL decomposition; Cleveland et al., "
                     "1990).",
}


class ForecastFigures(FigureMaker):
    def __init__(self, out_dir: str, name: str, index, **kw):
        super().__init__(out_dir, class_names=[], **kw)
        self.name = name
        self.index = index
        self.is_time = isinstance(index, pd.DatetimeIndex)

    def _x(self, positions):
        return self.index[np.asarray(positions)]

    def _future_x(self, n_future, unit):
        if not self.is_time:
            return np.arange(len(self.index), len(self.index) + n_future)
        if unit == "B":
            return pd.bdate_range(self.index[-1], periods=n_future + 1)[1:]
        freq = pd.infer_freq(self.index[-10:]) or {
            "h": "h", "D": "D", "W": "W", "M": "MS", "Q": "QS",
            "Y": "YS"}.get(unit)
        return pd.date_range(self.index[-1], periods=n_future + 1,
                             freq=freq)[1:]

    def forecast(self, y, dev_end, future, lo, hi, future_x,
                 model_name) -> Dict[str, str]:
        with self._style():
            fig, ax = plt.subplots(figsize=(9, 4.2))
            # Recent history only, so the forecast and its band stay visible.
            start = max(0, min(dev_end, len(y) - 8 * len(future)) - len(future))
            x = self._x(np.arange(start, len(y)))
            ax.plot(x, y[start:], color="black", lw=1.1, label="actual values")
            ax.axvspan(self._x([dev_end])[0], x[-1], color=self.color(4),
                       alpha=0.15, lw=0, label="final period (evaluation)")
            ax.fill_between(future_x, lo, hi, color=self.color(0), alpha=0.25,
                            lw=0, label="80% band")
            ax.plot(future_x, future, color=self.color(0), lw=2,
                    label=f"forecast ({model_name})")
            ax.set_ylabel(self.name)
            ax.set_title(f"{self.name}: history and forecast")
            ax.legend(loc="upper left", frameon=False, fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            return self._save(fig, "forecast")

    def final_period(self, y, final, dev_end, horizon) -> Dict[str, str]:
        start = max(0, dev_end - 3 * horizon)
        with self._style():
            fig, ax = plt.subplots(figsize=(9, 4))
            pos = np.arange(start, len(y))
            ax.plot(self._x(pos), y[start:], color="black", lw=1.2,
                    label="actual values")
            for i, o in enumerate(final.origins):
                f = final.forecast[i]
                k = int(np.sum(~np.isnan(f)))
                ax.plot(self._x(np.arange(o, o + k)), f[:k], lw=2,
                        color=self.color(i % 2), alpha=0.9)
                ax.axvline(self._x([o])[0], color="#bbbbbb", lw=0.6)
            ax.axvline(self._x([dev_end])[0], color="black", ls="--", lw=1)
            ax.set_ylabel(self.name)
            ax.set_title(f"Forecasts of {final.name} in the final period")
            handles = [Line2D([], [], color="black", label="actual values"),
                       Line2D([], [], color=self.color(0), lw=2,
                              label="forecast from each origin"),
                       Line2D([], [], color="black", ls="--",
                              label="start of the final period")]
            ax.legend(handles=handles, loc="upper left", frameon=False,
                      fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            return self._save(fig, "final_period")

    def comparison(self, results: List, selected: str, metric: str,
                   final_value: Optional[float], references: set
                   ) -> Dict[str, str]:
        names = [r.name for r in results][::-1]
        label = SELECTION_METRICS[metric]
        with self._style():
            fig, ax = plt.subplots(
                figsize=(7.5, max(2.8, 0.42 * len(names) + 1.4)))
            for row, r in enumerate(results[::-1]):
                col = (self.color(0) if r.name == selected else
                       "#b0b0b0" if r.key in references else "#707070")
                folds = [f[metric] for f in r.fold_scores]
                if folds:
                    jitter = np.linspace(-0.12, 0.12, len(folds))
                    ax.scatter(folds, row + jitter, s=16, color=col,
                               alpha=0.55, edgecolors="none", zorder=2)
                ax.scatter(r.metrics[metric], row, marker="|", s=330,
                           linewidths=2.6, color=col, zorder=3)
            if final_value is not None and selected in names:
                ax.scatter(final_value, names.index(selected), marker="D",
                           s=65, color=self.color(1), edgecolors="black",
                           zorder=4)
            if metric == "mase":
                ax.axvline(1, color="black", ls="--", lw=0.9)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            for lab, r in zip(ax.get_yticklabels(), results[::-1]):
                if r.name == selected:
                    lab.set_fontweight("bold")
                if r.key in references:
                    lab.set_style("italic")
                    lab.set_color("#707070")
            better = "lower is better" if metric != "mhsp" else \
                "higher is better"
            ax.set_xlabel(f"{label} at the comparison origins ({better})")
            ax.set_title("Comparison of forecasting models")
            handles = [Line2D([], [], marker="o", ls="", color="#8c8c8c",
                              alpha=0.6, label="one forecast origin"),
                       Line2D([], [], marker="|", ls="", color="#8c8c8c",
                              markersize=14, markeredgewidth=2.6,
                              label="overall")]
            if final_value is not None:
                handles.append(Line2D([], [], marker="D", ls="",
                                      color=self.color(1),
                                      markeredgecolor="black",
                                      label="final period"))
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            fig.legend(handles=handles, loc="lower center", ncol=3,
                       frameon=False)
            return self._save(fig, "comparison")

    def error_by_step(self, curves: Dict[str, np.ndarray]) -> Dict[str, str]:
        with self._style():
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for i, (label, v) in enumerate(curves.items()):
                steps = np.arange(1, len(v) + 1)
                ax.plot(steps, v, marker="o", ms=3.5, color=self.color(i),
                        ls=self.linestyle(i) if "rule" not in label else "--",
                        label=label)
            ax.set_xlabel("Steps ahead")
            ax.set_ylabel(f"RMSE ({self.name})")
            ax.set_title("Typical error by steps ahead")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            return self._save(fig, "error_by_step")

    def decomposition(self, y, season) -> Optional[Dict[str, str]]:
        if season <= 1 or len(y) < 2 * season + 1:
            return None
        from statsmodels.tsa.seasonal import STL
        res = STL(np.asarray(y, float), period=season, robust=True).fit()
        x = self._x(np.arange(len(y)))
        with self._style():
            fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
            for ax, (title, v) in zip(axes, [
                    (self.name, y), ("Trend", res.trend),
                    ("Seasonal pattern", res.seasonal),
                    ("Irregular part", res.resid)]):
                ax.plot(x, v, color=self.color(0) if title != self.name
                        else "black", lw=1)
                ax.set_ylabel(title, fontsize=8)
            axes[-1].axhline(0, color="black", lw=0.6)
            axes[0].set_title("Trend and seasonal pattern")
            fig.autofmt_xdate()
            fig.tight_layout()
            return self._save(fig, "decomposition")


def band(forecast, sd):
    return forecast - Z80 * sd, forecast + Z80 * sd
