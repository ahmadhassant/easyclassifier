"""Regression figures, drawn with EasyClassifier's themes (colour-blind safe
by default, 300 dpi PNG)."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from easyclassifier.figures import FigureMaker  # noqa: E402

from .evaluation import fmt  # noqa: E402

FIGURES = {
    "target_distribution": "Distribution of the predicted value",
    "missing_values": "Missing-value map",
    "comparison": "Model comparison",
    "predicted_vs_actual": "Predicted versus actual values",
    "residuals": "Prediction errors (residuals)",
    "feature_importance": "Which columns matter",
}

CAPTIONS = {
    "target_distribution": "How often each range of the predicted value "
                           "occurs. A long tail or isolated extreme values "
                           "are harder to predict.",
    "missing_values": "Empty cells in the data as loaded. Left: each dark "
                      "mark is an empty cell. Right: the share of empty "
                      "cells in each column.",
    "comparison": "R² of each model on rows it was not trained on. Dots are "
                  "individual test folds, the vertical bar is the overall "
                  "score. The dashed line at 0 is the level of always "
                  "predicting the average. The diamond, if shown, is the "
                  "final score of the selected model on data not used to "
                  "choose it.",
    "predicted_vs_actual": "Each dot is one row: its actual value "
                           "(horizontal) and the value predicted for it "
                           "without having seen it (vertical). Perfect "
                           "predictions lie on the diagonal line.",
    "residuals": "Prediction errors (actual minus predicted). Left: errors "
                 "should scatter evenly around zero at every level; a "
                 "funnel or curve shows where the model is less reliable. "
                 "Right: how large the errors usually are.",
    "feature_importance": "Permutation importance: how much R² drops when "
                          "a column is shuffled (mean and spread over "
                          "repeated shuffles), measured on rows not used "
                          "for training. When two columns carry almost the "
                          "same information, their importance can be "
                          "shared or unstable (a wide spread).",
}

STANDARD_FIGURES = ["target_distribution", "comparison",
                    "predicted_vs_actual", "residuals", "feature_importance"]


def default_figures(has_missing: bool) -> List[str]:
    keys = list(STANDARD_FIGURES)
    if has_missing:
        keys.append("missing_values")
    return [k for k in FIGURES if k in keys]


class RegressionFigures(FigureMaker):
    def __init__(self, out_dir: str, target: str, **kw):
        super().__init__(out_dir, class_names=[], **kw)
        self.target = target

    def target_distribution(self, y) -> Dict[str, str]:
        y = np.asarray(y, dtype=float)
        with self._style():
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.hist(y, bins=min(40, max(10, int(np.sqrt(len(y))))),
                    **self._bar_style(0))
            ax.axvline(np.mean(y), color="black", ls="--", lw=1,
                       label=f"average {fmt(np.mean(y))}")
            ax.set_xlabel(self.target)
            ax.set_ylabel("Number of rows")
            ax.set_title(f"Distribution of {self.target}")
            ax.legend(frameon=False)
            fig.tight_layout()
            return self._save(fig, "target_distribution")

    def comparison(self, results: List, selected: str,
                   final_score: Optional[float] = None,
                   final_label: str = "") -> Dict[str, str]:
        names = [r.name for r in results][::-1]
        lows = [min([r.selection_score] + r.fold_scores) for r in results]
        with self._style():
            fig, ax = plt.subplots(
                figsize=(7, max(2.6, 0.5 * len(names) + 1.3)))
            for row, r in enumerate(results[::-1]):
                col = self.color(0) if r.name == selected else "#8c8c8c"
                if r.fold_scores:
                    jitter = np.linspace(-0.12, 0.12, len(r.fold_scores))
                    ax.scatter(r.fold_scores, row + jitter, s=18, color=col,
                               alpha=0.55, zorder=2, edgecolors="none")
                ax.scatter(r.selection_score, row, marker="|", s=380,
                           linewidths=2.6, color=col, zorder=3)
            if final_score is not None and selected in names:
                ax.scatter(final_score, names.index(selected), marker="D",
                           s=70, color=self.color(1), edgecolors="black",
                           zorder=4)
            ax.axvline(0, color="black", ls="--", lw=0.9)
            left = min(lows + [0.0, final_score or 0.0])
            ax.set_xlim(max(left - 0.05, -1.0), 1.0)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            for lab in ax.get_yticklabels():
                if lab.get_text() == selected:
                    lab.set_fontweight("bold")
            ax.set_xlabel("R² on unseen rows (1 = perfect, 0 = average only)")
            ax.set_title("Comparison of models")
            handles = [
                Line2D([], [], marker="o", ls="", color="#8c8c8c",
                       alpha=0.6, label="one test fold"),
                Line2D([], [], marker="|", ls="", color="#8c8c8c",
                       markersize=14, markeredgewidth=2.6, label="overall")]
            if final_score is not None:
                handles.append(Line2D([], [], marker="D", ls="",
                                      color=self.color(1),
                                      markeredgecolor="black",
                                      label=final_label or "Final score"))
            fig.tight_layout(rect=(0, 0.1, 1, 1))
            fig.legend(handles=handles, loc="lower center", ncol=3,
                       frameon=False)
            return self._save(fig, "comparison")

    def predicted_vs_actual(self, result) -> Dict[str, str]:
        a, p = result.y_true, result.y_pred
        lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
        pad = 0.04 * (hi - lo or 1)
        with self._style():
            fig, ax = plt.subplots(figsize=(5.4, 5.0))
            ax.scatter(a, p, s=16, alpha=0.6, color=self.color(0),
                       edgecolors="none")
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black",
                    lw=1, ls="--")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")
            ax.set_xlabel(f"Actual {self.target}")
            ax.set_ylabel(f"Predicted {self.target}")
            ax.set_title("Predicted versus actual")
            m = result.metrics
            ax.text(0.03, 0.97, f"R² = {m['r2']:.3f}\nRMSE = {fmt(m['rmse'])}"
                    f"\nMAE = {fmt(m['mae'])}", transform=ax.transAxes,
                    va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="#cccccc"))
            fig.tight_layout()
            return self._save(fig, "predicted_vs_actual")

    def residuals(self, result) -> Dict[str, str]:
        res = result.y_true - result.y_pred
        with self._style():
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(9, 3.8), gridspec_kw=dict(width_ratios=[3, 2]))
            ax1.scatter(result.y_pred, res, s=14, alpha=0.6,
                        color=self.color(0), edgecolors="none")
            ax1.axhline(0, color="black", lw=1, ls="--")
            ax1.set_xlabel(f"Predicted {self.target}")
            ax1.set_ylabel("Error (actual − predicted)")
            ax1.set_title("Errors across predictions")
            ax2.hist(res, bins=min(40, max(10, int(np.sqrt(len(res))))),
                     orientation="horizontal", **self._bar_style(0))
            ax2.axhline(0, color="black", lw=1, ls="--")
            ax2.set_xlabel("Number of rows")
            ax2.set_title("Size of errors")
            ax2.sharey(ax1)
            fig.tight_layout()
            return self._save(fig, "residuals")

    def importance(self, table: pd.DataFrame, top: int = 20) -> Dict[str, str]:
        t = table.head(top).iloc[::-1]
        with self._style():
            fig, ax = plt.subplots(
                figsize=(6.5, max(2.6, 0.36 * len(t) + 1.2)))
            ax.barh([str(c) for c in t["column"]], t["importance"],
                    xerr=t["std"], ecolor="#555555", capsize=2,
                    **self._bar_style(0))
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Drop in R² when the column is shuffled")
            ax.set_title("Which columns matter")
            ax.grid(axis="y", visible=False)
            fig.tight_layout()
            return self._save(fig, "feature_importance")
