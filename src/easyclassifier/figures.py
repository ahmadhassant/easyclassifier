"""All figures, drawn in one of four colour themes.

* Every class keeps the same colour (and, in greyscale, the same hatch or
  line style) in every figure.
* Figures are saved as PNG at 300 dpi, the usual journal requirement, and
  optionally also as PDF or SVG (vector files that stay sharp at any size).
* A non-interactive matplotlib backend is used, so it works on servers.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    precision_recall_curve,
    roc_curve,
)


# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Theme:
    key: str
    name: str
    description: str
    palette: Sequence[str]
    cmap: str            # sequential (confusion matrices)
    cmap_div: str        # diverging (correlations, -1 .. +1)
    font_scale: float = 1.0
    linewidth: float = 1.8
    hatches: Sequence[str] = ("",)
    linestyles: Sequence[str] = ("-",)
    edge: str = "none"


THEMES: Dict[str, Theme] = {
    "colorblind": Theme(
        "colorblind", "Colour-blind safe",
        "Readable with the common forms of colour blindness (Okabe-Ito "
        "colours). Recommended for papers.",
        # Okabe & Ito (2008) palette
        ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
         "#D55E00", "#F0E442", "#000000"],
        cmap="cividis", cmap_div="RdBu_r"),
    "greyscale": Theme(
        "greyscale", "Greyscale",
        "Shades of grey with patterns and line styles, for printed journals "
        "and theses.",
        ["#1a1a1a", "#7f7f7f", "#bdbdbd", "#4d4d4d", "#a6a6a6", "#e0e0e0"],
        cmap="Greys", cmap_div="Greys",
        hatches=("", "//", "..", "xx", "\\\\", "oo"),
        linestyles=("-", "--", "-.", ":"), edge="black"),
    "high_contrast": Theme(
        "high_contrast", "High contrast",
        "Strong colours, thick lines and larger text, for slides and "
        "posters.",
        # Blue and orange first: the pair that stays distinct for readers
        # with red-green colour blindness.
        ["#0033CC", "#FF8C00", "#008000", "#8B00FF", "#E60000", "#00A6A6",
         "#000000"],
        cmap="viridis", cmap_div="coolwarm", font_scale=1.25,
        linewidth=3.0, edge="black"),
    "soft": Theme(
        "soft", "Soft",
        "Gentle pastel colours for reports and teaching material.",
        ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
         "#E5C494", "#B3B3B3"],
        cmap="PuBu", cmap_div="PiYG"),
}

DEFAULT_THEME = "colorblind"

FORMATS = {
    "png": "PNG images at 300 dpi",
    "png+pdf": "PNG + PDF (vector files, best for journals)",
    "png+svg": "PNG + SVG (vector files, for editing in Inkscape or "
               "Illustrator)",
}

# Figure key -> menu label (order = order in the report)
FIGURES = {
    "class_distribution": "Class distribution",
    "missing_values": "Missing-value map",
    "correlation": "Correlation matrix of numeric columns",
    "comparison": "Classifier comparison (with fold-to-fold spread)",
    "confusion_matrix": "Confusion matrix (counts and percentages)",
    "roc_curve": "ROC curves",
    "pr_curve": "Precision-recall curves",
    "feature_importance": "Feature importance (which columns matter)",
    "columns_by_class": "Most important columns, by class",
    "learning_curve": "Learning curve (would more data help?)",
}

# The figures most used in classification papers: always made by default.
STANDARD_FIGURES = ["class_distribution", "comparison", "confusion_matrix",
                    "roc_curve", "feature_importance"]
# Added by default only when the data call for them.
WHEN_NEEDED = {
    "pr_curve": "classes are imbalanced",
    "missing_values": "the data have empty cells",
}
# Optional extras, chosen in Advanced/Research mode.
OPTIONAL_FIGURES = ["correlation", "columns_by_class", "learning_curve"]


def default_figures(imbalanced: bool, has_missing: bool) -> List[str]:
    figs = list(STANDARD_FIGURES)
    if imbalanced:
        figs.append("pr_curve")
    if has_missing:
        figs.append("missing_values")
    return [k for k in FIGURES if k in figs]   # report order


# --------------------------------------------------------------------------- #
# Figure maker
# --------------------------------------------------------------------------- #

class FigureMaker:
    """Draws figures in a theme and saves them in the chosen formats.

    ``files`` maps each figure key to {"png": path, "pdf": path, ...}.
    """

    def __init__(self, out_dir: str, theme: str = DEFAULT_THEME,
                 formats: str = "png", dpi: int = 300,
                 class_names: Optional[List[str]] = None):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.theme = THEMES[theme]
        self.exts = formats.split("+")
        self.dpi = dpi
        self.class_names = list(class_names or [])
        self.files: Dict[str, Dict[str, str]] = {}

    # ---- styling ------------------------------------------------------ #

    def color(self, i: int) -> str:
        return self.theme.palette[i % len(self.theme.palette)]

    def hatch(self, i: int) -> str:
        return self.theme.hatches[i % len(self.theme.hatches)]

    def linestyle(self, i: int) -> str:
        return self.theme.linestyles[i % len(self.theme.linestyles)]

    @contextmanager
    def _style(self):
        s = self.theme.font_scale
        rc = {
            "font.size": 10 * s, "axes.titlesize": 11.5 * s,
            "axes.labelsize": 10.5 * s, "xtick.labelsize": 9 * s,
            "ytick.labelsize": 9 * s, "legend.fontsize": 9 * s,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
            "lines.linewidth": self.theme.linewidth,
            "hatch.linewidth": 0.8, "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
        with plt.rc_context(rc):
            yield

    def _save(self, fig, key: str) -> Dict[str, str]:
        paths = {}
        for ext in self.exts:
            p = os.path.join(self.out_dir, f"{key}.{ext}")
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
            paths[ext] = p
        plt.close(fig)
        self.files[key] = paths
        return paths

    def _bar_style(self, i: int) -> dict:
        return dict(color=self.color(i), hatch=self.hatch(i),
                    edgecolor=(self.theme.edge if self.theme.edge != "none"
                               else self.color(i)), linewidth=0.8)

    # ---- data description --------------------------------------------- #

    def class_distribution(self, y) -> Dict[str, str]:
        y = np.asarray(y)
        counts = [int((y == i).sum()) for i in range(len(self.class_names))]
        with self._style():
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            for i, (name, n) in enumerate(zip(self.class_names, counts)):
                ax.bar(i, n, **self._bar_style(i))
                ax.text(i, n, str(n), ha="center", va="bottom")
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(self.class_names, rotation=30, ha="right")
            ax.set_ylabel("Rows")
            ax.set_title("Rows in each class")
            ax.grid(axis="x", visible=False)
            return self._save(fig, "class_distribution")

    def missing_values(self, df: pd.DataFrame,
                       max_cols: int = 60) -> Optional[Dict[str, str]]:
        """Map of empty cells (rows x columns) and share missing per
        column. Returns None if nothing is missing."""
        miss = df.isna()
        if not miss.values.any():
            return None
        cols = list(miss.columns)
        if len(cols) > max_cols:  # prefer columns that have gaps
            with_gaps = [c for c in cols if miss[c].any()]
            cols = (with_gaps + [c for c in cols if c not in with_gaps])[
                :max_cols]
        m = miss[cols].to_numpy()
        share = m.mean(axis=0) * 100
        with self._style():
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(9, max(3.5, 0.22 * len(cols) + 1.5)),
                gridspec_kw={"width_ratios": [3, 1.3]}, sharey=True)
            ax1.imshow(m.T, aspect="auto", interpolation="nearest",
                       cmap=matplotlib.colors.ListedColormap(
                           ["#f2f2f2", self.color(0)]))
            ax1.set_yticks(range(len(cols)))
            ax1.set_yticklabels([str(c) for c in cols])
            ax1.set_xlabel("Row number")
            ax1.set_title("Empty cells (dark)")
            ax1.grid(False)
            ax2.barh(range(len(cols)), share, **self._bar_style(0))
            ax2.set_xlabel("% empty")
            ax2.set_xlim(0, max(5, share.max() * 1.15))
            ax2.set_title("Share empty")
            ax2.invert_yaxis()
            fig.tight_layout()
            return self._save(fig, "missing_values")

    def correlation(self, X: pd.DataFrame, max_cols: int = 25,
                    prefer: Optional[List[str]] = None
                    ) -> Optional[Dict[str, str]]:
        """Pearson correlations between numeric columns."""
        num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
               and X[c].nunique() > 1]
        if len(num) < 2:
            return None
        if len(num) > max_cols:
            order = [c for c in (prefer or []) if c in num]
            num = (order + [c for c in num if c not in order])[:max_cols]
        corr = X[num].corr().to_numpy()
        n = len(num)
        with self._style():
            size = max(4.5, 0.42 * n + 2)
            fig, ax = plt.subplots(figsize=(size + 1, size))
            im = ax.imshow(corr, cmap=self.theme.cmap_div, vmin=-1, vmax=1)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels([str(c) for c in num], rotation=45,
                               ha="right")
            ax.set_yticklabels([str(c) for c in num])
            ax.grid(False)
            if n <= 12:
                for i in range(n):
                    for j in range(n):
                        v = corr[i, j]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8 * self.theme.font_scale,
                                color="white" if abs(v) > 0.6 else "black")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Correlation")
            ax.set_title("Correlation between numeric columns")
            fig.tight_layout()
            return self._save(fig, "correlation")

    # ---- results -------------------------------------------------------- #

    def comparison(self, results: List, selected: str,
                   final_score: Optional[float] = None,
                   final_label: str = "") -> Dict[str, str]:
        """Balanced accuracy of each classifier: one dot per test fold and
        the overall score; the selected classifier is highlighted."""
        names = [r.classifier_name for r in results][::-1]
        with self._style():
            fig, ax = plt.subplots(
                figsize=(7, max(2.6, 0.5 * len(names) + 1.3)))
            for row, r in enumerate(results[::-1]):
                chosen = r.classifier_name == selected
                col = self.color(0) if chosen else "#8c8c8c"
                if r.fold_scores:
                    jitter = np.linspace(-0.12, 0.12, len(r.fold_scores))
                    ax.scatter(np.array(r.fold_scores) * 100, row + jitter,
                               s=18, color=col, alpha=0.55, zorder=2,
                               edgecolors="none")
                ax.scatter(r.selection_score * 100, row, marker="|", s=380,
                           linewidths=2.6, color=col, zorder=3)
            if final_score is not None:
                row = names.index(selected)
                ax.scatter(final_score * 100, row, marker="D", s=70,
                           color=self.color(1), edgecolors="black",
                           zorder=4, label=final_label or "Final score")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            for lab in ax.get_yticklabels():
                if lab.get_text() == selected:
                    lab.set_fontweight("bold")
            ax.set_xlabel("Balanced accuracy (%)")
            ax.set_title("Comparison of classifiers")
            from matplotlib.lines import Line2D
            handles = [
                Line2D([], [], marker="o", ls="", color="#8c8c8c",
                       alpha=0.6, label="one test fold"),
                Line2D([], [], marker="|", ls="", color="#8c8c8c",
                       markersize=14, markeredgewidth=2.6,
                       label="overall"),
            ]
            if final_score is not None:
                handles.append(Line2D([], [], marker="D", ls="",
                                      color=self.color(1),
                                      markeredgecolor="black",
                                      label=final_label or "Final score"))
            fig.tight_layout(rect=(0, 0.1, 1, 1))
            fig.legend(handles=handles, loc="lower center", ncol=3,
                       frameon=False)
            return self._save(fig, "comparison")

    def confusion(self, result) -> Dict[str, str]:
        """Counts and row percentages in one matrix. Colours follow the
        percentages, so small classes are as readable as large ones."""
        counts = result.confusion.astype(int)
        names = result.class_names
        rows = counts.sum(axis=1, keepdims=True)
        pct = np.divide(counts * 100.0, rows,
                        out=np.zeros(counts.shape, dtype=float),
                        where=rows > 0)
        n = len(names)
        with self._style():
            size = max(4.2, 0.75 * n + 2.4)
            fig, ax = plt.subplots(figsize=(size + 0.8, size))
            im = ax.imshow(pct, cmap=self.theme.cmap, vmin=0, vmax=100)
            for i in range(n):
                for j in range(n):
                    v = pct[i, j]
                    txt = f"{counts[i, j]}\n({v:.0f}%)"
                    rgba = im.cmap(im.norm(v))
                    light = (0.299 * rgba[0] + 0.587 * rgba[1]
                             + 0.114 * rgba[2]) > 0.55
                    ax.text(j, i, txt, ha="center", va="center",
                            color="black" if light else "white",
                            fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_yticklabels(names)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.grid(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("% of the true class")
            ax.set_title("Confusion matrix: rows (and % of each true class)")
            fig.tight_layout()
            return self._save(fig, "confusion_matrix")

    def _curves(self, result, kind: str) -> Optional[Dict[str, str]]:
        if result.y_proba is None:
            return None
        y, P, names = result.y_true, result.y_proba, result.class_names
        binary = len(names) == 2
        classes = [1] if binary else list(range(len(names)))
        with self._style():
            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            for c in classes:
                yt = (y == c).astype(int)
                if yt.min() == yt.max():
                    continue
                label = names[c] if not binary else f"{names[1]} (positive)"
                # One curve (two classes): always a solid line, so it cannot
                # be mistaken for the dotted random-guessing reference.
                style = dict(color=self.color(c),
                             ls="-" if binary else self.linestyle(c))
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(yt, P[:, c])
                    ax.plot(fpr, tpr, label=f"{label}  AUC={auc(fpr, tpr):.3f}",
                            **style)
                else:
                    prec, rec, _ = precision_recall_curve(yt, P[:, c])
                    ax.plot(rec, prec,
                            label=f"{label}  AUC={auc(rec, prec):.3f}",
                            **style)
                    ax.axhline(yt.mean(), color=self.color(c), lw=0.8,
                               ls=":", alpha=0.8)
            if kind == "roc":
                ax.plot([0, 1], [0, 1], ls=":", color="#9e9e9e", lw=1.2,
                        label="random guessing")
                ax.set_xlabel("False positive rate (1 - specificity)")
                ax.set_ylabel("True positive rate (sensitivity)")
                ax.set_title("ROC curve" + ("" if binary else
                                            "s (each class vs the rest)"))
            else:
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-recall curve" + (
                    "" if binary else "s (each class vs the rest)"))
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.02)
            ax.legend(loc="lower right" if kind == "roc" else "lower left")
            fig.tight_layout()
            return self._save(fig, "roc_curve" if kind == "roc"
                              else "pr_curve")

    def roc(self, result):
        return self._curves(result, "roc")

    def pr(self, result):
        return self._curves(result, "pr")

    # ---- which columns matter ----------------------------------------- #

    def importance(self, table: pd.DataFrame, top: int = 20
                   ) -> Dict[str, str]:
        t = table.head(top).iloc[::-1]
        with self._style():
            fig, ax = plt.subplots(figsize=(6.5, max(2.6, 0.36 * len(t)
                                                     + 1.2)))
            ax.barh([str(c) for c in t["column"]], t["importance"],
                    xerr=t["std"], ecolor="#555555", capsize=2,
                    **self._bar_style(0))
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Drop in balanced accuracy when the column is "
                          "shuffled")
            ax.set_title("Which columns matter")
            ax.grid(axis="y", visible=False)
            fig.tight_layout()
            return self._save(fig, "feature_importance")

    def columns_by_class(self, X: pd.DataFrame, y, columns: List[str]
                         ) -> Optional[Dict[str, str]]:
        """For the most important columns: box plots of numeric columns by
        class; for text columns, the share of each category by class."""
        cols = [c for c in columns if c in X.columns][:6]
        if not cols:
            return None
        y = np.asarray(y)
        k = len(self.class_names)
        ncols = min(3, len(cols))
        nrows = int(np.ceil(len(cols) / ncols))
        with self._style():
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4.2 * ncols, 3.6 * nrows),
                                     squeeze=False)
            for ax, col in zip(axes.flat, cols):
                s = X[col]
                if pd.api.types.is_numeric_dtype(s):
                    data = [s[y == i].dropna().to_numpy() for i in range(k)]
                    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                                    medianprops=dict(color="black", lw=1.4),
                                    flierprops=dict(markersize=3,
                                                    alpha=0.5))
                    for i, box in enumerate(bp["boxes"]):
                        box.set_facecolor(self.color(i))
                        box.set_alpha(0.85)
                        box.set_hatch(self.hatch(i))
                        box.set_edgecolor("black")
                    ax.set_xticks(range(1, k + 1))
                    ax.set_xticklabels(self.class_names, rotation=30,
                                       ha="right")
                    ax.set_ylabel(str(col))
                else:
                    cats = s.astype(str).value_counts().index[:8]
                    shares = np.array([[((s[y == i].astype(str) == c).mean()
                                         if (y == i).any() else 0) * 100
                                        for c in cats] for i in range(k)])
                    width = 0.8 / k
                    for i in range(k):
                        ax.bar(np.arange(len(cats)) + i * width
                               - 0.4 + width / 2, shares[i], width,
                               label=self.class_names[i],
                               **self._bar_style(i))
                    ax.set_xticks(range(len(cats)))
                    ax.set_xticklabels([str(c) for c in cats], rotation=30,
                                       ha="right")
                    ax.set_ylabel(f"% of class ({col})")
                    ax.legend(fontsize=7 * self.theme.font_scale)
                ax.set_title(str(col))
                ax.grid(axis="x", visible=False)
            for ax in list(axes.flat)[len(cols):]:
                ax.set_visible(False)
            fig.suptitle("Most important columns, by class")
            fig.tight_layout()
            return self._save(fig, "columns_by_class")

    # ---- more data? ----------------------------------------------------- #

    def learning_curve(self, d: Dict) -> Dict[str, str]:
        with self._style():
            fig, ax = plt.subplots(figsize=(6, 4.3))
            for key, label, i in (("train", "rows used for training", 1),
                                  ("val", "new rows (validation)", 0)):
                m, s = d[f"{key}_mean"] * 100, d[f"{key}_std"] * 100
                ax.plot(d["sizes"], m, marker="o", color=self.color(i),
                        ls=self.linestyle(i), label=label)
                ax.fill_between(d["sizes"], m - s, m + s,
                                color=self.color(i), alpha=0.15)
            ax.set_xlabel("Number of training rows")
            ax.set_ylabel("Balanced accuracy (%)")
            ax.set_title("Learning curve: would more data help?")
            ax.legend(loc="lower right")
            fig.tight_layout()
            return self._save(fig, "learning_curve")
