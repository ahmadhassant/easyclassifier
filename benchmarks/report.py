"""Turn cached benchmark results into tables (CSV, Markdown, LaTeX) and
figures. Run via ``python run_benchmarks.py --report``."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from datasets import DATASETS
from run_benchmarks import (
    KNN_EXCLUDE,
    REPEATS,
    RESULTS,
    _path,
    prepare,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIST_LABELS = {
    "hassanat": "Hassanat",
    "euclidean": "Euclidean",
    "manhattan": "Manhattan",
    "chebyshev": "Chebyshev",
    "canberra": "Canberra",
    "cosine": "Cosine",
    "euclidean_unscaled": "Euclidean (unscaled)",
    "hassanat_unscaled": "Hassanat (unscaled)",
    "hassanat_standardised": "Hassanat (standardised)",
}
REFERENCE = {"euclidean_unscaled", "hassanat_unscaled",
             "hassanat_standardised"}
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
             "#D55E00", "#000000"]


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _tex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("_", r"\_").replace("#", r"\#"))


# --------------------------------------------------------------------------- #
# Table 1: datasets
# --------------------------------------------------------------------------- #

def dataset_table() -> pd.DataFrame:
    rows = []
    for key, ds in DATASETS.items():
        _, y, names, _, info = prepare(key)
        rows.append({
            "Dataset": ds.name, "Field": ds.field, "Key": key,
            "Rows": info["rows"], "Columns": info["cols"],
            "Classes": info["classes"],
            "Smallest class": min(info["class_sizes"]),
            "Missing cells": info["missing_cells"],
            "Source": ds.source, "Notes": ds.notes,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Benchmark 1: honesty of the reported scores
# --------------------------------------------------------------------------- #

def honesty_rows() -> pd.DataFrame:
    rows = []
    for key in DATASETS:
        for s in REPEATS:
            p = _path(key, s, "B1")
            if os.path.exists(p):
                rows.append(_load(p))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["bias_naive"] = df["naive"] - df["external_naive"]
    df["bias_comparison"] = df["comparison"] - df["external_tool"]
    df["bias_final"] = df["final"] - df["external_tool"]
    return df


def honesty_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("dataset", sort=False)
    out = pd.DataFrame({
        "Dataset": [DATASETS[k].name for k in g.groups],
        "Final-score method": g["method"].agg(
            lambda m: {"final_test": "final test set",
                       "nested": "nested CV"}[m.iloc[0]]),
        "Naive estimate": g["naive"].mean(),
        "Comparison best": g["comparison"].mean(),
        "EasyClassifier final": g["final"].mean(),
        "External (truth)": g["external_tool"].mean(),
        "Bias naive": g["bias_naive"].mean(),
        "Bias comparison": g["bias_comparison"].mean(),
        "Bias final": g["bias_final"].mean(),
    })
    return out.reset_index(drop=True)


def honesty_overall(df: pd.DataFrame) -> dict:
    real = df[df["dataset"] != "noise"]
    res = {}
    for col in ("bias_naive", "bias_comparison", "bias_final"):
        v = real[col].to_numpy()
        res[col] = {"mean": float(v.mean()), "mean_abs": float(
            np.abs(v).mean()), "sd": float(v.std(ddof=1)), "n": len(v)}
    # paired test: is the naive estimate more optimistic than the final one?
    res["wilcoxon_naive_vs_final_p"] = float(stats.wilcoxon(
        real["bias_naive"], real["bias_final"],
        alternative="greater").pvalue)
    return res


def honesty_figure(summary: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(summary) + 1.6))
    y = np.arange(len(summary))[::-1]
    cols = [("Bias naive", "Naive estimate (leakage + best-of)", 0),
            ("Bias comparison", "Comparison best (best-of only)", 1),
            ("Bias final", "EasyClassifier final score", 2)]
    for j, (c, lab, k) in enumerate(cols):
        ax.scatter(summary[c] * 100, y + (j - 1) * 0.22, s=36,
                   color=OKABE_ITO[k], marker="osD"[j], label=lab, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["Dataset"])
    ax.set_xlabel("Estimate minus true performance (percentage points)\n"
                  "< 0 pessimistic     0 = honest     optimistic > 0")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path + ".png", dpi=300)
    fig.savefig(path + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Benchmark 2: KNN distances
# --------------------------------------------------------------------------- #

def knn_scores() -> pd.DataFrame:
    rows = []
    for key in DATASETS:
        if key in KNN_EXCLUDE:
            continue
        p = _path(key, "B2v2")
        if not os.path.exists(p):
            continue
        for dist, scores in _load(p).items():
            rows.append({"dataset": key, "distance": dist,
                         "mean": float(np.mean(scores)),
                         "sd": float(np.std(scores, ddof=1))})
    return pd.DataFrame(rows)


def knn_analysis(scores: pd.DataFrame) -> dict:
    wide = scores.pivot(index="dataset", columns="distance", values="mean")
    wide = wide[[d for d in DIST_LABELS if d in wide.columns]]
    tool = [d for d in wide.columns if d not in REFERENCE]
    ranks = wide[tool].rank(axis=1, ascending=False, method="average")
    res = {"wide": wide, "ranks": ranks,
           "mean_rank": ranks.mean().sort_values()}
    if len(wide) >= 3:
        res["friedman_p"] = float(stats.friedmanchisquare(
            *[wide[d] for d in tool]).pvalue)
    pair = {}
    for d in wide.columns:
        if d == "hassanat":
            continue
        diff = wide["hassanat"] - wide[d]
        try:
            p = float(stats.wilcoxon(wide["hassanat"], wide[d]).pvalue)
        except ValueError:
            p = float("nan")
        pair[d] = {"wins": int((diff > 1e-9).sum()),
                   "ties": int((diff.abs() <= 1e-9).sum()),
                   "losses": int((diff < -1e-9).sum()),
                   "mean_diff": float(diff.mean()), "p": p}
    res["pairwise"] = pair
    return res


def knn_figure(res: dict, path: str):
    wide = res["wide"] * 100
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(wide) + 1.8))
    y = np.arange(len(wide))[::-1]
    names = [DATASETS[k].name for k in wide.index]
    ref_marker = {"euclidean_unscaled": "x", "hassanat_unscaled": "v",
                  "hassanat_standardised": "^"}
    offered = [d for d in wide.columns if d not in REFERENCE]
    for j, d in enumerate(offered):
        ax.scatter(wide[d], y, s=46 if d == "hassanat" else 24,
                   marker="D" if d == "hassanat" else "o",
                   color=OKABE_ITO[j % len(OKABE_ITO)],
                   edgecolors="black" if d == "hassanat" else "none",
                   label=DIST_LABELS[d], zorder=4 if d == "hassanat" else 3)
    for d in wide.columns:
        if d in REFERENCE:
            ax.scatter(wide[d], y, s=34, marker=ref_marker.get(d, "+"),
                       facecolors="none" if d != "euclidean_unscaled"
                       else "#555555", edgecolors="#555555",
                       label=DIST_LABELS[d] + " - reference", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Balanced accuracy of KNN (k = 5), % (mean of 10 test "
                  "folds)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path + ".png", dpi=300)
    fig.savefig(path + ".pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Everything
# --------------------------------------------------------------------------- #

def _pct(v):
    return f"{v * 100:.1f}"


def build_report():
    os.makedirs(RESULTS, exist_ok=True)
    md = ["# EasyClassifier benchmark results", ""]

    dt = dataset_table()
    dt.to_csv(os.path.join(RESULTS, "datasets.csv"), index=False)
    md += ["## Datasets", "",
           dt[["Dataset", "Field", "Rows", "Columns", "Classes",
               "Smallest class", "Missing cells"]].to_markdown(index=False),
           ""]

    h = honesty_rows()
    if not h.empty:
        h.drop(columns=["comparison_all"]).to_csv(
            os.path.join(RESULTS, "honesty_runs.csv"), index=False)
        s = honesty_summary(h)
        s.to_csv(os.path.join(RESULTS, "honesty_summary.csv"), index=False)
        o = honesty_overall(h)
        with open(os.path.join(RESULTS, "honesty_overall.json"), "w") as fh:
            json.dump(o, fh, indent=2)
        honesty_figure(s, os.path.join(RESULTS, "honesty"))
        show = s.copy()
        for c in show.columns[2:]:
            show[c] = show[c].map(_pct)
        md += ["## Benchmark 1: are the reported scores honest?", "",
               f"{len(h)} runs (datasets x {len(REPEATS)} random 50/50 "
               "splits). Balanced accuracy in %. Bias = estimate minus "
               "performance on the untouched external half (0 = honest, "
               "positive = optimistic).", "",
               show.to_markdown(index=False), "",
               "Across the real datasets (noise control excluded): mean bias "
               f"naive {_pct(o['bias_naive']['mean'])}, comparison best "
               f"{_pct(o['bias_comparison']['mean'])}, EasyClassifier final "
               f"{_pct(o['bias_final']['mean'])} percentage points; mean "
               f"absolute bias {_pct(o['bias_naive']['mean_abs'])} / "
               f"{_pct(o['bias_comparison']['mean_abs'])} / "
               f"{_pct(o['bias_final']['mean_abs'])}. One-sided Wilcoxon "
               "signed-rank test that the naive estimate is more optimistic "
               f"than the final score: p = {o['wilcoxon_naive_vs_final_p']:.3g}"
               f" (n = {o['bias_final']['n']} runs).", ""]
        _latex_honesty(s)

    k = knn_scores()
    if not k.empty:
        k.to_csv(os.path.join(RESULTS, "knn_scores.csv"), index=False)
        r = knn_analysis(k)
        knn_figure(r, os.path.join(RESULTS, "knn_distances"))
        wide = r["wide"].copy()
        wide.index = [DATASETS[i].name for i in wide.index]
        wide.columns = [DIST_LABELS[c] for c in wide.columns]
        md += ["## Benchmark 2: KNN distances", "",
               "Balanced accuracy (%) of KNN, k = 5, repeated stratified "
               "5-fold CV (2 repeats), with EasyClassifier's automatic "
               "preprocessing: Hassanat on data scaled to 0-1, the other "
               "distances on standardised data. References: Euclidean and "
               "Hassanat on unscaled data, Hassanat on standardised data.",
               "",
               (wide * 100).round(1).to_markdown(), "",
               "Mean rank (1 = best) among the six distances offered: "
               + ", ".join(f"{DIST_LABELS[d]} {v:.2f}"
                           for d, v in r["mean_rank"].items()) + ".",
               ""]
        if "friedman_p" in r:
            md += [f"Friedman test across datasets: p = "
                   f"{r['friedman_p']:.3g}.", ""]
        md += ["Hassanat vs each other distance (wins / ties / losses over "
               "datasets, mean difference in percentage points, Wilcoxon "
               "signed-rank p):", ""]
        for d, v in r["pairwise"].items():
            md.append(f"* vs {DIST_LABELS[d]}: {v['wins']}/{v['ties']}/"
                      f"{v['losses']}, {v['mean_diff'] * 100:+.2f}, "
                      f"p = {v['p']:.3g}")
        md.append("")
        _latex_knn(r)

    with open(os.path.join(RESULTS, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("Report written to", RESULTS)


def _latex_honesty(s: pd.DataFrame):
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"Dataset & Final score by & & Naive & Comparison & "
             r"EasyClassifier & External \\",
             r"\midrule"]
    for _, r in s.iterrows():
        lines.append(
            f"{_tex(r['Dataset'])} & {r['Final-score method']} & & "
            f"{_pct(r['Naive estimate'])} & {_pct(r['Comparison best'])} & "
            f"{_pct(r['EasyClassifier final'])} & "
            f"{_pct(r['External (truth)'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RESULTS, "table_honesty.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _latex_knn(r: dict):
    wide = r["wide"] * 100
    cols = list(wide.columns)
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Dataset & " + " & ".join(DIST_LABELS[c] for c in cols)
             + r" \\", r"\midrule"]
    for key, row in wide.iterrows():
        best = row[[c for c in cols if c not in REFERENCE]].max()
        cells = [(r"\textbf{%.1f}" % v) if abs(v - best) < 1e-9
                 and c not in REFERENCE else f"{v:.1f}"
                 for c, v in row.items()]
        lines.append(_tex(DATASETS[key].name) + " & " + " & ".join(cells)
                     + r" \\")
    lines += [r"\midrule", "Mean rank & " + " & ".join(
        f"{r['mean_rank'][c]:.2f}" if c in r["mean_rank"] else "--"
        for c in cols) + r" \\", r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RESULTS, "table_knn.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
