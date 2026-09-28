"""Analysis report as a LaTeX file, compiled to PDF when LaTeX is available.

The report is written for non-programmers and contains:

1. the data and the question (what was predicted),
2. a ready-to-adapt *Methods* paragraph for a paper or thesis,
3. results: the honest final score, the comparison of classifiers,
   figures, and which columns mattered,
4. what each number means, in plain words,
5. references to cite.

``report.tex`` is always written. If a LaTeX engine (tectonic, pdflatex,
xelatex or lualatex) is installed, ``report.pdf`` is produced from it;
otherwise the .tex file can be uploaded to an online editor such as Overleaf.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .distances import HASSANAT_CITATION
from .evaluation import METRICS, SELECTION_LABEL, VALIDATION
from .help_texts import explain

SKLEARN_CITATION = ("Pedregosa, F., et al. (2011). Scikit-learn: Machine "
                    "Learning in Python. Journal of Machine Learning "
                    "Research, 12, 2825-2830.")
BREIMAN_CITATION = ("Breiman, L. (2001). Random Forests. Machine Learning, "
                    "45, 5-32.")

PERCENT_METRICS = {"accuracy", "precision", "recall", "f1",
                   "balanced_accuracy", "specificity"}


# --------------------------------------------------------------------------- #
# Everything the report needs, collected by the wizard
# --------------------------------------------------------------------------- #

@dataclass
class ReportContext:
    software_citation: str
    version: str
    dataset: str
    rows_loaded: int
    cols_loaded: int
    target: str
    target_grouping: str
    class_counts: Dict[str, int]
    rows_used: int
    predictor_columns: List[str]
    left_out: Dict[str, str]
    data_notes: List[str]
    impute: Optional[str]           # None if no missing values were filled
    encoding: Optional[str]         # None if no text columns
    scaled: Dict[str, List[str]]    # "standard"/"minmax" -> classifier names
    classifiers: List[str]
    knn_text: str                   # "" if KNN not used
    hassanat_used: bool
    validation: str
    selection: Optional[str]
    n_dev: int
    n_test: int
    best_name: str
    final: object                   # Result or None
    results: List[object]           # Results, ranked
    metrics: List[str]
    # figure key -> {format: path relative to the results folder}
    figures: Dict[str, Dict[str, str]] = field(default_factory=dict)
    importance: Optional[pd.DataFrame] = None
    comparison_text: str = ""
    learning_text: str = ""


# --------------------------------------------------------------------------- #
# LaTeX helpers
# --------------------------------------------------------------------------- #

_TEX = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}",
}


def tex(s) -> str:
    """Escape text for LaTeX. Characters pdfLaTeX cannot typeset are
    replaced with '?', so unusual column names never break compilation."""
    out = []
    for ch in str(s):
        if ch in _TEX:
            out.append(_TEX[ch])
        elif ord(ch) < 256:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def _val(metric: str, v: float) -> str:
    return f"{v * 100:.2f}\\%" if metric in PERCENT_METRICS else f"{v:.4f}"


# Captions: what the figure shows and how to read it, in plain words.
CAPTIONS = {
    "class_distribution": "Number of rows in each class. Very unequal "
                          "classes make some measures (e.g.\\ accuracy) "
                          "look better than they are; balanced accuracy "
                          "takes this into account.",
    "missing_values": "Empty cells in the data as loaded. Left: each dark "
                      "mark is an empty cell (rows run left to right). "
                      "Right: the share of empty cells in each column.",
    "correlation": "Correlation between numeric columns, from $-1$ (one "
                   "goes up when the other goes down) through 0 (no linear "
                   "relation) to $+1$ (they go up together). Values near "
                   "$\\pm 1$ mean two columns carry almost the same "
                   "information.",
    "comparison": "Balanced accuracy of each classifier. Dots are the "
                  "individual test folds, the vertical bar is the overall "
                  "score. The wider the dots are spread, the less certain "
                  "the ranking. The diamond, if shown, is the final score "
                  "of the selected classifier on data not used to choose "
                  "it.",
    "confusion_matrix": "Confusion matrix. Rows are the true classes, "
                        "columns the predicted classes; each cell shows the "
                        "number of rows and the percentage of that true "
                        "class. The diagonal holds the correct predictions; "
                        "off-diagonal cells show which classes are "
                        "confused.",
    "roc_curve": "ROC curve(s). The closer a curve comes to the top-left "
                 "corner, the better that class is separated from the "
                 "others; the dotted diagonal is random guessing. AUC "
                 "summarises each curve (1 = perfect, 0.5 = random).",
    "pr_curve": "Precision-recall curve(s), most informative when a class is "
                "rare. The higher and further right a curve, the better. "
                "The dotted line is the level a random guess would reach.",
    "feature_importance": "Permutation importance: how much the balanced "
                          "accuracy drops when a column is shuffled "
                          "(mean and spread over repeated shuffles).",
    "columns_by_class": "The most important columns, shown for each class. "
                        "For numbers, each box covers the middle half of the "
                        "values, the line is the median. For text columns, "
                        "bars show how common each category is in each "
                        "class.",
    "learning_curve": "Learning curve. The lower line is the score on new "
                      "rows as more training rows are used; shaded bands "
                      "show the variation between folds. A line still "
                      "rising at the right means more data would probably "
                      "help; a flat line means it probably would not. A "
                      "large gap between the two lines means the model "
                      "partly memorises its training rows.",
}


def _fig(c, key: str, caption: str, width: str = "0.62") -> str:
    """Figure block for ``key``, preferring the vector PDF version."""
    files = c.figures.get(key)
    if not files:
        return ""
    path = files.get("pdf") or files.get("png")
    if not path:
        return ""
    return _figure(path, caption, width)


def _figure(path: str, caption: str, width: str = "0.62") -> str:
    return (
        "\\begin{figure}[H]\n\\centering\n"
        f"\\includegraphics[width={width}\\textwidth]"
        f"{{{path.replace(os.sep, '/')}}}\n"
        f"\\caption{{{caption}}}\n\\end{{figure}}\n"
    )


def _join(items: List[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def _methods_paragraph(c: ReportContext) -> str:
    k = len(c.class_counts)
    parts = [
        f"Classification was carried out with EasyClassifier version "
        f"{tex(c.version)} \\cite{{easyclassifier}}, which is built on "
        f"scikit-learn \\cite{{sklearn}}.",
        f"The dataset ({tex(c.dataset)}) contained {c.rows_loaded} rows and "
        f"{c.cols_loaded} columns. The outcome to predict was "
        f"\\emph{{{tex(c.target)}}} with {k} classes.",
    ]
    if c.target_grouping:
        parts.append(
            f"Because {tex(c.target)} is a numeric measurement, it was "
            f"divided into groups before modelling: "
            f"{tex(c.target_grouping)}.")
    if c.left_out:
        parts.append(
            "Columns that could not contribute to prediction were excluded ("
            + tex(_join([f"{col}: {why}" for col, why in c.left_out.items()]))
            + ").")
    for note in c.data_notes:
        parts.append(tex(note))
    parts.append(f"The analysis used {c.rows_used} rows and "
                 f"{len(c.predictor_columns)} predictor columns.")

    prep = []
    if c.impute:
        label = "most frequent value" if c.impute == "mode" else c.impute
        prep.append(f"missing numeric values were filled with the {label} "
                    "and missing text values with the most frequent value")
    if c.encoding:
        enc = {
            "onehot": "one-hot encoded",
            "label": "ordinal-encoded",
            "auto": ("one-hot encoded when they had at most 10 categories "
                     "and ordinal-encoded otherwise"),
        }[c.encoding]
        prep.append(f"text columns were {enc}")
    how = []
    if c.scaled.get("standard"):
        how.append("standardised (mean 0, standard deviation 1) for "
                   + tex(_join(c.scaled["standard"])))
    if c.scaled.get("minmax"):
        how.append("scaled to the range 0--1 for "
                   + tex(_join(c.scaled["minmax"])))
    if how:
        prep.append("numeric columns were " + ", and ".join(how))
    if prep:
        parts.append("Before modelling, " + "; ".join(prep) + ". All of these "
                     "steps were learned from the training data only, "
                     "separately within every validation split, so that no "
                     "information from test data entered model training.")

    parts.append(
        "The following classifiers were used with their default parameters, "
        "without hyperparameter tuning: " + tex(_join(c.classifiers)) + ".")
    if c.knn_text:
        cite = " \\cite{hassanat2014}" if c.hassanat_used else ""
        parts.append(tex(c.knn_text) + cite + ".")
    parts.append(f"Performance was estimated with "
                 f"{tex(VALIDATION[c.validation].lower())}.")

    if c.final is not None and c.selection == "final_test":
        parts.append(
            f"The classifier with the highest {SELECTION_LABEL.lower()} was "
            f"selected. To avoid optimistic bias from this selection, "
            f"{c.n_test} rows (20\\%, stratified by class) were set aside "
            f"before any analysis; classifiers were compared on the "
            f"remaining {c.n_dev} rows only, and the selected classifier was "
            f"evaluated once on the untouched test rows.")
    elif c.final is not None and c.selection == "nested":
        parts.append(
            f"The classifier with the highest {SELECTION_LABEL.lower()} was "
            f"selected. To avoid optimistic bias from this selection, "
            f"performance was estimated with nested 5-fold cross-validation, "
            f"in which the complete comparison was repeated within each "
            f"outer training fold and the winner was evaluated on the outer "
            f"test fold.")
    if c.importance is not None:
        parts.append(
            "The contribution of each predictor was estimated by permutation "
            "importance \\cite{breiman2001}, i.e.\\ the drop in "
            f"{SELECTION_LABEL.lower()} when that column's values were "
            "randomly shuffled, measured on the "
            f"{tex(c.importance.attrs.get('measured_on', 'held-out rows'))}.")
    return " ".join(parts)


def _metric_table(result, metrics: List[str]) -> str:
    rows = []
    for m in metrics:
        if m in result.metrics:
            rows.append(f"{tex(METRICS[m])} & {_val(m, result.metrics[m])} & "
                        f"\\parbox[t]{{0.55\\textwidth}}{{\\small "
                        f"{tex(explain(m))}}}\\\\[3pt]")
    return ("\\begin{tabular}{llp{0.57\\textwidth}}\n\\toprule\n"
            "Measure & Value & What it means\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def _comparison_table(results, metrics: List[str]) -> str:
    extra = [m for m in ("accuracy", "f1", "roc_auc")
             if m in metrics and m != "balanced_accuracy"]
    head = "Classifier & " + SELECTION_LABEL + "".join(
        f" & {tex(METRICS[m])}" for m in extra)
    lines = []
    for r in results:
        cells = [tex(r.classifier_name),
                 _val("balanced_accuracy", r.selection_score)]
        cells += [_val(m, r.metrics[m]) if m in r.metrics else "--"
                  for m in extra]
        lines.append(" & ".join(cells) + "\\\\")
    cols = "l" + "r" * (1 + len(extra))
    return (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\\\\\n"
            "\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")


def build_tex(c: ReportContext) -> str:
    date = _dt.date.today().strftime("%d %B %Y")
    shown = c.final if c.final is not None else (c.results[0]
                                                 if c.results else None)
    L = []
    L.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\begin{document}
""")
    L.append("\\begin{center}\n{\\LARGE EasyClassifier analysis report}\\\\"
             f"[4pt]\n{{\\large {tex(c.dataset)}}}\\\\[2pt]\n"
             f"{date} \\quad$\\cdot$\\quad EasyClassifier "
             f"{tex(c.version)}\n\\end{{center}}\n")

    # 1. Summary ---------------------------------------------------------- #
    L.append("\\section*{Summary}\n")
    if shown is not None:
        key = ("accuracy" if "accuracy" in shown.metrics else
               next(iter(shown.metrics), None))
        what = (f"\\emph{{{tex(c.target)}}}"
                + (f" (grouped: {tex(c.target_grouping)})"
                   if c.target_grouping else ""))
        s = (f"The goal was to predict {what} from "
             f"{len(c.predictor_columns)} other columns, using "
             f"{c.rows_used} rows. ")
        if len(c.results) > 1:
            s += (f"{len(c.results)} classifiers were compared and "
                  f"\\textbf{{{tex(c.best_name)}}} was selected. ")
        else:
            s += f"The classifier used was \\textbf{{{tex(c.best_name)}}}. "
        if key:
            s += (f"Its {'honest final ' if c.final is not None else ''}"
                  f"{tex(METRICS[key].lower())} was "
                  f"\\textbf{{{_val(key, shown.metrics[key])}}}")
            chance = 100.0 / max(1, len(c.class_counts))
            s += (f" (guessing at random would give about {chance:.0f}\\% "
                  "with equally sized classes).")
        L.append(s + "\n")

    # 2. Data -------------------------------------------------------------- #
    L.append("\\section{Data and question}\n")
    L.append(f"File: {tex(c.dataset)}; {c.rows_loaded} rows and "
             f"{c.cols_loaded} columns were loaded. Rows used in the "
             f"analysis: {c.rows_used}.\n\n")
    total = sum(c.class_counts.values()) or 1
    L.append("\\begin{table}[H]\n\\centering\n\\begin{tabular}{lrr}\n"
             "\\toprule\nClass & Rows & Share\\\\\n\\midrule\n")
    for cls, n in c.class_counts.items():
        L.append(f"{tex(cls)} & {n} & {100 * n / total:.1f}\\%\\\\\n")
    L.append("\\bottomrule\n\\end{tabular}\n\\caption{Classes of "
             f"\\emph{{{tex(c.target)}}}.}}\n\\end{{table}}\n")
    L.append("Predictor columns: " + tex(", ".join(map(str,
                                                      c.predictor_columns)))
             + ".\n\n")
    if c.left_out:
        L.append("Left out: " + tex("; ".join(
            f"{k} ({v})" for k, v in c.left_out.items())) + ".\n\n")
    L.append(_fig(c, "class_distribution", CAPTIONS["class_distribution"]))
    L.append(_fig(c, "missing_values", CAPTIONS["missing_values"],
                  width="0.85"))
    L.append(_fig(c, "correlation", CAPTIONS["correlation"], width="0.7"))

    # 3. Methods ------------------------------------------------------------ #
    L.append("\\section{Methods (ready to adapt for your paper)}\n")
    L.append("The paragraph below describes exactly what was done. You may "
             "copy and adapt it for the methods section of a paper or "
             "thesis.\n\n")
    L.append("\\begin{quote}\n" + _methods_paragraph(c) + "\n\\end{quote}\n")

    # 4. Results ------------------------------------------------------------ #
    L.append("\\section{Results}\n")
    if shown is not None:
        if c.final is not None:
            where = ("the untouched 20\\% test set"
                     if c.selection == "final_test"
                     else "nested cross-validation")
            L.append(f"\\subsection*{{Final result for {tex(c.best_name)} "
                     f"(measured on {where}) -- report these numbers}}\n")
            if getattr(c.final, "note", ""):
                L.append(tex(c.final.note) + ".\n\n")
        else:
            L.append(f"\\subsection*{{Result for {tex(c.best_name)}}}\n")
        L.append("\\begin{table}[H]\n\\centering\n"
                 + _metric_table(shown, c.metrics)
                 + "\\end{table}\n")
    if len(c.results) > 1:
        L.append("\\subsection*{Comparison of classifiers}\n")
        if c.final is not None:
            L.append("These scores were used only to choose the best "
                     "classifier. They are slightly optimistic, because the "
                     "winner partly won by luck; report the final result "
                     "above instead.\n\n")
        L.append("\\begin{table}[H]\n\\centering\n\\small\n"
                 + _comparison_table(c.results, c.metrics)
                 + "\\caption{Classifiers ranked by "
                 f"{SELECTION_LABEL.lower()}.}}\n\\end{{table}}\n")
        if c.comparison_text:
            L.append(tex(c.comparison_text) + "\n\n")
        L.append(_fig(c, "comparison", CAPTIONS["comparison"],
                      width="0.8"))

    if any(k in c.figures for k in ("confusion_matrix", "roc_curve",
                                    "pr_curve")):
        L.append("\\subsection*{Figures for "
                 f"{tex(c.best_name)}"
                 + (" (final result)" if c.final is not None else "")
                 + "}\n")
    for key in ("confusion_matrix", "roc_curve", "pr_curve"):
        L.append(_fig(c, key, CAPTIONS[key]))

    if c.importance is not None and len(c.importance):
        L.append("\\subsection*{Which columns mattered}\n")
        L.append("Each column's values were shuffled, and the drop in "
                 f"{SELECTION_LABEL.lower()} was measured on rows the model "
                 "had not been trained on. A larger drop means the model "
                 "relies more on that column. Values near zero (or below) "
                 "mean the column did not help. Importance shows what the "
                 "model uses; it does not prove cause and effect.\n\n")
        L.append(_fig(c, "feature_importance",
                      CAPTIONS["feature_importance"], width="0.72"))
        top = c.importance.head(10)
        L.append("\\begin{table}[H]\n\\centering\n\\small\n"
                 "\\begin{tabular}{lrr}\n\\toprule\nColumn & Importance & "
                 "Spread\\\\\n\\midrule\n")
        for _, r in top.iterrows():
            L.append(f"{tex(r['column'])} & {r['importance']:.4f} & "
                     f"{r['std']:.4f}\\\\\n")
        cap = ("The ten most important columns." if len(c.importance) > 10
               else "Importance of each column.")
        L.append(f"\\bottomrule\n\\end{{tabular}}\n\\caption{{{cap}}}\n"
                 "\\end{table}\n")
        L.append(_fig(c, "columns_by_class", CAPTIONS["columns_by_class"],
                      width="0.95"))

    if "learning_curve" in c.figures:
        L.append("\\subsection*{Would more data help?}\n")
        if c.learning_text:
            L.append(tex(c.learning_text) + "\n\n")
        L.append(_fig(c, "learning_curve", CAPTIONS["learning_curve"]))

    # 5. Notes -------------------------------------------------------------- #
    L.append("\\section{Good practice when reporting}\n")
    L.append("\\begin{itemize}\n"
             "\\item Report the final result, not the best score from the "
             "comparison table.\n"
             "\\item Mention the number of rows and the class sizes; results "
             "for small classes are less reliable.\n"
             "\\item Classifiers used default parameters (no tuning), so the "
             "results can be reproduced exactly with the same data and "
             "EasyClassifier version.\n"
             "\\item All choices and steps are recorded in "
             "\\texttt{log.txt} in the Results folder.\n"
             "\\end{itemize}\n")

    # 6. References --------------------------------------------------------- #
    L.append("\\begin{thebibliography}{9}\n")
    L.append(f"\\bibitem{{easyclassifier}} {tex(c.software_citation)}\n")
    L.append(f"\\bibitem{{sklearn}} {tex(SKLEARN_CITATION)}\n")
    if c.hassanat_used:
        L.append(f"\\bibitem{{hassanat2014}} {tex(HASSANAT_CITATION)}\n")
    if c.importance is not None:
        L.append(f"\\bibitem{{breiman2001}} {tex(BREIMAN_CITATION)}\n")
    L.append("\\end{thebibliography}\n\\end{document}\n")
    return "".join(L)


# --------------------------------------------------------------------------- #
# Writing and compiling
# --------------------------------------------------------------------------- #

ENGINES = ["tectonic", "pdflatex", "xelatex", "lualatex"]


def find_engine() -> Optional[str]:
    for e in ENGINES:
        if shutil.which(e):
            return e
    return None


def write_and_compile(c: ReportContext, out_dir: str,
                      compile_pdf: bool = True,
                      timeout: int = 180) -> Tuple[str, Optional[str], str]:
    """Write report.tex; compile it to report.pdf if possible.

    Returns (tex_path, pdf_path or None, message for the user).
    """
    tex_path = os.path.join(out_dir, "report.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(build_tex(c))
    if not compile_pdf:
        return tex_path, None, "PDF not requested."

    engine = find_engine()
    if engine is None:
        return tex_path, None, (
            "No LaTeX program was found on this computer, so only "
            "report.tex was written. To get the PDF, upload report.tex and "
            "the 'figures' folder to an online LaTeX editor such as "
            "Overleaf, or install a LaTeX distribution (e.g. MiKTeX on "
            "Windows, MacTeX on macOS, TeX Live on Linux).")

    pdf_path = os.path.join(out_dir, "report.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if engine == "tectonic":
        runs = [["tectonic", "report.tex"]]
    else:
        cmd = [engine, "-interaction=nonstopmode", "-halt-on-error",
               "report.tex"]
        runs = [cmd, cmd]  # twice, for references
    try:
        for cmd in runs:
            proc = subprocess.run(cmd, cwd=out_dir, capture_output=True,
                                  timeout=timeout)
            if proc.returncode != 0:
                break
    except (subprocess.TimeoutExpired, OSError) as exc:
        return tex_path, None, f"LaTeX ({engine}) did not finish: {exc}."
    finally:
        for ext in (".aux", ".out", ".toc"):
            p = os.path.join(out_dir, "report" + ext)
            if os.path.exists(p):
                os.remove(p)

    log = os.path.join(out_dir, "report.log")
    if os.path.exists(pdf_path) and proc.returncode == 0:
        if os.path.exists(log):
            os.remove(log)
        return tex_path, pdf_path, f"PDF created with {engine}."
    if os.path.exists(log):
        os.replace(log, os.path.join(out_dir, "report_latex_errors.log"))
    return tex_path, None, (
        f"LaTeX ({engine}) could not build the PDF; details are in "
        "report_latex_errors.log. report.tex can still be opened in an "
        "online editor such as Overleaf.")
