"""Regression report as LaTeX, compiled to PDF when LaTeX is installed.

Same structure as EasyClassifier's report: the question, a Methods paragraph
to adapt for a paper, the honest final result, the comparison, figures,
which columns mattered, what each number means, and references.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
from easyclassifier.latex_report import _figure, _join, tex

from .. import EASYCLASSIFIER_CITATION
from ..common import references as refs
from ..common.hassanat import MHSP_EXPLANATION
from ..common.output import compile_tex
from ..common.references import cite

from .evaluation import METRICS, VALIDATION, fmt
from .figures import CAPTIONS

EXPLAIN = {
    "r2": "R² (coefficient of determination): the share of the differences "
          "between rows that the model explains. 1 means perfect "
          "predictions; 0 means no better than always predicting the "
          "average; below 0 means worse than that.",
    "rmse": "RMSE (root mean squared error): the typical size of an error, "
            "in the units of the predicted value. Large errors count more "
            "than small ones.",
    "mae": "MAE (mean absolute error): the average size of an error, in the "
           "units of the predicted value.",
    "medae": "Median absolute error: half of the predictions were closer "
             "to the actual value than this. Not affected by a few very "
             "large errors.",
    "mape": "MAPE (mean absolute percentage error): the average error as a "
            "percentage of the actual value. Only given when no actual "
            "value is zero or changes sign.",
    "mhsp": MHSP_EXPLANATION,
}


@dataclass
class ReportContext:
    software_citation: str
    version: str
    dataset: str
    dataset_citation: str
    rows_loaded: int
    cols_loaded: int
    target: str
    rows_used: int
    predictor_columns: List[str]
    left_out: Dict[str, str]
    data_notes: List[str]
    impute: Optional[str]
    encoding: Optional[str]
    scaled: Dict[str, List[str]]
    models: List[str]
    hassanat_used: bool
    hassanat_text: str
    hassanat_citations: List[str]
    validation: str
    method: str                     # nested | final_test | holdout | cross_validation
    n_dev: int
    n_test: int
    selected: str
    final: object
    baseline: object
    results: List[object]
    summary: List[str]
    comparison_text: str
    figures: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Registry keys of the models (same order as ``models``), so that each
    # model is cited.
    model_keys: List[str] = field(default_factory=list)
    hassanat_signed: bool = False   # signed form applied (negative values)
    importance: Optional[pd.DataFrame] = None
    unit_target_models: List[str] = field(default_factory=list)


def _v(key: str, v: float) -> str:
    if key == "r2":
        return f"{v:.4f}"
    if key == "mape":
        return f"{v * 100:.2f}\\%"
    if key == "mhsp":
        return f"{v:.2f}\\%"
    return tex(fmt(v))


def _fig(c, key, width="0.62"):
    files = c.figures.get(key)
    if not files:
        return ""
    path = files.get("pdf") or files.get("png")
    return _figure(path, tex(CAPTIONS[key]), width) if path else ""


def _table(rows, metrics) -> str:
    head = "Model & " + " & ".join(tex(METRICS[m]).replace("²", "$^2$")
                                   for m in metrics)
    lines = []
    for r in rows:
        cells = [tex(r.name)] + [_v(m, r.metrics[m]) if m in r.metrics
                                 else "--" for m in metrics]
        lines.append(" & ".join(cells) + "\\\\")
    return ("\\begin{tabular}{l" + "r" * len(metrics) + "}\n\\toprule\n"
            + head + "\\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")


def _methods(c: ReportContext) -> str:
    p = [f"Regression was carried out with EasyResearch "
         f"(version {tex(c.version)}) \\cite{{software}}, which uses the data "
         "preparation of EasyClassifier \\cite{easyclassifier} and is built "
         "on scikit-learn \\cite{sklearn}, NumPy \\cite{numpy} and pandas "
         "\\cite{pandas}."]
    src = f" \\cite{{dataset}}" if c.dataset_citation else ""
    p.append(f"The dataset ({tex(c.dataset)}){src} contained "
             f"{c.rows_loaded} rows and {c.cols_loaded} columns; the value "
             f"to predict was {tex(c.target)}.")
    if c.left_out:
        p.append("Columns that could not inform the prediction were left out ("
                 + _join([f"{tex(k)}: {tex(v)}" for k, v in c.left_out.items()])
                 + ").")
    for n in c.data_notes:
        p.append(tex(n))
    p.append(f"{c.rows_used} rows and {len(c.predictor_columns)} predictor "
             "columns were analysed.")
    prep = []
    if c.impute:
        prep.append(f"missing numeric values were filled with the "
                    f"{'most common value' if c.impute == 'mode' else c.impute}"
                    " and missing text values with the most common category")
    if c.encoding:
        prep.append({"onehot": "text columns were one-hot encoded",
                     "label": "text columns were label encoded"}.get(
            c.encoding, "text columns with up to 10 categories were one-hot "
            "encoded and those with more categories label encoded"))
    if c.scaled.get("standard"):
        prep.append("numeric columns were standardised (mean 0, standard "
                    "deviation 1) for " + tex(_join(c.scaled["standard"])))
    if c.scaled.get("minmax"):
        prep.append("numeric columns were scaled to the range 0--1 for "
                    + tex(_join(c.scaled["minmax"])))
    if prep:
        p.append("All learned preparation steps were fitted on the training "
                 "rows of each split only, so no information from test rows "
                 "reached the models: " + "; ".join(prep) + ".")
    if len(c.model_keys) == len(c.models):
        named = [tex(n) + cite(refs.REGRESSION_MODEL_REFERENCES.get(k, []))
                 for n, k in zip(c.models, c.model_keys)]
    else:
        named = [tex(n) for n in c.models]
    p.append("The models compared were " + _join(named)
             + ", all with their default parameters (no tuning) and fixed "
             "random seeds.")
    if c.unit_target_models:
        p.append("For " + tex(_join(c.unit_target_models)) + ", the predicted"
                 " value was standardised during training (learned on "
                 "training rows) and transformed back for predictions.")
    if c.hassanat_used:
        p.append("K-nearest-neighbour regression used k = 5 and the Hassanat "
                 "distance \\cite{hassanat2022,hassanat2014}, predicting the average value of "
                 "the five nearest training rows; "
                 + (tex(c.hassanat_text.rstrip("."))
                    + cite(refs.HASSANAT_SIGNED_REFERENCES) + "."
                    if c.hassanat_signed else tex(c.hassanat_text)))
    p.append("Models were compared by R² on held-out rows using "
             + tex(VALIDATION.get(c.validation, c.validation))
             + cite(refs.VALIDATION_REFERENCES.get(c.validation, [])) + ".")
    if c.method == "nested":
        p.append("Because choosing the best of several models and reporting "
                 "its comparison score is optimistic, the complete procedure "
                 "(compare all models, select the best) was evaluated with "
                 "5-fold nested cross-validation; the scores reported are "
                 "from the outer test folds"
                 + cite(refs.SELECTION_REFERENCES["nested"]) + ".")
    elif c.method == "final_test":
        p.append(f"Before the comparison, {c.n_test} rows (20\\%) were set "
                 f"aside; the selected model was trained on the other "
                 f"{c.n_dev} rows and scored once on these unseen rows"
                 + cite(refs.SELECTION_REFERENCES["final_test"]) + ".")
    p.append("Performance is reported as R², root mean squared error (RMSE), "
             "mean absolute error (MAE), median absolute error and the mean "
             "Hassanat similarity percentage (MHSP) \\cite{mhsp}, and "
             "compared with always predicting the average of the training "
             "rows.")
    if c.importance is not None:
        p.append("Column importance was measured as permutation importance "
                 "\\cite{breiman2001} (the drop in R² when a column's values "
                 "are shuffled) on rows not used for training.")
    if c.figures:
        p.append("Figures were drawn with Matplotlib \\cite{matplotlib}.")
    return " ".join(p)


def build_tex(c: ReportContext) -> str:
    date = _dt.date.today().strftime("%d %B %Y")
    L = [r"""\documentclass[11pt,a4paper]{article}
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
"""]
    L.append("\\begin{center}\n{\\LARGE Regression analysis report}\\\\[4pt]"
             f"\n{{\\large {tex(c.dataset)}}}\\\\[2pt]\n{date}\n"
             "\\end{center}\n")

    L.append("\\section{The question}\n")
    L.append(f"Predict \\textbf{{{tex(c.target)}}} (a number) from "
             f"{len(c.predictor_columns)} other columns: "
             f"{tex(_join(c.predictor_columns[:12]))}"
             + (" and others" if len(c.predictor_columns) > 12 else "")
             + ".\n\n")

    L.append("\\section{Methods (ready to adapt for a paper)}\n")
    L.append(_methods(c) + "\n\n")

    L.append("\\section{Results}\n")
    shown = c.final
    metrics = [m for m in METRICS if m in shown.metrics]
    if c.method in ("nested", "final_test"):
        L.append(f"Selected model: \\textbf{{{tex(c.selected)}}}. "
                 "These are the numbers to report:\n\n")
    else:
        L.append(f"Model: \\textbf{{{tex(c.selected)}}}.\n\n")
    L.append("\\begin{table}[H]\n\\centering\n"
             + _table([shown, c.baseline], metrics)
             + "\\caption{Final result"
             + (f" ({tex(shown.note)})" if shown.note else "")
             + ", and the reference of always predicting the average.}\n"
             "\\end{table}\n")
    for line in c.summary:
        L.append(tex(line) + "\n\n")
    L.append(_fig(c, "predicted_vs_actual", "0.55"))
    L.append(_fig(c, "residuals", "0.95"))

    if len(c.results) > 1:
        L.append("\\subsection*{Comparison of models}\n")
        L.append("These scores were used only to choose the model and are "
                 "slightly optimistic; report the final result above.\n\n")
        L.append("\\begin{table}[H]\n\\centering\n\\small\n"
                 + _table(c.results, [m for m in METRICS
                                      if m in c.results[0].metrics])
                 + "\\caption{Models ranked by R².}\n\\end{table}\n"
                 .replace("R²", "R$^2$"))
        if c.comparison_text:
            L.append(tex(c.comparison_text) + "\n\n")
        L.append(_fig(c, "comparison", "0.8"))

    L.append("\\subsection*{The data}\n")
    L.append(_fig(c, "target_distribution", "0.6"))
    L.append(_fig(c, "missing_values", "0.9"))

    if c.importance is not None:
        L.append("\\section{Which columns matter}\n")
        L.append("Measured on the "
                 + tex(c.importance.attrs.get("measured_on", "test rows"))
                 + ". A value near 0 means the model does not rely on the "
                 "column.\n\n")
        L.append(_fig(c, "feature_importance", "0.7"))

    L.append("\\section{What the numbers mean}\n\\begin{itemize}\n")
    for m in metrics:
        L.append("\\item " + tex(EXPLAIN[m]).replace("R²", "R$^2$") + "\n")
    L.append("\\end{itemize}\n")

    L.append("\\section{Good practice when reporting}\n\\begin{itemize}\n"
             "\\item Report the final result, not the best score from the "
             "comparison table.\n"
             "\\item Give R² together with an error in the original units "
             "(RMSE or MAE); R² alone does not show how large the errors "
             "are.\n"
             "\\item Look at the residual figure: errors that grow with the "
             "predicted value, or a curved pattern, mean the model is less "
             "reliable in that range.\n"
             "\\item Models used default parameters (no tuning), so the "
             "results can be reproduced exactly with the same data and "
             "software version. Every step is recorded in "
             "\\texttt{log.txt}.\n\\end{itemize}\n")

    # References: every reference cited above, in order of citation.
    body = "".join(L)
    body += refs.bibliography(body, report_entries(c), escape=tex)
    return (body + "\\end{document}\n").replace("R²", "R$^2$")


def report_entries(c: ReportContext) -> Dict[str, str]:
    """Report-specific bibliography entries: the software and the dataset."""
    entries = {"software": c.software_citation,
               "easyclassifier": EASYCLASSIFIER_CITATION}
    if c.dataset_citation:
        entries["dataset"] = c.dataset_citation
    return entries


def citation_lines(c: ReportContext) -> List[str]:
    """citations.txt: exactly the references cited in the report."""
    return refs.citation_lines(build_tex(c), report_entries(c), refs.TOPICS)


def write_and_compile(c: ReportContext, out_dir: str,
                      timeout: int = 180) -> Tuple[str, Optional[str], str]:
    return compile_tex(build_tex(c), out_dir, timeout)
