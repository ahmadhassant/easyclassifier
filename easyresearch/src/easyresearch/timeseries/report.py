"""Forecasting report as LaTeX (compiled to PDF when LaTeX is installed)."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List

from easyclassifier.latex_report import _figure, _join, tex

from .. import CITATION
from ..common import references as refs
from ..common.hassanat import MHSP_EXPLANATION
from ..common.output import compile_tex, fmt
from ..common.references import cite
from .evaluation import METRICS, SELECTION_METRICS
from .figures import CAPTIONS

# Kept for compatibility; the texts live in common/references.py.
MASE_CITATION = refs.REFERENCES["hyndman2006"]
TASHMAN_CITATION = refs.REFERENCES["tashman2000"]
STL_CITATION = refs.REFERENCES["cleveland1990"]

EXPLAIN = {
    "mase": "MASE (mean absolute scaled error): the average error divided "
            "by the average change from one season (or one step) earlier in "
            "the known history. Below 1 means more accurate than simply "
            "repeating that earlier value; it can be compared across series "
            "with different units.",
    "mae": "MAE (mean absolute error): the average size of a forecast "
           "error, in the units of the series.",
    "rmse": "RMSE (root mean squared error): the typical size of an error in "
            "the units of the series; large errors count more.",
    "smape": "sMAPE (symmetric mean absolute percentage error): the average "
             "error as a percentage of the average of the actual and "
             "forecast values (0% is perfect, 200% the maximum).",
    "mhsp": MHSP_EXPLANATION,
}


@dataclass
class ForecastReport:
    version: str
    dataset: str
    dataset_citation: str
    series: object
    horizon: int
    window: int
    plan: object
    results: List
    best: object
    final: object
    refs_final: Dict
    metric: str
    summary: List[str]
    comparison_text: str
    coverage: float
    figures: Dict[str, Dict[str, str]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    registry: Dict = field(default_factory=dict)
    rows_loaded: int = 0
    deep_device: str = "cpu"


def _v(key, v):
    if key == "mhsp":
        return f"{v:.3f}\\%"
    if key == "smape":
        return f"{v:.2f}\\%"
    if key == "mase":
        return f"{v:.3f}"
    return tex(fmt(v))


def _table(rows) -> str:
    head = "Model & " + " & ".join(METRICS.values())
    lines = [" & ".join([tex(name)] + [_v(k, r.metrics[k]) for k in METRICS])
             + "\\\\" for name, r in rows]
    return ("\\begin{tabular}{l" + "r" * len(METRICS) + "}\n\\toprule\n"
            + head + "\\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")


def _fig(c, key, width="0.85"):
    files = c.figures.get(key)
    if not files:
        return ""
    path = files.get("pdf") or files.get("png")
    return _figure(path, tex(CAPTIONS[key]), width) if path else ""


def _methods(c: ForecastReport) -> str:
    s, reg = c.series, c.registry
    used = [r.key for r in c.results]
    fam = {f: [k for k in used if reg[k].family == f]
           for f in ("statistical", "ml", "deep")}

    def named(keys):
        return _join([tex(reg[k].name.replace("Deep: ", ""))
                      + cite(refs.FORECAST_MODEL_REFERENCES.get(k, []))
                      for k in keys])

    p = [f"Forecasting was carried out with EasyResearch (version "
         f"{tex(c.version)}) \\cite{{software}}, using statsmodels "
         "\\cite{statsmodels}, scikit-learn \\cite{sklearn}, NumPy "
         "\\cite{numpy}, pandas \\cite{pandas}"
         + (" and PyTorch \\cite{pytorch}" if fam["deep"] else "") + "."]
    src = " \\cite{dataset}" if c.dataset_citation else ""
    p.append(f"The series {tex(s.name)} ({tex(c.dataset)}){src} contained "
             f"{len(s.values)} {tex(s.label)} values"
             + (f" (time column {tex(s.time_column)})" if s.time_column
                else "") + ".")
    p += [tex(n) for n in s.notes]
    p.append(f"Forecasts were made {c.horizon} step(s) ahead"
             + (f"; a seasonal cycle of {s.season} steps was assumed"
                if s.season > 1 else "; no seasonal cycle was assumed")
             + ".")
    p.append("Two reference rules were always included: the naive forecast "
             "(repeat the last value)"
             + (" and the seasonal naive forecast (repeat the value one "
                "season earlier)" if s.season > 1 else "")
             + cite(refs.FORECAST_MODEL_REFERENCES["naive"]) + ".")
    if fam["statistical"]:
        # Describe only the settings of the statistical models actually used.
        detail = []
        if "ets" in fam["statistical"]:
            detail.append("exponential smoothing with a damped additive trend"
                          + (" and additive seasonality" if s.season > 1
                             else ""))
        if "theta" in fam["statistical"]:
            detail.append("the Theta method"
                          + (" on the seasonally adjusted series"
                             if s.season > 1 else ""))
        p.append("Statistical models: " + named(fam["statistical"])
                 + " (" + "; ".join(detail + ["parameters estimated by "
                                              "statsmodels"]) + ").")
    if fam["ml"] or fam["deep"]:
        p.append(f"Machine-learning and deep models predicted the next "
                 f"{c.horizon} values at once from the last {c.window} "
                 "values (a direct multi-output strategy "
                 "\\cite{bontempi2013}), expressed relative to the last value "
                 "of each window and divided by the standard deviation of the "
                 "one-step changes of the training series.")
    if fam["ml"]:
        p.append("Machine-learning models: " + named(fam["ml"])
                 + (", the nearest-neighbour model using the Hassanat "
                    "distance \\cite{hassanat2022,hassanat2014}"
                    if "knn" in used else "") + ".")
    if fam["deep"]:
        p.append("Deep networks: " + named(fam["deep"])
                 + ". They were trained with Adam \\cite{kingma2015} "
                 "(learning rate 0.001, batches of "
                 "32, mean squared error) for at most 200 epochs, stopping after "
                 "20 epochs without improvement on the most recent 10\\% of the "
                 "training windows and keeping the best weights.")
    p.append("All models used default settings without tuning, with fixed "
             "random seeds.")
    p.append(f"Models were compared by rolling-origin evaluation "
             f"\\cite{{tashman2000}}: at {len(c.plan.dev_origins)} origins "
             f"before the final period, each model was trained on all earlier "
             f"values and forecast the next {c.horizon}. The model with the "
             f"best {SELECTION_METRICS[c.metric]} was selected. It was then "
             f"evaluated in the final {len(s.values) - c.plan.dev_end} values, "
             "which were not used for the comparison, at "
             f"{len(c.plan.test_origins)} further origins, retraining before "
             "each one.")
    p.append("Accuracy is reported as MASE \\cite{hyndman2006}, MAE, RMSE, "
             "sMAPE \\cite{makridakis1993} and the mean Hassanat similarity "
             "percentage (MHSP) \\cite{mhsp}."
             " The 80\\% uncertainty band of the final forecast was derived "
             "from the errors of the selected model at each step ahead "
             "\\cite{hyndman2021}.")
    if c.figures:
        p.append("Figures were drawn with Matplotlib \\cite{matplotlib}"
                 + ("; the trend and seasonal pattern were separated by STL "
                    "decomposition \\cite{cleveland1990}"
                    if "decomposition" in c.figures else "") + ".")
    return " ".join(p)


def build_tex(c: ForecastReport) -> str:
    s, reg = c.series, c.registry
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
    L.append("\\begin{center}\n{\\LARGE Forecasting report}\\\\[4pt]\n"
             f"{{\\large {tex(c.dataset)}}}\\\\[2pt]\n{date}\n\\end{{center}}\n")
    L.append("\\section{The question}\n")
    L.append(f"Forecast the next {c.horizon} {tex(s.label)} values of "
             f"\\textbf{{{tex(s.name)}}} from its own history.\n\n")
    L.append("\\section{Methods (ready to adapt for a paper)}\n"
             + _methods(c) + "\n\n")

    L.append("\\section{Results}\n")
    L.append(f"Selected model: \\textbf{{{tex(c.best.name)}}}. Results in the "
             "final period -- these are the numbers to report:\n\n")
    rows = [(c.best.name, c.final)] + [
        (reg[k].name, r) for k, r in c.refs_final.items()]
    L.append("\\begin{table}[H]\n\\centering\n\\small\n" + _table(rows)
             + "\\caption{Final period: the selected model and the reference "
             "rules, forecasting from the same origins.}\n\\end{table}\n")
    L += [tex(x) + "\n\n" for x in c.summary]
    L.append(_fig(c, "final_period"))
    L.append(_fig(c, "forecast"))

    L.append("\\subsection*{Comparison of models}\n")
    L.append("These scores were used only to choose the model and are "
             "slightly optimistic; report the final period above.\n\n")
    L.append("\\begin{table}[H]\n\\centering\n\\small\n"
             + _table([(r.name, r) for r in c.results])
             + f"\\caption{{Comparison origins, ranked by "
             f"{SELECTION_METRICS[c.metric]}.}}\n\\end{{table}}\n")
    if c.comparison_text:
        L.append(tex(c.comparison_text) + "\n\n")
    if c.skipped:
        L.append("Not evaluated: " + tex("; ".join(c.skipped)) + ".\n\n")
    L.append(_fig(c, "comparison", "0.8"))
    L.append(_fig(c, "error_by_step", "0.65"))
    if "decomposition" in c.figures:
        L.append("\\subsection*{The series}\n" + _fig(c, "decomposition"))

    L.append("\\section{What the numbers mean}\n\\begin{itemize}\n")
    L += ["\\item " + tex(EXPLAIN[k]) + "\n" for k in METRICS]
    L.append("\\end{itemize}\n")
    L.append("\\section{Good practice when reporting}\n\\begin{itemize}\n"
             "\\item Report the final-period results, together with the "
             "reference rules: a forecast is only useful if it beats them.\n"
             "\\item State the horizon: errors usually grow with the number "
             "of steps ahead (see the error-by-step figure).\n"
             "\\item The future forecast assumes the patterns of the past "
             "continue; events that never occurred in the history cannot be "
             "foreseen.\n"
             "\\item Every step is recorded in \\texttt{log.txt}; the same "
             "data and software version reproduce the results.\n"
             "\\end{itemize}\n")

    # References: every reference cited above, in order of citation.
    body = "".join(L)
    return (body + refs.bibliography(body, report_entries(c), escape=tex)
            + "\\end{document}\n")


def report_entries(c: ForecastReport) -> Dict[str, str]:
    """Report-specific bibliography entries: the software and the dataset."""
    entries = {"software": CITATION}
    if c.dataset_citation:
        entries["dataset"] = c.dataset_citation
    return entries


def citation_lines(c: ForecastReport) -> List[str]:
    """citations.txt: exactly the references cited in the report."""
    return refs.citation_lines(build_tex(c), report_entries(c), refs.TOPICS)


def write_report(c: ForecastReport, out_dir: str) -> str:
    return compile_tex(build_tex(c), out_dir)[2]
