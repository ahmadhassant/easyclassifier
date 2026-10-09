"""Signal classification: one class label per recording (ECG beats, EEG
epochs, sensor windows), from table to saved results.

Evaluation is EasyClassifier's: models are compared by balanced accuracy
with stratified cross-validation; the selected model gets a separate final
score - nested cross-validation (up to 2,000 recordings, when no deep
network is compared) or an untouched, stratified 20% test set. Deep networks
take minutes to train on a normal computer, so nested cross-validation
(about 30 trainings per network) is not used automatically with them.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from easyclassifier import preprocessing as pp  # noqa: E402
from easyclassifier import recommend as rec  # noqa: E402
from easyclassifier import reporting as rep  # noqa: E402
from easyclassifier.diagnostics import interpret_comparison  # noqa: E402
from easyclassifier.evaluation import (METRICS, evaluate,  # noqa: E402
                                       evaluate_on_test, fit_final_model,
                                       nested_cv, recommend_selection,
                                       split_final_test)
from easyclassifier.figures import FigureMaker  # noqa: E402
from easyclassifier.latex_report import _figure, _join, tex  # noqa: E402
from easyclassifier.logbook import LogBook  # noqa: E402

from .. import CITATION, EASYCLASSIFIER_CITATION, __version__  # noqa: E402
from ..common import references as refs  # noqa: E402
from ..common.output import (compile_tex, figure_style,  # noqa: E402
                             results_folder, save_model, write_run_record)
from ..common.references import cite  # noqa: E402
from . import signal_data as sd  # noqa: E402
from .signal_models import build_registry  # noqa: E402

REPORT_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall",
                  "f1", "roc_auc", "mcc"]
PERCENT = {"accuracy", "balanced_accuracy", "precision", "recall", "f1"}


@dataclass
class SignalSettings:
    models: List[str] = field(default_factory=lambda: [
        k for k, s in build_registry().items() if s.default_selected])
    validation: str = "auto"         # auto | kfold5 | kfold10 | holdout
    selection: str = "auto"          # auto | nested | final_test
    figures: bool = True
    figure_theme: str = "colorblind"  # colorblind | greyscale | high_contrast
    figure_format: str = "png"        # png | png+pdf | png+svg
    figure_keys: Optional[List[str]] = None   # None = all five figures


def say(text: str) -> None:
    print(text, flush=True)


class SignalFigures(FigureMaker):
    def signals_by_class(self, X: pd.DataFrame, y) -> Dict[str, str]:
        A = np.asarray(X, dtype=float)
        t = np.arange(A.shape[1])
        k = len(self.class_names)
        with self._style():
            fig, axes = plt.subplots(1, k, figsize=(min(4 * k, 14), 3.4),
                                     sharey=True, squeeze=False)
            for i, (ax, name) in enumerate(zip(axes[0], self.class_names)):
                rows = A[np.asarray(y) == i]
                for r in rows[:5]:
                    ax.plot(t, r, color=self.color(i), alpha=0.25, lw=0.7)
                m, s = rows.mean(0), rows.std(0)
                ax.fill_between(t, m - s, m + s, color=self.color(i),
                                alpha=0.2, lw=0)
                ax.plot(t, m, color=self.color(i), lw=2)
                ax.set_title(f"{name} (n={len(rows)})")
                ax.set_xlabel("Time point")
            axes[0][0].set_ylabel("Value")
            fig.suptitle("Recordings by class: average (line), ± 1 SD "
                         "(band) and five examples", fontsize=10)
            fig.tight_layout()
            return self._save(fig, "signals_by_class")


CAPTIONS = {
    "signals_by_class": "Recordings of each class: the average shape (thick "
                        "line), the spread of one standard deviation "
                        "(band) and five individual recordings (thin "
                        "lines). Clear differences in shape make the "
                        "classes easier to tell apart.",
    "class_distribution": "Number of recordings in each class.",
    "comparison": "Balanced accuracy of each model in the comparison. Dots "
                  "are the individual test folds, the vertical bar is the "
                  "overall score; the diamond, if shown, is the final score "
                  "of the selected model on data not used to choose it.",
    "confusion_matrix": "Confusion matrix of the final evaluation. Rows are "
                        "the true classes, columns the predicted classes; "
                        "the diagonal holds the correct predictions.",
    "roc_curve": "ROC curve(s) of the final evaluation (1 = perfect "
                 "separation, 0.5 = random).",
}


def run_signal_classification(df: pd.DataFrame, label: str,
                              settings: SignalSettings, out_root: str,
                              source_name: str, source_stem: str,
                              dataset_citation: str = "",
                              progress: Callable[[str], None] = say) -> Dict:
    log = LogBook()
    log.add(f"EasyResearch {__version__} - signal classification")
    registry = build_registry()
    keys = list(dict.fromkeys(settings.models))
    if not keys or any(k not in registry for k in keys):
        raise ValueError("Select at least one known signal classifier.")
    style, chosen = figure_style(settings, CAPTIONS)  # refused before training
    progress("Preparing the recordings ...")
    recs = sd.prepare(df, label)
    for n in recs.notes:
        log.add("Data: " + n)
    X = recs.X
    y, class_names, _ = pp.encode_target(recs.labels)
    n, L = X.shape
    log.add(f"{n} recordings of {L} values; classes: " + ", ".join(
        f"{c} ({int((y == i).sum())})" for i, c in enumerate(class_names)))

    validation = (rec.validation_method(y) if settings.validation == "auto"
                  else settings.validation)
    deep_used = any(registry[k].family == "deep" for k in keys)
    method = None
    if len(keys) > 1:
        if settings.selection == "auto":
            method = "final_test" if deep_used else recommend_selection(y)
        else:
            method = settings.selection
    counts = np.bincount(y)
    if method == "nested" and counts.min() < 3:
        raise ValueError("Nested cross-validation needs at least three "
                         "recordings in every class.")

    specs = {k: registry[k] for k in keys}
    candidates = ["majority"] + [k for k in keys if k != "majority"]
    dev = test = None
    if method == "final_test":
        dev, test = split_final_test(X, y)
        Xc, yc = X.iloc[dev].reset_index(drop=True), y[dev]
        progress(f"{len(test)} recordings (20%) set aside as a final test "
                 "set.")
    else:
        Xc, yc = X, y

    results, failed = [], []
    for k in candidates:
        spec = registry[k]
        progress(f"Training {spec.name} ...")
        try:
            r = evaluate(spec, Xc, yc, class_names, validation,
                         REPORT_METRICS)
            results.append(r)
            log.add(f"{spec.name}: balanced accuracy "
                    f"{r.selection_score:.4f}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{spec.name}: {exc}")
            log.add(f"{spec.name} FAILED: {exc}")
    learned = [r for r in results if r.classifier_key != "majority"]
    if not learned:
        raise RuntimeError("No classifier could be trained. "
                           + " ".join(failed))
    learned.sort(key=lambda r: r.selection_score, reverse=True)
    reference = next((r for r in results if r.classifier_key == "majority"),
                     None)
    best = learned[0]
    comparison_text = interpret_comparison(learned)

    final = None
    if method and len(learned) > 1:
        if method == "final_test":
            progress(f"Scoring {best.classifier_name} once on the untouched "
                     "test set ...")
            final = evaluate_on_test(specs[best.classifier_key],
                                     X.iloc[dev], y[dev], X.iloc[test],
                                     y[test], class_names, REPORT_METRICS)
            final.note = (f"{best.classifier_name}, trained on {len(dev)} "
                          f"recordings, tested once on {len(test)} unseen "
                          "recordings")
        else:
            progress("Running nested cross-validation ...")
            final = nested_cv([specs[r.classifier_key] for r in learned], X,
                              y, class_names, validation, REPORT_METRICS,
                              progress=lambda f, w: progress(
                                  f"fold {f}: {w} selected"))
        log.add("Final: " + ", ".join(f"{METRICS[m]}={v:.4f}"
                                      for m, v in final.metrics.items())
                + f" ({final.note})")
    else:
        method = None
    shown = final if final is not None else best
    eval_method = method or ("holdout" if validation == "holdout"
                             else "cross_validation")

    summary = [f"The selected model, {best.classifier_name}, classified "
               f"{shown.metrics['accuracy'] * 100:.1f}% of the recordings "
               "correctly"
               + (" on recordings it had not seen during the comparison"
                  if final is not None else "")
               + f" (balanced accuracy {shown.selection_score * 100:.1f}%; "
               f"guessing the most frequent class would give "
               f"{100 / len(class_names):.1f}% balanced accuracy)."]
    if comparison_text:
        summary.append(comparison_text)

    # ---- outputs ------------------------------------------------------- #
    out = results_folder(out_root, source_stem)
    table = list(learned) + ([reference] if reference else [])
    if final is not None:
        label_ = ("FINAL - untouched 20% test set" if method == "final_test"
                  else "FINAL - nested cross-validation")
        table = [dataclasses.replace(
            final, classifier_name=f"{label_} ({best.classifier_name})")
        ] + table
    rep.save_summary_csv(table, os.path.join(out, "summary.csv"))
    rep.save_excel(table, os.path.join(out, "results.xlsx"))
    rows = (recs.file_rows[test] if method == "final_test"
            else recs.file_rows)
    pd.DataFrame({"row_in_file": rows,
                  "actual": [class_names[int(v)] for v in shown.y_true],
                  "predicted": [class_names[int(v)] for v in shown.y_pred]}
                 ).to_csv(os.path.join(out, "predictions.csv"), index=False)
    progress(f"Training the final {best.classifier_name} on all {n} "
             "recordings ...")
    model = fit_final_model(specs[best.classifier_key], X, y)
    save_problem = save_model(dict(model=model, class_names=class_names,
                                   columns=list(X.columns)),
                              os.path.join(out, "trained_model.pkl"))
    if save_problem:
        log.add(save_problem)

    fig_files: Dict[str, Dict[str, str]] = {}
    if settings.figures:
        progress("Drawing figures ...")
        fm = SignalFigures(os.path.join(out, "figures"),
                           class_names=class_names, **style)
        jobs = {
            "signals_by_class": lambda: fm.signals_by_class(X, y),
            "class_distribution": lambda: fm.class_distribution(y),
            "comparison": lambda: fm.comparison(
                learned, best.classifier_name,
                final.selection_score if final is not None else None,
                "Final score") if len(learned) > 1 else None,
            "confusion_matrix": lambda: fm.confusion(shown),
            "roc_curve": lambda: fm.roc(shown),
        }
        for key, job in jobs.items():
            if chosen is not None and key not in chosen:
                continue
            try:
                job()
            except Exception as exc:  # noqa: BLE001
                log.add(f"Figure {key} FAILED: {exc}")
        fig_files = {k: {e: os.path.relpath(p, out) for e, p in v.items()}
                     for k, v in fm.files.items()}

    progress("Writing the report ...")
    warnings = list(failed) + ([save_problem] if save_problem else [])
    try:
        tex_text = _build_tex(source_name, dataset_citation, recs, n, L,
                              class_names, y, registry, keys, learned,
                              reference, best, final, shown, eval_method,
                              validation, dev, test, summary, fig_files)
        # citations.txt lists exactly the references cited in the report.
        with open(os.path.join(out, "citations.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(refs.citation_lines(
                tex_text, _entries(dataset_citation), refs.TOPICS)) + "\n")
        msg = compile_tex(tex_text, out)[2]
        log.add("Report: " + msg)
    except Exception as exc:  # noqa: BLE001
        log.add(f"Report FAILED: {exc}")
        warnings.append(f"Report FAILED: {exc}")
    write_run_record(out, "signal classification", source_name, label,
                     settings, {"models": keys, "validation": validation,
                                "final_evaluation": eval_method,
                                "recordings": n, "values_per_recording": L,
                                "signal_columns": [str(X.columns[0]),
                                                   str(X.columns[-1])],
                                "classes": list(class_names)})
    log.save(os.path.join(out, "log.txt"))
    return dict(out=out, results=learned, reference=reference, best=best,
                final=final, shown=shown, method=eval_method,
                validation=validation, summary=summary, notes=recs.notes,
                warnings=warnings, class_names=class_names, n=n, length=L,
                dev_rows=len(dev) if dev is not None else None,
                test_rows=len(test) if test is not None
                else len(shown.y_true), figures=fig_files)


def _val(m, v):
    return f"{v * 100:.2f}\\%" if m in PERCENT else f"{v:.4f}"


def _table(rows):
    head = "Model & " + " & ".join(METRICS[m] for m in REPORT_METRICS)
    lines = [" & ".join([tex(r.classifier_name)] + [
        _val(m, r.metrics[m]) if m in r.metrics else "--"
        for m in REPORT_METRICS]) + "\\\\" for r in rows]
    return ("\\begin{tabular}{l" + "r" * len(REPORT_METRICS) + "}\n\\toprule\n"
            + head + "\\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")


def _build_tex(source_name, dataset_citation, recs, n, L, class_names, y,
               registry, keys, learned, reference, best, final, shown,
               method, validation, dev, test, summary, figs) -> str:
    def fig(key, width="0.7"):
        f = figs.get(key)
        return _figure(f.get("pdf") or f["png"], tex(CAPTIONS[key]), width) \
            if f else ""
    named = [tex(registry[k].name)
             + cite(refs.SIGNAL_MODEL_REFERENCES.get(k, [])) for k in keys]
    deep = [registry[k].name.replace("Deep: ", "") for k in keys
            if registry[k].family == "deep"]
    torch_used = bool(deep) or "rocket" in keys
    src = " \\cite{dataset}" if dataset_citation else ""
    p = [f"Signal classification was carried out with EasyResearch "
         f"(version {tex(__version__)}) \\cite{{software}}, using the "
         "evaluation procedure of EasyClassifier \\cite{easyclassifier}, "
         "scikit-learn \\cite{sklearn}, NumPy \\cite{numpy}, pandas "
         "\\cite{pandas}" + (" and PyTorch \\cite{pytorch}" if torch_used
                              else "") + ".",
         f"The data ({tex(source_name)}){src} contained {n} recordings of {L} "
         "values each, in the classes " + tex(_join([
             f"{c} ({int((y == i).sum())})"
             for i, c in enumerate(class_names)])) + "."]
    p += [tex(x) for x in recs.notes]
    p.append("The models compared were " + _join(named)
             + ", with their default settings and fixed random seeds, and "
             "the reference rule of always predicting the most frequent "
             "class.")
    if "knn" in keys:
        p.append("For KNN each recording was scaled to the range 0--1 and "
                 "the Hassanat distance \\cite{hassanat2022,hassanat2014} "
                 "was used with k = 5.")
    if deep:
        p.append("The deep networks (" + tex(_join(deep)) + ") received each "
                 "recording standardised to mean 0 and standard deviation 1 "
                 "and were trained with Adam \\cite{kingma2015} (learning "
                 "rate 0.001, batches of "
                 "16, cross-entropy) for at most 150 epochs, stopping after "
                 "20 epochs without improvement on a stratified 10\\% of the "
                 "training recordings.")
    p.append("Models were compared by balanced accuracy"
             + cite(refs.METRIC_REFERENCES["balanced_accuracy"]) + " using "
             + ("a single 80/20 split" if validation == "holdout"
                else "stratified cross-validation"
                + cite(refs.VALIDATION_REFERENCES.get(validation,
                                                      ["kohavi1995"])))
             + ".")
    if method == "nested":
        p.append("The complete comparison was evaluated with nested 5-fold "
                 "cross-validation to obtain an unbiased final score"
                 + cite(refs.SELECTION_REFERENCES["nested"]) + ".")
    elif method == "final_test":
        p.append(f"Before the comparison, a stratified 20\\% of the "
                 f"recordings ({len(test)}) was set aside; the selected model "
                 f"was trained on the other {len(dev)} and tested once on "
                 "them" + cite(refs.SELECTION_REFERENCES["final_test"]) + ".")
    p.append("The measures reported are " + _join([
        tex(METRICS[m]) + cite(refs.METRIC_REFERENCES.get(m, []))
        for m in REPORT_METRICS if m in METRICS]) + ".")
    if figs:
        p.append("Figures were drawn with Matplotlib \\cite{matplotlib}"
                 + (", including ROC curves"
                    + cite(refs.FIGURE_REFERENCES["roc_curve"])
                    if "roc_curve" in figs else "") + ".")
    L_ = [r"""\documentclass[11pt,a4paper]{article}
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
    L_.append("\\begin{center}\n{\\LARGE Signal classification report}"
              f"\\\\[4pt]\n{{\\large {tex(source_name)}}}\\\\[2pt]\n"
              f"{_dt.date.today().strftime('%d %B %Y')}\n\\end{{center}}\n")
    L_.append("\\section{Methods (ready to adapt for a paper)}\n"
              + " ".join(p) + "\n\n")
    L_.append("\\section{Results}\n")
    L_.append(f"Selected model: \\textbf{{{tex(best.classifier_name)}}}"
              + (". These are the numbers to report:" if final is not None
                 else ".") + "\n\n")
    L_.append("\\begin{table}[H]\n\\centering\n\\scriptsize\n"
              + _table([shown] + ([reference] if reference else []))
              + "\\caption{Final result" + (f" ({tex(shown.note)})"
                                            if shown.note else "")
              + " and the most-frequent-class reference.}\n\\end{table}\n")
    L_ += [tex(s) + "\n\n" for s in summary]
    L_.append(fig("confusion_matrix", "0.55") + fig("roc_curve", "0.55"))
    if len(learned) > 1:
        L_.append("\\subsection*{Comparison of models}\nUsed only to choose "
                  "the model; slightly optimistic.\n\n"
                  "\\begin{table}[H]\n\\centering\n\\scriptsize\n"
                  + _table(learned) + "\\caption{Comparison, ranked by "
                  "balanced accuracy.}\n\\end{table}\n" + fig("comparison",
                                                             "0.8"))
    L_.append("\\subsection*{The recordings}\n" + fig("signals_by_class",
                                                       "0.95")
              + fig("class_distribution", "0.5"))
    # References: every reference cited above, in order of citation.
    body = "".join(L_)
    return (body + refs.bibliography(body, _entries(dataset_citation),
                                     escape=tex) + "\\end{document}\n")


def _entries(dataset_citation: str) -> Dict[str, str]:
    """Report-specific bibliography entries: the software and the dataset."""
    entries = {"software": CITATION, "easyclassifier": EASYCLASSIFIER_CITATION}
    if dataset_citation:
        entries["dataset"] = dataset_citation
    return entries
