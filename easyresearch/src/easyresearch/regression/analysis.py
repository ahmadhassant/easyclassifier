"""A complete regression analysis, from a loaded table to saved results.

No user interaction: every choice arrives in ``Settings``. Messages are
printed (the desktop worker turns them into progress events).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from easyclassifier import preprocessing as pp
from easyclassifier import recommend as rec
from easyclassifier import target as tg
from easyclassifier.distances import hassanat_form
from easyclassifier.logbook import LogBook

from .. import CITATION, __version__
from ..common.hassanat import HASSANAT_CITATIONS
from ..common.output import (figure_style, results_folder, save_model,
                             write_run_record)
from . import data as rd
from . import evaluation as ev
from .figures import FIGURES, RegressionFigures, default_figures
from .models import build_registry
from .report import ReportContext, write_and_compile
from .report import citation_lines as report_citation_lines


@dataclass
class Settings:
    models: List[str]
    validation: str = "auto"          # auto | kfold5 | kfold10 | holdout
    selection: str = "auto"           # auto | nested | final_test
    figures: bool = True
    figure_theme: str = "colorblind"  # colorblind | greyscale | high_contrast
    figure_format: str = "png"        # png | png+pdf | png+svg
    figure_keys: Optional[List[str]] = None   # None = the usual set


@dataclass
class Prepared:
    X: pd.DataFrame
    y: np.ndarray
    file_rows: np.ndarray             # spreadsheet row of each kept row
    cfg: pp.PrepConfig
    notes: List[str] = field(default_factory=list)
    left_out: Dict[str, str] = field(default_factory=dict)
    impute: Optional[str] = None
    encoding: Optional[str] = None


def say(text: str) -> None:
    print(text, flush=True)


def prepare(df: pd.DataFrame, target: str, log: LogBook) -> Prepared:
    """Row-level cleaning (nothing learned) and the preprocessing plan."""
    check = rd.check_target(df[target])
    if not check.suitable:
        raise ValueError(f"'{target}' cannot be predicted with regression. "
                         + check.note)
    df = df.copy()
    df[target] = rd.numeric_values(df[target])
    df["__file_row__"] = np.arange(len(df)) + 2     # header is row 1
    notes: List[str] = []
    if check.note:
        notes.append(check.note)

    # Columns that cannot help: identifiers and single-value columns.
    left_out = {}
    for c in df.columns:
        if c in (target, "__file_row__"):
            continue
        kind = tg.describe_column(df[c]).kind
        if kind == tg.ID_LIKE:
            left_out[str(c)] = "different in every row (ID or name)"
        elif kind in (tg.CONSTANT, tg.EMPTY):
            left_out[str(c)] = "only one value"
    if left_out:
        df = df.drop(columns=list(left_out))
        log.add("Left out columns: " + ", ".join(
            f"{c} ({w})" for c, w in left_out.items()))
    if df.shape[1] < 3:
        raise ValueError("No usable predictor columns remain after leaving "
                         "out identifiers and single-value columns.")

    n = int(df[target].isna().sum())
    if n:
        df = df[df[target].notna()]
        notes.append(f"{n} row(s) with no value for {target} were removed.")
        log.add(f"Removed {n} rows missing the target")

    cfg = pp.PrepConfig()
    impute = None
    preds = df.drop(columns=[target, "__file_row__"])
    if preds.isna().any().any():
        frac = preds.isna().sum().sum() / max(1, preds.size)
        if frac >= 0.2:
            before = len(df)
            df = df[preds.notna().all(axis=1)]
            notes.append(f"{before - len(df)} row(s) with missing values "
                         "were removed (more than 20% of cells were empty).")
        else:
            cfg.impute = impute = "median"
            log.add("Missing values: median / most common value, learned "
                    "on training rows")

    body = df.drop(columns=["__file_row__"])
    dup = body.duplicated()
    if dup.any():
        if pp.duplicates_expected_by_chance(body):
            notes.append(f"{int(dup.sum())} identical rows were kept, because "
                         "with so few possible value combinations different "
                         "cases are expected to coincide.")
        else:
            df = df[~dup.to_numpy()]
            notes.append(f"{int(dup.sum())} duplicate row(s) were removed.")
    log.add("Data notes: " + " ".join(notes) if notes else "No data notes")

    if len(df) < rd.MIN_ROWS:
        raise ValueError(f"Regression needs at least {rd.MIN_ROWS} complete "
                         f"rows; {len(df)} remain after cleaning.")
    file_rows = df["__file_row__"].to_numpy()
    y = df[target].to_numpy(dtype=float)
    X = pp.cast_categoricals(df.drop(columns=[target, "__file_row__"])
                             .reset_index(drop=True))
    encoding = None
    if pp.categorical_columns(X):
        cfg.encoding = encoding = rec.encoding_method(
            df.drop(columns=["__file_row__"]), target)
        log.add(f"Encoding: {encoding}")
    return Prepared(X, y, file_rows, cfg, notes, left_out, impute, encoding)


def _pipeline_spec(spec, X, cfg):
    return replace(spec, factory=lambda: pp.build_pipeline(
        X, cfg, spec.factory(), spec.scaling))


def run_analysis(df: pd.DataFrame, target: str, settings: Settings,
                 out_root: str, source_name: str, source_stem: str,
                 dataset_citation: str = "",
                 progress: Callable[[str], None] = say) -> Dict:
    log = LogBook()
    log.add(f"EasyResearch {__version__} - regression")
    log.add(f"Dataset: {source_name}; target: {target}")
    registry = build_registry()
    keys = list(dict.fromkeys(settings.models))
    if not keys or any(k not in registry or not registry[k].available
                       for k in keys):
        raise ValueError("Select at least one available regression model.")
    style, chosen = figure_style(settings, FIGURES)   # refused before training

    progress("Preparing the data ...")
    rows_loaded, cols_loaded = len(df), df.shape[1]
    prep = prepare(df, target, log)
    X, y = prep.X, prep.y
    n = len(y)
    validation = "kfold5" if settings.validation == "auto" else \
        settings.validation
    if validation not in ev.VALIDATION:
        raise ValueError("Unknown validation setting.")
    method = None
    if len(keys) > 1:
        method = (ev.recommend_selection(n) if settings.selection == "auto"
                  else settings.selection)
        if method not in ("nested", "final_test"):
            raise ValueError("Unknown final evaluation setting.")
    log.add(f"Rows used: {n}; predictors: {X.shape[1]}; validation: "
            f"{validation}; final evaluation: {method or 'none (one model)'}")

    specs = {k: _pipeline_spec(registry[k], X, prep.cfg) for k in keys}
    hassanat = None
    if "knn" in keys:
        Xt = pp.describe_transformed(X, prep.cfg, "minmax")
        hassanat = hassanat_form(Xt)
        hassanat["text"] = (
            "columns were scaled to 0-1 on the training rows, so values are "
            "non-negative and the standard form of the formula applies; a "
            "test value outside the training range can become slightly "
            "negative, and for such values the signed form "
            "1 - (1+min+|min|)/(1+max+|min|) is used automatically.")
        log.add("Hassanat distance: " + hassanat["text"])

    dev = test = None
    if method == "final_test":
        dev, test = ev.split_final_test(n)
        Xc, yc, rows_c = X.iloc[dev], y[dev], dev
        progress(f"{len(test)} rows (20%) set aside as a final test set.")
    else:
        Xc, yc, rows_c = X, y, np.arange(n)

    results, failed = [], []
    for k in keys:
        progress(f"Training {registry[k].name} ...")
        try:
            r = ev.evaluate(specs[k], Xc.reset_index(drop=True), yc,
                            validation, rows_c)
            results.append(r)
            log.add(f"{r.name}: " + ", ".join(
                f"{ev.METRICS[m]}={v:.4f}" for m, v in r.metrics.items()))
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{registry[k].name} FAILED: {exc}")
            log.add(f"{registry[k].name} FAILED: {exc}")
            progress(f"{registry[k].name} could not be trained: {exc}")
    if not results:
        raise RuntimeError("No model could be trained. " + " ".join(failed))
    results.sort(key=lambda r: r.selection_score, reverse=True)
    best = results[0]
    comparison_text = ev.interpret_comparison(results)
    log.add(f"Selected: {best.name} (highest R²). {comparison_text}")

    final = None
    if method and len(results) > 1:
        ok = [registry[r.key] for r in results]
        if method == "final_test":
            progress(f"Scoring {best.name} once on the untouched test set ...")
            final = ev.evaluate_on_test(specs[best.key], X, y, dev, test)
        else:
            progress("Running nested cross-validation (the whole comparison "
                     "is repeated inside each of 5 folds) ...")
            final = ev.nested_cv([specs[s.key] for s in ok], X, y, validation,
                                 progress=lambda f, w: progress(
                                     f"fold {f}: {w} selected"))
        log.add("Final estimate: " + ", ".join(
            f"{ev.METRICS[m]}={v:.4f}" for m, v in final.metrics.items())
            + f" ({final.note})")
    elif method:
        method = None       # only one model could be trained
    shown = final if final is not None else best
    eval_method = method or ("holdout" if validation == "holdout"
                             else "cross_validation")
    baseline = ev.baseline_like(shown, X, y, eval_method, validation,
                                dev, test)
    summary = ev.interpret_final(shown, baseline, target)
    for s in summary:
        log.add("Summary: " + s)

    # ---- output folder ------------------------------------------------- #
    out = results_folder(out_root, source_stem)
    fig_dir = os.path.join(out, "figures")

    importance = None
    if settings.figures:
        progress("Measuring which columns matter ...")
        try:
            importance = ev.importance(specs[best.key], X, y, validation,
                                       dev, test)
            importance.round(6).to_csv(
                os.path.join(out, "feature_importance.csv"), index=False)
            log.add("Permutation importance (top 5): " + ", ".join(
                f"{r.column}={r.importance:.4f}"
                for r in importance.head(5).itertuples()))
        except Exception as exc:  # noqa: BLE001
            log.add(f"Importance FAILED: {exc}")
            importance = None

    fig_files: Dict[str, Dict[str, str]] = {}
    if settings.figures:
        progress("Drawing figures ...")
        fm = RegressionFigures(fig_dir, target, **style)
        final_label = ("Final score (untouched test set)"
                       if method == "final_test" else "Final score (nested CV)")
        jobs = {
            "target_distribution": lambda: fm.target_distribution(y),
            "missing_values": lambda: fm.missing_values(df),
            "comparison": lambda: (fm.comparison(
                results, best.name,
                final.selection_score if final is not None else None,
                final_label) if len(results) > 1 else None),
            "predicted_vs_actual": lambda: fm.predicted_vs_actual(shown),
            "residuals": lambda: fm.residuals(shown),
            "feature_importance": lambda: (fm.importance(importance)
                                           if importance is not None
                                           else None),
        }
        for key in (chosen if chosen is not None
                    else default_figures(bool(df.isna().any().any()))):
            try:
                jobs[key]()
            except Exception as exc:  # noqa: BLE001
                log.add(f"Figure {key} FAILED: {exc}")
        fig_files = {k: {e: os.path.relpath(p, out) for e, p in v.items()}
                     for k, v in fm.files.items()}

    # ---- tables, predictions, model ------------------------------------ #
    table_rows = ([replace(shown, name=(
        "FINAL - untouched 20% test set" if method == "final_test"
        else "FINAL - nested cross-validation") + f" ({best.name})")]
        if final is not None else []) + results + [baseline]
    summary_df = pd.DataFrame([{"Model": r.name, **{
        ev.METRICS[m]: round(r.metrics[m], 6) for m in ev.METRICS
        if m in r.metrics}} for r in table_rows])
    summary_df.to_csv(os.path.join(out, "summary.csv"), index=False)
    pred_df = pd.DataFrame({
        "row_in_file": prep.file_rows[shown.rows],
        "actual": shown.y_true, "predicted": shown.y_pred,
        "error": shown.y_true - shown.y_pred})
    pred_df.to_csv(os.path.join(out, "predictions.csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(out, "results.xlsx")) as xl:
            summary_df.to_excel(xl, sheet_name="Results", index=False)
            pred_df.to_excel(xl, sheet_name="Predictions", index=False)
            if importance is not None:
                importance.to_excel(xl, sheet_name="Column importance",
                                    index=False)
    except Exception as exc:  # noqa: BLE001
        log.add(f"Excel FAILED: {exc}")

    progress(f"Training the final {best.name} on all {n} rows ...")
    model = specs[best.key].factory().fit(X, y)
    save_problem = save_model(model, os.path.join(out, "trained_model.pkl"))
    log.add(save_problem or f"Saved model: {best.name}, refitted on all {n} rows")

    progress("Writing the report ...")
    names = {k: registry[k].name for k in keys}
    scaled = {}
    for k in keys:
        if registry[k].scaling:
            scaled.setdefault(registry[k].scaling, []).append(names[k])
    ctx = ReportContext(
        software_citation=CITATION, version=__version__,
        dataset=source_name, dataset_citation=dataset_citation,
        rows_loaded=rows_loaded, cols_loaded=cols_loaded, target=target,
        rows_used=n, predictor_columns=[str(c) for c in X.columns],
        left_out=prep.left_out, data_notes=prep.notes, impute=prep.impute,
        encoding=prep.encoding, scaled=scaled,
        models=[names[k] for k in keys], hassanat_used=bool(hassanat),
        hassanat_text=hassanat["text"] if hassanat else "",
        hassanat_citations=HASSANAT_CITATIONS, validation=validation,
        method=eval_method, n_dev=len(dev) if dev is not None else 0,
        n_test=len(test) if test is not None else 0, selected=best.name,
        final=shown, baseline=baseline, results=results, summary=summary,
        comparison_text=comparison_text, figures=fig_files,
        importance=importance,
        unit_target_models=[names[k] for k in keys
                            if k in ("svr", "neural_network")],
        model_keys=list(keys),
        hassanat_signed=bool(hassanat and hassanat["form"] == "signed"))
    warnings = list(failed) + ([save_problem] if save_problem else [])
    # citations.txt lists exactly the references cited in the report.
    cites = report_citation_lines(ctx)
    if hassanat:
        cites += ["", "Hassanat distance, formula form applied: "
                  + hassanat["text"]]
    with open(os.path.join(out, "citations.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(cites) + "\n")
    try:
        _, pdf, msg = write_and_compile(ctx, out)
        log.add("Report: " + msg)
    except Exception as exc:  # noqa: BLE001
        log.add(f"Report FAILED: {exc}")
        warnings.append(f"Report FAILED: {exc}")
    write_run_record(out, "regression", source_name, target, settings,
                     {"models": keys, "validation": validation,
                      "final_evaluation": eval_method, "rows_used": n,
                      "predictors": list(X.columns)})
    log.save(os.path.join(out, "log.txt"))

    return dict(out=out, results=results, final=final, best=best,
                shown=shown, baseline=baseline, method=eval_method,
                summary=summary, comparison_text=comparison_text,
                notes=prep.notes, warnings=warnings, rows_used=n,
                predictors=[str(c) for c in X.columns],
                dev_rows=len(dev) if dev is not None else None,
                test_rows=len(test) if test is not None else len(shown.y_true),
                figures=fig_files, hassanat=hassanat)
