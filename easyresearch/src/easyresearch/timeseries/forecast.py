"""A complete forecasting analysis of one series, from table to saved
results. No user interaction: every choice arrives in ``ForecastSettings``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from easyclassifier.logbook import LogBook

from .. import __version__
from ..common.output import (figure_style, results_folder, save_model,
                             write_run_record)
from . import data as td
from . import evaluation as ev
from .deep import device
from .figures import FIGURES, ForecastFigures, band
from .models import (REFERENCE_KEYS, build_registry, spec_unusable,
                     window_length)
from .report import ForecastReport, write_report
from .report import citation_lines as report_citation_lines


@dataclass
class ForecastSettings:
    models: List[str] = field(default_factory=lambda: [
        k for k, s in build_registry().items() if s.default_selected])
    horizon: Union[str, int] = "auto"
    selection_metric: str = "mase"       # mase | rmse | mae | mhsp
    time_column: Optional[str] = None
    figures: bool = True
    figure_theme: str = "colorblind"     # colorblind | greyscale | high_contrast
    figure_format: str = "png"           # png | png+pdf | png+svg
    figure_keys: Optional[List[str]] = None   # None = all five figures


def say(text: str) -> None:
    print(text, flush=True)


def horizon_fits(n: int, horizon: int, season: int) -> bool:
    """Whether a series of n values leaves enough history for an honest
    evaluation (comparison origins plus a final period) at this horizon."""
    try:
        ev.plan_split(n, horizon, window_length(int(0.6 * n), season, horizon),
                      season)
        return True
    except ValueError:
        return False


def shortest_series(season: int, horizon: int = 1) -> int:
    """About how many values are needed to forecast `horizon` steps."""
    n = td.MIN_POINTS
    while not horizon_fits(n, horizon, season) and n < 5000:
        n += 1
    return n


def choose_horizon(series) -> int:
    """The automatic horizon: the usual one for the time unit (a year of
    months, a week of days ...), shortened when the series is too short to
    evaluate it honestly."""
    n, m = len(series.values), series.season
    usual = td.default_horizon(series)
    for h in range(usual, 0, -1):
        if horizon_fits(n, h, m):
            if h < usual:
                series.notes.append(
                    f"The automatic horizon was shortened from {usual} to {h} "
                    f"step{'s' if h > 1 else ''}: a longer one would leave too "
                    "little history for an honest evaluation of this series.")
            return h
    raise ValueError(
        f"The series is too short for an honest forecast evaluation: it has "
        f"{n} values, and even one step ahead needs about "
        f"{shortest_series(m)}. Add more history.")


def run_forecast(df: pd.DataFrame, value: str, settings: ForecastSettings,
                 out_root: str, source_name: str, source_stem: str,
                 dataset_citation: str = "",
                 progress: Callable[[str], None] = say) -> Dict:
    log = LogBook()
    log.add(f"EasyResearch {__version__} - forecasting")
    log.add(f"Dataset: {source_name}; series: {value}")
    registry = build_registry()
    keys = list(dict.fromkeys(settings.models))
    if any(k not in registry for k in keys):
        raise ValueError("Unknown forecasting model: " + ", ".join(
            k for k in keys if k not in registry))
    if settings.selection_metric not in ev.SELECTION_METRICS:
        raise ValueError("Unknown selection measure.")
    metric = settings.selection_metric
    style, chosen = figure_style(settings, FIGURES)   # refused before training

    progress("Preparing the series ...")
    series = td.prepare_series(df, value, settings.time_column)
    y, n, m = series.values, len(series.values), series.season
    for note in series.notes:
        log.add("Data: " + note)
    if settings.horizon == "auto":
        before = len(series.notes)
        H = choose_horizon(series)
        for note in series.notes[before:]:
            log.add("Horizon: " + note)
    else:
        H = int(settings.horizon)
        if not 1 <= H <= n // 5:
            raise ValueError(f"The horizon must be between 1 and {n // 5} "
                             "steps for this series.")
    L = window_length(int(0.6 * n), m, H)
    plan = ev.plan_split(n, H, L, m)
    if plan.overlapping:
        note = ("The series is short, so the forecasts used to compare the "
                "models overlap in time; treat small differences between "
                "models with caution. The final period is unaffected.")
        series.notes.append(note)
        log.add("Plan: " + note)
    log.add(f"{series.label} series, {n} values, season {m}, horizon {H}, "
            f"window {L}; comparison origins {len(plan.dev_origins)}, final "
            f"period {n - plan.dev_end} values ({len(plan.test_origins)} "
            "origins)")

    candidates = [k for k in REFERENCE_KEYS if not spec_unusable(
        registry[k], m)] + [k for k in keys if k not in REFERENCE_KEYS]
    skipped: List[str] = []
    results: List[ev.Result] = []
    for k in candidates:
        spec = registry[k]
        why = spec_unusable(spec, m)
        if why:
            skipped.append(f"{spec.name}: {why}")
            continue
        progress(f"Training {spec.name} at {len(plan.dev_origins)} "
                 "comparison origins ...")
        try:
            r = ev.rolling(spec, lambda s=spec: s.make(L, H, m), y,
                           plan.dev_origins, H, m)
            results.append(r)
            log.add(f"{spec.name}: " + ", ".join(
                f"{ev.METRICS[k2]}={v:.4f}" for k2, v in r.metrics.items()))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{spec.name}: {exc}")
            log.add(f"{spec.name} FAILED: {exc}")
    learned = [r for r in results if r.key not in REFERENCE_KEYS]
    if not learned:
        raise RuntimeError("No forecasting model could be trained. "
                           + " ".join(skipped))
    results.sort(key=lambda r: ev.selection_value(r, metric), reverse=True)
    best = results[0]
    comparison_text = ev.interpret_comparison(results, metric)
    log.add(f"Selected: {best.name} (best {ev.SELECTION_METRICS[metric]}). "
            + comparison_text)

    progress(f"Scoring {best.name} in the final period "
             f"({len(plan.test_origins)} forecast origins) ...")
    spec = registry[best.key]
    final = ev.rolling(spec, lambda: spec.make(L, H, m), y, plan.test_origins,
                       H, m)
    refs_final = {k: ev.rolling(registry[k], lambda k=k: registry[k].make(
        L, H, m), y, plan.test_origins, H, m)
        for k in REFERENCE_KEYS if not spec_unusable(registry[k], m)}
    coverage = ev.interval_coverage(best, final)
    log.add("Final period: " + ", ".join(
        f"{ev.METRICS[k2]}={v:.4f}" for k2, v in final.metrics.items())
        + f"; 80% band coverage {coverage:.2f}")
    summary = ev.interpret(final, refs_final.get("naive"),
                           refs_final.get("seasonal_naive"), series.label
                           .replace("in row order", "").replace("-daily",
                                                                "-day")
                           .strip() or "", H, m, coverage)
    summary = [s.replace("  ", " ") for s in summary]

    progress(f"Training the final {best.name} on all {n} values ...")
    model = spec.make(L, H, m).fit(y)
    future = model.predict(H)
    sd = ev.error_by_step([best, final])
    lo, hi = band(future, sd)

    # ---- outputs -------------------------------------------------------- #
    out = results_folder(out_root, source_stem)
    fig_dir = os.path.join(out, "figures")
    fm = ForecastFigures(fig_dir, value, series.index, **style)
    future_x = fm._future_x(H, series.unit)
    time_name = series.time_column or "position"
    pd.DataFrame({time_name: future_x, "forecast": future,
                  "lower_80": lo, "upper_80": hi}).to_csv(
        os.path.join(out, "forecast.csv"), index=False)

    rows = []
    for i, o in enumerate(final.origins):
        for step in range(H):
            if np.isnan(final.actual[i, step]):
                continue
            rows.append({"forecast_origin": series.index[o],
                         time_name: series.index[o + step],
                         "steps_ahead": step + 1,
                         "actual": final.actual[i, step],
                         "forecast": final.forecast[i, step],
                         "error": final.actual[i, step]
                         - final.forecast[i, step]})
    final_df = pd.DataFrame(rows)
    final_df.to_csv(os.path.join(out, "final_period_forecasts.csv"),
                    index=False)

    def row(name, r, family):
        return {"Model": name, "Type": family, **{
            ev.METRICS[k2]: round(v, 6) for k2, v in r.metrics.items()}}
    table = [row(f"FINAL PERIOD - {best.name}", final, "selected")]
    table += [row(f"FINAL PERIOD - {registry[k].name}", r, "reference")
              for k, r in refs_final.items()]
    table += [row(r.name, r, registry[r.key].family) for r in results]
    summary_df = pd.DataFrame(table)
    summary_df.to_csv(os.path.join(out, "summary.csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(out, "results.xlsx")) as xl:
            summary_df.to_excel(xl, sheet_name="Results", index=False)
            pd.read_csv(os.path.join(out, "forecast.csv")).to_excel(
                xl, sheet_name="Forecast", index=False)
            final_df.to_excel(xl, sheet_name="Final period", index=False)
    except Exception as exc:  # noqa: BLE001
        log.add(f"Excel FAILED: {exc}")
    save_problem = save_model(model, os.path.join(out, "trained_model.pkl"))
    log.add(save_problem or f"Saved model: {best.name}, trained on all {n} values; "
            "predict(h) gives the next h values (h <= horizon)")

    fig_files: Dict[str, Dict[str, str]] = {}
    if settings.figures:
        progress("Drawing figures ...")
        refs = set(REFERENCE_KEYS)
        jobs = {
            "forecast": lambda: fm.forecast(y, plan.dev_end, future, lo, hi,
                                            future_x, best.name),
            "final_period": lambda: fm.final_period(y, final, plan.dev_end,
                                                    H),
            "comparison": lambda: fm.comparison(
                results, best.name, metric, final.metrics[metric], refs),
            "error_by_step": lambda: fm.error_by_step(_step_curves(
                best, final, results, refs_final, registry)),
            "decomposition": lambda: fm.decomposition(y, m),
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
    warnings = list(skipped) + ([save_problem] if save_problem else [])
    try:
        rep = ForecastReport(
            version=__version__, dataset=source_name,
            dataset_citation=dataset_citation, series=series, horizon=H,
            window=L, plan=plan, results=results, best=best, final=final,
            refs_final=refs_final, metric=metric, summary=summary,
            comparison_text=comparison_text, coverage=coverage,
            figures=fig_files, skipped=skipped, registry=registry,
            rows_loaded=len(df), deep_device=str(device()))
        # citations.txt lists exactly the references cited in the report.
        with open(os.path.join(out, "citations.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(report_citation_lines(rep)) + "\n")
        msg = write_report(rep, out)
        log.add("Report: " + msg)
    except Exception as exc:  # noqa: BLE001
        log.add(f"Report FAILED: {exc}")
        warnings.append(f"Report FAILED: {exc}")
    write_run_record(out, "forecasting", source_name, value, settings,
                     {"time_column": series.time_column, "unit": series.label,
                      "season": m, "horizon": H, "window": L,
                      "selection_metric": metric,
                      "comparison_origins": len(plan.dev_origins),
                      "final_period_values": n - plan.dev_end,
                      "values_used": n})
    log.save(os.path.join(out, "log.txt"))

    return dict(out=out, series=series, horizon=H, window=L, plan=plan,
                results=results, best=best, final=final,
                refs_final=refs_final, metric=metric, summary=summary,
                comparison_text=comparison_text, coverage=coverage,
                future=future, lower=lo, upper=hi, future_index=future_x,
                figures=fig_files, warnings=warnings, notes=series.notes)


def _step_curves(best, final, results, refs_final, registry):
    ref_key = "seasonal_naive" if "seasonal_naive" in refs_final else "naive"
    ref_dev = next(r for r in results if r.key == ref_key)
    name = registry[ref_key].name.split(" (")[0].lower()
    return {f"{best.name} - comparison": ev.error_by_step([best]),
            f"{best.name} - final period": ev.error_by_step([final]),
            f"{name} rule - comparison": ev.error_by_step([ref_dev]),
            f"{name} rule - final period": ev.error_by_step(
                [refs_final[ref_key]])}
