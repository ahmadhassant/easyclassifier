"""Command line for the EasyResearch cores (no programming needed beyond one
command). Examples::

    easyresearch forecast sales.csv --value Sales --horizon 12
    easyresearch forecast --demo
    easyresearch regress houses.xlsx --target Price
    easyresearch signals ECG200_TRAIN.tsv --label label
    easyresearch models forecast           # list the available models
    easyresearch regress --demo --figures residuals,comparison --figure-format png+pdf

Results are written to a new folder inside ./Results (or --out).
A terminal wizard and the desktop/web interfaces will drive the same cores.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__


def _models(arg, registry):
    if not arg:
        return None
    if arg == "all":
        return [k for k, s in registry.items() if getattr(s, "available", True)]
    return [m.strip() for m in arg.split(",") if m.strip()]


def _figure_settings(st, a):
    """Figure colours, file formats and the figures to draw (default: the
    task's usual set)."""
    st.figure_theme, st.figure_format = a.figure_theme, a.figure_format
    if a.figures:
        st.figure_keys = [k.strip() for k in a.figures.split(",") if k.strip()]
    return st


def main(argv=None) -> int:
    from easyclassifier.figures import DEFAULT_THEME, FORMATS, THEMES
    p = argparse.ArgumentParser(prog="easyresearch",
                                description="Machine learning without "
                                "programming - analysis cores.")
    p.add_argument("--version", action="version",
                   version=f"EasyResearch {__version__}")
    sub = p.add_subparsers(dest="task", required=True)

    for name, help_ in (("forecast", "forecast one time series"),
                        ("regress", "predict a number from a table"),
                        ("signals", "classify recordings (ECG, EEG, ...)")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("data", nargs="?", help="CSV, Excel or UCR .tsv file")
        s.add_argument("--demo", action="store_true",
                       help="use the built-in example data")
        s.add_argument("--models", help="comma-separated model ids, or 'all'"
                       " (default: the fast set)")
        s.add_argument("--out", default=".", help="folder for Results/")
        s.add_argument("--no-figures", action="store_true")
        s.add_argument("--figures", help="comma-separated figure names "
                       "(default: the usual set; see the report's list)")
        s.add_argument("--figure-theme", default=DEFAULT_THEME,
                       choices=list(THEMES), help="figure colours")
        s.add_argument("--figure-format", default="png", choices=list(FORMATS),
                       help="figure files: png, png+pdf or png+svg")
        if name == "forecast":
            s.add_argument("--value", help="column to forecast")
            s.add_argument("--time", help="date/time column (auto-detected)")
            s.add_argument("--horizon", default="auto",
                           help="steps ahead (default: automatic)")
            s.add_argument("--metric", default="mase",
                           choices=["mase", "rmse", "mae", "mhsp"],
                           help="measure used to choose the model")
        elif name == "regress":
            s.add_argument("--target", help="numeric column to predict")
        else:
            s.add_argument("--label", help="column holding the class")
    lm = sub.add_parser("models", help="list the models of a task")
    lm.add_argument("which", choices=["forecast", "regress", "signals"])
    a = p.parse_args(argv)

    if a.task == "models":
        if a.which == "forecast":
            from .timeseries.models import build_registry
        elif a.which == "regress":
            from .regression.models import build_registry
        else:
            from .timeseries.signal_models import build_registry
        for k, s in build_registry().items():
            mark = "*" if getattr(s, "default_selected", False) else " "
            print(f" {mark} {k:20s} {s.name}")
        print("\n * = in the default (fast) set")
        return 0

    if not a.demo and not a.data:
        p.error("give a data file or --demo")
    out = os.path.abspath(a.out)
    try:
        if a.task == "forecast":
            from .timeseries import ForecastSettings, run_forecast
            from .timeseries import data as td
            from .timeseries.models import build_registry
            df = td.load_demo() if a.demo else _load(a.data)
            value = a.value or ("CO2_ppm" if a.demo else td.suggest_value(df))
            st = ForecastSettings(time_column=a.time, figures=not a.no_figures,
                                  selection_metric=a.metric,
                                  horizon=a.horizon if a.horizon == "auto"
                                  else int(a.horizon))
            st.models = _models(a.models, build_registry()) or st.models
            _figure_settings(st, a)
            r = run_forecast(df, value, st, out,
                             td.DEMO_NAME if a.demo else Path(a.data).name,
                             "demo_co2" if a.demo else Path(a.data).stem,
                             td.DEMO_CITATION if a.demo else "")
            head = f"Selected: {r['best'].name}"
        elif a.task == "regress":
            from .regression import Settings, run_analysis
            from .regression import data as rd
            from .regression.models import build_registry
            df = rd.load_demo() if a.demo else _load(a.data)
            target = a.target or rd.suggest_target(df)
            reg = build_registry()
            models = _models(a.models, reg) or [
                k for k, s in reg.items() if s.default_selected]
            st = _figure_settings(Settings(models=models,
                                           figures=not a.no_figures), a)
            r = run_analysis(df, target, st, out, rd.DEMO_NAME if a.demo else
                             Path(a.data).name, "demo_diabetes" if a.demo
                             else Path(a.data).stem,
                             rd.DEMO_CITATION if a.demo else "")
            head = f"Selected: {r['best'].name}"
        else:
            from .timeseries import signal_data as sd
            from .timeseries.classify import (SignalSettings,
                                              run_signal_classification)
            from .timeseries.signal_models import build_registry
            df = sd.load_demo() if a.demo else sd.load_table(a.data)
            label = a.label or sd.suggest_label(df)
            st = SignalSettings(figures=not a.no_figures)
            st.models = _models(a.models, build_registry()) or st.models
            _figure_settings(st, a)
            r = run_signal_classification(
                df, label, st, out, sd.DEMO_NAME if a.demo else
                Path(a.data).name, "demo_heartbeats" if a.demo
                else Path(a.data).stem)
            head = f"Selected: {r['best'].classifier_name}"
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"\nCould not run the analysis: {exc}", file=sys.stderr)
        return 1
    print("\n" + head)
    for line in r["summary"]:
        print("  " + line)
    print(f"\nAll results: {r['out']}")
    return 0


def _load(path):
    from easyclassifier.dataset import clean_path, load_csv
    return load_csv(clean_path(path))


if __name__ == "__main__":
    raise SystemExit(main())
