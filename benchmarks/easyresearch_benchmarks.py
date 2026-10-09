"""Benchmarks for the EasyResearch cores: regression, forecasting and signal
classification (classification has its own suite in run_benchmarks.py).

For every dataset the analysis runs as a beginner would run it - the fast
default set of models, automatic settings - and the script records:

* the honest final score (nested cross-validation, an untouched test set or a
  final forecasting period, exactly as reported to the user);
* the run time and the peak memory of the analysis process;
* agreement with an independent reference implementation:
    - regression: the linear model's cross-validated R² against a plain
      scikit-learn pipeline built here on the same folds;
    - forecasting: the seasonal naive rule's final-period error recomputed
      here from the raw series, without EasyResearch code;
    - signals: EasyResearch's ROCKET against aeon's RocketClassifier on the
      official UCR train/test split.

All datasets are public and ship inside pip packages (scikit-learn,
statsmodels, pydataset, aeon), so the benchmark needs no downloads:

    pip install -r requirements-easyresearch.txt
    python easyresearch_benchmarks.py                # everything
    python easyresearch_benchmarks.py --task signals # one task

Results: results/easyresearch/{regression,forecasting,signals}.csv and
results/easyresearch/RESULTS.md. Each dataset runs in its own process, so the
peak memory is that of one analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'easyresearch'


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

def _r(name):
    from pydataset import data
    return data(name).reset_index(drop=True)


def _monthly(name, column, start):
    s = _r(name)
    return pd.DataFrame({'month': pd.period_range(start, periods=len(s), freq='M').strftime('%Y-%m'),
                         column: s[name].to_numpy(dtype=float)})


def _quarterly(name, column, start):
    s = _r(name)
    q = pd.period_range(start, periods=len(s), freq='Q').to_timestamp()
    return pd.DataFrame({'quarter': q.strftime('%Y-%m-%d'), column: s[name].to_numpy(dtype=float)})


def _yearly(values, column, start):
    return pd.DataFrame({'year': np.arange(start, start + len(values)), column: np.asarray(values, float)})


def _diabetes():
    from easyresearch.regression.data import load_demo
    return load_demo()


def _friedman():
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=500, noise=1.0, random_state=2026)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1, 11)])
    df['y'] = y
    return df


def _co2():
    from easyresearch.timeseries.data import load_demo
    return load_demo()


def _sunspots():
    import statsmodels.api as sm
    d = sm.datasets.sunspots.load_pandas().data
    return _yearly(d['SUNACTIVITY'], 'sunspots', int(d['YEAR'].iloc[0]))


def _nile():
    import statsmodels.api as sm
    d = sm.datasets.nile.load_pandas().data
    return _yearly(d['volume'], 'flow', int(d['year'].iloc[0]))


def _ucr(loader):
    def load(split=None):
        import warnings
        import aeon.datasets as ad
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = getattr(ad, loader)
            if split:
                X, y = f(split=split)
                return X[:, 0, :], np.asarray(y).astype(str)
            parts = [f(split=s) for s in ('train', 'test')]
        X = np.vstack([p[0][:, 0, :] for p in parts])
        y = np.concatenate([np.asarray(p[1]).astype(str) for p in parts])
        df = pd.DataFrame(X, columns=[f't{i}' for i in range(1, X.shape[1] + 1)])
        df.insert(0, 'label', y)
        return df
    return load


REGRESSION = [
    ('diabetes', 'Diabetes progression', _diabetes, 'Progression',
     'Efron et al. (2004), Annals of Statistics 32(2); via scikit-learn'),
    ('mtcars', 'Car fuel economy (mtcars)', lambda: _r('mtcars'), 'mpg',
     'Henderson & Velleman (1981), Biometrics 37(2); via pydataset (R datasets)'),
    ('prestige', 'Occupational prestige (Canada)', lambda: _r('Prestige'), 'prestige',
     'Fox & Weisberg (2011), An R Companion to Applied Regression; via pydataset'),
    ('airquality', 'New York air quality (ozone)', lambda: _r('airquality'), 'Ozone',
     'Chambers et al. (1983), Graphical Methods for Data Analysis; via pydataset. Has missing values'),
    ('friedman1', 'Friedman #1 (synthetic, noise sd 1)', _friedman, 'y',
     'Friedman (1991), Annals of Statistics 19(1); generated with scikit-learn, seed 2026'),
]

FORECASTING = [
    ('co2', 'Mauna Loa CO2 (monthly)', _co2, 'CO2_ppm', 'Keeling & Whorf; via statsmodels'),
    ('airpassengers', 'Airline passengers (monthly)', lambda: _monthly('AirPassengers', 'passengers', '1949-01'),
     'passengers', 'Box & Jenkins (1976); via pydataset'),
    ('nottem', 'Nottingham temperature (monthly)', lambda: _monthly('nottem', 'temperature_F', '1920-01'),
     'temperature_F', 'Anderson (1976); via pydataset'),
    ('ukgas', 'UK gas consumption (quarterly)', lambda: _quarterly('UKgas', 'gas', '1960Q1'), 'gas',
     'Durbin & Koopman (2001); via pydataset'),
    ('sunspots', 'Sunspot activity (yearly)', _sunspots, 'sunspots', 'Yearly sunspot numbers; via statsmodels'),
    ('nile', 'Nile river flow (yearly)', _nile, 'flow', 'Cobb (1978), Biometrika 65(2); via statsmodels'),
]

SIGNALS = [
    ('gunpoint', 'GunPoint (motion)', _ucr('load_gunpoint'), 'label', 'UCR archive (Dau et al., 2019); via aeon'),
    ('italypower', 'ItalyPowerDemand (electricity)', _ucr('load_italy_power_demand'), 'label',
     'UCR archive (Dau et al., 2019); via aeon'),
    ('arrowhead', 'ArrowHead (shape outlines)', _ucr('load_arrow_head'), 'label', 'UCR archive (Dau et al., 2019); via aeon'),
    ('osuleaf', 'OSULeaf (leaf outlines, 6 classes)', _ucr('load_osuleaf'), 'label',
     'UCR archive (Dau et al., 2019); via aeon'),
]

TASKS = {'regression': REGRESSION, 'forecasting': FORECASTING, 'signals': SIGNALS}


# --------------------------------------------------------------------------- #
# One analysis (runs in its own process)
# --------------------------------------------------------------------------- #

def peak_memory_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, 'peak_wset', None)        # Windows
        if peak:
            return peak / 2 ** 20
    except ImportError:
        pass
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2 ** 20 if sys.platform == 'darwin' else 2 ** 10)


def run_regression(df, target, out):
    from easyclassifier.logbook import LogBook
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from easyresearch.regression import Settings, run_analysis
    from easyresearch.regression import analysis as ra
    from easyresearch.regression import evaluation as ev
    from easyresearch.regression.models import build_registry
    defaults = [k for k, s in build_registry().items() if s.default_selected]
    start = time.perf_counter()
    run = run_analysis(df, target, Settings(models=defaults), out, 'bench', 'bench', progress=lambda m: None)
    seconds = time.perf_counter() - start
    final = run['shown']
    # Reference: the linear model, cross-validated on the same folds by a plain scikit-learn pipeline.
    prep = ra.prepare(df, target, LogBook())
    ours = ev.evaluate(ra._pipeline_spec(build_registry()['linear_regression'], prep.X, prep.cfg),
                       prep.X, prep.y, 'kfold5').metrics['r2']
    num = [c for c in prep.X.columns if pd.api.types.is_numeric_dtype(prep.X[c])]
    cat = [c for c in prep.X.columns if c not in num]
    reference = make_pipeline(ColumnTransformer(
        [('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num)]
        + ([('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), cat)]
           if cat else [])), LinearRegression())
    pred = cross_val_predict(reference, prep.X, prep.y, cv=KFold(5, shuffle=True, random_state=ev.SPLIT_SEED))
    ref = float(r2_score(prep.y, pred))
    return dict(rows=run['rows_used'], evaluation=run['method'], selected=run['best'].name,
                r2=final.metrics['r2'], rmse=final.metrics['rmse'], mae=final.metrics['mae'],
                mhsp=final.metrics['mhsp'], baseline_r2=run['baseline'].metrics['r2'],
                linear_r2_easyresearch=ours, linear_r2_reference=ref,
                agreement_difference=abs(ours - ref), seconds=seconds)


def run_forecasting(df, value, out):
    from easyresearch.timeseries import ForecastSettings, run_forecast
    from easyresearch.timeseries import data as td
    start = time.perf_counter()
    run = run_forecast(df, value, ForecastSettings(), out, 'bench', 'bench', progress=lambda m: None)
    seconds = time.perf_counter() - start
    final, refs, plan, H = run['final'], run['refs_final'], run['plan'], run['horizon']
    # Reference: the seasonal (or plain) naive rule recomputed here from the raw series.
    series = td.prepare_series(df, value)
    y, m = series.values, series.season
    key = 'seasonal_naive' if 'seasonal_naive' in refs else 'naive'
    errors = []
    for o in plan.test_origins:
        cycle = y[o - m:o] if key == 'seasonal_naive' else y[o - 1:o]
        for step in range(min(H, len(y) - o)):
            errors.append(abs(y[o + step] - cycle[step % len(cycle)]))
    ref = float(np.mean(errors))
    ours = refs[key].metrics['mae']
    return dict(values=len(y), unit=series.label, horizon=H, selected=run['best'].name,
                mase=final.metrics['mase'], mae=final.metrics['mae'], smape=final.metrics['smape'],
                mhsp=final.metrics['mhsp'], reference_rule=key, reference_mase=refs[key].metrics['mase'],
                beats_reference=bool(final.metrics['mae'] < ours), band_coverage=run['coverage'],
                naive_mae_easyresearch=ours, naive_mae_recomputed=ref,
                agreement_difference=abs(ours - ref), seconds=seconds)


def run_signals(df, label, out, loader):
    from sklearn.metrics import accuracy_score
    from easyresearch.timeseries.classify import SignalSettings, run_signal_classification
    from easyresearch.timeseries.signal_models import build_registry
    start = time.perf_counter()
    run = run_signal_classification(df, label, SignalSettings(), out, 'bench', 'bench', progress=lambda m: None)
    seconds = time.perf_counter() - start
    shown = run['shown']
    # Reference: ROCKET on the official UCR split, EasyResearch's and aeon's.
    Xtr, ytr = loader('train')
    Xte, yte = loader('test')
    ours = build_registry()['rocket'].factory().fit(pd.DataFrame(Xtr), ytr)
    acc_ours = float(accuracy_score(yte, ours.predict(pd.DataFrame(Xte))))
    from aeon.classification.convolution_based import RocketClassifier
    theirs = RocketClassifier(n_kernels=10000, random_state=0).fit(Xtr[:, None, :], ytr)
    acc_ref = float(accuracy_score(yte, theirs.predict(Xte[:, None, :])))
    return dict(recordings=run['n'], length=run['length'], classes=len(run['class_names']),
                evaluation=run['method'], selected=run['best'].classifier_name,
                accuracy=shown.metrics['accuracy'], balanced_accuracy=shown.metrics['balanced_accuracy'],
                reference_rule_balanced_accuracy=1 / len(run['class_names']),
                rocket_official_split_easyresearch=acc_ours, rocket_official_split_aeon=acc_ref,
                agreement_difference=abs(acc_ours - acc_ref), seconds=seconds)


def run_one(task, key):
    entry = next(e for e in TASKS[task] if e[0] == key)
    _, name, loader, column, source = entry
    df = loader()
    with tempfile.TemporaryDirectory() as out:
        if task == 'regression':
            row = run_regression(df, column, out)
        elif task == 'forecasting':
            row = run_forecasting(df, column, out)
        else:
            row = run_signals(df, column, out, loader)
    row = dict(dataset=name, key=key, **row, peak_memory_mb=peak_memory_mb(), source=source)
    print('RESULT ' + json.dumps(row, default=float))


# --------------------------------------------------------------------------- #
# Driver and report
# --------------------------------------------------------------------------- #

def fmt(v, digits=3):
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return f'{v:,}'
    if isinstance(v, float):
        return f'{v:.{digits}f}'
    return str(v)


def markdown(rows, columns):
    head = '| ' + ' | '.join(label for _, label in columns) + ' |'
    rule = '|' + '|'.join('---' for _ in columns) + '|'
    body = ['| ' + ' | '.join(fmt(r.get(k, '')) for k, _ in columns) + ' |' for r in rows]
    return '\n'.join([head, rule] + body)


TABLES = {
    'regression': [('dataset', 'Dataset'), ('rows', 'Rows'), ('selected', 'Selected model'), ('evaluation', 'Final score'),
                   ('r2', 'R²'), ('baseline_r2', 'R² of average'), ('mhsp', 'MHSP %'),
                   ('linear_r2_easyresearch', 'Linear R² (EasyResearch)'), ('linear_r2_reference', 'Linear R² (scikit-learn)'),
                   ('seconds', 'Seconds'), ('peak_memory_mb', 'Peak MB')],
    'forecasting': [('dataset', 'Dataset'), ('values', 'Values'), ('horizon', 'Horizon'), ('selected', 'Selected model'),
                    ('mase', 'MASE'), ('reference_mase', 'MASE of naive rule'), ('beats_reference', 'Beats rule'),
                    ('band_coverage', '80% band coverage'), ('naive_mae_easyresearch', 'Rule MAE (EasyResearch)'),
                    ('naive_mae_recomputed', 'Rule MAE (recomputed)'), ('seconds', 'Seconds'), ('peak_memory_mb', 'Peak MB')],
    'signals': [('dataset', 'Dataset'), ('recordings', 'Recordings'), ('length', 'Length'), ('classes', 'Classes'),
                ('selected', 'Selected model'), ('evaluation', 'Final score'), ('balanced_accuracy', 'Balanced accuracy'),
                ('rocket_official_split_easyresearch', 'ROCKET, official split (EasyResearch)'),
                ('rocket_official_split_aeon', 'ROCKET, official split (aeon)'), ('seconds', 'Seconds'),
                ('peak_memory_mb', 'Peak MB')],
}


def report():
    lines = ['# EasyResearch benchmark results', '',
             'Produced by `easyresearch_benchmarks.py`. Every analysis ran as a beginner would run it: the fast '
             'default set of models and automatic settings. Scores are the honest final scores reported to the '
             'user. Times are for the computer below and include every model in the comparison and the final '
             'evaluation.', '']
    meta = OUT / 'environment.json'
    if meta.exists():
        env = json.loads(meta.read_text(encoding='utf-8'))
        lines += ['Computer: ' + ', '.join(f'{k} {v}' for k, v in env.items()), '']
    for task in TASKS:
        path = OUT / f'{task}.csv'
        if not path.exists():
            continue
        rows = pd.read_csv(path).to_dict(orient='records')
        lines += [f'## {task.capitalize()}', '', markdown(rows, TABLES[task]), '']
        worst = max(r['agreement_difference'] for r in rows)
        lines += [f'Largest difference from the reference implementation: {worst:.4f}.', '']
        lines += ['Sources: ' + '; '.join(f"{r['dataset']}: {r['source']}" for r in rows), '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print((OUT / 'RESULTS.md').read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='EasyResearch benchmarks')
    parser.add_argument('--task', choices=list(TASKS))
    parser.add_argument('--one', nargs=2, metavar=('TASK', 'KEY'), help=argparse.SUPPRESS)
    parser.add_argument('--report', action='store_true', help='only rebuild RESULTS.md')
    args = parser.parse_args(argv)
    if args.one:
        run_one(*args.one)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report:
        from importlib.metadata import version
        import os as _os
        env = {'Python': platform.python_version(), 'OS': platform.platform(terse=True),
               'CPUs': _os.cpu_count(), 'easyresearch': version('easyresearch'), 'torch': version('torch')}
        (OUT / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
        for task in ([args.task] if args.task else TASKS):
            rows = []
            for key, *_ in TASKS[task]:
                print(f'{task}: {key} ...', flush=True)
                proc = subprocess.run([sys.executable, __file__, '--one', task, key], capture_output=True, text=True,
                                      encoding='utf-8', env=dict(os.environ, PYTHONIOENCODING='utf-8'))
                line = next((l for l in proc.stdout.splitlines() if l.startswith('RESULT ')), None)
                if line is None:
                    print(proc.stdout[-2000:], proc.stderr[-4000:])
                    raise SystemExit(f'{task} {key} failed')
                rows.append(json.loads(line[7:]))
                print(f"  done in {rows[-1]['seconds']:.0f} s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f'{task}.csv', index=False)
    report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
