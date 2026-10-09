"""Desktop adapter for time-series forecasting (the shared EasyResearch core).

The desktop shows, after a file is read, the choices this task needs besides
the column to forecast: the date column, how far ahead to forecast and the
measure used to choose the model. They are sent back in the inspection result
as ``parameters`` (data-dependent drop-down lists), so the interface needs no
forecasting-specific code.
"""
from __future__ import annotations
import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import uuid

import pandas as pd
from easyclassifier import dataset as ds

import easyresearch
from easyresearch.common.hassanat import HASSANAT_CITATIONS
from easyresearch.common.output import fmt
from easyresearch.timeseries import data as td
from easyresearch.timeseries import deep
from easyresearch.timeseries import evaluation as ev
from easyresearch.timeseries.forecast import ForecastSettings, choose_horizon, horizon_fits, run_forecast
from easyresearch.timeseries.figures import FIGURES, CAPTIONS
from easyresearch.timeseries.models import REFERENCE_KEYS, build_registry

from . import advice, options
from progress import ProgressTracker, forecast_rules

METRIC_SPECS = [
    dict(key='mase', label='MASE', format='decimal3', higher_is_better=False),
    dict(key='mae', label='MAE', format='number', higher_is_better=False),
    dict(key='rmse', label='RMSE', format='number', higher_is_better=False),
    # sMAPE and MHSP are already percentages (0-100), so they are shown as numbers.
    dict(key='smape', label='sMAPE (%)', format='number', higher_is_better=False),
    dict(key='mhsp', label='MHSP (%)', format='number', higher_is_better=True),
]
METRIC_CHOICES = [
    dict(id='mase', name='MASE - error compared with simply repeating past values (recommended)'),
    dict(id='rmse', name='RMSE - typical error, large errors weigh more'),
    dict(id='mae', name='MAE - average error, in the units of the series'),
    dict(id='mhsp', name='MHSP - mean Hassanat similarity percentage'),
]
# Words for "steps ahead" by time unit (keys of timeseries.data.UNITS).
STEP_WORDS = {'h': ('hour', 'hours'), 'D': ('day', 'days'), 'B': ('business day', 'business days'),
              'W': ('week', 'weeks'), 'M': ('month', 'months'), 'Q': ('quarter', 'quarters'),
              'Y': ('year', 'years'), None: ('step', 'steps')}
# Horizons people usually ask for, by time unit (one day, a week, a month ...).
HORIZON_CANDIDATES = {'h': [1, 6, 12, 24, 48, 72, 168], 'D': [1, 7, 14, 30, 60, 90], 'B': [1, 5, 10, 20, 60],
                      'W': [1, 4, 8, 13, 26, 52], 'M': [1, 3, 6, 12, 18, 24, 36], 'Q': [1, 2, 4, 8, 12],
                      'Y': [1, 2, 3, 5, 10], None: [1, 5, 10, 20, 50]}


def load_data(path):
    return td.load_demo() if path == 'demo' else ds.load_csv(path)


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def steps(n, unit):
    one, many = STEP_WORDS.get(unit, STEP_WORDS[None])
    return f'{n} {one if n == 1 else many}'


def result_row(result, reference=False, name=None):
    return dict(key=result.key, name=name or result.name, note='', reference=reference,
                metrics={key: number(value) for key, value in result.metrics.items()},
                selection_score=None)


def time_label(value, unit):
    """A future time point as people write it (2002-01, 2002-Q1, 2002-01-05 ...)."""
    if not isinstance(value, pd.Timestamp):
        return str(value)
    if unit == 'Y':
        return f'{value.year}'
    if unit == 'Q':
        return f'{value.year}-Q{value.quarter}'
    if unit == 'M':
        return value.strftime('%Y-%m')
    if unit == 'h':
        return value.strftime('%Y-%m-%d %H:%M')
    return value.strftime('%Y-%m-%d')


def kind(series):
    return series.label if series.unit else 'equally spaced'


def horizon_parameter(df, value):
    """Horizon choices for the suggested series. Only horizons the series can
    evaluate honestly are offered, so a beginner never picks one that the run
    would refuse; the run still checks the final choice."""
    option = dict(id='auto', name='Automatic (recommended)')
    hint = 'How many time steps beyond the last value to forecast. Longer horizons are harder to forecast well.'
    if value is None:
        return dict(id='horizon', label='How far ahead?', hint=hint, options=[option], default='auto')
    try:
        series = td.prepare_series(df, value, None)
        auto = choose_horizon(series)
    except ValueError as exc:
        return dict(id='horizon', label='How far ahead?', hint=str(exc), options=[option], default='auto')
    n, unit, season = len(series.values), series.unit, series.season
    usual = td.default_horizon(series)
    option['name'] = f'Automatic: {steps(auto, unit)} ahead (recommended)'
    values = sorted(h for h in set(HORIZON_CANDIDATES.get(unit, HORIZON_CANDIDATES[None])) - {auto}
                    if 1 <= h <= n // 5 and horizon_fits(n, h, season))
    longest = max([auto] + values)
    options = [option] + [dict(id=str(h), name=f'{steps(h, unit)} ahead') for h in values]
    hint += f' This series has {n:,} {kind(series)} values, so horizons up to {steps(longest, unit)} can be evaluated honestly.'
    if auto < usual:
        hint += f' Forecasting {steps(usual, unit)} ahead needs a longer series.'
    return dict(id='horizon', label='How far ahead?', hint=hint, options=options, default='auto')


def forecast_advice(df, value):
    if value is None:
        return ['No column of numbers that change over time was found. Each row should be one time point, '
                'with the measured value in its own column.']
    try:
        series = td.prepare_series(df, value, None)
    except ValueError as exc:
        return [f"'{value}' cannot be forecast yet: {exc}"]
    out = [f"'{value}': {len(series.values):,} {kind(series)} values"
           + (f', from {time_label(series.index[0], series.unit)} to {time_label(series.index[-1], series.unit)}.'
              if series.unit else '.')]
    if series.season > 1:
        out.append(f'A repeating pattern every {steps(series.season, series.unit)} is taken into account.')
    out += series.notes
    out.append('The last part of the series (about 20%) is kept back. Models are compared on the part before it, and the '
               'selected one is scored on it, each forecast using only the values before it.')
    out.append(advice.deep_note('MLP, LSTM, GRU, TCN, N-BEATS, Transformer'))
    return out


class ForecastingModule:
    def __init__(self, manifest):
        self.manifest = manifest
        installed = version(manifest['package'])
        if installed != manifest['package_version']:
            raise RuntimeError(f'The forecasting module requires EasyResearch {manifest["package_version"]}; installed version is {installed}.')

    def describe(self):
        result = {key: value for key, value in self.manifest.items() if key != 'adapter'}
        result['engine_version'] = easyresearch.__version__
        # The naive rules are always included as references, so they are not offered as choices.
        result['models'] = [dict(id=key, name=spec.name, available=spec.available,
                                 default_selected=spec.default_selected, reason=spec.reason)
                            for key, spec in build_registry().items() if key not in REFERENCE_KEYS]
        result['metrics'] = METRIC_SPECS
        result['headline'] = ['mase', 'mae', 'mhsp']
        # Forecasts are always evaluated in time order (rolling origin), so there is nothing to choose here.
        result['validation_options'] = []
        result['selection_options'] = []
        result['advanced_choices'], result['advanced_lists'] = options.figure_options(FIGURES, list(FIGURES))
        result['labels'] = dict(
            target_title='Which series do you want to forecast?',
            target_hint='Choose the column of numbers measured over time, such as monthly sales, daily admissions or hourly temperature. Each row is one time point.',
            models_title='Compare forecasting models',
            model_noun='model', model_noun_plural='models',
            demo_button='Try CO₂ example',
            selection_metric='MASE',
            prep_note='The two simple rules (repeat the last value; repeat the value one season earlier) are always included for reference. Models are compared on forecasts made inside the history; the last part of the series is kept back for the final evaluation. Deep models take longer.')
        return result

    def inspect(self, dataset):
        df = load_data(dataset)
        if df.empty or df.shape[1] < 1:
            raise ValueError('Choose a spreadsheet with at least one column of numbers measured over time.')
        if df.columns.duplicated().any():
            raise ValueError('Column names must be unique.')
        info = ds.inspect(df)
        tcols = td.time_columns(df)
        columns = []
        for c in df.columns:
            ok, note, label = td.check_value_column(df[c], tcols)
            columns.append(dict(name=str(c), description=label, kind='numeric' if ok else 'other',
                                suitable=ok, note=note))
        suggested = td.suggest_value(df)
        if tcols:
            time_options = [dict(id='auto', name=f'{tcols[0]} (found automatically)')] + \
                           [dict(id=c, name=c) for c in tcols[1:]]
            time_hint = 'Dates must be evenly spaced: hourly, daily, weekly, monthly, quarterly or yearly.'
        else:
            time_options = [dict(id='auto', name='No date column found: rows are taken in file order')]
            time_hint = 'Without dates the rows are treated as equally spaced and no seasonal pattern is assumed. Add a date column to use weekly, monthly or yearly patterns.'
        parameters = [
            dict(id='time', label='Date or time column', hint=time_hint, options=time_options, default='auto'),
            horizon_parameter(df, suggested),
            dict(id='metric', label='Choose the best model by', options=METRIC_CHOICES, default='mase',
                 hint='Used only to rank the models in the comparison. The final evaluation reports all measures.'),
        ]
        return dict(dataset=dataset, rows=info.n_rows, columns_count=info.n_cols,
                    missing_cells=info.missing_total, duplicates=info.duplicate_rows,
                    suggested_target=suggested, columns=columns, parameters=parameters,
                    advice=advice.other_example(dataset, 'Time-series forecasting') + advice.safely(lambda: forecast_advice(df, suggested)),
                    preview=df.head(8).fillna('').astype(str).to_dict(orient='records'))

    def run(self, request, emitter):
        dataset = request.get('dataset', '')
        df = load_data(dataset)
        value = request.get('target', '')
        if value not in df.columns:
            raise ValueError('Choose the column to forecast from the inspected dataset.')
        settings = request.get('settings') or {}
        keys = settings.get('models', [])
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            raise ValueError('Select at least one forecasting model.')
        time = settings.get('time', 'auto')
        if time != 'auto' and time not in df.columns:
            raise ValueError('The chosen date column is not in the dataset.')
        horizon = str(settings.get('horizon', 'auto'))
        if horizon != 'auto' and not horizon.isdigit():
            raise ValueError('The forecast horizon must be a whole number of steps.')
        metric = settings.get('metric', 'mase')
        if metric not in ev.SELECTION_METRICS:
            raise ValueError('Unknown measure for choosing the model.')
        if not isinstance(settings.get('figures', True), bool):
            raise ValueError('The figures setting must be true or false.')
        theme, formats, figure_keys = options.read_figures(settings, FIGURES, list(FIGURES))
        output_root = Path(request.get('output_root', '')).expanduser()
        if not str(request.get('output_root', '')).strip() or not output_root.is_absolute():
            raise ValueError('Choose an absolute output folder.')

        session = output_root / ('analysis_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
        session.mkdir(parents=True)
        (session / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
        (session / 'status.json').write_text(json.dumps({'state': 'running'}), encoding='utf-8')
        emitter.send('progress', message='Analysis folder created.', session_dir=str(session))
        old_cwd = Path.cwd()
        try:
            os.chdir(session)
            emitter.tracker = ProgressTracker(*forecast_rules(len(set(keys) - set(REFERENCE_KEYS)), figures=settings.get('figures', True)))
            deep.EPOCH_HOOK = emitter.epoch
            os.environ['MPLCONFIGDIR'] = str(session / 'plot-cache')
            demo = dataset == 'demo'
            st = ForecastSettings(models=keys, horizon='auto' if horizon == 'auto' else int(horizon),
                                  selection_metric=metric, time_column=None if time == 'auto' else time,
                                  figures=settings.get('figures', True), figure_theme=theme,
                                  figure_format=formats, figure_keys=figure_keys)
            run = run_forecast(df, value, st, str(session),
                               source_name=td.DEMO_NAME if demo else Path(dataset).name,
                               source_stem='demo_co2' if demo else Path(dataset).stem,
                               dataset_citation=td.DEMO_CITATION if demo else '')
            out = Path(run['out'])
            series, plan, H = run['series'], run['plan'], run['horizon']
            n = len(series.values)
            best, final, refs = run['best'], run['final'], run['refs_final']
            ref_key = 'seasonal_naive' if 'seasonal_naive' in refs else 'naive'
            registry = build_registry()
            figures = [dict(key=k, title=FIGURES[k], caption=CAPTIONS[k], path=str(out / files['png']))
                       for k, files in run['figures'].items() if 'png' in files]
            figures.sort(key=lambda f: list(FIGURES).index(f['key']))
            unit = series.unit
            final_values = n - plan.dev_end
            forecast_table = dict(
                title='The forecast',
                hint=f'{steps(H, unit)} beyond the last value, made by {best.name} retrained on all {n:,} values. '
                     'In about 8 of 10 cases the true value should fall inside the 80% band. Also saved as forecast.csv.',
                columns=[series.time_column or 'Step', 'Forecast', 'Lower 80%', 'Upper 80%'],
                rows=[[time_label(t, unit), fmt(f), fmt(lo), fmt(hi)]
                      for t, f, lo, hi in zip(run['future_index'], run['future'], run['lower'], run['upper'])])
            result = dict(module_id=self.manifest['id'], package_version=easyresearch.__version__,
                          selected_model=best.name, evaluation_method='rolling_origin',
                          evaluation_text=(f'Final evaluation on the last {steps(final_values, unit)} of the series '
                                           f'({len(plan.test_origins)} forecasts of {steps(H, unit)} each). This period was not used '
                                           'to choose the model; before each forecast the model was retrained on the values before it.'),
                          final=result_row(final),
                          baseline=result_row(refs[ref_key], True, registry[ref_key].name + ' - final period'),
                          comparison=[result_row(r, r.key in REFERENCE_KEYS) for r in run['results']],
                          selection_metric=ev.SELECTION_METRICS[run['metric']],
                          rows_loaded=len(df), rows_used=n,
                          development_rows=plan.dev_end, test_rows=final_values,
                          summary=run['summary'] + ([run['comparison_text']] if run['comparison_text'] else []),
                          notes=[f'{n:,} {kind(series)} values; seasonal cycle of {series.season}; '
                                 f'forecast horizon {steps(H, unit)}.'] + run['notes']
                                + (['KNN used the Hassanat distance; please cite: ' + ' '.join(HASSANAT_CITATIONS)]
                                   if any(r.key == 'knn' for r in run['results']) else []),
                          warnings=run['warnings'],
                          tables=[forecast_table],
                          figures=figures,
                          output_dir=str(out), session_dir=str(session),
                          artifacts=[dict(name=str(p.relative_to(out)), path=str(p)) for p in sorted(out.rglob('*')) if p.is_file()],
                          environment={name: version(name) for name in ['easyresearch', 'easyclassifier', 'pandas', 'numpy', 'scikit-learn',
                                                                        'statsmodels', 'torch', 'matplotlib', 'joblib']})
            (out / 'desktop-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            result['artifacts'].append(dict(name='desktop-result.json', path=str(out / 'desktop-result.json')))
            (session / 'status.json').write_text(json.dumps({'state': 'completed', 'output_dir': str(out)}), encoding='utf-8')
            return result
        except Exception:
            (session / 'status.json').write_text(json.dumps({'state': 'failed'}), encoding='utf-8')
            raise
        finally:
            deep.EPOCH_HOOK = None
            os.chdir(old_cwd)
