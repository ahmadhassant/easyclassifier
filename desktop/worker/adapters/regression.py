from __future__ import annotations
import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import uuid

from easyclassifier import dataset as ds

# The regression engine is the shared EasyResearch core, the same code the
# command line runs; the desktop keeps no copy of its own.
import easyresearch
from easyresearch.common.hassanat import HASSANAT_CITATIONS
from easyresearch.regression import data as rd
from easyresearch.regression import evaluation as ev
from easyresearch.regression.analysis import Settings, run_analysis
from easyresearch.regression.figures import FIGURES, CAPTIONS, STANDARD_FIGURES
from easyresearch.regression.models import build_registry

from . import advice, options
from progress import ProgressTracker, table_rules

METRIC_SPECS = [
    dict(key='r2', label='R²', format='decimal3', higher_is_better=True),
    dict(key='rmse', label='RMSE', format='number', higher_is_better=False),
    dict(key='mae', label='MAE', format='number', higher_is_better=False),
    dict(key='medae', label='Median AE', format='number', higher_is_better=False),
    dict(key='mape', label='MAPE', format='percent', higher_is_better=False),
    # MHSP is already a percentage (0-100), so it is shown as a plain number.
    dict(key='mhsp', label='MHSP (%)', format='number', higher_is_better=True),
]

def load_data(path):
    return rd.load_demo() if path == 'demo' else ds.load_csv(path)

def number(value):
    value = float(value)
    return value if math.isfinite(value) else None

def result_row(result, reference=False):
    return dict(key=result.key, name=result.name, note=result.note, reference=reference,
                metrics={key: number(value) for key, value in result.metrics.items()},
                selection_score=number(result.selection_score))

class RegressionModule:
    def __init__(self, manifest):
        self.manifest = manifest
        installed = version(manifest['package'])
        if installed != manifest['package_version']:
            raise RuntimeError(f'The regression module requires EasyResearch {manifest["package_version"]}; installed version is {installed}.')

    def describe(self):
        result = {key: value for key, value in self.manifest.items() if key != 'adapter'}
        result['engine_version'] = easyresearch.__version__
        result['models'] = [dict(id=key, name=spec.name, available=spec.available,
                                 default_selected=spec.default_selected, reason=spec.reason)
                            for key, spec in build_registry().items()]
        result['metrics'] = METRIC_SPECS
        result['headline'] = ['r2', 'rmse', 'mae']
        result['validation_options'] = [dict(id='auto', name='Automatic (recommended)'),
                                        dict(id='kfold5', name='5-fold cross-validation'),
                                        dict(id='kfold10', name='10-fold cross-validation'),
                                        dict(id='holdout', name='Single 80/20 validation split')]
        result['selection_options'] = [dict(id=k, name=v) for k, v in ev.SELECTION.items()]
        result['advanced_choices'], result['advanced_lists'] = options.figure_options(
            FIGURES, STANDARD_FIGURES, 'The usual figures are ticked; the missing-value map is added when the data have empty cells.')
        result['labels'] = dict(
            target_title='What number do you want to predict?',
            target_hint='Choose a measurement, such as a price, a test result, a yield or a duration. Categories belong in Classification.',
            models_title='Compare regression models',
            model_noun='model', model_noun_plural='models',
            demo_button='Try diabetes example',
            selection_metric='R²',
            prep_note='Missing values, encoding and scaling use automatic preparation learned on training rows only. Multiple models receive a separate final evaluation.')
        return result

    def inspect(self, dataset):
        df = load_data(dataset)
        if df.empty or df.shape[1] < 2:
            raise ValueError('Choose a spreadsheet with observations and at least two columns.')
        if df.columns.duplicated().any():
            raise ValueError('Column names must be unique.')
        info = ds.inspect(df)
        columns = []
        for c in df.columns:
            check = rd.check_target(df[c])
            columns.append(dict(name=str(c), description=check.label, kind='numeric' if check.suitable else 'other',
                                suitable=check.suitable, note=check.note))
        suggested = rd.suggest_target(df)
        return dict(dataset=dataset, rows=info.n_rows, columns_count=info.n_cols,
                    missing_cells=info.missing_total, duplicates=info.duplicate_rows,
                    suggested_target=suggested, columns=columns,
                    advice=advice.other_example(dataset, 'Regression') + advice.safely(lambda: advice.regression(df, info, suggested)),
                    preview=df.head(8).fillna('').astype(str).to_dict(orient='records'))

    def run(self, request, emitter):
        dataset = request.get('dataset', '')
        df = load_data(dataset)
        target = request.get('target', '')
        if target not in df.columns:
            raise ValueError('Choose the column to predict from the inspected dataset.')
        settings = request.get('settings') or {}
        keys = settings.get('models', settings.get('classifiers', []))
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError('Select at least one available regression model.')
        validation = settings.get('validation', 'auto')
        selection = settings.get('selection', 'auto')
        if validation not in {'auto', *ev.VALIDATION} or selection not in ev.SELECTION:
            raise ValueError('Unknown validation or final evaluation setting.')
        if not isinstance(settings.get('figures', True), bool):
            raise ValueError('The figures setting must be true or false.')
        theme, formats, figure_keys = options.read_figures(settings, FIGURES, STANDARD_FIGURES)
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
            # Progress bar: the same rules as the core's choice of final evaluation.
            method = None if len(keys) < 2 else (selection if selection != 'auto' else ('nested' if len(df) <= ev.NESTED_MAX_ROWS else 'final_test'))
            figures_on = settings.get('figures', True)
            emitter.tracker = ProgressTracker(*table_rules(len(set(keys)), method, figures=figures_on, importance=figures_on))
            os.environ['MPLCONFIGDIR'] = str(session / 'plot-cache')
            demo = dataset == 'demo'
            run = run_analysis(df, target, Settings(models=keys, validation=validation, selection=selection,
                                                    figures=settings.get('figures', True), figure_theme=theme,
                                                    figure_format=formats, figure_keys=figure_keys),
                               str(session),
                               source_name=rd.DEMO_NAME if demo else Path(dataset).name,
                               source_stem='demo_diabetes' if demo else Path(dataset).stem,
                               dataset_citation=rd.DEMO_CITATION if demo else '')
            out = Path(run['out'])
            final, best, shown = run['final'], run['best'], run['shown']
            figures = [dict(key=k, title=FIGURES[k], caption=CAPTIONS[k], path=str(out / files['png']))
                       for k, files in run['figures'].items() if 'png' in files]
            figures.sort(key=lambda f: list(FIGURES).index(f['key']))
            hassanat = run['hassanat']
            result = dict(module_id=self.manifest['id'], package_version=easyresearch.__version__,
                          selected_model=best.name, evaluation_method=run['method'],
                          final=result_row(shown), baseline=result_row(run['baseline'], True),
                          comparison=[result_row(r) for r in run['results']] + [result_row(run['baseline'], True)],
                          selection_metric='R²',
                          rows_loaded=len(df), rows_used=run['rows_used'], predictors=run['predictors'],
                          development_rows=run['dev_rows'], test_rows=run['test_rows'],
                          summary=run['summary'] + ([run['comparison_text']] if run['comparison_text'] else []),
                          notes=run['notes'] + (['KNN used the Hassanat distance; please cite: ' + ' '.join(HASSANAT_CITATIONS)] if hassanat else []),
                          warnings=run['warnings'],
                          figures=figures,
                          output_dir=str(out), session_dir=str(session),
                          artifacts=[dict(name=str(p.relative_to(out)), path=str(p)) for p in sorted(out.rglob('*')) if p.is_file()],
                          environment={name: version(name) for name in ['easyresearch', 'easyclassifier', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'joblib']})
            (out / 'desktop-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            result['artifacts'].append(dict(name='desktop-result.json', path=str(out / 'desktop-result.json')))
            (session / 'status.json').write_text(json.dumps({'state': 'completed', 'output_dir': str(out)}), encoding='utf-8')
            return result
        except Exception:
            (session / 'status.json').write_text(json.dumps({'state': 'failed'}), encoding='utf-8')
            raise
        finally:
            os.chdir(old_cwd)
