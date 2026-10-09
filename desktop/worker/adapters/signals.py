"""Desktop adapter for signal classification (the shared EasyResearch core).

One row is one recording from a single channel (an ECG beat, an EEG epoch, a
sensor window) with its class label; the signal values are the numeric
columns in file order. UCR/UEA archive files (.tsv/.txt, label first, no
header) are read directly. The first and last signal columns are offered as
drop-down choices after inspection, so extra numeric columns (age, a second
measurement) can be left out.
"""
from __future__ import annotations
import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import uuid

from easyclassifier import dataset as ds
from easyclassifier import target as tg
from easyclassifier.evaluation import SELECTION, VALIDATION

import easyresearch
from easyresearch.common.hassanat import HASSANAT_CITATIONS
from easyresearch.timeseries import deep
from easyresearch.timeseries import signal_data as sd
from easyresearch.timeseries.classify import CAPTIONS, SignalSettings, run_signal_classification
from easyresearch.timeseries.signal_models import build_registry

from . import advice, options
from progress import ProgressTracker, table_rules

METRIC_SPECS = [
    dict(key='accuracy', label='Accuracy', format='percent', higher_is_better=True),
    dict(key='balanced_accuracy', label='Balanced accuracy', format='percent', higher_is_better=True),
    dict(key='precision', label='Precision', format='percent', higher_is_better=True),
    dict(key='recall', label='Recall', format='percent', higher_is_better=True),
    dict(key='f1', label='F1', format='percent', higher_is_better=True),
    dict(key='roc_auc', label='ROC AUC', format='decimal4', higher_is_better=True),
    dict(key='mcc', label='MCC', format='decimal4', higher_is_better=True),
]
FIGURES = {
    'signals_by_class': 'Recordings by class',
    'class_distribution': 'Class sizes',
    'comparison': 'Model comparison',
    'confusion_matrix': 'Confusion matrix',
    'roc_curve': 'ROC curve',
}
SINGLE_CHANNEL = ('Each row must be one recording from a single channel (one sensor, one ECG lead or one EEG electrode). '
                  'Recordings with several channels at once, such as a 12-lead ECG, are not supported yet; '
                  'analyse one channel at a time.')
UNSUITABLE = {
    tg.MEASUREMENT: 'It holds measurements; choose the column with the class of each recording.',
    tg.ID_LIKE: 'It is different in every row and looks like an identifier.',
    tg.CONSTANT: 'It has only one value, so there is nothing to classify.',
    tg.EMPTY: 'It is empty.',
    tg.MANY_TEXT: 'It has too many different text values to be a set of classes.',
}


def load_data(path):
    return sd.load_demo() if path == 'demo' else sd.load_table(path)


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def result_row(result, reference=False):
    return dict(key=result.classifier_key, name=result.classifier_name, note=result.note or '', reference=reference,
                metrics={key: number(value) for key, value in result.metrics.items()},
                balanced_accuracy=number(result.selection_score),
                selection_score=number(result.selection_score))


def signal_range(df, label, first, last):
    """The signal columns from `first` to `last` (inclusive, file order)."""
    names = [str(c) for c in df.columns]
    if first not in names or last not in names:
        raise ValueError('Choose the first and last signal columns from the inspected file.')
    i, j = names.index(first), names.index(last)
    if i > j:
        raise ValueError(f"The first signal column ('{first}') must come before the last one ('{last}') in the file.")
    usable = set(map(str, sd.signal_columns(df, label)))
    return [c for c in df.columns[i:j + 1] if str(c) in usable]


def signal_advice(df, label):
    if label is None:
        return ['No column with class labels was found. Each row should be one recording, with its class in one column '
                'and the signal values in the others.']
    try:
        recs = sd.prepare(df, label)
    except ValueError as exc:
        return [f"These recordings cannot be classified yet: {exc}"]
    n, length = recs.X.shape
    out = [f'{n:,} recordings of {length} values each.']
    out += advice.classes(recs.labels.rename(str(label)), 'recordings')
    out.append('When a deep model is compared, 20% of the recordings are set aside and used only once, for the final '
               'score; otherwise nested cross-validation is used (up to 2,000 recordings).')
    out += recs.notes
    out.append(advice.deep_note('FCN, ResNet, InceptionTime, LSTM'))
    return out


class SignalsModule:
    def __init__(self, manifest):
        self.manifest = manifest
        installed = version(manifest['package'])
        if installed != manifest['package_version']:
            raise RuntimeError(f'The signal classification module requires EasyResearch {manifest["package_version"]}; installed version is {installed}.')

    def describe(self):
        result = {key: value for key, value in self.manifest.items() if key != 'adapter'}
        result['engine_version'] = easyresearch.__version__
        result['models'] = [dict(id=key, name=spec.name, available=spec.available,
                                 default_selected=spec.default_selected, reason=spec.reason)
                            for key, spec in build_registry().items() if key != 'majority']
        result['metrics'] = METRIC_SPECS
        result['headline'] = ['accuracy', 'balanced_accuracy', 'roc_auc']
        result['validation_options'] = [dict(id='auto', name='Automatic (recommended)'),
                                        dict(id='kfold5', name='Stratified 5-fold cross-validation'),
                                        dict(id='kfold10', name='Stratified 10-fold cross-validation'),
                                        dict(id='holdout', name='Single 80/20 validation split')]
        result['selection_options'] = [dict(id='auto', name='Automatic (recommended)'),
                                       dict(id='nested', name='Nested cross-validation'),
                                       dict(id='final_test', name='Separate 20% final test')]
        result['advanced_choices'], result['advanced_lists'] = options.figure_options(FIGURES, list(FIGURES))
        result['labels'] = dict(
            target_title='Which column holds the class of each recording?',
            target_hint='For example a diagnosis (normal / abnormal), an activity or a sleep stage. ' + SINGLE_CHANNEL,
            models_title='Compare signal classifiers',
            model_noun='model', model_noun_plural='models',
            demo_button='Try heartbeat example',
            selection_metric='Balanced accuracy',
            prep_note='Gaps inside a recording are filled from that recording only. The rule "always predict the most frequent class" is included for reference. '
                      'Deep models take several minutes; when one is compared, the final score uses an untouched 20% test set.')
        result['file_filter'] = 'Recordings (CSV, Excel, UCR .tsv/.txt)|*.csv;*.xlsx;*.xlsm;*.tsv;*.txt|All files|*.*'
        return result

    def inspect(self, dataset):
        df = load_data(dataset)
        if df.empty or df.shape[1] < 2:
            raise ValueError('Choose a file with one recording per row: a class column and the signal values.')
        if df.columns.duplicated().any():
            raise ValueError('Column names must be unique.')
        info = ds.inspect(df)
        suggested = sd.suggest_label(df)
        columns = []
        for c in df.columns:
            d = tg.describe_column(df[c])
            columns.append(dict(name=str(c), description=d.label, kind=d.kind,
                                suitable=d.kind == tg.CATEGORY, note=UNSUITABLE.get(d.kind, '')))
        signal = [str(c) for c in sd.signal_columns(df, suggested)]
        if signal:
            first = [dict(id=c, name=f'{c} (column {list(map(str, df.columns)).index(c) + 1})') for c in signal]
            span = f'{len(signal)} numeric columns, from {signal[0]} to {signal[-1]}'
            parameters = [
                dict(id='first', label='Signal values start at', options=first, default=signal[0],
                     hint=f'Found {span}. Change the first or last column to leave out numbers that are not part of the signal, such as age. '
                          f'At least {sd.MIN_LENGTH} values are needed.'),
                dict(id='last', label='Signal values end at', options=first, default=signal[-1],
                     hint=SINGLE_CHANNEL),
            ]
        else:
            parameters = []
        return dict(dataset=dataset, rows=info.n_rows, columns_count=info.n_cols,
                    missing_cells=info.missing_total, duplicates=info.duplicate_rows,
                    suggested_target=None if suggested is None else str(suggested),
                    columns=columns, parameters=parameters,
                    advice=advice.other_example(dataset, 'Signal classification') + advice.safely(lambda: signal_advice(df, suggested)),
                    preview=df.iloc[:8, :30].fillna('').astype(str).to_dict(orient='records'))

    def run(self, request, emitter):
        dataset = request.get('dataset', '')
        df = load_data(dataset)
        df.columns = [str(c) for c in df.columns]
        label = request.get('target', '')
        if label not in df.columns:
            raise ValueError('Choose the column with the class of each recording.')
        settings = request.get('settings') or {}
        keys = settings.get('models', [])
        registry = build_registry()
        if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or k not in registry or k == 'majority'
                                                         or not registry[k].available for k in keys):
            raise ValueError('Select at least one available signal classifier.')
        validation = settings.get('validation', 'auto')
        selection = settings.get('selection', 'auto')
        if validation not in {'auto', *VALIDATION} or selection not in SELECTION:
            raise ValueError('Unknown validation or final evaluation setting.')
        if not isinstance(settings.get('figures', True), bool):
            raise ValueError('The figures setting must be true or false.')
        theme, formats, figure_keys = options.read_figures(settings, FIGURES, list(FIGURES))
        signal = [str(c) for c in sd.signal_columns(df, label)]
        if not signal:
            raise ValueError(f'No numeric signal values were found. Each recording needs at least {sd.MIN_LENGTH} numeric columns.')
        cols = signal_range(df, label, settings.get('first', signal[0]), settings.get('last', signal[-1]))
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
            chosen = list(dict.fromkeys(keys))
            deep_used = any(registry[k].family == 'deep' for k in chosen)
            method = None if len(chosen) < 2 else (selection if selection != 'auto' else
                                                   ('final_test' if deep_used or len(df) > 2000 else 'nested'))
            emitter.tracker = ProgressTracker(*table_rules(len(chosen), method, references=1,
                                                           figures=settings.get('figures', True), importance=False))
            deep.EPOCH_HOOK = emitter.epoch
            os.environ['MPLCONFIGDIR'] = str(session / 'plot-cache')
            demo = dataset == 'demo'
            run = run_signal_classification(
                df[[label] + cols], label,
                SignalSettings(models=list(dict.fromkeys(keys)), validation=validation, selection=selection,
                               figures=settings.get('figures', True), figure_theme=theme,
                               figure_format=formats, figure_keys=figure_keys),
                str(session), source_name=sd.DEMO_NAME if demo else Path(dataset).name,
                source_stem='demo_heartbeats' if demo else Path(dataset).stem)
            out = Path(run['out'])
            best, shown, reference = run['best'], run['shown'], run['reference']
            class_names = list(run['class_names'])
            figures = [dict(key=k, title=FIGURES[k], caption=CAPTIONS.get(k, ''), path=str(out / files['png']))
                       for k, files in run['figures'].items() if 'png' in files and k in FIGURES]
            figures.sort(key=lambda f: list(FIGURES).index(f['key']))
            explanation = {
                'final_test': f'Final evaluation on {run["test_rows"]:,} recordings (20%) kept untouched while the models were compared.',
                'nested': 'Nested cross-validation evaluates the whole selection procedure, so the score is not inflated by choosing the best model.',
            }.get(run['method'], '')
            result = dict(module_id=self.manifest['id'], package_version=easyresearch.__version__,
                          selected_model=best.classifier_name, evaluation_method=run['method'],
                          evaluation_text=explanation,
                          final=result_row(shown),
                          baseline=result_row(reference, True) if reference is not None else None,
                          comparison=[result_row(r) for r in run['results']] + ([result_row(reference, True)] if reference is not None else []),
                          selection_metric='Balanced accuracy',
                          rows_loaded=len(df), rows_used=run['n'],
                          development_rows=run['dev_rows'], test_rows=run['test_rows'],
                          class_names=class_names, positive_class=class_names[1] if len(class_names) == 2 else None,
                          summary=run['summary'],
                          notes=[f'{run["n"]:,} recordings of {run["length"]} values each (columns {cols[0]} to {cols[-1]}), single channel.']
                                + (['These are synthetic heartbeats for trying the tool, not real patient data.'] if demo else [])
                                + run['notes']
                                + (['KNN used the Hassanat distance; please cite: ' + ' '.join(HASSANAT_CITATIONS)]
                                   if any(r.classifier_key == 'knn' for r in run['results']) else []),
                          warnings=run['warnings'],
                          figures=figures,
                          output_dir=str(out), session_dir=str(session),
                          artifacts=[dict(name=str(p.relative_to(out)), path=str(p)) for p in sorted(out.rglob('*')) if p.is_file()],
                          environment={name: version(name) for name in ['easyresearch', 'easyclassifier', 'pandas', 'numpy', 'scikit-learn',
                                                                        'torch', 'matplotlib', 'joblib']})
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
