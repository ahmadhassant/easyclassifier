from __future__ import annotations
import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from easyclassifier import dataset as ds, preprocessing as pp, target as tg, recommend as rec
from easyclassifier.demo_data import load_demo
from easyclassifier.evaluation import recommend_selection, VALIDATION, SELECTION
from easyclassifier.distances import DISTANCES, make_knn
from easyclassifier.models import build_registry
from easyclassifier.wizard import Wizard
from easyclassifier.figures import FIGURES, STANDARD_FIGURES
from easyclassifier.latex_report import CAPTIONS

from . import advice, options
from .record import write_run_record
from progress import ProgressTracker, table_rules

METRIC_SPECS = [
    dict(key='accuracy', label='Accuracy', format='percent', higher_is_better=True),
    dict(key='balanced_accuracy', label='Balanced accuracy', format='percent', higher_is_better=True),
    dict(key='precision', label='Precision', format='percent', higher_is_better=True),
    dict(key='recall', label='Recall', format='percent', higher_is_better=True),
    dict(key='f1', label='F1', format='percent', higher_is_better=True),
    dict(key='specificity', label='Specificity', format='percent', higher_is_better=True),
    dict(key='roc_auc', label='ROC AUC', format='decimal4', higher_is_better=True),
    dict(key='mcc', label='MCC', format='decimal4', higher_is_better=True),
    dict(key='cohen_kappa', label="Cohen's kappa", format='decimal4', higher_is_better=True),
    dict(key='log_loss', label='Log loss', format='decimal4', higher_is_better=False),
]
# Measured unless the user changes the list in Advanced options.
USUAL_METRICS = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']
METRIC_HINTS = {'specificity': 'Specificity (macro)', 'log_loss': 'Log loss (lower is better)'}

# Advanced options: the choices of EasyClassifier's Advanced and Research modes.
NEIGHBOURS = ['1', '3', '5', '7', '9', '11', '15']
MISSING = [('auto', 'Automatic (recommended)'), ('drop', 'Remove rows with empty cells'),
           ('mean', 'Fill with the mean'), ('median', 'Fill with the median'),
           ('mode', 'Fill with the most common value')]
ENCODING = [('auto', 'Automatic (recommended)'), ('label', 'Label encoding (one number per category)'),
            ('onehot', 'One-hot encoding (one column per category)')]
SCALING = [('auto', 'Automatic: decided for each classifier (recommended)'), ('standard', 'Standardise all (mean 0, SD 1)'),
           ('minmax', 'Scale all to 0-1'), ('none', 'No scaling')]
SCALE_SETTING = {'standard': ('standard', True), 'minmax': ('minmax', True), 'none': ('none', False)}

UNSUITABLE = {
    tg.MEASUREMENT: 'It holds numeric measurements; use Regression to predict a number.',
    tg.ID_LIKE: 'It is different in every row and looks like an identifier.',
    tg.CONSTANT: 'It has only one value, so there is nothing to predict.',
    tg.EMPTY: 'It is empty.',
    tg.MANY_TEXT: 'It has too many different text values to be a set of classes.',
}

def plain_caption(text):
    """LaTeX captions from the package, as plain text for the interface."""
    for a, b in (('e.g.\\ ', 'e.g. '), ('i.e.\\ ', 'i.e. '), ('$\\pm 1$', '±1'), ('$-1$', '−1'), ('$+1$', '+1'), ('$', '')):
        text = text.replace(a, b)
    return text

def load_data(path):
    return load_demo() if path == 'demo' else ds.load_csv(path)

def number(value):
    value = float(value)
    return value if math.isfinite(value) else None

def result_row(result):
    return dict(key=result.classifier_key, name=result.classifier_name,
                metrics={key:number(value) for key,value in result.metrics.items()},
                balanced_accuracy=number(result.selection_score),
                selection_score=number(result.selection_score), note=result.note, reference=False)

class DesktopWizard(Wizard):
    """Reuse package evaluation and exporters, replacing only interaction."""
    captured = None
    missing_choice = 'auto'     # Advanced options; 'auto' keeps the package's recommendation
    encoding_choice = 'auto'

    def preprocess(self, auto):
        # The package's automatic preparation, with the user's missing-value and
        # encoding choices in place of its two recommendations.
        saved = rec.missing_strategy, rec.encoding_method
        if self.missing_choice != 'auto':
            rec.missing_strategy = lambda insp: self.missing_choice
        if self.encoding_choice != 'auto':
            rec.encoding_method = lambda df, target: self.encoding_choice
        try:
            return super().preprocess(auto)
        finally:
            rec.missing_strategy, rec.encoding_method = saved

    def show_results(self, best, results, metrics, final=None):
        if len(results) > 1 and final is None:
            raise RuntimeError('Final evaluation failed. Comparison scores cannot be reported as a final result. Try cross-validation or use classes with more observations.')
        self.captured = dict(best=best, results=results, final=final)
        super().show_results(best, results, metrics, final)

class ClassificationModule:
    def __init__(self, manifest):
        self.manifest = manifest
        installed = version('easyclassifier')
        if installed != manifest['package_version']:
            raise RuntimeError(f'This adapter requires EasyClassifier {manifest["package_version"]}; installed version is {installed}.')

    def describe(self):
        defaults = {'decision_tree', 'random_forest', 'logistic_regression'}
        result = {key:value for key,value in self.manifest.items() if key != 'adapter'}
        result['models'] = [dict(id=key, name=spec.name, available=spec.available,
                                 default_selected=key in defaults, reason=spec.reason)
                            for key,spec in build_registry().items()]
        result['metrics'] = METRIC_SPECS
        result['headline'] = ['accuracy', 'balanced_accuracy', 'roc_auc']
        result['validation_options'] = [dict(id='auto', name='Automatic (recommended)'),
                                        dict(id='kfold5', name='Stratified 5-fold cross-validation'),
                                        dict(id='kfold10', name='Stratified 10-fold cross-validation'),
                                        dict(id='holdout', name='Single 80/20 validation split'),
                                        dict(id='loo', name='Leave-one-out (slow; for very small datasets)')]
        figure_choices, figure_lists = options.figure_options(FIGURES, STANDARD_FIGURES,
            'The usual figures are ticked; precision-recall curves and the missing-value map are added when the data call for them. '
            'The learning curve trains the selected classifier 25 more times.')
        result['advanced_choices'] = [
            options.choice('knn_distance', 'KNN distance', [(k, v[0]) for k, v in DISTANCES.items()], 'hassanat',
                           'Used only when KNN is selected. The Hassanat distance is robust to outliers and to the scale of the columns.'),
            options.choice('knn_k', 'KNN neighbours (k)', [(k, k) for k in NEIGHBOURS], '5',
                           'How many nearest rows vote on the class.'),
            options.choice('missing', 'Empty cells', MISSING, 'auto',
                           'Automatic fills them with the median when few cells are empty and removes incomplete rows otherwise. Fill values are learned from training rows only.'),
            options.choice('encoding', 'Text columns', ENCODING, 'auto',
                           'Automatic uses one-hot encoding when every text column has at most 10 categories.'),
            options.choice('scaling', 'Scaling of numeric columns', SCALING, 'auto',
                           'Automatic scales only for the classifiers that need it (KNN, SVM, logistic regression, neural network); scaling is learned from training rows only.'),
        ] + figure_choices
        result['advanced_lists'] = [
            options.tick_list('metrics', 'Measures', [(m['key'], METRIC_HINTS.get(m['key'], m['label'])) for m in METRIC_SPECS],
                              USUAL_METRICS, 'Balanced accuracy is always used to choose the classifier; the other measures are reported.'),
        ] + figure_lists
        result['selection_options'] = [dict(id='auto', name='Automatic (recommended)'),
                                       dict(id='nested', name='Nested cross-validation'),
                                       dict(id='final_test', name='Separate 20% final test')]
        result['labels'] = dict(
            target_title='What do you want to predict?',
            target_hint='Choose a category, such as a diagnosis, species or outcome. Numeric measurements belong in Regression.',
            models_title='Compare classifiers',
            model_noun='classifier', model_noun_plural='classifiers',
            demo_button='Try Iris example',
            selection_metric='Balanced accuracy',
            prep_note="Missing values, encoding and scaling use the package's automatic preparation. Multiple classifiers receive a separate final evaluation.")
        return result

    def inspect(self, dataset):
        df=load_data(dataset)
        if df.empty or df.shape[1] < 2:
            raise ValueError('Choose a spreadsheet with observations and at least two columns.')
        if df.columns.duplicated().any():
            raise ValueError('Column names must be unique.')
        info=ds.inspect(df)
        suggested=tg.suggest_target(df)
        return dict(dataset=dataset, rows=info.n_rows, columns_count=info.n_cols,
                    missing_cells=info.missing_total, duplicates=info.duplicate_rows,
                    suggested_target=suggested,
                    advice=advice.other_example(dataset, 'Classification') + advice.safely(lambda: advice.classification(df, info, suggested)),
                    columns=[self._column(df[c]) for c in df.columns],
                    preview=df.head(8).fillna('').astype(str).to_dict(orient='records'))

    @staticmethod
    def _column(series):
        info = tg.describe_column(series)
        return dict(name=str(series.name), description=info.label, kind=info.kind,
                    suitable=info.kind == tg.CATEGORY, note=UNSUITABLE.get(info.kind, ''))

    def run(self, request, emitter):
        dataset=request.get('dataset', '')
        df=load_data(dataset)
        target=request.get('target', '')
        if target not in df.columns:
            raise ValueError('Choose a target column from the inspected dataset.')
        kind=tg.describe_column(df[target]).kind
        if kind in (tg.EMPTY, tg.CONSTANT, tg.ID_LIKE, tg.MEASUREMENT, tg.MANY_TEXT):
            raise ValueError('Choose a categorical target with repeated class labels. Numeric measurements require regression; automatic grouping is not performed by this interface.')
        settings=request.get('settings') or {}
        keys=settings.get('models', settings.get('classifiers', []))
        registry=build_registry()
        if not isinstance(keys,list) or not keys or any(not isinstance(key,str) or key not in registry or not registry[key].available for key in keys):
            raise ValueError('Select at least one available classifier.')
        keys=list(dict.fromkeys(keys))
        validation=settings.get('validation','auto')
        selection=settings.get('selection','auto')
        if validation not in {'auto', *VALIDATION} or selection not in SELECTION:
            raise ValueError('Unknown validation or final evaluation setting.')
        if not isinstance(settings.get('figures',True),bool):
            raise ValueError('The figures setting must be true or false.')
        metrics=options.read_list(settings,'metrics',[m['key'] for m in METRIC_SPECS],'measures')
        metrics=list(USUAL_METRICS) if metrics is None else metrics
        if not metrics:
            raise ValueError('Tick at least one measure in Advanced options.')
        if isinstance(settings.get('knn_k'),int) and not isinstance(settings.get('knn_k'),bool):
            settings=dict(settings,knn_k=str(settings['knn_k']))
        knn_distance=options.read_choice(settings,'knn_distance',DISTANCES,'hassanat','KNN distance')
        knn_k=int(options.read_choice(settings,'knn_k',NEIGHBOURS,'5','number of neighbours'))
        missing=options.read_choice(settings,'missing',[k for k,_ in MISSING],'auto','choice for empty cells')
        encoding=options.read_choice(settings,'encoding',[k for k,_ in ENCODING],'auto','encoding')
        scaling=options.read_choice(settings,'scaling',[k for k,_ in SCALING],'auto','scaling')
        theme,formats,figure_keys=options.read_figures(settings,FIGURES,STANDARD_FIGURES)
        figures_on=settings.get('figures',True) and figure_keys!=[]
        output_root=Path(request.get('output_root','')).expanduser()
        if not str(request.get('output_root','')).strip() or not output_root.is_absolute():
            raise ValueError('Choose an absolute output folder.')

        session=output_root / ('analysis_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        session.mkdir(parents=True)
        old_cwd=Path.cwd()
        original_request=dict(request)
        (session/'request.json').write_text(json.dumps(original_request,ensure_ascii=False,indent=2),encoding='utf-8')
        (session/'status.json').write_text(json.dumps({'state':'running'}),encoding='utf-8')
        emitter.send('progress',message='Preparing the data and checking class sizes.',session_dir=str(session))
        try:
            os.chdir(session)
            os.environ['MPLCONFIGDIR']=str(session/'plot-cache')
            wizard=DesktopWizard()
            wizard.missing_choice, wizard.encoding_choice = missing, encoding
            wizard.df=df.copy()
            wizard.df_loaded=df.copy()
            wizard.source_name='Iris demonstration' if dataset=='demo' else Path(dataset).name
            wizard.source_stem='demo_iris' if dataset=='demo' else Path(dataset).stem
            wizard.rows_loaded=len(df)
            wizard.cols_loaded=df.shape[1]
            wizard.target=target
            wizard.target_display=target
            wizard.mode='beginner'
            wizard.insp=ds.inspect(df)
            wizard.drop_useless_columns(auto=True)
            if wizard.df.shape[1] < 2:
                raise ValueError('No usable predictor columns remain after excluding identifiers and constant columns.')
            wizard.insp=ds.inspect(wizard.df)
            if missing=='drop' and wizard.df.drop(columns=[target]).isna().any(axis=1).all():
                raise ValueError('Every row has at least one empty cell, so removing incomplete rows would leave nothing. Choose to fill the empty cells instead (Advanced options).')
            X,y,class_names=wizard.preprocess(auto=True)
            counts=np.unique(y,return_counts=True)[1]
            if len(counts)<2 or counts.min()<2:
                raise ValueError('At least two classes are required, each with at least two observations after cleaning.')
            if scaling!='auto':
                wizard.prep_cfg.scale_method,wizard.prep_cfg.scale=SCALE_SETTING[scaling]
                wizard.log.add(f'Scaling choice (Advanced options): {scaling}')
            if 'knn' in keys:
                wizard.knn_distance=knn_distance
                wizard.knn_k=knn_k
                spec=wizard.registry['knn']
                spec.factory=lambda d=knn_distance,kk=knn_k: make_knn(d,kk)
                spec.name=f"KNN ({DISTANCES[knn_distance][0].split(' (')[0]}, k={knn_k})"
                wizard.log.add(f'KNN distance: {knn_distance}, k={knn_k}')
            validation=rec.validation_method(y) if validation=='auto' else validation
            # A holdout or outer CV can leave too few rare-class rows for model selection.
            final_method=recommend_selection(y) if selection=='auto' else selection
            if len(keys)>1 and final_method=='nested' and counts.min()<3:
                raise ValueError('Comparing multiple classifiers with nested validation needs at least three rows in every class. Choose one classifier or use more observations.')
            if validation=='holdout' or (len(keys)>1 and final_method=='final_test'):
                from sklearn.model_selection import train_test_split
                train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
            wizard.selection=final_method if len(keys)>1 else None
            if not figures_on:
                figures=[]
            elif figure_keys is None:
                figures=wizard.choose_figures(True,y)
            else:
                figures=figure_keys
                wizard.log.add('Figures (Advanced options): '+', '.join(figures))
            wizard.fig_theme,wizard.fig_formats=theme,formats
            if figures and (theme,formats)!=('colorblind','png'):
                wizard.log.add(f'Figure theme {theme}; format {formats}')
            if metrics!=USUAL_METRICS:
                wizard.log.add('Measures (Advanced options): '+', '.join(metrics))
            rules,total=table_rules(len(keys), wizard.selection, figures=bool(figures),
                                    importance=bool({'feature_importance','columns_by_class'} & set(figures)), final_fit=False)
            if 'learning_curve' in figures:
                rules,total=rules+[(r'^Computing the learning curve', 3, 'start')],total+3
            emitter.tracker=ProgressTracker(rules,total)
            wizard.log.add('Desktop configuration; EasyClassifier '+version('easyclassifier'))
            wizard.execute(X,y,class_names,keys,validation,metrics,figures)
            if wizard.captured is None:
                raise RuntimeError('No classifier could be evaluated. Inspect the analysis log for details.')
            captured=wizard.captured
            out=next((session/'Results').iterdir())
            shown=captured['final'] or captured['best']
            method=wizard.selection or ('holdout' if validation=='holdout' else 'cross_validation')
            figures=[dict(key=key,title=FIGURES[key],caption=plain_caption(CAPTIONS.get(key,'')),path=str(out/'figures'/f'{key}.png'))
                     for key in FIGURES if (out/'figures'/f'{key}.png').exists()]
            asked=dict(models=keys, validation=settings.get('validation', 'auto'),
                       selection=settings.get('selection', 'auto'), figures=settings.get('figures', True),
                       metrics=metrics, knn_distance=knn_distance, knn_k=knn_k, missing=missing,
                       encoding=encoding, scaling=scaling, **options.describe_choices(theme, formats, figure_keys))
            summary=[text for text in (wizard.comparison_text, wizard.learning_text) if text]
            write_run_record(str(out), 'classification', wizard.source_name, target,
                             asked,
                             dict(models=keys, validation=validation, final_evaluation=method,
                                  rows_used=len(y), predictors=list(X.columns), classes=list(class_names),
                                  knn_distance=knn_distance if 'knn' in keys else None,
                                  knn_k=knn_k if 'knn' in keys else None, metrics=metrics,
                                  missing_values=wizard.impute_used or ('removed rows' if missing=='drop' else None),
                                  encoding=wizard.encoding_used, scaling=wizard._scale_for,
                                  figures=[f['key'] for f in figures], figure_theme=theme, figure_format=formats))
            result=dict(module_id=self.manifest['id'],package_version=version('easyclassifier'),
                        selected_model=captured['best'].classifier_name,
                        evaluation_method=method, final=result_row(shown),
                        comparison=[result_row(row) for row in captured['results']],
                        selection_metric='Balanced accuracy', summary=summary, figures=figures,
                        rows_loaded=len(df),rows_used=len(y),predictors=list(X.columns),
                        class_names=class_names, positive_class=class_names[1] if len(class_names)==2 else None,
                        class_counts={name:int((y==i).sum()) for i,name in enumerate(class_names)},
                        development_rows=len(wizard._dev) if wizard._dev is not None else None,
                        test_rows=len(wizard._test) if wizard._test is not None else len(shown.y_true),
                        notes=wizard.data_notes,
                        output_dir=str(out),session_dir=str(session),
                        artifacts=[dict(name=str(p.relative_to(out)),path=str(p)) for p in sorted(out.rglob('*')) if p.is_file()],
                        warnings=[line.strip() for line in (out/'log.txt').read_text(encoding='utf-8').splitlines() if 'FAILED' in line],
                        environment={name:version(name) for name in ['easyclassifier','pandas','numpy','scikit-learn','matplotlib','joblib']})
            (out/'desktop-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            (session/'status.json').write_text(json.dumps({'state':'completed','output_dir':str(out)}),encoding='utf-8')
            return result
        except Exception:
            (session/'status.json').write_text(json.dumps({'state':'failed'}),encoding='utf-8')
            raise
        finally:
            os.chdir(old_cwd)
