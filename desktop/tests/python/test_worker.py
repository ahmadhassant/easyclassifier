"""Protocol and adapter checks against the installed PyPI package."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'worker'))

def request(action,**fields):
    data=dict(protocol_version=1,request_id=uuid.uuid4().hex,action=action,**fields)
    process=subprocess.run([sys.executable,str(ROOT/'worker/worker.py'),'--manifest-dir',str(ROOT/'modules')],
                           input=json.dumps(data)+'\n',text=True,encoding='utf-8',capture_output=True,timeout=180)
    events=[json.loads(line) for line in process.stdout.splitlines()]
    assert all(e['request_id']==data['request_id'] for e in events)
    assert all(e['protocol_version']==1 for e in events)
    return process,events

class WorkerTests(unittest.TestCase):
    def test_catalog_and_inspection(self):
        p,events=request('catalog')
        self.assertEqual(p.returncode,0)
        module=events[-1]['result']['modules'][0]
        self.assertEqual(module['id'],'classification')
        # XGBoost and LightGBM are offered when their packages are installed (they are in the lock file).
        import importlib.util
        boosters=sum(importlib.util.find_spec(name) is not None for name in ('xgboost','lightgbm'))
        self.assertEqual(sum(m['available'] for m in module['models']),7+boosters)
        p,events=request('inspect',module_id='classification',dataset='demo')
        self.assertEqual(p.returncode,0)
        self.assertEqual(events[-1]['result']['rows'],150)
        self.assertEqual(events[-1]['result']['suggested_target'],'Species')

    def test_broken_module_does_not_hide_the_others(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(ROOT/'modules'/'classification.json',folder)
            (Path(folder)/'broken.json').write_text(json.dumps(dict(protocol_version=1,id='broken',title='Broken',
                adapter='adapters.does_not_exist:Module')),encoding='utf-8')
            data=dict(protocol_version=1,request_id='r1',action='catalog')
            process=subprocess.run([sys.executable,str(ROOT/'worker/worker.py'),'--manifest-dir',folder],
                                   input=json.dumps(data)+'\n',text=True,encoding='utf-8',capture_output=True,timeout=180)
            events=[json.loads(line) for line in process.stdout.splitlines()]
            self.assertEqual(process.returncode,0)
            modules={m['id']:m for m in events[-1]['result']['modules']}
            self.assertEqual(list(modules),['broken','classification'])
            self.assertNotIn('unavailable_reason',modules['classification'])
            self.assertEqual(modules['broken']['title'],'Broken')
            self.assertIn('Python package',modules['broken']['unavailable_reason'])
            self.assertTrue(any('Broken module could not be loaded' in e.get('message','') for e in events))

    def test_protocol_is_utf8_under_a_windows_code_page(self):
        # Without the desktop's PYTHONIOENCODING, Windows Python writes cp1252;
        # the worker must still speak UTF-8 (R², CO₂, Arabic file names).
        import os, pandas as pd
        env=dict(os.environ,PYTHONIOENCODING='cp1252')
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'بيانات المبيعات.csv'
            pd.DataFrame({'x':range(40),'المبيعات':[float(i%7) for i in range(40)]}).to_csv(data,index=False)
            for action,fields in (('catalog',{}),('inspect',dict(module_id='regression',dataset=str(data)))):
                payload=dict(protocol_version=1,request_id='r1',action=action,**fields)
                process=subprocess.run([sys.executable,str(ROOT/'worker/worker.py'),'--manifest-dir',str(ROOT/'modules')],
                                       input=(json.dumps(payload,ensure_ascii=False)+'\n').encode('utf-8'),capture_output=True,timeout=180,env=env)
                events=[json.loads(line) for line in process.stdout.decode('utf-8').splitlines()]
                self.assertEqual(process.returncode,0,events[-1])
            self.assertEqual(events[-1]['result']['dataset'],str(data))
            self.assertIn('المبيعات',[c['name'] for c in events[-1]['result']['columns']])

    def test_unknown_module_returns_protocol_error(self):
        p,events=request('inspect',module_id='missing',dataset='demo')
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(events[-1]['type'],'error')
        self.assertIn('Unknown analysis module',events[-1]['message'])

    def test_missing_file_returns_protocol_error(self):
        p,events=request('inspect',module_id='classification',dataset='does-not-exist.csv')
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(events[-1]['type'],'error')

    def test_continuous_target_is_not_silently_binned(self):
        with tempfile.TemporaryDirectory() as folder:
            p,events=request('run',module_id='classification',dataset='demo',target='sepal_length',
                             output_root=folder,settings={'classifiers':['decision_tree']})
            self.assertNotEqual(p.returncode,0)
            self.assertIn('categorical target',events[-1]['message'])

    def test_nested_demo_exports_actual_final_predictions(self):
        import pandas as pd
        from sklearn.metrics import accuracy_score
        with tempfile.TemporaryDirectory() as folder:
            p,events=request('run',module_id='classification',dataset='demo',target='Species',output_root=folder,
                             settings={'classifiers':['decision_tree','logistic_regression'],'figures':False})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['evaluation_method'],'nested')
            self.assertEqual(result['rows_used'],149)
            self.assertEqual(len(result['comparison']),2)
            self.assertEqual(result['final']['key'],'nested')
            prediction=pd.read_csv(Path(result['output_dir'])/'predictions.csv')
            self.assertAlmostEqual(accuracy_score(prediction.actual,prediction.predicted),result['final']['metrics']['accuracy'])
            self.assertEqual(json.loads((Path(result['session_dir'])/'status.json').read_text())['state'],'completed')
            self.assertTrue((Path(result['output_dir'])/'trained_model.pkl').exists())

    def test_rare_class_error_does_not_report_optimistic_fallback(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'rare.csv'
            pd.DataFrame({'x':[i/21 for i in range(22)],'Outcome':['A']*20+['B']*2}).to_csv(data,index=False)
            p,events=request('run',module_id='classification',dataset=str(data),target='Outcome',output_root=folder,
                             settings={'classifiers':['decision_tree','random_forest'],'figures':False})
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(events[-1]['type'],'error')
            self.assertIn('at least three rows',events[-1]['message'])
            self.assertFalse(any(e['type']=='completed' for e in events))

    def test_inspection_marks_unsuitable_columns(self):
        p,events=request('inspect',module_id='classification',dataset='demo')
        columns={c['name']:c for c in events[-1]['result']['columns']}
        self.assertTrue(columns['Species']['suitable'])
        self.assertFalse(columns['sepal_length']['suitable'])
        self.assertIn('Regression',columns['sepal_length']['note'])

    def test_classification_result_lists_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            p,events=request('run',module_id='classification',dataset='demo',target='Species',output_root=folder,
                             settings={'models':['decision_tree','logistic_regression']})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertTrue(result['figures'])
            self.assertTrue(all(Path(f['path']).exists() and '$' not in f['caption'] for f in result['figures']))
            self.assertEqual(result['selection_metric'],'Balanced accuracy')


class RegressionTests(unittest.TestCase):
    def test_desktop_uses_the_shared_core(self):
        # One regression engine for the desktop and the command line.
        self.assertFalse(list((ROOT/'worker'/'easyregressor').glob('*.py')))
        source=(ROOT/'worker'/'adapters'/'regression.py').read_text(encoding='utf-8')
        self.assertIn('from easyresearch.regression',source)

    def test_catalog_describes_regression(self):
        p,events=request('catalog')
        modules={m['id']:m for m in events[-1]['result']['modules']}
        module=modules['regression']
        self.assertEqual(module['headline'],['r2','rmse','mae'])
        self.assertIn('mhsp',{m['key'] for m in module['metrics']})
        self.assertEqual(module['package'],'easyresearch')
        self.assertGreaterEqual(sum(m['available'] for m in module['models']),8)
        self.assertEqual({m['id'] for m in module['models'] if m['default_selected']},
                         {'linear_regression','random_forest','gradient_boosting'})

    def test_demo_inspection_suggests_numeric_target(self):
        p,events=request('inspect',module_id='regression',dataset='demo')
        result=events[-1]['result']
        self.assertEqual((result['rows'],result['suggested_target']),(442,'Progression'))
        columns={c['name']:c for c in result['columns']}
        self.assertFalse(columns['sex']['suitable'])        # two values: classification
        self.assertTrue(columns['bmi']['suitable'])

    def test_nested_demo_scores_match_saved_predictions(self):
        import pandas as pd, joblib
        from sklearn.metrics import r2_score, mean_absolute_error
        from easyresearch.regression.data import load_demo
        with tempfile.TemporaryDirectory() as folder:
            p,events=request('run',module_id='regression',dataset='demo',target='Progression',output_root=folder,
                             settings={'models':['linear_regression','random_forest','knn'],'figures':True})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['evaluation_method'],'nested')
            self.assertEqual(result['final']['key'],'nested')
            out=Path(result['output_dir'])
            pred=pd.read_csv(out/'predictions.csv')
            self.assertEqual(len(pred),442)
            self.assertEqual(sorted(pred.row_in_file),list(range(2,444)))
            self.assertAlmostEqual(r2_score(pred.actual,pred.predicted),result['final']['metrics']['r2'])
            self.assertAlmostEqual(mean_absolute_error(pred.actual,pred.predicted),result['final']['metrics']['mae'])
            from easyresearch.common.hassanat import mhsp
            self.assertAlmostEqual(mhsp(pred.actual,pred.predicted),result['final']['metrics']['mhsp'])
            self.assertIn('MHSP',(out/'report.tex').read_text(encoding='utf-8'))
            self.assertGreater(result['final']['metrics']['r2'],.4)
            self.assertLess(abs(result['baseline']['metrics']['r2']),.05)
            self.assertTrue(result['comparison'][-1]['reference'])
            self.assertEqual({f['key'] for f in result['figures']},
                             {'target_distribution','comparison','predicted_vs_actual','residuals','feature_importance'})
            model=joblib.load(out/'trained_model.pkl')
            demo=load_demo()
            self.assertEqual(len(model.predict(demo.drop(columns=['Progression']).head(3))),3)
            self.assertIn('Hassanat',(out/'citations.txt').read_text(encoding='utf-8'))
            self.assertNotIn('Journal of American Science',(out/'report.tex').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((Path(result['session_dir'])/'status.json').read_text())['state'],'completed')

    def test_text_target_is_refused(self):
        import pandas as pd
        from easyclassifier.demo_data import load_demo
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'iris.csv'
            load_demo().to_csv(data,index=False)
            p,events=request('run',module_id='regression',dataset=str(data),target='Species',output_root=folder,
                             settings={'models':['linear_regression']})
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Classification',events[-1]['message'])

    def test_large_data_uses_final_test_and_handles_text_and_gaps(self):
        import numpy as np, pandas as pd
        rng=np.random.default_rng(1)
        n=2400
        x=rng.normal(size=n); group=rng.choice(['north','south','east'],n)
        y=3*x+np.where(group=='north',2,0)+rng.normal(scale=.5,size=n)
        x_missing=x.copy(); x_missing[rng.choice(n,100,replace=False)]=np.nan
        frame=pd.DataFrame({'id':np.arange(n),'x':x_missing,'region':group,'Yield':y})
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'large.csv'
            frame.to_csv(data,index=False)
            p,events=request('run',module_id='regression',dataset=str(data),target='Yield',output_root=folder,
                             settings={'models':['linear_regression','decision_tree'],'figures':False})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['evaluation_method'],'final_test')
            self.assertEqual(result['development_rows']+result['test_rows'],result['rows_used'])
            self.assertNotIn('id',result['predictors'])
            self.assertGreater(result['final']['metrics']['r2'],.9)

    def test_single_model_uses_cross_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            p,events=request('run',module_id='regression',dataset='demo',target='bmi',output_root=folder,
                             settings={'models':['ridge'],'validation':'kfold10','figures':False})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['evaluation_method'],'cross_validation')
            self.assertEqual(len(result['comparison']),2)

    def test_hassanat_knn_regressor_matches_brute_force(self):
        import numpy as np
        from easyresearch.regression.models import HassanatKNNRegressor
        from easyclassifier.distances import hassanat_distance
        rng=np.random.default_rng(0)
        X=rng.normal(size=(40,3)); y=rng.normal(size=40); Q=rng.normal(size=(6,3))
        got=HassanatKNNRegressor(5).fit(X,y).predict(Q)
        for q,value in zip(Q,got):
            d=np.array([hassanat_distance(q,row) for row in X])
            self.assertAlmostEqual(value,y[np.argsort(d)[:5]].mean())

class ForecastingTests(unittest.TestCase):
    def test_catalog_describes_forecasting(self):
        p,events=request('catalog')
        modules={m['id']:m for m in events[-1]['result']['modules']}
        module=modules['forecasting']
        self.assertEqual(module['package'],'easyresearch')
        self.assertEqual(module['headline'],['mase','mae','mhsp'])
        self.assertEqual((module['validation_options'],module['selection_options']),([],[]))
        ids={m['id'] for m in module['models']}
        self.assertFalse(ids & {'naive','seasonal_naive'})            # always included as references
        self.assertTrue({'ets','theta','ridge','knn','lstm'} <= ids)

    def test_demo_inspection_offers_date_horizon_and_measure(self):
        p,events=request('inspect',module_id='forecasting',dataset='demo')
        self.assertEqual(p.returncode,0,events[-1])
        result=events[-1]['result']
        self.assertEqual(result['suggested_target'],'CO2_ppm')
        columns={c['name']:c for c in result['columns']}
        self.assertFalse(columns['month']['suitable'])
        params={q['id']:q for q in result['parameters']}
        self.assertEqual(list(params),['time','horizon','metric'])
        for q in params.values():
            self.assertIn(q['default'],[o['id'] for o in q['options']])
        self.assertIn('month',params['time']['options'][0]['name'])
        self.assertIn('12 months',params['horizon']['options'][0]['name'])
        self.assertTrue(all(int(o['id'])<=526//5 for o in params['horizon']['options'][1:]))

    def test_demo_run_is_time_ordered_and_scores_match_saved_forecasts(self):
        import numpy as np, pandas as pd, joblib
        from easyresearch.timeseries import data as td
        with tempfile.TemporaryDirectory() as folder:
            p,events=request('run',module_id='forecasting',dataset='demo',target='CO2_ppm',output_root=folder,
                             settings={'models':['ridge','theta'],'horizon':'6','metric':'mase','figures':True})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['evaluation_method'],'rolling_origin')
            self.assertEqual(result['rows_used'],526)
            self.assertEqual(result['development_rows']+result['test_rows'],526)
            self.assertTrue(result['baseline']['reference'])
            self.assertEqual({r['name'] for r in result['comparison'] if r['reference']},
                             {'Naive (last value)','Seasonal naive (value one season ago)'})
            out=Path(result['output_dir'])
            final=pd.read_csv(out/'final_period_forecasts.csv')
            self.assertAlmostEqual(final.error.abs().mean(),result['final']['metrics']['mae'])
            self.assertAlmostEqual(float(np.sqrt((final.error**2).mean())),result['final']['metrics']['rmse'])
            # The final period starts where the comparison stopped, and every forecast is of later values.
            index=td.prepare_series(td.load_demo(),'CO2_ppm').index
            origins=pd.to_datetime(final.forecast_origin)
            self.assertEqual(origins.min(),index[result['development_rows']])
            self.assertTrue((pd.to_datetime(final.month)>=origins).all())
            forecast=pd.read_csv(out/'forecast.csv')
            table=result['tables'][0]
            self.assertEqual(len(forecast),6)
            self.assertEqual(len(table['rows']),6)
            self.assertEqual(table['columns'][0],'month')
            self.assertEqual(table['rows'][0][0],'2002-01')
            self.assertEqual({f['key'] for f in result['figures']},
                             {'forecast','final_period','comparison','error_by_step','decomposition'})
            self.assertTrue(all(Path(f['path']).exists() for f in result['figures']))
            self.assertEqual(len(joblib.load(out/'trained_model.pkl').predict(6)),6)
            self.assertIn('statsmodels',(out/'citations.txt').read_text(encoding='utf-8').lower())
            self.assertIn('MHSP',(out/'report.tex').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((Path(result['session_dir'])/'status.json').read_text())['state'],'completed')

    def test_offered_horizons_are_always_accepted(self):
        import numpy as np, pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            for months in (30,40,60):
                data=Path(folder)/f'short_{months}.csv'
                t=np.arange(months)
                pd.DataFrame({'month':pd.date_range('2019-01-01',periods=months,freq='MS').strftime('%Y-%m'),
                              'sales':100+t+10*np.sin(2*np.pi*t/12)}).to_csv(data,index=False)
                p,events=request('inspect',module_id='forecasting',dataset=str(data))
                horizon={q['id']:q for q in events[-1]['result']['parameters']}['horizon']
                for option in horizon['options']:
                    with self.subTest(months=months,horizon=option['id']):
                        p,events=request('run',module_id='forecasting',dataset=str(data),target='sales',output_root=folder,
                                         settings={'models':['ridge','theta'],'horizon':option['id'],'figures':False})
                        self.assertEqual(p.returncode,0,events[-1])

    def test_series_without_dates_uses_row_order(self):
        import numpy as np, pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'readings.csv'
            t=np.arange(120)
            pd.DataFrame({'reading':10+0.05*t+np.sin(t/3)}).to_csv(data,index=False)
            p,events=request('inspect',module_id='forecasting',dataset=str(data))
            params={q['id']:q for q in events[-1]['result']['parameters']}
            self.assertIn('file order',params['time']['options'][0]['name'])
            self.assertIn('steps',params['horizon']['options'][0]['name'])
            p,events=request('run',module_id='forecasting',dataset=str(data),target='reading',output_root=folder,
                             settings={'models':['ridge'],'figures':False})
            self.assertEqual(p.returncode,0,events[-1])
            self.assertTrue(any('file order' in n for n in events[-1]['result']['notes']))

    def test_unsuitable_requests_are_refused_in_plain_language(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            short=Path(folder)/'short.csv'
            pd.DataFrame({'month':pd.date_range('2020-01-01',periods=12,freq='MS').strftime('%Y-%m'),
                          'sales':range(12)}).to_csv(short,index=False)
            p,events=request('run',module_id='forecasting',dataset=str(short),target='sales',output_root=folder,
                             settings={'models':['ridge']})
            self.assertNotEqual(p.returncode,0)
            self.assertIn('at least 30',events[-1]['message'])
            p,events=request('run',module_id='forecasting',dataset='demo',target='month',output_root=folder,
                             settings={'models':['ridge']})
            self.assertIn('cannot be forecast',events[-1]['message'])
            p,events=request('run',module_id='forecasting',dataset='demo',target='CO2_ppm',output_root=folder,
                             settings={'models':['ridge'],'horizon':'500'})
            self.assertIn('horizon must be between',events[-1]['message'])
            p,events=request('run',module_id='forecasting',dataset='demo',target='CO2_ppm',output_root=folder,
                             settings={'models':[]})
            self.assertIn('Select at least one',events[-1]['message'])

def write_ucr(path,n=60,length=50,extra_column=False):
    """A small UCR-style file: label first, no header; two shape classes."""
    import numpy as np
    rng=np.random.default_rng(3)
    t=np.linspace(0,1,length)
    lines=[]
    for i in range(n):
        cls=1+i%2
        wave=np.sin(2*np.pi*3*t) if cls==1 else np.sign(np.sin(2*np.pi*3*t))
        values=list(wave+rng.normal(0,.3,length))+([float(rng.integers(20,80))] if extra_column else [])
        lines.append('\t'.join([str(cls)]+[f'{v:.4f}' for v in values]))
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')

class SignalsTests(unittest.TestCase):
    def test_catalog_describes_signals(self):
        p,events=request('catalog')
        module={m['id']:m for m in events[-1]['result']['modules']}['signals']
        self.assertEqual(module['package'],'easyresearch')
        self.assertIn('*.tsv',module['file_filter'])
        self.assertNotIn('majority',{m['id'] for m in module['models']})     # always included as reference
        self.assertIn('single channel',module['labels']['target_hint'])
        self.assertTrue(module['validation_options'] and module['selection_options'])

    def test_demo_inspection_offers_signal_range(self):
        p,events=request('inspect',module_id='signals',dataset='demo')
        self.assertEqual(p.returncode,0,events[-1])
        result=events[-1]['result']
        self.assertEqual((result['rows'],result['suggested_target']),(300,'label'))
        params={q['id']:q for q in result['parameters']}
        self.assertEqual((params['first']['default'],params['last']['default']),('t1','t140'))
        self.assertEqual(len(params['first']['options']),140)
        self.assertIn('single channel',params['last']['hint'])

    def test_ucr_file_runs_and_predictions_match_scores(self):
        import pandas as pd
        from sklearn.metrics import accuracy_score
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'Waves_TRAIN.tsv'
            write_ucr(data)
            p,events=request('inspect',module_id='signals',dataset=str(data))
            self.assertEqual(events[-1]['result']['suggested_target'],'label')
            p,events=request('run',module_id='signals',dataset=str(data),target='label',output_root=folder,
                             settings={'models':['knn','features_rf'],'figures':True})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['evaluation_method'],'nested')
            self.assertEqual(result['class_names'],['1','2'])
            self.assertTrue(result['baseline']['reference'])
            self.assertAlmostEqual(result['baseline']['metrics']['balanced_accuracy'],.5)
            out=Path(result['output_dir'])
            pred=pd.read_csv(out/'predictions.csv')
            self.assertEqual(len(pred),60)
            self.assertAlmostEqual(accuracy_score(pred.actual.astype(str),pred.predicted.astype(str)),result['final']['metrics']['accuracy'])
            self.assertGreater(result['final']['metrics']['balanced_accuracy'],.8)
            self.assertIn('signals_by_class',{f['key'] for f in result['figures']})
            self.assertTrue(all(Path(f['path']).exists() for f in result['figures']))
            self.assertTrue(any('Hassanat' in n for n in result['notes']))
            self.assertTrue((out/'trained_model.pkl').exists())

    def test_signal_range_leaves_out_extra_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'with_age.tsv'
            write_ucr(data,extra_column=True)        # t51 is an age, not part of the signal
            p,events=request('run',module_id='signals',dataset=str(data),target='label',output_root=folder,
                             settings={'models':['knn'],'first':'t1','last':'t50','figures':False})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['evaluation_method'],'cross_validation')
            self.assertIn('50 values each (columns t1 to t50)',result['notes'][0])
            p,events=request('run',module_id='signals',dataset=str(data),target='label',output_root=folder,
                             settings={'models':['knn'],'first':'t40','last':'t10'})
            self.assertIn('must come before',events[-1]['message'])
            p,events=request('run',module_id='signals',dataset=str(data),target='label',output_root=folder,
                             settings={'models':['knn'],'first':'t1','last':'t5'})
            self.assertIn('at least 10',events[-1]['message'])

    def test_unsuitable_label_and_models_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'waves.tsv'
            write_ucr(data)
            p,events=request('run',module_id='signals',dataset=str(data),target='t3',output_root=folder,
                             settings={'models':['knn']})
            self.assertIn('class labels',events[-1]['message'])
            p,events=request('run',module_id='signals',dataset=str(data),target='label',output_root=folder,
                             settings={'models':['majority']})
            self.assertIn('Select at least one',events[-1]['message'])

class BeginnerWorkflowTests(unittest.TestCase):
    EXAMPLES=ROOT/'examples'
    CASES=[('classification','classification_iris.csv','Species'),
           ('regression','regression_diabetes.csv','Progression'),
           ('forecasting','forecasting_co2_monthly.csv','CO2_ppm'),
           ('signals','signals_heartbeats_synthetic.csv','label')]

    def test_every_task_has_an_example_file_that_inspects_cleanly(self):
        self.assertTrue((self.EXAMPLES/'README.txt').exists())
        for module,name,target in self.CASES:
            with self.subTest(module=module):
                p,events=request('inspect',module_id=module,dataset=str(self.EXAMPLES/name))
                self.assertEqual(p.returncode,0,events[-1])
                result=events[-1]['result']
                self.assertEqual(result['suggested_target'],target)
                self.assertTrue(result['advice'],'every task explains what it noticed')

    def test_example_files_match_the_example_buttons(self):
        import numpy as np, pandas as pd
        from easyclassifier.demo_data import load_demo as iris
        from easyresearch.regression.data import load_demo as diabetes
        from easyresearch.timeseries.data import load_demo as co2
        from easyresearch.timeseries.signal_data import load_demo as beats
        for (module,name,target),demo in zip(self.CASES,(iris,diabetes,co2,beats)):
            with self.subTest(module=module):
                saved,built=pd.read_csv(self.EXAMPLES/name),demo()
                self.assertEqual(list(saved.columns),[str(c) for c in built.columns])
                self.assertEqual(len(saved),len(built))
                numeric=built.select_dtypes('number').columns
                self.assertTrue(np.allclose(saved[numeric].to_numpy(float),built[numeric].to_numpy(float),atol=1e-3,equal_nan=True))

    def test_advice_names_left_out_columns_and_warns_about_rare_classes(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'patients.csv'
            rows=[dict(patient_id=f'P{i:03d}',site='A',age=30+i%40,dose=i%5,outcome='rare' if i<2 else ('yes' if i%3 else 'no')) for i in range(60)]
            pd.DataFrame(rows).to_csv(data,index=False)
            p,events=request('inspect',module_id='classification',dataset=str(data))
            advice=' '.join(events[-1]['result']['advice'])
            self.assertIn("'patient_id' is different in every row",advice)
            self.assertIn("'site' has a single value",advice)
            self.assertIn("'rare' has fewer than three rows",advice)
            self.assertIn('nested cross-validation',advice)
            copies=Path(folder)/'copies.csv'
            frame=pd.DataFrame(rows).drop(columns='patient_id')
            pd.concat([frame,frame.iloc[[10]]]).to_csv(copies,index=False)
            p,events=request('inspect',module_id='classification',dataset=str(copies))
            self.assertRegex(' '.join(events[-1]['result']['advice']),r'\d+ rows? (is an exact copy|are exact copies) of another row')

    def test_forecasting_and_signals_advice_describe_the_data(self):
        p,events=request('inspect',module_id='forecasting',dataset='demo')
        advice=events[-1]['result']['advice']
        self.assertIn("'CO2_ppm': 526 monthly values, from 1958-03 to 2001-12.",advice)
        self.assertTrue(any('every 12 months' in a for a in advice))
        p,events=request('inspect',module_id='signals',dataset='demo')
        advice=events[-1]['result']['advice']
        self.assertEqual(advice[0],'300 recordings of 140 values each.')
        self.assertTrue(any('not in the fast set' in a for a in advice))

class PlainErrorTests(unittest.TestCase):
    """Problems a beginner can cause get one plain sentence; the traceback still reaches the log."""
    def inspect(self,path,module='classification'):
        p,events=request('inspect',module_id=module,dataset=str(path))
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(events[-1]['type'],'error')
        self.assertIn('Traceback',events[-1]['detail'])
        return events[-1]['message']

    def test_unreadable_files_are_explained(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'empty.csv').write_text('',encoding='utf-8')
            (folder/'binary.csv').write_bytes(bytes(range(128,256))*8)
            (folder/'fake.xlsx').write_text('not a workbook',encoding='utf-8')
            for module in ('classification','forecasting','signals'):
                with self.subTest(module=module):
                    self.assertIn('The file is empty',self.inspect(folder/'empty.csv',module))
                    self.assertIn("save it as 'CSV UTF-8'",self.inspect(folder/'binary.csv',module))
                    self.assertIn('not a valid Excel workbook',self.inspect(folder/'fake.xlsx',module))
                    self.assertIn('was not found',self.inspect(folder/'gone.csv',module))

    def test_messages_from_the_analysis_code_are_kept(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'tiny.csv'
            pd.DataFrame({'x':range(8),'y':[1.5,2,3,4.5,5,6,7.2,8]}).to_csv(data,index=False)
            p,events=request('run',module_id='regression',dataset=str(data),target='y',output_root=folder,
                             settings={'models':['linear_regression']})
            self.assertEqual(events[-1]['message'],'Regression needs at least 20 complete rows; 8 remain after cleaning.')

    def test_translation_rules(self):
        from errors import plain_message
        self.assertIn('close it and try again',plain_message(PermissionError(13,'Permission denied','C:\\data\\open.xlsx')))
        self.assertIn('ran out of memory',plain_message(MemoryError()))
        self.assertIn("package 'torch' is not installed",plain_message(ModuleNotFoundError("No module named 'torch.nn'",name='torch.nn')))
        try:
            {}['missing']
        except KeyError as exc:                        # not raised by the analysis code
            self.assertIn('unexpected problem',plain_message(exc))
            self.assertIn('activity log',plain_message(exc))

class AdviceWordingTests(unittest.TestCase):
    def test_whole_number_labels_and_wrong_task_examples(self):
        p,events=request('inspect',module_id='classification',dataset=str(ROOT/'examples'/'regression_diabetes.csv'))
        advice=events[-1]['result']['advice']
        self.assertEqual(advice[0],'This is the example file for Regression; choose Regression in the sidebar to use it as intended.')
        self.assertIn("'sex' has 2 classes: 1 (235), 2 (207).",advice)
        p,events=request('inspect',module_id='regression',dataset=str(ROOT/'examples'/'regression_diabetes.csv'))
        self.assertFalse(any('example file for' in a for a in events[-1]['result']['advice']))

class ReproducibilityTests(unittest.TestCase):
    """The same analysis run twice gives identical scores and identical saved predictions."""
    def twice(self,module,dataset,target,settings,files):
        results=[]
        with tempfile.TemporaryDirectory() as folder:
            for _ in range(2):
                p,events=request('run',module_id=module,dataset=dataset,target=target,output_root=folder,
                                 settings=dict(settings,figures=False))
                self.assertEqual(p.returncode,0,events[-1])
                r=events[-1]['result']
                out=Path(r['output_dir'])
                results.append((r['final'],[(c['name'],c['metrics']) for c in r['comparison']],
                                {f:(out/f).read_text(encoding='utf-8') for f in files}))
        self.assertEqual(results[0],results[1])

    def test_classification(self):
        self.twice('classification','demo','Species',{'models':['random_forest','knn','logistic_regression']},['predictions.csv'])

    def test_regression(self):
        self.twice('regression','demo','Progression',{'models':['random_forest','knn','gradient_boosting']},['predictions.csv'])

    def test_forecasting_with_a_deep_model(self):
        self.twice('forecasting','demo','CO2_ppm',{'models':['ridge','random_forest','mlp'],'horizon':'6'},
                   ['forecast.csv','final_period_forecasts.csv'])

    def test_signals(self):
        with tempfile.TemporaryDirectory() as folder:
            data=Path(folder)/'waves.tsv'
            write_ucr(data)
            self.twice('signals',str(data),'label',{'models':['rocket','features_rf']},['predictions.csv'])

class StandardOutputTests(unittest.TestCase):
    """Every task saves the same kinds of files in its own results folder."""
    COMMON=['report.tex','summary.csv','results.xlsx','trained_model.pkl','settings.json','versions.txt',
            'log.txt','citations.txt','desktop-result.json']
    CASES=[('classification','Species',['decision_tree','logistic_regression'],['predictions.csv']),
           ('regression','Progression',['linear_regression','ridge'],['predictions.csv']),
           ('forecasting','CO2_ppm',['ridge','theta'],['forecast.csv','final_period_forecasts.csv']),
           ('signals','label',['knn','features_rf'],['predictions.csv'])]

    def test_every_task_saves_the_standard_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for module,target,models,specific in self.CASES:
                with self.subTest(module=module):
                    p,events=request('run',module_id=module,dataset='demo',target=target,output_root=folder,
                                     settings={'models':models,'figures':True})
                    self.assertEqual(p.returncode,0,events[-1])
                    result=events[-1]['result']
                    out=Path(result['output_dir'])
                    for name in self.COMMON+specific:
                        self.assertTrue((out/name).is_file(),f'{module}: {name} is missing')
                    self.assertGreaterEqual(len(list((out/'figures').glob('*.png'))),4)
                    record=json.loads((out/'settings.json').read_text(encoding='utf-8'))
                    self.assertEqual(record['column'],target)
                    self.assertEqual(record['settings']['models'],models)
                    self.assertTrue(record['used'])
                    versions=(out/'versions.txt').read_text(encoding='utf-8')
                    self.assertIn('easyclassifier:',versions)
                    self.assertIn('scikit-learn:',versions)
                    self.assertIn('settings.json',{a['name'] for a in result['artifacts']})

class ProgressTests(unittest.TestCase):
    """Every task reports the share of the work done; it only moves forward and stays below 100% until completion."""
    CASES=[('classification','Species',['decision_tree','knn','logistic_regression'],{}),
           ('regression','Progression',['linear_regression','ridge','random_forest'],{}),
           ('forecasting','CO2_ppm',['ridge','theta','mlp'],{'horizon':'6'}),
           ('signals','label',['knn','features_rf'],{})]

    def test_every_task_reports_forward_only_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            for module,target,models,extra in self.CASES:
                with self.subTest(module=module):
                    p,events=request('run',module_id=module,dataset='demo',target=target,output_root=folder,
                                     settings=dict({'models':models,'figures':True},**extra))
                    self.assertEqual(p.returncode,0,events[-1])
                    fractions=[e['fraction'] for e in events if e['type']=='progress' and 'fraction' in e]
                    self.assertGreaterEqual(len(fractions),8)
                    self.assertTrue(all(0<=f<1 for f in fractions))
                    self.assertEqual(fractions,sorted(fractions))
                    self.assertGreater(fractions[-1],.75)
                    if module=='forecasting':   # the deep model's epochs move the bar without log lines
                        self.assertTrue(any(e['type']=='progress' and e['message']=='' and 'fraction' in e for e in events))

    def test_tracker_rules(self):
        from progress import ProgressTracker, table_rules
        tracker=ProgressTracker(*table_rules(2,'nested',figures=False,importance=False,final_fit=False))
        steps=[tracker.message(m) for m in ['Training A ...','Training B ...','Running nested cross-validation ...',
                                             'fold 1: A selected','fold 2: B selected','fold 3: A selected',
                                             'fold 4: A selected','fold 5: A selected','Writing the report ...']]
        self.assertEqual(steps[0],0)
        self.assertEqual(steps,sorted(steps))
        self.assertAlmostEqual(steps[-1],12/13,places=3)   # 2 models + 5 folds x 2 + report = 13 units
        self.assertIsNone(tracker.message('Some other line'))


class AdvancedOptionsTests(unittest.TestCase):
    """The Advanced options section: EasyClassifier's Advanced/Research-mode
    choices for classification, and figure choices for every task."""

    @staticmethod
    def iris_with_gaps(folder):
        import numpy as np
        from easyclassifier.demo_data import load_demo
        df=load_demo()
        df.insert(0,'site',np.array(['north','south','east'])[np.arange(len(df))%3])
        df.loc[[3,17,40,77,101,130],'petal_width']=np.nan
        path=Path(folder)/'iris_gaps.csv'
        df.to_csv(path,index=False)
        return path

    def test_catalog_offers_the_advanced_options(self):
        p,events=request('catalog')
        modules={m['id']:m for m in events[-1]['result']['modules']}
        for module in modules.values():
            with self.subTest(module=module['id']):
                choices={c['id']:c for c in module['advanced_choices']}
                lists={l['id']:l for l in module['advanced_lists']}
                self.assertTrue({'figure_theme','figure_format'} <= set(choices))
                self.assertIn('figure_keys',lists)
                for c in choices.values():
                    self.assertIn(c['default'],[o['id'] for o in c['options']])
                for l in lists.values():
                    self.assertTrue(l['default'] and set(l['default']) <= {o['id'] for o in l['options']})
        cls=modules['classification']
        choices={c['id']:c for c in cls['advanced_choices']}
        lists={l['id']:l for l in cls['advanced_lists']}
        self.assertEqual([o['id'] for o in choices['knn_distance']['options']],
                         ['hassanat','euclidean','manhattan','chebyshev','canberra','cosine'])
        self.assertEqual((choices['knn_distance']['default'],choices['knn_k']['default']),('hassanat','5'))
        self.assertEqual({o['id'] for o in choices['missing']['options']},{'auto','drop','mean','median','mode'})
        self.assertEqual({o['id'] for o in choices['scaling']['options']},{'auto','standard','minmax','none'})
        self.assertEqual(len(lists['figure_keys']['options']),10)
        self.assertEqual(len(lists['metrics']['options']),10)
        self.assertEqual(len(lists['metrics']['default']),7)
        self.assertEqual({m['key'] for m in cls['metrics']},{o['id'] for o in lists['metrics']['options']})
        self.assertIn('loo',[o['id'] for o in cls['validation_options']])

    def test_classification_research_mode_choices_are_used(self):
        with tempfile.TemporaryDirectory() as folder:
            data=self.iris_with_gaps(folder)
            p,events=request('run',module_id='classification',dataset=str(data),target='Species',output_root=folder,
                             settings={'models':['knn','decision_tree'],'knn_distance':'euclidean','knn_k':'3',
                                       'missing':'drop','encoding':'label','scaling':'minmax',
                                       'figure_keys':['correlation','confusion_matrix','learning_curve'],
                                       'figure_theme':'greyscale','figure_format':'png+pdf',
                                       'metrics':['balanced_accuracy','specificity','cohen_kappa','log_loss']})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            out=Path(result['output_dir'])
            self.assertEqual({f['key'] for f in result['figures']},{'correlation','confusion_matrix','learning_curve'})
            for key in ('correlation','confusion_matrix','learning_curve'):
                self.assertTrue((out/'figures'/f'{key}.pdf').exists(),key)
            self.assertEqual(set(result['final']['metrics']),{'balanced_accuracy','specificity','cohen_kappa','log_loss'})
            self.assertTrue(all(set(r['metrics'])==set(result['final']['metrics']) for r in result['comparison']))
            self.assertEqual(result['rows_used'],144)          # the 6 incomplete rows removed
            self.assertTrue(any('missing values were removed' in n for n in result['notes']))
            self.assertIn('KNN (Euclidean distance, k=3)',[r['name'] for r in result['comparison']])
            record=json.loads((out/'settings.json').read_text(encoding='utf-8'))
            self.assertEqual((record['settings']['knn_distance'],record['settings']['knn_k']),('euclidean',3))
            used=record['used']
            self.assertEqual((used['knn_distance'],used['knn_k'],used['encoding']),('euclidean',3,'label'))
            self.assertEqual(used['scaling'],{'knn':'minmax','decision_tree':'minmax'})
            self.assertEqual((used['figure_theme'],used['figure_format']),('greyscale','png+pdf'))
            log=(out/'log.txt').read_text(encoding='utf-8')
            self.assertIn('KNN distance: euclidean, k=3',log)
            cites=(out/'citations.txt').read_text(encoding='utf-8')
            for author in ('Cover, T.','Deza, M. M.','Breiman, L., Friedman','Brodersen','Cohen, J.','Matplotlib'):
                self.assertIn(author,cites)       # every method used is cited
            self.assertNotIn('Hassanat distance',cites)     # not used here
            self.assertIn('Learning curve',log)
            fractions=[e['fraction'] for e in events if e['type']=='progress' and 'fraction' in e]
            self.assertEqual(fractions,sorted(fractions))
            self.assertGreater(fractions[-1],.75)

    def test_fill_choice_and_usual_figures_keep_the_automatic_extras(self):
        from easyclassifier.figures import STANDARD_FIGURES
        with tempfile.TemporaryDirectory() as folder:
            data=self.iris_with_gaps(folder)
            p,events=request('run',module_id='classification',dataset=str(data),target='Species',output_root=folder,
                             settings={'models':['decision_tree'],'missing':'mean','encoding':'onehot','scaling':'none',
                                       'figure_keys':list(STANDARD_FIGURES)})
            self.assertEqual(p.returncode,0,events[-1])
            result=events[-1]['result']
            self.assertEqual(result['rows_used'],150)          # nothing removed: the gaps are filled
            self.assertIn('missing_values',{f['key'] for f in result['figures']})   # added because of the gaps
            used=json.loads((Path(result['output_dir'])/'settings.json').read_text(encoding='utf-8'))['used']
            self.assertEqual((used['missing_values'],used['encoding'],used['scaling']),('mean','onehot',{'decision_tree':None}))
            self.assertEqual(set(result['final']['metrics']),{'accuracy','balanced_accuracy','precision','recall','f1','roc_auc','mcc'})

    def test_other_tasks_draw_only_the_chosen_figures(self):
        cases=[('regression','Progression',['linear_regression'],{},['predicted_vs_actual','residuals'],'png+svg','svg'),
               ('forecasting','CO2_ppm',['ridge'],{'horizon':'6'},['forecast','error_by_step'],'png+pdf','pdf'),
               ('signals','label',['knn'],{},['confusion_matrix'],'png+pdf','pdf')]
        with tempfile.TemporaryDirectory() as folder:
            for module,target,models,extra,keys,formats,ext in cases:
                with self.subTest(module=module):
                    p,events=request('run',module_id=module,dataset='demo',target=target,output_root=folder,
                                     settings=dict({'models':models,'figure_keys':keys,'figure_theme':'high_contrast',
                                                    'figure_format':formats},**extra))
                    self.assertEqual(p.returncode,0,events[-1])
                    result=events[-1]['result']
                    self.assertEqual([f['key'] for f in result['figures']],keys)
                    out=Path(result['output_dir'])
                    self.assertTrue(all((out/'figures'/f'{k}.{ext}').exists() for k in keys))
                    asked=json.loads((out/'settings.json').read_text(encoding='utf-8'))['settings']
                    self.assertEqual((asked['figure_theme'],asked['figure_keys']),('high_contrast',keys))

    def test_unknown_choices_are_refused_in_plain_words(self):
        cases=[('classification','Species',{'models':['knn'],'figure_theme':'neon'},'Unknown figure colour theme'),
               ('classification','Species',{'models':['knn'],'knn_k':'4'},'Unknown number of neighbours'),
               ('classification','Species',{'models':['knn'],'metrics':[]},'Tick at least one measure'),
               ('classification','Species',{'models':['knn'],'figure_keys':['pie_chart']},'Unknown figures: pie_chart'),
               ('regression','Progression',{'models':['ridge'],'figure_format':'gif'},'Unknown figure file format'),
               ('signals','label',{'models':['knn'],'figure_keys':['residuals']},'Unknown figures: residuals')]
        with tempfile.TemporaryDirectory() as folder:
            for module,target,settings,message in cases:
                with self.subTest(settings=settings):
                    p,events=request('run',module_id=module,dataset='demo',target=target,output_root=folder,settings=settings)
                    self.assertNotEqual(p.returncode,0)
                    self.assertEqual(events[-1]['type'],'error')
                    self.assertIn(message,events[-1]['message'])
            self.assertEqual(list(Path(folder).iterdir()),[])     # refused before any folder was made

if __name__=='__main__':
    unittest.main()
