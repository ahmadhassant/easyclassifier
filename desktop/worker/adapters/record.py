"""settings.json and versions.txt for runs made by EasyClassifier.

The EasyResearch cores write the same two files themselves
(easyresearch.common.output.write_run_record). Classification runs through
the EasyClassifier package, so the desktop writes them here, in the same
format, without depending on EasyResearch being installed.
"""
from __future__ import annotations

import datetime
from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import sys

PACKAGES = ('easyresearch', 'easyclassifier', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib',
            'joblib', 'statsmodels', 'torch', 'openpyxl', 'xgboost', 'lightgbm')


def write_run_record(out_dir, task, data, column, settings, used):
    record = dict(task=task, data=data, column=column, settings=settings, used=used,
                  created=datetime.datetime.now().isoformat(timespec='seconds'))
    with open(os.path.join(out_dir, 'settings.json'), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2, default=str)
    found = {'Python': platform.python_version(),
             'Operating system': f'{platform.system()} {platform.release()} ({platform.machine()})'}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    with open(os.path.join(out_dir, 'versions.txt'), 'w', encoding='utf-8') as fh:
        fh.write('Software used for these results\n\n' + '\n'.join(f'{k}: {v}' for k, v in found.items())
                 + f'\n\nPython executable: {sys.executable}\n')
