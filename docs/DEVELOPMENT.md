# Development guide

This guide is for working on the code. Users of the desktop application or
of `pip install easyclassifier` do not need it.

## One environment for everything

Use Python 3.12, the version the portable desktop application ships, and the
exact package versions it ships:

```bat
conda create -n easyresearch python=3.12 -y
conda activate easyresearch
python -m pip install -r desktop\requirements.lock.txt
python -m pip install --no-deps -e . -e easyresearch
python -m pip install pytest
```

The last two lines link EasyClassifier (`src/easyclassifier`) and the
EasyResearch cores (`easyresearch/src/easyresearch`) to this folder, so edits
take effect without reinstalling. On Linux or macOS, install PyTorch first
from `https://download.pytorch.org/whl/cpu` to avoid the large GPU build.

## Running the checks

```bat
python tools\run_all_tests.py           :: everything
python tools\run_all_tests.py --quick   :: without the slow desktop suite
```

| Suite | What it covers | Time |
|---|---|---|
| `tests/` | EasyClassifier: leakage-free preparation, honest final scores, reports, figures | ~1 min |
| `easyresearch/tests/` | Regression, forecasting and signal cores, including evaluation integrity | ~2 min |
| `desktop/tests/python/` | The desktop adapters end to end: every task, error messages, reproducibility | ~10 min |
| `desktop/build.ps1` | C# build, the adapter tests above, and the C# integration checks (Windows) | ~12 min |

The same checks run on GitHub for every push (`.github/workflows/tests.yml`).

## The desktop application

* `powershell -ExecutionPolicy Bypass -File desktop\build.ps1` builds the
  application and runs its tests. It records the Python it tested with in
  `python-path.txt` next to the Release and Debug builds, and the application
  uses that Python however it is started (double-click or Visual Studio).
* `desktop\publish.ps1` makes the portable application, with its own Python
  and the packages from `desktop\requirements.lock.txt`.
* The activity log in the application starts with the Python it uses and why.

## Rules that every change keeps

* Anything learned from data is fitted on training rows only, inside the
  model pipeline.
* Default parameters with fixed random seeds; no hyperparameter tuning.
* When several models are compared, the selected one gets a separate final
  score (nested cross-validation up to 2,000 rows, otherwise an untouched
  20% test set; forecasting uses a final period after the comparison).
* Messages are written for researchers who do not program.
* Every fix or feature comes with a test.
