# EasyResearch

Machine learning without programming: analysis cores for researchers who do
not code. They follow the rules of
[EasyClassifier](https://github.com/ahmadhassant/easyclassifier):

* every step that learns from data uses training data only;
* models use their authors' default settings with fixed random seeds (no
  tuning), so results are reproducible;
* when several models are compared, the selected one gets a separate,
  honest final score;
* every run writes a plain-language report with a Methods paragraph,
  figures, tables, predictions, the trained model, citations and a log.

The cores have no user interaction (settings in, results out), so a terminal
wizard, the desktop application or a web server can all drive them.

| Core | Task | Status |
|---|---|---|
| Classification | predict a category from a table | EasyClassifier 0.8.1 |
| `easyresearch.regression` | predict a number from a table | ready |
| `easyresearch.timeseries` – forecasting | forecast one series | ready |
| `easyresearch.timeseries` – signals | classify recordings (ECG, EEG, sensors) | ready (one channel) |
| Computer vision | images | next |

## Install and run

```bash
pip install -e .            # from this folder; installs PyTorch and statsmodels
easyresearch forecast --demo
easyresearch forecast sales.csv --value Sales --horizon 12
easyresearch regress houses.xlsx --target Price
easyresearch signals ECG200_TRAIN.tsv
easyresearch models forecast            # list models; * = default fast set
```

Add `--models all`, or a list such as `--models ets,lstm,nbeats`, to choose
models. Results go to a new folder in `./Results` (or `--out`). Deep models
use a GPU automatically when PyTorch finds one.

Figures: `--figures residuals,comparison` draws only the named figures (the
default is the task's usual set), `--figure-theme` chooses colour-blind safe
(default), greyscale or high-contrast colours, and `--figure-format png+pdf`
or `png+svg` adds vector files for journals.

## Forecasting

* **Data:** a CSV/Excel table with a date column (detected automatically)
  and the value to forecast. Hourly, daily, business-daily, weekly, monthly,
  quarterly and yearly series are recognised; the season length follows
  (24, 7, 5, 52, 12, 4, none). Repeated time points are averaged; missing
  points are filled with the **last known value** (never a later one).
  Without a date column, rows are taken in order with no season.
* **Models:** reference rules (naive, seasonal naive) are always included.
  Statistical: exponential smoothing (damped trend, seasonal) and Theta.
  Machine learning on the last *L* values: Ridge, Random Forest, Gradient
  Boosting, KNN with the Hassanat distance. Deep (PyTorch): MLP, LSTM, GRU,
  TCN, N-BEATS, Transformer. Default set: ETS, Theta, Ridge, Random Forest
  (deep models are optional because each takes minutes).
* **Honest evaluation:** never shuffled. Models are compared at up to five
  rolling origins before a final period (the last ~20%); the selected model
  is then scored in the final period, which played no part in choosing it,
  retrained before each origin. The automatic horizon is the usual one for
  the time unit, shortened (with a note) when the series is too short to
  evaluate it honestly; a series too short for any horizon is refused with
  the number of values it needs.
* **Measures:** MASE (used to choose, by default), MAE, RMSE, sMAPE and
  **MHSP** (mean Hassanat similarity percentage) – any of MASE, RMSE, MAE or
  MHSP can choose the model (`--metric mhsp`).
* **Outputs:** the future forecast with an 80% band (its coverage in the
  final period is checked and reported), final-period forecasts, comparison
  table, five figures, report, trained model, `settings.json` (what was
  asked and what was used) and `versions.txt` (the software used).

### MHSP

For each prediction, the Hassanat similarity between actual *a* and
forecast *f* is (1 + min(a, f)) / (1 + max(a, f)), with the signed form
(1 + min + |min|) / (1 + max + |min|) when the smaller value is negative.
MHSP is 100 × the mean of these similarities (100% = perfect), computed on
the values as they are. It is also reported by the regression core. Reference:
Hassanat, A. B., et al. (2024). A novel outlier-robust accuracy measure for
machine learning regression using a non-convex distance metric. *Mathematics*,
12(22), 3623. https://doi.org/10.3390/math12223623

## Signal classification (ECG, EEG, sensors)

* **Data:** one recording per row – a CSV/Excel table with a label column and
  the signal values in the other numeric columns, or UCR/UEA archive files
  (`.tsv`/`.txt`, label first). Gaps inside a recording are filled from its
  own neighbouring values.
* **Models:** KNN with the Hassanat distance (each recording scaled to 0–1),
  ROCKET (10,000 random convolution kernels), signal features + Random
  Forest, and deep FCN, ResNet, InceptionTime and LSTM.
* **Evaluation:** EasyClassifier's own – stratified cross-validation to
  compare; nested cross-validation for the final score (up to 2,000
  recordings, without deep networks) or an untouched stratified 20% test set
  (larger data, or when deep networks are compared, because they take
  minutes each on a normal computer).
* The demo (`--demo`) uses **synthetic** heartbeats for trying the tool; it
  is not patient data.
* Not yet: several channels per recording (multichannel EEG) and long
  recordings to be cut into windows.

## Regression

See the EasyResearch Desktop README for details: linear and ridge regression,
decision tree, random forest, gradient boosting, SVR, KNN with the Hassanat
distance and a neural network; R², RMSE, MAE, median absolute error, MAPE
and MHSP against the reference of always predicting the average.
XGBoost and LightGBM are added when installed (`pip install -e ".[full]"`).

## Development

```bash
pip install -e ".[test]"
python -m pytest tests
```

From the repository root, `python tools/run_all_tests.py` runs every suite
(EasyClassifier, these cores and the desktop adapters); see
`docs/DEVELOPMENT.md`. `benchmarks/easyresearch_benchmarks.py` measures
accuracy, run time, memory and agreement with reference implementations.

The tests include a check that no forecasting model – statistical, machine
learning or deep – changes its forecast when every value after the forecast
origin is changed.

## Licence

MIT. Ahmad Hassanat.
