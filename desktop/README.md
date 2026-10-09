# EasyResearch Desktop

A Windows WPF application built around the published EasyClassifier 0.8.1 package and the EasyResearch analysis cores. Four modules work: **Classification** predicts categorical targets with the EasyClassifier package; **Regression** (new in 0.2) predicts a numeric target; **Time-series forecasting** (new in 0.3) forecasts one series measured over time; and **Signal classification** (new in 0.3) classifies single-channel recordings such as ECG beats, EEG epochs or sensor windows. Regression, forecasting and signal classification run the cores of the EasyResearch package (`../easyresearch`), the same code its command line runs. All produce reports, figures, predictions and saved models from CSV/Excel data (signal classification also reads UCR archive `.tsv`/`.txt` files). Computer vision (step 4) appears in the sidebar as a roadmap entry and cannot be selected until its module exists.

## Run the portable application

Open `release/EasyResearchDesktop/EasyResearchDesktop.exe`, keeping its complete folder intact. The release bundles .NET and an isolated Python runtime; no separate Python installation is required. Use **Try Iris example** for a first run. Every task also has an example file (**Example files…**, the `examples` folder) that shows how data should be laid out. The default fast set compares Decision Tree, Random Forest and Logistic Regression; **Select all** includes all seven standard classifiers, plus XGBoost and LightGBM, which are now bundled (they are in `requirements.lock.txt`). When the application uses your own Python, XGBoost and LightGBM become selectable after `python -m pip install xgboost-cpu==3.4.1 lightgbm==4.7.0`; until then they are greyed out and the reason is shown when hovered.

An analysis uses automatic preprocessing from EasyClassifier. Model comparison and final evaluation are shown separately. The automatic method uses nested CV for smaller datasets, or a separate 20% final test for larger datasets with adequate class counts. The worker refuses to report a comparison score as a final result when final evaluation fails. A single preselected model is evaluated using the chosen validation method.

Output goes to a new uniquely named analysis session under the chosen results folder. Each session includes `request.json`, `status.json`, and package outputs; `desktop-result.json` adds full-precision scores and dependency versions. Optional-output failures are surfaced as warnings. PDF generation requires LaTeX; it is not bundled. Excel, CSV and LaTeX source remain available without it. Every results folder also contains `settings.json` (what was asked and what was used, such as the evaluation method or the forecast horizon) and `versions.txt` (Python, the operating system and every analysis package), so a result can be repeated and reported exactly.

## Regression module

The regression core (`easyresearch.regression`) follows EasyClassifier's rules and reuses its data loading and leakage-free preprocessing: learned steps are fitted on training rows only, models use default parameters with fixed seeds (no tuning), and when several models are compared the selected one gets a separate final score (nested CV up to 2,000 rows, otherwise an untouched 20% test set). Models are ranked by R² on held-out rows.

* Models: Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting (histogram-based), SVR, KNN with the Hassanat distance (k = 5, columns scaled to 0–1 on training rows; the signed form is applied automatically to negative values) and a neural network (MLP); XGBoost and LightGBM when installed. The fast set is Linear Regression, Random Forest and Gradient Boosting. SVR and the neural network learn on a standardised target (fitted on training rows), because their defaults assume values of roughly unit size.
* Measures: R², RMSE, MAE, median absolute error, MHSP (mean Hassanat similarity percentage; Hassanat et al. 2024, Mathematics 12(22), 3623), and MAPE when no actual value is zero or changes sign. Every result is shown next to the reference of always predicting the training average, and explained in plain sentences.
* Column to predict: numeric columns only. Text categories, yes/no columns, identifiers and single-value columns are greyed out with the reason; a column with ten or fewer whole-number values is allowed with a note.
* Outputs: `report.tex`/`report.pdf` with a Methods paragraph, `summary.csv`, `results.xlsx` (results, predictions, column importance), `predictions.csv` (row in file, actual, predicted, error), `feature_importance.csv`, `trained_model.pkl` (full pipeline), `citations.txt`, `settings.json`, `versions.txt`, `log.txt` and five figures (distribution of the target, model comparison, predicted versus actual, residuals, column importance; plus a missing-value map when needed).
* Example data: **Try diabetes example** loads the diabetes progression data shipped with scikit-learn (442 patients, Efron et al. 2004).

## Forecasting module

Forecasts one series measured over time with the forecasting core (`easyresearch.timeseries`).

* Choices after the file is read: the column to forecast, the date column (found automatically; without one, rows are taken in file order), how far ahead (only horizons the series can evaluate honestly are offered; the automatic horizon is shortened, with a note, when the series is short; 30 values are enough for a few steps ahead, and about 53 monthly values for a 12-month horizon), and the measure that chooses the model (MASE by default; RMSE, MAE or MHSP).
* Models: the naive and seasonal naive rules are always included for reference; ETS and Theta; Ridge, Random Forest, Gradient Boosting and KNN with the Hassanat distance on past values; deep MLP, LSTM, GRU, TCN, N-BEATS and Transformer. The fast set is ETS, Theta, Ridge and Random Forest; deep models are optional because each takes minutes.
* Evaluation never shuffles: models are compared on forecasts made before a final period (about the last 20%), and the selected model is scored in that final period, retrained before each forecast on the values before it.
* Results: the forecast table with an 80% band, the final-period scores (MASE, MAE, RMSE, sMAPE, MHSP) next to the naive rule, the model comparison and five figures. Example data: monthly Mauna Loa CO₂, 1958–2001.

## Signal classification module

Classifies recordings with the signal core (`easyresearch.timeseries`). One row is one recording from a single channel: a class column and the signal values in the other numeric columns, or a UCR archive file (label first, no header).

* Choices after the file is read: the class column and the first and last signal columns (to leave out numbers such as age). Several channels per recording (for example a 12-lead ECG) are not supported yet; the screen says so.
* Models: KNN with the Hassanat distance, ROCKET, signal features with Random Forest, and deep FCN, ResNet, InceptionTime and LSTM; the most-frequent-class rule is shown for reference. The fast set is KNN, ROCKET and features with Random Forest; deep models are optional because each takes minutes.
* Evaluation is EasyClassifier's: nested cross-validation for the final score, or an untouched 20% test set for larger data or whenever a deep model is compared.
* Example data: synthetic heartbeats (not patient data).

## Guidance for beginners

* After a file is read, **What we noticed** explains in plain words what the analysis will do with it: identifier or constant columns that will be left out, empty cells, repeated rows, class sizes and rare classes, how the final score will be obtained, the length and season of a series, and that deep models take minutes. Opening another task's example file points to the right task.
* Step **5 Check and run** summarises the whole analysis before it starts, or names the step that is still missing; Run is available only when nothing is missing.
* A task that cannot load stays in the sidebar, greyed out, with the reason when hovered and in the activity log.
* Problems such as an empty or damaged file, a file open in another program or a missing folder are explained in one sentence with what to do; the technical details go to the activity log.
* **Advanced options** (in step 3, closed by default) bring back the choices of EasyClassifier's Advanced and Research modes, with the recommended settings preselected and a button to restore them. Every task: which figures to draw, their colours (colour-blind safe, greyscale, high contrast) and files (PNG, PNG + PDF, PNG + SVG). Classification also: the measures to report (adding specificity, Cohen's kappa and log loss), the KNN distance (Hassanat, Euclidean, Manhattan, Chebyshev, Canberra, cosine) and k, how empty cells are handled (remove the rows, or fill with the mean, median or most common value), the encoding of text columns (label or one-hot) and scaling (standardise, 0–1 or none), and leave-one-out among the evaluation settings. Extra figures for classification are the correlation matrix, precision-recall curves, the missing-value map, the most important columns by class and the learning curve. Step 5 lists every choice that differs from the recommended one, and `settings.json` records what was asked and what was used.

## Interface

The sidebar lists the installed modules and the roadmap steps. Switching module keeps your spreadsheet and re-reads it for the new task. Labels, the column check, the result tiles, the comparison table and the evaluation options come from each module's `describe()` result, so a new module needs no interface changes when its result fits the tabular contract. Results show the module's headline measures, a plain-language summary, the comparison table (with the reference row for regression), and a gallery of figures that open full size on click. Tables use neutral column ids with the real names as headers, so column names containing dots, brackets or slashes display correctly.

## Open in Visual Studio

Open `EasyResearchDesktop.sln` using Visual Studio with .NET 10 support and the **.NET desktop development** workload. The solution contains:

- `EasyResearch.Core`: typed JSON contracts and the cancellable Python process client, with no WPF dependency.
- `EasyResearch.Desktop`: WPF views, view models, file dialogs and results browsing.
- `EasyResearch.IntegrationTests`: executable integration checks using real worker processes.

Install Python 3.12 dependencies using `python -m pip install -r requirements.lock.txt`, then link the two packages to the repository with `python -m pip install --no-deps -e .. -e ..\easyresearch` (see `docs/DEVELOPMENT.md` in the repository root). `build.ps1` records the Python it tested with in `python-path.txt` next to the Release and Debug builds, and the application uses that Python however it is started. The order is: the bundled interpreter, then `python-path.txt`, then the `EASYRESEARCH_PYTHON` setting. The first line of the activity log names the Python in use.

`build.ps1 -Dotnet <dotnet.exe> -Python <python.exe>` builds the solution and runs adapter plus C# integration checks. `publish.ps1` creates a self-contained Windows x64 release and downloads the official Python 3.12.8 embeddable distribution. Python dependencies are pinned to the environment already used to test EasyClassifier. Downloads occur only during publishing, not while running an analysis.

## Add another analysis module

1. Add a manifest to `modules/`, giving it a unique id, protocol version, adapter class and input kind.
2. Implement `describe()`, `inspect(dataset)` and `run(request, emitter)` in a Python adapter. Package-specific work belongs in that adapter; stdout is reserved for the versioned protocol.
3. Register a view for a new input kind, or reuse the tabular workflow where its configuration and result contracts fit. Task-specific measures come from `describe()`; data-dependent choices (such as the forecast horizon or the signal range) are returned by `inspect` as `parameters` and shown as drop-down lists, and extra result tables (such as the forecast) as `tables`. Image input needs a different data view. Do not label an unimplemented module as available.
4. Include the package dependencies in the lock file and add protocol, evaluation and UI tests.

The JSON-lines protocol is documented in `docs/PROTOCOL.md`. Every job runs in a separate Python process, so a module cannot share working-directory or plotting state with another job. Cancellation terminates the process tree. Each worker is noninteractive: no terminal-menu answer automation or parsing of human-readable score tables is used.

## Current scope

Automatic preprocessing unless other choices are made in Advanced options; categorical (classification) or numeric (regression) targets; standard classifier defaults (no hyperparameter tuning, by design); Hassanat KNN with k=5 unless another distance or k is chosen; the package's standard figure set unless other figures are ticked. The interface explicitly identifies the encoded positive class for binary precision/recall/F1. The worker is tied to EasyClassifier 0.8.1 because it adapts package internals; a package upgrade requires adapter tests. Its preprocessing choices retain the underlying package's current behavior, including category-count decisions made before splitting. Grouped subjects (several rows per person) need future dedicated controls, and signal classification is single-channel only.

The original package and manuscript are not modified by this desktop project. It lives in the `desktop` subfolder of the existing repository and should be reviewed as a new preview application before public distribution.
