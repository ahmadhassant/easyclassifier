# EasyResearch Desktop changelog

## 0.3.0 – unreleased

Forecasting and signal classification, one shared analysis engine, and guidance for beginners.

* New **Time-series forecasting** module: the date column, how far ahead and the measure that chooses the model are offered as choices after the data is read; only horizons the series can evaluate honestly are offered. Results show the forecast with an 80% band, the final-period evaluation against the naive rules and five figures. Example: monthly CO₂ at Mauna Loa.
* New **Signal classification** module for single-channel recordings (CSV, Excel or UCR files), with a choice of the first and last signal columns and a clear note that several channels per recording are not supported yet. Example: synthetic heartbeats.
* Regression now runs the shared EasyResearch core instead of a separate copy in `worker/easyregressor`, so the desktop and the command line give the same results; results also report MHSP (mean Hassanat similarity percentage).
* **What we noticed**: plain-language advice after the data is read (columns left out and why, empty cells, repeated rows, class sizes, how the final score will be obtained, series length and season, slow deep models).
* Step **5 Check and run** summarises the analysis before it starts, or says which step is missing.
* **Example files** for every task, with a README giving their sources.
* Every results folder contains `settings.json` and `versions.txt`.
* Problems are explained in one plain sentence (empty or damaged files, files open in another program, missing folders); a task that cannot load stays visible with the reason instead of disappearing.
* Development builds remember the Python they were tested with (`python-path.txt`); the activity log names the Python in use.
* The protocol always uses UTF-8, so Arabic file names and labels such as R² and CO₂ work whatever the Windows language settings.
* Fixed: a forecast whose best model was Random Forest or Gradient Boosting on past values could not be saved and stopped the analysis at the end.
* A progress bar with a percentage while an analysis runs: it follows every model trained, every fold of nested cross-validation and the final evaluation, and moves during the training of deep models.
* Deep models are no longer in the fast set of forecasting and signal classification (they take minutes each); tick them to include them.
* Shorter series can be forecast: 30 values are enough for a few steps ahead (before: about 50), and about 53 monthly values for a 12-month horizon (before: about 82).
* The Hassanat distance is now cited with the IEEE ETCEA 2022 paper and the arXiv preprint.
* **Advanced options** (closed by default): the choices of EasyClassifier's Advanced and Research modes are back. Every task chooses its figures, their colours and their file formats (PNG, PDF, SVG). Classification also chooses the measures (now including specificity, Cohen's kappa and log loss), the KNN distance and k, how empty cells are handled, the encoding of text columns and scaling, and offers leave-one-out validation and five more figures (correlation, precision-recall, missing values, important columns by class, learning curve). A wrong choice is refused before any training starts.
* Every report cites every method that was used (each model, the KNN distance, the evaluation design, the measures, the figures that need a source and the software), and `citations.txt` lists exactly the report's references with what each is cited for.
* XGBoost and LightGBM are in `requirements.lock.txt`, so the portable application bundles them and they can be selected in classification and regression.

## 0.2.0 – 2026-10-08

Regression (roadmap step 2) and a module-aware interface.

* New **Regression** module (`worker/easyregressor`, adapter `adapters/regression.py`): predicts a numeric target with Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN with the Hassanat distance and a neural network (XGBoost and LightGBM when installed). It uses EasyClassifier's leakage-free preprocessing, default parameters and fixed seeds, and the same honest final evaluation (nested CV up to 2,000 rows, otherwise a 20% final test). It reports R², RMSE, MAE, median absolute error and MAPE against the reference of always predicting the average, with a plain-language summary, a LaTeX/PDF report, five figures, predictions, column importance and the saved pipeline. Example data: diabetes progression (scikit-learn).
* The interface takes its labels, measures, headline tiles, comparison columns and evaluation options from each module, so no screen is hard-wired to classification any more.
* Sidebar shows the four roadmap steps; forecasting and computer vision are visible but cannot be selected until implemented.
* Switching module keeps the loaded spreadsheet and re-reads it for the new task.
* The column-to-predict list greys out columns that the task cannot use and explains why (hover); a caution is shown for borderline choices.
* Results show a plain-language summary and a gallery of figures (click to open full size).
* Restyled tabs, sidebar selection and tables; the Run button text is now white as intended.
* Fixed: preview and result tables no longer break on column names containing dots, brackets or slashes, or names differing only in case.
* Classification now also returns its figures, summary sentences and per-column suitability to the interface. Its scores and outputs are unchanged.
