EasyResearch Desktop 0.3 — Windows x64 preview

Double-click EasyResearchDesktop.exe. Keep the entire folder together:
the runtime, worker, modules and DLLs are required. Python and .NET are
bundled; they do not need to be installed separately.

1. Choose Classification (predict a category), Regression (predict a
   number), Time-series forecasting (forecast future values) or Signal
   classification (classify ECG, EEG or sensor recordings), then Browse,
   open one of the Example files, or try the example data. Read "What we
   noticed" under the data.
2. Choose the column to predict. Columns that cannot be used for the
   chosen task are greyed out; hover over them to see why.
3. Choose classifiers. Fast set selects three; Select all selects every
   installed classifier, including XGBoost and LightGBM (bundled).
   Advanced options, under the classifiers, offer more figures and
   measures, the KNN distance and k, and how the data are prepared.
4. Choose a results folder, check the summary in step 5, and click Run
   analysis.
5. Review the Final evaluation before reporting model performance.
   Open the report or results folder, or double-click a saved artifact.

Cancel stops the current worker process. An interrupted analysis can leave
partial files; the session's status.json records cancellation and no final
result is displayed. Earlier completed analyses are not overwritten.

CSV and Excel .xlsx/.xlsm are supported (signal classification also reads
UCR archive .tsv/.txt files). Continuous targets and single-row
classes are rejected; grouping a numeric outcome is not done automatically.
The GUI uses EasyClassifier 0.8.1 automatic preprocessing. PDF generation
requires an installed LaTeX distribution; report.tex, CSV and Excel files
are available without one. Detailed package messages appear in Activity log.

This preview includes classification, regression, time-series forecasting
and signal classification (one channel per recording). Computer vision is
shown as an upcoming step and is not implemented yet.

Engines: EasyClassifier 0.8.1 and EasyResearch 0.1.0 by Ahmad Hassanat (MIT).
Other bundled components retain their own licenses. Python: LICENSE.txt
in runtime/python. Package license files are in the *.dist-info folders.
