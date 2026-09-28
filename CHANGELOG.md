# Changelog

All notable changes to EasyClassifier. Version numbers follow
[semantic versioning](https://semver.org/); before 1.0 the interface may
still change.

## 0.8.1 – 2026-09-28

Changes based on the benchmarks (`benchmarks/`):

* The final score of the selected classifier uses nested cross-validation
  for datasets of up to 2,000 rows (previously only below 100 rows); it was
  the more accurate method on small and medium data. Larger datasets keep
  the faster 20% final test set.
* KNN with the Hassanat distance now uses columns scaled to 0–1 (learned on
  training data only); this improved it on 9 of 10 benchmark datasets. The
  menu no longer labels it "recommended"; it remains the default.
* Benchmarks: eleven public datasets, reproducible with
  `python benchmarks/run_benchmarks.py --jobs 4`.

## 0.8.0 – 2026-09-28

* Figures redesigned: five standard figures by default (class distribution,
  classifier comparison with fold-to-fold spread, confusion matrix with
  counts and percentages, ROC curves – now also for three or more classes –
  and feature importance); precision-recall curves and a missing-value map
  added when the data need them; correlation matrix, columns by class and a
  learning curve available in Advanced mode.
* Four colour themes (colour-blind safe default, greyscale with patterns,
  high contrast, soft); a class keeps its colour in every figure.
* Figures at 300 dpi, optionally also PDF or SVG; the PDF report uses vector
  figures.
* The report and screen state in words whether the winner is clearly better
  than the runner-up and whether more data would help.
* Fixed random seeds for all classifiers, so results are exactly
  reproducible.
* Identical rows are kept when they are expected between different cases
  (few possible value combinations, e.g. survey answers) instead of always
  being removed.

## 0.7.0

* Step-by-step user guide (`docs/USER_GUIDE.md`, also as PDF).
* Requires Python 3.10 or newer (as the scientific libraries do).
* Reads Excel `.xlsx` files; files dragged into the terminal are found;
  decimal commas in semicolon-separated files are handled.
* Every run writes to its own dated folder inside `Results/`.
* `--version` and `--help` options; ENTER selects all classifiers.
* Warnings before slow classifiers on large data; memory use of the
  Hassanat KNN bounded.

## 0.6.0

* Report as LaTeX, compiled to PDF when LaTeX is installed, with a
  ready-to-adapt Methods paragraph and plain-language explanations.
* Permutation importance measured on held-out rows.

## 0.5.0

* Checks on the column to predict: plain-language column descriptions,
  suggested column, paging and search, refusal of single-value columns,
  warnings for IDs, grouping of numeric measurements into categories,
  handling of classes with a single row.
* ID-like and single-value predictor columns left out automatically.

## 0.4.0

* Honest final score for the selected classifier: untouched 20% test set or
  nested cross-validation; ranking by balanced accuracy.

## 0.3.0

* Leakage-free evaluation: imputation, encoding and scaling learned inside
  each training fold (scikit-learn pipelines).

## 0.2.0

* KNN distance selection, including the Hassanat distance, with citation and
  report of the formula form (normal or signed) applied.

## 0.1.0

* First version of the guided wizard.
