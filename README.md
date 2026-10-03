# EasyClassifier

[![tests](https://github.com/ahmadhassant/easyclassifier/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmadhassant/easyclassifier/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/easyclassifier.svg)](https://pypi.org/project/easyclassifier/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ahmadhassant/easyclassifier/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23122902.svg)](https://doi.org/10.5281/zenodo.23122902)

**Machine Learning without Programming.**

EasyClassifier is a guided, menu-driven assistant that lets anyone build and
evaluate machine learning classification models from a CSV or Excel file —
no Python code, no scripting, and no machine learning jargon required
(unless you want it).

**New to Python or the terminal? Read the step-by-step
[User Guide](https://github.com/ahmadhassant/easyclassifier/blob/main/docs/USER_GUIDE.md).** It covers installing Python, installing
EasyClassifier, running it, what each question means, and what the results
files contain.

## Quick start

**Requirements:** Python 3.10 or newer (3.12–3.14 recommended) on Windows,
macOS or Linux. About 400 MB of disk space for the scientific libraries.

| | Windows (Command Prompt) | macOS (Terminal) |
|---|---|---|
| Install | `py -m pip install easyclassifier` | `python3 -m pip install easyclassifier` |
| Check | `py -m easyclassifier --version` | `python3 -m easyclassifier --version` |
| Run | `py -m easyclassifier` | `python3 -m easyclassifier` |

On Linux (and with Homebrew Python on macOS), install into a separate
environment:

```bash
python3 -m venv ~/easyclassifier-env
~/easyclassifier-env/bin/python -m pip install easyclassifier
~/easyclassifier-env/bin/python -m easyclassifier
```

The shorter command `easyclassifier` also works when Python's scripts folder
is on your PATH.

**First run:** type `demo` when asked for a data file, then press ENTER at
every question. It takes under a minute and produces a complete example
report.

**What you get:** a new folder `Results/<file>_<date>_<time>/` for every run,
with `report.pdf` (plain-language report with a ready-to-adapt Methods
paragraph), `report.tex`, score tables (`summary.csv`, `results.xlsx`),
predictions, column importance, figures, the trained model, `citations.txt`
and a full `log.txt`. The PDF is made when a LaTeX program is installed
(MiKTeX, MacTeX, TeX Live); otherwise `report.tex` can be opened in Overleaf.
An example is in [`examples/iris/report.pdf`](https://github.com/ahmadhassant/easyclassifier/blob/main/examples/iris/report.pdf).

**Optional extras:** `pip install "easyclassifier[full]"` adds XGBoost and
LightGBM.

## What the wizard does

One simple question at a time:

1. Load your data file (`.csv` or `.xlsx`; you can drag the file into the
   window)
2. Inspect the data (rows, columns, missing values, duplicates, types)
3. Choose the column to predict, with checks that it makes sense
4. Prepare the data (missing values, duplicates, text columns, scaling)
5. Choose classifiers — or try them all
6. Review your choices, then train and evaluate with honest scores
7. Save everything in a new results folder

## Figures and colour themes

By default each run draws the five figures most used in classification
papers: class distribution, classifier comparison (with the score of every
test fold, so you can see whether the winner is clearly better), confusion
matrix (counts and percentages), ROC curves (one per class for three or more
classes) and feature importance. Precision-recall curves are added when
classes are imbalanced, and a missing-value map when the data have empty
cells. Advanced mode can add a correlation matrix, the most important
columns by class, and a learning curve ("would more data help?"). The report
explains how to read each figure, and states in words whether the winner is
clearly better and whether more data would help.

Figures are saved as PNG at 300 dpi, optionally also as PDF or SVG (vector).
Four colour themes: colour-blind safe (default, Okabe-Ito colours),
greyscale with patterns (for print), high contrast (slides) and soft. Each
class keeps the same colour in every figure.

## Choosing what to predict

Every column is listed with a short plain-language description, for example
`Purchased - 2 categories (No, Yes)` or `income - numbers from 39000 to
120000 (13 different)`. The most likely class column is marked as suggested
and chosen by pressing ENTER. Long lists are shown 20 at a time; typing part
of a name searches them.

The choice is checked before continuing:

* **One value only** - refused; there is nothing to predict.
* **Different in every row** (an ID or a name) - warning; choose another
  column or continue anyway.
* **A measurement** (numbers with many different values) - EasyClassifier
  predicts groups, not exact numbers (regression is not supported yet). The
  user can split the numbers into 2, 3 or 4 equal-sized groups, split at a
  value of their choice, or pick another column. The cut-points are shown and
  recorded in the log and results.
* **Classes with a single row** - these can never be tested; the user can
  remove those rows or pick another column. Classes with fewer than 5 rows
  trigger a reliability note.

Predictor columns that cannot help (IDs, names, single-value columns) are
left out automatically, with a note; manual mode asks first.

## The report (LaTeX and PDF)

Every run writes `Results/report.tex` and, when a LaTeX program is installed
(TeX Live, MiKTeX, MacTeX or tectonic), compiles it to `Results/report.pdf`.
Without LaTeX, upload `report.tex` together with the `figures` folder to an
online editor such as Overleaf. The report contains:

* a short summary of the question and the honest final result;
* the data: rows, class sizes, columns used and left out;
* a **Methods paragraph ready to adapt** for a paper or thesis, describing
  exactly what was done (cleaning, encoding, scaling, classifiers with default
  parameters, validation, final-score method, KNN distance);
* the results, with a plain-language explanation of every measure, the
  comparison of classifiers, and the figures;
* **which columns mattered**: permutation importance, measured on rows the
  model was not trained on (also saved as `feature_importance.csv`);
* references to cite.

Column names in non-Latin scripts (e.g. Arabic) are shown as `?` in the PDF,
because pdfLaTeX cannot typeset them.

## Design choice: default parameters, no tuning

Classifiers use their default parameters, without hyperparameter tuning.
Default values were chosen by the methods' developers after evaluation across
many datasets, so they perform well on average; parameters tuned on a single
dataset tend to fit its particular characteristics and transfer poorly to new
data. Fixed defaults also make results exactly reproducible. For the same
reason, and to keep the tool simple, feature selection and extraction (e.g.
PCA) are not included; they are planned as future work.

## Honest evaluation (no data leakage)

Filling missing values, scaling and encoding are learned from the training
part only, separately for every hold-out split and every cross-validation
fold, using a scikit-learn Pipeline. Only steps that learn nothing from the
data (removing duplicate rows, removing incomplete rows) run before splitting.
The saved `trained_model.pkl` is the complete pipeline, so it can be applied
directly to new raw data with the same columns. The tests in `tests/` check
this (`python -m pytest tests`).

## An honest score for the "best" classifier

When several classifiers are compared, the winner's score is slightly too
optimistic: it partly won by luck on those particular splits. EasyClassifier
therefore reports a separate final score for the selected classifier:

* **Nested cross-validation** (datasets up to 2,000 rows): the whole
  comparison is repeated inside each of 5 folds using only that fold's
  training rows, and the winner is scored on the fold's test rows. In the
  benchmarks it gave the most accurate final scores on small and medium data.
* **Final test set** (larger datasets): 20% of the rows are set aside before
  anything else, the classifiers are compared on the remaining 80%, and the
  winner is scored once on the untouched 20%. Accurate at this size and
  about five times faster.

The choice is automatic; Advanced and Research modes can pick either one.
Classifiers are ranked by balanced accuracy. The comparison table is still
shown and saved, clearly marked as used only for choosing. The saved model is
the selected classifier refitted on all rows. The tests in
`tests/test_selection.py` check that no row is ever predicted by a model
trained on it, and that on random data the honest estimates stay near chance
while the winner's comparison score does not.

## KNN and the Hassanat distance

When you choose KNN, the wizard asks which distance to use: Hassanat (the
default), Euclidean, Manhattan, Chebyshev, Canberra, or Cosine. For the
Hassanat distance, EasyClassifier scales the numeric columns to 0–1 (learned
on training data only), which improved it on 9 of 10 benchmark datasets and
keeps all values non-negative; the other distances use standardised columns.
The report says whether the normal form (all values >= 0) or the signed form
(negative values present) of the formula was applied, and the citation is
shown on screen and saved to `citations.txt`:

> Hassanat, A. B. (2014). Dimensionality Invariant Similarity Measure.
> Journal of American Science, 10(8). arXiv:1409.0923.

## Benchmarks

On eleven public datasets from medicine, botany, chemistry, computer vision,
sociology, political science and psychology plus a random-label control
([details](https://github.com/ahmadhassant/easyclassifier/blob/main/benchmarks/README.md), [results](https://github.com/ahmadhassant/easyclassifier/blob/main/benchmarks/results/RESULTS.md)):

* The common practice of reporting the best cross-validation score after
  fitting preprocessing on all rows was optimistic by 2.9 percentage points
  of balanced accuracy on average (up to about 10 on small datasets).
  EasyClassifier's final score was within 0.7 points on average and had the
  smallest average error (2.3 vs 3.0 points). On random labels it reported
  48.8% (truth: 50%), while the naive estimate said 52.9%.
* Among the KNN distances, with EasyClassifier's preprocessing, no distance
  was significantly better than the others (Friedman p = 0.053); scaling to
  0–1 improved the Hassanat distance on 9 of 10 datasets.

## Try it without your own data

At the first question, type `demo` to load a built-in sample dataset so you
can see the whole workflow immediately.

## Philosophy

> The user should never write Python code, never edit scripts, and never
> understand machine learning terminology unless they want to.

## Citing

If you use EasyClassifier in published work, please cite it (GitHub's
"Cite this repository" button uses [`CITATION.cff`](CITATION.cff)); each
run also writes the exact references to `citations.txt`.

## Contributing

Problem reports and suggestions are very welcome – see
[CONTRIBUTING.md](https://github.com/ahmadhassant/easyclassifier/blob/main/CONTRIBUTING.md).

## License

MIT – see [LICENSE](https://github.com/ahmadhassant/easyclassifier/blob/main/LICENSE).
