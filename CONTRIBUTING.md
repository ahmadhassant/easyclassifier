# Contributing to EasyClassifier

Thank you for helping. EasyClassifier is for researchers who do not
program, so the most valuable contributions are often **not code**:

* **Report a problem.** Open an issue on GitHub and describe what you did,
  what you expected, and what happened. Attach `log.txt` from the results
  folder – it lists every step. Please do **not** attach your data unless it
  is public.
* **Tell us what was confusing.** If a question in the wizard or a sentence
  in the report was unclear, that is a bug too.
* **Suggest an improvement** by opening an issue.

## Working on the code

```bash
git clone https://github.com/ahmadhassant/easyclassifier.git
cd easyclassifier
python -m pip install -e ".[test]"
python -m pytest tests
```

Guidelines:

* Keep the user's side free of jargon; every new choice needs a
  plain-language explanation (see `help_texts.py`) and a sensible default.
* Anything that learns from data (filling gaps, scaling, encoding …) must go
  inside the scikit-learn pipeline so that it is fitted on training rows
  only (see `preprocessing.py` and `tests/test_leakage.py`).
* Classifiers use their default parameters with a fixed random seed; there
  is deliberately no hyperparameter tuning.
* Add a test for every fix or feature. The test suite runs automatically
  on Windows, macOS and Linux with Python 3.10–3.14.
* Describe user-visible changes in `CHANGELOG.md`.

## Adding a classifier, figure or measure

* Classifier: add a `ClassifierSpec` in `models.py` and an explanation in
  `help_texts.py`.
* Figure: add a method to `FigureMaker` in `figures.py`, an entry in
  `FIGURES`, and a caption in `latex_report.py`.
* Measure: add it to `METRICS` and `compute_metrics` in `evaluation.py` and
  explain it in `help_texts.py`.

## Code of conduct

Be kind and constructive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/),
version 2.1. Problems can be reported to ahmad.hassanat@gmail.com.
