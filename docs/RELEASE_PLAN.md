# EasyResearch release plan

## Release 1 (EasyResearch Desktop 0.3, EasyResearch 0.1.0, EasyClassifier 0.8.1)

**Goal:** a new user installs the desktop application, runs the example of
every supported task, and gets a reproducible report without writing code.

### In scope

| Task | Data | Notes |
|---|---|---|
| Classification | CSV, Excel | EasyClassifier 0.8.1, unchanged |
| Regression | CSV, Excel | shared EasyResearch core |
| Time-series forecasting | CSV, Excel; one series | horizons limited to what the series can evaluate honestly |
| Signal classification | CSV, Excel, UCR `.tsv`/`.txt`; one channel per recording | |

For every task: example file and example button, plain-language advice,
a check-and-run summary, cancellation, plain error messages, and a results
folder with report, comparison table, predictions or forecasts, figures,
trained model, `settings.json`, `versions.txt`, citations and log.

Also in release 1 (added 9 October 2026): Advanced options in the desktop,
closed by default (figures, colours and file formats for every task;
measures, KNN distance and k, empty cells, encoding, scaling and
leave-one-out for classification), and XGBoost and LightGBM bundled in the
portable application.

### Not in release 1 (next release)

* Computer vision (images).
* Multichannel signals (for example 12-lead ECG, multi-electrode EEG) and
  long recordings cut into windows.
* Using the UCR archive's own TRAIN/TEST split as the final test.
* Several series at once, and forecasts with extra explanatory columns.
* Grouped subjects (several rows per person) in cross-validation.
* Advanced options in the desktop beyond those listed above. There is
  still no hyperparameter tuning, by design.
* A web or server version with GPU access.

New features wait for the next release; release 1 only receives fixes.

### Release gates

1. All automated checks pass on GitHub (`.github/workflows/tests.yml`):
   EasyClassifier on Windows, macOS and Linux; the EasyResearch cores; the
   desktop build, adapter tests and C# integration checks.
2. `desktop\publish.ps1` makes the portable folder from a clean checkout.
3. `desktop\verify_portable.ps1` passes on a clean Windows computer, offline,
   with the application in a folder whose name has spaces and Arabic letters.
4. The benchmark results (`benchmarks/results/easyresearch/RESULTS.md` and the
   classification benchmark) are regenerated with the release versions.
5. Versions, changelogs, READMEs and citation files agree.
6. The new source folders (`desktop/`, `easyresearch/`, `tools/`, `docs/`)
   are committed; build output and test results are not (see `.gitignore`).

### Gate status (9 October 2026)

* Gate 4 done: `benchmarks/results/easyresearch` regenerated with the release
  code (fast sets without deep models). The classification benchmark stays
  valid: EasyClassifier's code has changed only in its citation text since
  the benchmark ran.
* Gate 5 checked: desktop 0.3.0, EasyResearch 0.1.0 and EasyClassifier 0.8.1
  agree in the project files, manifests, lock file, footer and changelogs.
* Still to do, on the Windows computer and GitHub (not done yet): gate 1
  (all checks pass on GitHub), gate 2 (the portable folder), gate 3 (the
  clean-computer check) and gate 6 (the new folders are still untracked in
  git; commit them).
* Still to do before the release: publish EasyClassifier 0.8.2 (see the open
  decisions); until then the PyPI package lacks the new citations.

### Decisions

* Deep models are not in the default (fast) set of any task; users tick them.
* Short series: each comparison point needs ten training windows and one
  season of history; when a series is too short for comparison points a whole
  horizon apart, they are placed closer together (overlapping forecasts, with
  a note). 30 values are enough for a few steps ahead; about 53 monthly values
  for a 12-month horizon. The final period is never used for the comparison.
* The Hassanat distance is cited with the IEEE ETCEA 2022 paper and the arXiv
  preprint (2014).

### Open decisions

* Whether the portable application must also run on computers without the
  Microsoft Visual C++ runtime (PyTorch may need it; the clean-computer check
  will show this).
* How EasyResearch is cited (a Zenodo DOI for the desktop release).
* EasyClassifier's citation text changed after 0.8.1 was published on PyPI,
  and its reports now cite every method used; publish it as 0.8.2 (and
  update the version in `pyproject.toml`, `src/easyclassifier/__init__.py`,
  `CITATION.cff`, the desktop classification manifest and
  `desktop/requirements.lock.txt`).
