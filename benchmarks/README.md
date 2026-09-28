# EasyClassifier benchmarks

Two questions, answered on eleven public datasets from medicine,
obstetrics, botany, chemistry, computer vision, sociology/economics,
political science and psychology, plus a random-label control:

1. **Are the scores EasyClassifier reports honest?** Each dataset is split
   in half (stratified; three different random splits). The first half is
   analysed exactly as EasyClassifier's automatic mode does; the second half
   is never touched and gives the true performance. Three estimates are
   compared with it:
   * *naive* – preprocessing fitted on all rows before cross-validation
     (leakage) and the best classifier's cross-validation score reported
     (selection bias): common practice;
   * *comparison best* – EasyClassifier's leakage-free comparison, best
     score (selection bias only);
   * *EasyClassifier final* – the score EasyClassifier tells users to
     report.
2. **How do the KNN distances compare**, including the Hassanat distance,
   with EasyClassifier's automatic preprocessing?

All classifiers use their default parameters with fixed random seeds. The
results are in [`results/RESULTS.md`](results/RESULTS.md), with CSV files,
LaTeX tables (`table_*.tex`) and figures (`*.png`, `*.pdf`).

## Datasets

All datasets are installed with pip; nothing else is downloaded. Sources
and notes (e.g. a removed column that would leak the answer) are in
[`datasets.py`](datasets.py) and `results/datasets.csv`.

## Reproduce

```bash
pip install easyclassifier
pip install -r requirements.txt
python run_benchmarks.py --jobs 4        # use 4 CPU cores
```

This takes roughly 15–60 minutes depending on the computer. Finished pieces
are saved in `results/cache/`, so an interrupted run continues where it
stopped; delete that folder to start from scratch. To rebuild only the
tables and figures:

```bash
python run_benchmarks.py --report
```

Results are deterministic: the same EasyClassifier version gives the same
numbers.
