"""EasyClassifier benchmarks.

Benchmark 1 - Are the reported scores honest?
    Each dataset is split in half (stratified), repeated with 3 seeds.
    The first half ("development") is analysed exactly as EasyClassifier's
    automatic mode would; the second half ("external") is never touched
    until the end and serves as the truth. Three estimates are compared
    with the external performance of the model they describe:

      naive      preprocessing fitted on all development rows before cross-
                 validation (leakage) + reporting the best classifier's
                 cross-validation score (selection bias) - common practice;
      comparison EasyClassifier's leakage-free comparison, best score
                 (selection bias only);
      final      EasyClassifier's reported final score (final test set or
                 nested cross-validation).

Benchmark 2 - KNN distances.
    Repeated stratified 5-fold cross-validation (2 repeats) of KNN (k = 5)
    with each distance offered by EasyClassifier, using the tool's own
    preprocessing (the Hassanat distance on unscaled data, the others on
    standardised data, as EasyClassifier does automatically), plus Euclidean
    on unscaled data for reference.

Usage:
    python run_benchmarks.py              # run everything, then report
    python run_benchmarks.py --report     # only rebuild tables and figures
    python run_benchmarks.py --jobs 4     # use 4 CPU cores
    python run_benchmarks.py --budget 35  # stop after ~35 s (resumable)

Results of finished pieces are cached in results/cache/, so an interrupted
run continues where it stopped.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from datasets import DATASETS  # noqa: E402
from easyclassifier import preprocessing as pp  # noqa: E402
from easyclassifier import recommend as rec  # noqa: E402
from easyclassifier import target as tg  # noqa: E402
from easyclassifier.dataset import inspect  # noqa: E402
from easyclassifier.distances import DISTANCES, make_knn  # noqa: E402
from easyclassifier.evaluation import (  # noqa: E402
    evaluate,
    evaluate_on_test,
    recommend_selection,
    split_final_test,
)
from easyclassifier.models import build_registry  # noqa: E402

warnings.filterwarnings("ignore")

RESULTS = os.path.join(HERE, "results")
# Results of EasyClassifier 0.8.1+ (Hassanat on 0-1 scaled data, nested CV
# up to 2,000 rows). Earlier runs lived in results/cache/ directly.
CACHE = os.path.join(RESULTS, "cache", "v081")
REPEATS = [0, 1, 2]
KNN_REPEATS = 2
KNN_EXCLUDE = {"noise"}           # a control, not a real comparison


# --------------------------------------------------------------------------- #
# Data preparation, exactly as EasyClassifier's automatic mode
# --------------------------------------------------------------------------- #

def prepare(key: str):
    ds = DATASETS[key]
    df = ds.load()
    info = {"rows_loaded": len(df), "cols_loaded": df.shape[1] - 1}
    df, _ = pp.drop_missing_target(df, ds.target)
    n_dup = 0
    if not pp.duplicates_expected_by_chance(df):   # as the tool does
        df, n_dup = pp.remove_duplicates(df)
    left_out = [c for c in df.columns if c != ds.target and
                tg.describe_column(df[c]).kind in (tg.ID_LIKE, tg.CONSTANT,
                                                   tg.EMPTY)]
    df = df.drop(columns=left_out)
    tiny, _ = tg.tiny_and_small_classes(df[ds.target])
    if tiny:
        df = df[~df[ds.target].astype(str).isin(tiny)].reset_index(drop=True)
    cfg = pp.PrepConfig()
    feats = df.drop(columns=[ds.target])
    if feats.isna().values.any():
        strat = rec.missing_strategy(inspect(df))
        if strat == "drop":
            df, _ = pp.drop_missing_rows(df)
        else:
            cfg.impute = "median"
    if pp.categorical_columns(df.drop(columns=[ds.target])):
        cfg.encoding = rec.encoding_method(df, ds.target)
    X = pp.cast_categoricals(df.drop(columns=[ds.target]))
    y, names, _ = pp.encode_target(df[ds.target].astype(str))
    info.update(rows=len(y), cols=X.shape[1], classes=len(names),
                duplicates_removed=n_dup, left_out=left_out,
                missing_cells=int(feats.isna().values.sum()),
                class_sizes=[int((y == i).sum()) for i in range(len(names))])
    return X, y, names, cfg, info


def pipeline_specs(X, cfg) -> Dict[str, object]:
    """All available classifiers as leakage-free pipelines, with the tool's
    per-classifier scaling rule (KNN uses the default Hassanat distance)."""
    specs = {}
    for key, spec in build_registry().items():
        if not spec.available:
            continue
        scale = rec.scaling_for(key, "hassanat")
        specs[key] = dataclasses.replace(
            spec, factory=lambda s=spec, sc=scale: pp.build_pipeline(
                X, cfg, s.factory(), sc))
    return specs


# --------------------------------------------------------------------------- #
# Cached tasks
# --------------------------------------------------------------------------- #

def _path(*parts) -> str:
    return os.path.join(CACHE, "__".join(str(p) for p in parts) + ".json")


def cached(parts, fn):
    """Return the cached result for ``parts`` or compute and store it."""
    p = _path(*parts)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    value = fn()
    os.makedirs(CACHE, exist_ok=True)
    with open(p + ".tmp", "w") as fh:
        json.dump(value, fh)
    os.replace(p + ".tmp", p)
    return value


class OutOfTime(Exception):
    pass


class Budget:
    def __init__(self, seconds):
        self.end = time.time() + seconds if seconds else None

    def check(self):
        if self.end and time.time() > self.end:
            raise OutOfTime


def _split_half(y, seed):
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=0.5, random_state=seed,
                            stratify=y)


def _naive_cv(clf_factory, X, y, cfg, seed) -> Tuple[float, List[int]]:
    """Common (flawed) practice: fit imputation, encoding and scaling on all
    rows, then cross-validate the classifier on the transformed data."""
    Xt = pp.build_preprocessor(X, cfg, scale=True).fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf_factory(), Xt, y, cv=cv)
    return float(balanced_accuracy_score(y, pred))


def benchmark1(key: str, seed: int, budget: Budget) -> dict:
    X, y, names, cfg, _ = prepare(key)
    dev, ext = _split_half(y, seed)
    Xd, yd = X.iloc[dev].reset_index(drop=True), y[dev]
    Xe, ye = X.iloc[ext].reset_index(drop=True), y[ext]
    specs = pipeline_specs(Xd, cfg)
    raw = {k: s for k, s in build_registry().items() if s.available}
    validation = rec.validation_method(yd)
    method = recommend_selection(yd)

    # naive: leakage + best-of
    naive = {}
    for k in specs:
        budget.check()
        naive[k] = cached((key, seed, "naive", k), lambda k=k: _naive_cv(
            raw[k].factory, Xd, yd, cfg, seed))
    naive_winner = max(naive, key=naive.get)

    # EasyClassifier comparison (on 80% of development rows if a final test
    # set is kept aside, otherwise on all development rows)
    if method == "final_test":
        d2, t2 = split_final_test(Xd, yd)
    else:
        d2, t2 = np.arange(len(yd)), None
    comp = {}
    for k, spec in specs.items():
        budget.check()
        comp[k] = cached((key, seed, "comparison", k), lambda spec=spec: evaluate(
            spec, Xd.iloc[d2], yd[d2], names, validation, [],
            with_proba=False).selection_score)
    tool_winner = max(comp, key=comp.get)       # ties: registry order

    # EasyClassifier final estimate
    if method == "final_test":
        budget.check()
        final = cached((key, seed, "final_test", tool_winner), lambda: (
            evaluate_on_test(specs[tool_winner], Xd.iloc[d2], yd[d2],
                             Xd.iloc[t2], yd[t2], names, []).selection_score))
    else:
        outer = StratifiedKFold(
            n_splits=max(2, min(5, int(np.bincount(yd).min()))),
            shuffle=True, random_state=42)
        pred = np.empty_like(yd)
        inner = "kfold5" if validation == "loo" else validation
        for f, (tr, te) in enumerate(outer.split(Xd, yd)):
            # the complete comparison, repeated inside this outer fold
            inner_scores = {}
            for k, spec in specs.items():
                budget.check()
                inner_scores[k] = cached(
                    (key, seed, "nested", f, k),
                    lambda spec=spec, tr=tr: evaluate(
                        spec, Xd.iloc[tr], yd[tr], names, inner, [],
                        with_proba=False).selection_score)
            win = max(inner_scores, key=inner_scores.get)
            budget.check()
            p = cached((key, seed, "nested", f, "winner", win),
                       lambda tr=tr, te=te, win=win: specs[win].factory()
                       .fit(Xd.iloc[tr], yd[tr]).predict(Xd.iloc[te])
                       .tolist())
            pred[te] = p
        final = float(balanced_accuracy_score(yd, pred))

    # External truth: each winner refitted on all development rows
    budget.check()
    ext_tool = cached((key, seed, "external_tool", tool_winner), lambda: float(
        balanced_accuracy_score(ye, specs[tool_winner].factory().fit(
            Xd, yd).predict(Xe))))

    def _ext_naive():
        prep = pp.build_preprocessor(Xd, cfg, scale=True).fit(Xd)
        m = raw[naive_winner].factory().fit(prep.transform(Xd), yd)
        return float(balanced_accuracy_score(ye, m.predict(
            prep.transform(Xe))))
    budget.check()
    ext_naive = cached((key, seed, "external_naive", naive_winner),
                       _ext_naive)

    return dict(dataset=key, seed=seed, method=method, validation=validation,
                n_dev=len(yd), n_ext=len(ye),
                naive_winner=naive_winner, naive=naive[naive_winner],
                external_naive=ext_naive,
                tool_winner=tool_winner, comparison=comp[tool_winner],
                final=final, external_tool=ext_tool,
                comparison_all=comp)


def knn_specs(X, cfg) -> Dict[str, object]:
    specs = {}
    for d in DISTANCES:
        scale = rec.scaling_for("knn", d)   # EasyClassifier's automatic rule
        specs[d] = (lambda d=d, sc=scale:
                    pp.build_pipeline(X, cfg, make_knn(d, 5), sc))
    # References: other ways of preparing the data
    specs["euclidean_unscaled"] = (lambda: pp.build_pipeline(
        X, cfg, make_knn("euclidean", 5), None))
    specs["hassanat_unscaled"] = (lambda: pp.build_pipeline(
        X, cfg, make_knn("hassanat", 5), None))
    specs["hassanat_standardised"] = (lambda: pp.build_pipeline(
        X, cfg, make_knn("hassanat", 5), "standard"))
    return specs


def benchmark2(key: str, budget: Budget) -> dict:
    X, y, names, cfg, _ = prepare(key)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=KNN_REPEATS,
                                 random_state=42)
    out = {}
    for dist, factory in knn_specs(X, cfg).items():
        def run(factory=factory):
            scores = []
            for tr, te in cv.split(X, y):
                m = factory().fit(X.iloc[tr], y[tr])
                scores.append(float(balanced_accuracy_score(
                    y[te], m.predict(X.iloc[te]))))
            return scores
        budget.check()
        out[dist] = cached((key, "knn", dist), run)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def all_jobs():
    jobs = [("b1", k, s) for s in REPEATS for k in DATASETS]
    jobs += [("b2", k, None) for k in DATASETS if k not in KNN_EXCLUDE]
    return jobs


def run(budget_s: float, worker: int, n_workers: int) -> bool:
    """Run pending jobs; returns True when everything is finished."""
    budget = Budget(budget_s)
    done = True
    for i, (kind, key, seed) in enumerate(all_jobs()):
        if i % n_workers != worker:
            continue
        try:
            if kind == "b1":
                cached((key, seed, "B1"), lambda: benchmark1(key, seed,
                                                              budget))
            else:
                cached((key, "B2v2"), lambda: benchmark2(key, budget))
        except OutOfTime:
            done = False
            break
    return done


def status() -> str:
    jobs = all_jobs()
    n = sum(os.path.exists(_path(k, s, "B1") if kind == "b1"
                           else _path(k, "B2v2")) for kind, k, s in jobs)
    return f"{n}/{len(jobs)} benchmark jobs finished"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=0,
                    help="stop after about this many seconds (0 = no limit)")
    ap.add_argument("--worker", default="0/1",
                    help="run only part of the jobs, e.g. 0/2 and 1/2")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many worker processes in parallel "
                         "(e.g. the number of CPU cores)")
    ap.add_argument("--report", action="store_true",
                    help="only build tables and figures from finished jobs")
    ap.add_argument("--no-report", action="store_true",
                    help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.report and a.jobs > 1:
        import subprocess
        procs = [subprocess.Popen(
            [sys.executable, __file__, "--budget", str(a.budget),
             "--worker", f"{i}/{a.jobs}", "--no-report"])
            for i in range(a.jobs)]
        for p in procs:
            p.wait()
        print(status(), flush=True)
    elif not a.report:
        w, n = map(int, a.worker.split("/"))
        run(a.budget, w, n)
        print(status(), flush=True)
    if a.no_report:
        return
    if a.report or a.budget == 0:
        from report import build_report
        build_report()


if __name__ == "__main__":
    main()
