"""Model training, validation, and metrics (Phases 11-13, 18)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)


# Available metrics: key -> friendly label
METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced Accuracy",
    "mcc": "MCC",
    "cohen_kappa": "Cohen Kappa",
    "roc_auc": "ROC AUC",
    "log_loss": "Log Loss",
}

VALIDATION = {
    "holdout": "Hold-Out (single 80/20 split)",
    "kfold5": "5-Fold Cross Validation",
    "kfold10": "10-Fold Cross Validation",
    "stratified": "Stratified 5-Fold Cross Validation",
    "loo": "Leave-One-Out",
}


@dataclass
class Result:
    classifier_key: str
    classifier_name: str
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    class_names: List[str]
    confusion: np.ndarray = field(default=None)
    selection_score: float = 0.0   # balanced accuracy, used for ranking
    note: str = ""                 # e.g. which classifier won in each fold
    fold_scores: List[float] = field(default_factory=list)  # per test fold

    @property
    def primary_score(self) -> float:
        return self.selection_score


# The single criterion used to rank classifiers. Balanced accuracy works for
# two or more classes, is robust to class imbalance, and higher is better.
SELECTION_METRIC = "balanced_accuracy"
SELECTION_LABEL = "Balanced accuracy"


def _specificity(y_true, y_pred, labels) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs = []
    total = cm.sum()
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specs.append(tn / denom if denom else 0.0)
    return float(np.mean(specs))


def compute_metrics(y_true, y_pred, y_proba, selected: List[str],
                    class_names: List[str]) -> Dict[str, float]:
    labels = list(range(len(class_names)))
    binary = len(class_names) == 2
    avg = "binary" if binary else "macro"
    out: Dict[str, float] = {}
    for key in selected:
        try:
            if key == "accuracy":
                out[key] = accuracy_score(y_true, y_pred)
            elif key == "precision":
                out[key] = precision_score(y_true, y_pred, average=avg,
                                           zero_division=0)
            elif key == "recall":
                out[key] = recall_score(y_true, y_pred, average=avg,
                                        zero_division=0)
            elif key == "f1":
                out[key] = f1_score(y_true, y_pred, average=avg,
                                    zero_division=0)
            elif key == "specificity":
                out[key] = _specificity(y_true, y_pred, labels)
            elif key == "balanced_accuracy":
                out[key] = balanced_accuracy_score(y_true, y_pred)
            elif key == "mcc":
                out[key] = matthews_corrcoef(y_true, y_pred)
            elif key == "cohen_kappa":
                out[key] = cohen_kappa_score(y_true, y_pred)
            elif key == "roc_auc":
                if y_proba is None:
                    continue
                if binary:
                    out[key] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    out[key] = roc_auc_score(y_true, y_proba,
                                             multi_class="ovr", average="macro")
            elif key == "log_loss":
                if y_proba is None:
                    continue
                out[key] = log_loss(y_true, y_proba, labels=labels)
        except Exception:  # noqa: BLE001
            # A metric that can't be computed on this data is simply skipped.
            continue
    return out


def _cv_splitter(method: str, y):
    if method == "loo":
        return LeaveOneOut()
    wanted = 10 if method == "kfold10" else 5
    # Stratified folds need at least n_splits members in every class.
    smallest = int(np.min(np.unique(y, return_counts=True)[1]))
    n_splits = max(2, min(wanted, smallest))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def evaluate(spec, X, y, class_names, validation: str,
             metrics: List[str], with_proba: bool = True) -> Result:
    """Train and evaluate a single classifier, returning a :class:`Result`.

    ``spec.factory()`` returns an unfitted Pipeline (preprocessing + model).
    It is fitted on training data only - once for hold-out, once per fold for
    cross-validation - so test data never influences preprocessing.
    ``X`` stays a DataFrame so the pipeline can select columns by name.
    """
    y = np.asarray(y)

    if validation == "holdout":
        strat = y if len(np.unique(y)) > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat
        )
        model = spec.factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = (_safe_proba(model, X_te, class_names)
                   if with_proba else None)
        y_true = y_te
        fold_scores = [float(balanced_accuracy_score(y_te, y_pred))]
    else:
        splitter = _cv_splitter(validation, y)
        model = spec.factory()
        y_pred = cross_val_predict(model, X, y, cv=splitter)
        y_proba = None
        if with_proba:
            try:
                y_proba = cross_val_predict(model, X, y, cv=splitter,
                                            method="predict_proba")
            except Exception:  # noqa: BLE001
                y_proba = None
        y_true = y
        fold_scores = _fold_scores(splitter, X, y, y_pred)

    return _make_result(spec.key, spec.name, y_true, y_pred, y_proba,
                        class_names, metrics, fold_scores=fold_scores)


def _fold_scores(splitter, X, y, y_pred) -> List[float]:
    """Balanced accuracy of each cross-validation test fold (same folds as
    used for the predictions). Empty for leave-one-out, where a single-row
    fold has no meaningful balanced accuracy."""
    if isinstance(splitter, LeaveOneOut):
        return []
    scores = []
    for _, te in splitter.split(X, y):
        scores.append(float(balanced_accuracy_score(y[te], y_pred[te])))
    return scores


def _make_result(key, name, y_true, y_pred, y_proba, class_names,
                 metrics, note="", fold_scores=None) -> Result:
    m = compute_metrics(y_true, y_pred, y_proba, metrics, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return Result(
        classifier_key=key,
        classifier_name=name,
        metrics=m,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_proba=y_proba,
        class_names=class_names,
        confusion=cm,
        selection_score=float(balanced_accuracy_score(y_true, y_pred)),
        note=note,
        fold_scores=list(fold_scores or []),
    )


# --------------------------------------------------------------------------- #
# Honest estimate for the selected (best) classifier
# --------------------------------------------------------------------------- #
#
# Picking the best of several classifiers by their validation scores and then
# reporting that same score is optimistic: the winner partly won by luck on
# those particular splits. Two remedies are offered:
#
# * "final_test": keep 20% of the rows aside before anything else. Classifiers
#   are compared on the other 80% only; the winner is scored once on the
#   untouched 20%.
# * "nested": nested cross-validation. In each outer fold the whole selection
#   procedure (compare all classifiers with inner validation, pick the best)
#   runs on the training part only, and the winner is scored on the outer
#   test part. Better for small datasets, where 20% would be too few rows.

SELECTION = {
    "auto": "Automatic (recommended)",
    "final_test": "Keep 20% aside as a final, untouched test set",
    "nested": "Nested cross-validation (best for small datasets)",
}

FINAL_TEST_SIZE = 0.2


# Up to this many rows, nested CV is used: in EasyClassifier's benchmarks it
# gave more accurate final scores than a 20% test set on small and medium
# data (a 20% test set of a few hundred rows is noisy). Larger datasets use
# the final test set, which is accurate there and about 5 times faster.
NESTED_MAX_ROWS = 2000


def recommend_selection(y) -> str:
    """Nested CV up to NESTED_MAX_ROWS rows, otherwise a final test set
    (provided every class has enough rows for a meaningful test set)."""
    counts = np.unique(np.asarray(y), return_counts=True)[1]
    if len(y) > NESTED_MAX_ROWS and counts.min() * FINAL_TEST_SIZE >= 5:
        return "final_test"
    return "nested"


def split_final_test(X, y, seed: int = 42):
    """Stratified split into development (80%) and final test (20%) rows."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    dev, test = train_test_split(idx, test_size=FINAL_TEST_SIZE,
                                 random_state=seed, stratify=y)
    return dev, test


def evaluate_on_test(spec, X_dev, y_dev, X_test, y_test, class_names,
                     metrics) -> Result:
    """Fit on the development rows, score once on the untouched test rows."""
    model = spec.factory()
    model.fit(X_dev, np.asarray(y_dev))
    y_pred = model.predict(X_test)
    y_proba = _safe_proba(model, X_test, class_names)
    return _make_result(spec.key, spec.name, np.asarray(y_test), y_pred,
                        y_proba, class_names, metrics)


def nested_cv(specs, X, y, class_names, validation: str,
              metrics: List[str], progress=None) -> Result:
    """Nested cross-validation estimate of the 'pick the best' procedure."""
    y = np.asarray(y)
    inner_validation = "kfold5" if validation == "loo" else validation
    outer = _cv_splitter("kfold5", y)
    y_pred = np.empty_like(y)
    y_proba = np.zeros((len(y), len(class_names)))
    have_proba = True
    winners: List[str] = []

    for fold, (tr, te) in enumerate(outer.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr], y[tr]
        best_spec, best_score = None, -np.inf
        for spec in specs:
            try:
                # Selection needs predicted classes only, not probabilities.
                r = evaluate(spec, X_tr, y_tr, class_names,
                             inner_validation, [], with_proba=False)
            except Exception:  # noqa: BLE001
                continue
            if r.selection_score > best_score:
                best_spec, best_score = spec, r.selection_score
        if best_spec is None:
            raise RuntimeError("no classifier could be trained in a fold")
        winners.append(best_spec.name)
        if progress:
            progress(fold, best_spec.name)
        model = best_spec.factory()
        model.fit(X_tr, y_tr)
        X_te = X.iloc[te]
        y_pred[te] = model.predict(X_te)
        p = _safe_proba(model, X_te, class_names)
        if p is None:
            have_proba = False
        else:
            y_proba[te] = p

    counts = {w: winners.count(w) for w in dict.fromkeys(winners)}
    note = "; ".join(f"{w} chosen in {c}/{len(winners)} folds"
                     for w, c in counts.items())
    return _make_result("nested", "Best classifier (nested CV estimate)",
                        y, y_pred, y_proba if have_proba else None,
                        class_names, metrics, note=note)


def _safe_proba(model, X, class_names):
    try:
        proba = model.predict_proba(X)
        # Ensure full class width even if a class was absent in training fold.
        if proba.shape[1] == len(class_names):
            return proba
    except Exception:  # noqa: BLE001
        pass
    return None


def fit_final_model(spec, X, y):
    """Fit a model on all data for saving/predictions."""
    model = spec.factory()
    model.fit(X, np.asarray(y))
    return model
