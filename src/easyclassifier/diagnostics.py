"""Learning curve and plain-language interpretations of results.

* ``learning_curve_data`` - how the score changes as more training rows are
  used ("would more data help?").
* ``interpret_learning_curve`` / ``interpret_comparison`` - one or two
  sentences a non-specialist can act on.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, learning_curve

from .evaluation import SELECTION_LABEL

TRAIN_SIZES = np.linspace(0.2, 1.0, 5)


def learning_curve_data(spec, X, y, seed: int = 42) -> Optional[Dict]:
    """Balanced accuracy on training and validation rows for 20%...100% of
    the available training rows (5-fold cross-validation). Returns None if
    the data are too small to compute it."""
    y = np.asarray(y)
    smallest = int(np.min(np.unique(y, return_counts=True)[1]))
    n_splits = max(2, min(5, smallest))
    if len(y) < 20 or smallest < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sizes, train, val = learning_curve(
        spec.factory(), X, y, train_sizes=TRAIN_SIZES, cv=cv,
        scoring="balanced_accuracy", shuffle=True, random_state=seed,
        error_score=np.nan)
    keep = ~np.all(np.isnan(val), axis=1)
    if keep.sum() < 2:
        return None
    return {
        "sizes": sizes[keep],
        "train_mean": np.nanmean(train[keep], axis=1),
        "train_std": np.nanstd(train[keep], axis=1),
        "val_mean": np.nanmean(val[keep], axis=1),
        "val_std": np.nanstd(val[keep], axis=1),
    }


def interpret_learning_curve(d: Dict) -> str:
    """Would more data probably help? Based on the last part of the curve."""
    v = d["val_mean"]
    gain = float(v[-1] - v[-2])
    gap = float(d["train_mean"][-1] - v[-1])
    if gain > 0.01:
        s = ("The score was still rising when all rows were used, so "
             "collecting more data would probably improve the results.")
    elif gain < -0.01:
        s = ("The score did not improve at the end of the curve; more data "
             "of the same kind is unlikely to help much.")
    else:
        s = ("The score had levelled off, so more data of the same kind is "
             "unlikely to improve the results much.")
    if gap > 0.15:
        s += (" The model does much better on its training rows than on new "
              "rows, a sign that it partly memorises the training data.")
    return s


def interpret_comparison(results: List) -> str:
    """Is the best classifier clearly better than the runner-up?"""
    if len(results) < 2:
        return ""
    a, b = results[0], results[1]
    diff = a.selection_score - b.selection_score
    spread = (float(np.std(a.fold_scores)) if len(a.fold_scores) > 1
              else None)
    label = SELECTION_LABEL.lower()
    if diff < 0.0005:
        tied = [r.classifier_name for r in results
                if a.selection_score - r.selection_score < 0.0005]
        return (f"{', '.join(tied[:-1])} and {tied[-1]} had the same "
                f"{label}; {a.classifier_name} was selected because it is "
                "listed first. They perform equally well on this data.")
    if spread is None:
        return (f"{a.classifier_name} had the highest {label}; with a "
                "single test split the uncertainty of this ranking cannot be "
                "judged.")
    if diff < spread:
        return (f"{a.classifier_name} had the highest {label}, but its lead "
                f"over {b.classifier_name} ({diff * 100:.1f} percentage "
                f"points) is smaller than the variation between test folds "
                f"({spread * 100:.1f} points). The top classifiers perform "
                "similarly; the choice between them is not clear-cut.")
    return (f"{a.classifier_name} had the highest {label}, ahead of "
            f"{b.classifier_name} by {diff * 100:.1f} percentage points, "
            "which is larger than the variation between test folds "
            f"({spread * 100:.1f} points).")
