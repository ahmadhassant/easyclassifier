"""Which columns matter? Permutation importance, measured on held-out rows.

For each original column, its values are shuffled and the drop in balanced
accuracy is measured. A large drop means the model relies on that column.

* It works with every classifier (including KNN with the Hassanat distance).
* It is computed on the *original* columns, so a text column such as "city"
  is one variable, not several encoded pieces.
* It is measured on rows the model was not trained on (the final test set,
  or the test part of each cross-validation fold), so it is not inflated.
* It changes nothing in the model; it only describes it.

Reference: Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split

N_REPEATS = 5
MAX_EVAL_ROWS = 500   # rows used per evaluation, to keep it fast


def _perm(model, X_eval, y_eval, seed: int) -> np.ndarray:
    if len(y_eval) > MAX_EVAL_ROWS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_eval), MAX_EVAL_ROWS, replace=False)
        X_eval, y_eval = X_eval.iloc[idx], y_eval[idx]
    r = permutation_importance(model, X_eval, y_eval,
                               scoring="balanced_accuracy",
                               n_repeats=N_REPEATS, random_state=seed)
    return r.importances  # shape (n_columns, n_repeats)


def honest_importance(spec, X: pd.DataFrame, y, validation: str,
                      dev: Optional[np.ndarray] = None,
                      test: Optional[np.ndarray] = None,
                      seed: int = 0) -> pd.DataFrame:
    """Permutation importance of ``spec`` (an unfitted pipeline factory).

    With ``dev``/``test`` (final test set): fit on dev, measure on test.
    Otherwise: hold-out split or 5-fold CV, measured on each test part.
    Returns a table sorted from most to least important.
    """
    y = np.asarray(y)
    parts = []
    if dev is not None and test is not None:
        model = spec.factory().fit(X.iloc[dev], y[dev])
        parts.append(_perm(model, X.iloc[test], y[test], seed))
        how = "untouched final test set"
    elif validation == "holdout":
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.2, random_state=42,
                                  stratify=y)
        model = spec.factory().fit(X.iloc[tr], y[tr])
        parts.append(_perm(model, X.iloc[te], y[te], seed))
        how = "hold-out test part"
    else:
        smallest = int(np.min(np.unique(y, return_counts=True)[1]))
        cv = StratifiedKFold(n_splits=max(2, min(5, smallest)),
                             shuffle=True, random_state=42)
        for i, (tr, te) in enumerate(cv.split(X, y)):
            model = spec.factory().fit(X.iloc[tr], y[tr])
            parts.append(_perm(model, X.iloc[te], y[te], seed + i))
        how = "test part of each cross-validation fold"

    allv = np.concatenate(parts, axis=1)
    table = pd.DataFrame({
        "column": list(X.columns),
        "importance": allv.mean(axis=1),
        "std": allv.std(axis=1),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    table.attrs["measured_on"] = how
    return table
