"""Smart recommendations (Phase 22).

Inspects the dataset and suggests sensible methods in plain language.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .dataset import Inspection
from .preprocessing import duplicates_expected_by_chance, imbalance_ratio


def _cat_cols(df, target) -> List[str]:
    """Non-numeric feature columns (works for object and string dtypes)."""
    return [c for c in df.columns
            if c != target and not pd.api.types.is_numeric_dtype(df[c])]


def missing_strategy(insp: Inspection) -> str:
    """Recommend a default missing-value strategy key."""
    if not insp.has_missing:
        return "none"
    # If missingness is small, filling is safer than dropping.
    frac = insp.missing_total / max(1, insp.n_rows * insp.n_cols)
    return "median" if frac < 0.2 else "drop"


def encoding_method(df, target) -> str:
    """Recommend an encoding method key based on categorical cardinality."""
    cat_cols = _cat_cols(df, target)
    if not cat_cols:
        return "auto"
    high_card = any(df[c].nunique() > 10 for c in cat_cols)
    return "auto" if high_card else "onehot"


# Distance- and margin-based classifiers depend on the units of the columns.
SCALED_CLASSIFIERS = {"svm", "logistic_regression", "neural_network", "knn"}


def scaling_for(key: str, knn_distance: Optional[str] = None,
                choice: Optional[bool] = None,
                method: str = "standard") -> Optional[str]:
    """How to scale numeric columns for one classifier.

    Returns None (no scaling), "standard" (mean 0, SD 1) or "minmax" (0-1).
    ``choice`` is the user's answer: True / False, or None for automatic.

    Automatic: tree-based models and Naive Bayes are not scaled; the others
    are standardised; KNN with the Hassanat distance is scaled to 0-1, which
    keeps all values non-negative (the formula's standard form) and gave
    better results than unscaled data in EasyClassifier's benchmarks.
    """
    if choice is False:
        return None
    if choice is True:
        return method
    if key == "knn" and knn_distance == "hassanat":
        return "minmax"
    return "standard" if key in SCALED_CLASSIFIERS else None


def validation_method(y) -> str:
    """Recommend a validation strategy key based on class balance."""
    ratio = imbalance_ratio(y)
    if ratio > 1.5:
        return "stratified"
    return "kfold5"


def build_notes(insp: Inspection, df, target, y) -> List[str]:
    """Return a list of plain-language recommendation strings to show once."""
    notes: List[str] = []

    if insp.has_missing:
        rec = missing_strategy(insp)
        pct = 100 * insp.missing_total / max(1, insp.n_rows * insp.n_cols)
        if rec == "drop":
            notes.append(
                f"{insp.missing_total} missing values ({pct:.1f}% of cells) "
                "detected. That's a lot, so removing incomplete rows is a "
                "reasonable default."
            )
        else:
            notes.append(
                f"{insp.missing_total} missing values ({pct:.1f}% of cells) "
                "detected. Filling them with the median keeps all your rows."
            )

    if insp.has_duplicates:
        if duplicates_expected_by_chance(df):
            notes.append(
                f"{insp.duplicate_rows} identical rows found. Your columns "
                "allow only a few value combinations, so different cases "
                "(e.g. people giving the same answers) are expected to "
                "coincide. They will be kept.")
        else:
            notes.append(
                f"{insp.duplicate_rows} duplicate rows detected. They are "
                "probably accidental copies; removing them avoids "
                "over-counting repeated records.")

    cat_cols = _cat_cols(df, target)
    if cat_cols:
        high = [c for c in cat_cols if df[c].nunique() > 10]
        if high:
            notes.append(
                "Some text columns have many categories. Automatic encoding "
                "will label-encode those and one-hot encode the simpler ones."
            )
        else:
            notes.append(
                "Text columns detected with few categories each. One-hot "
                "encoding is a safe choice."
            )

    ratio = imbalance_ratio(y)
    if ratio > 1.5:
        notes.append(
            f"Classes are imbalanced (largest is {ratio:.1f}x the smallest). "
            "Stratified cross-validation is recommended so each fold keeps the "
            "same class mix."
        )

    return notes
