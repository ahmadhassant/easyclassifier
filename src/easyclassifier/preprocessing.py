"""Data cleaning, encoding, and scaling - leakage-safe.

Two kinds of step are handled differently:

* Row-level steps that learn nothing from the data (removing duplicate rows,
  removing rows with missing values, casting text columns to text) are applied
  once, before any train/test split.

* Steps that *learn* something from the data (the mean/median used to fill
  gaps, the scaling range, the list of categories) are placed inside a
  scikit-learn Pipeline. During hold-out or cross-validation the pipeline is
  re-fitted on each training part only, so no information from the test part
  leaks into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# --------------------------------------------------------------------------- #
# Row-level cleaning (no learned statistics -> safe before splitting)
# --------------------------------------------------------------------------- #

def drop_missing_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, int]:
    """Rows without a class label cannot be used for training or testing."""
    before = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    return df, before - len(df)


def drop_missing_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    return df, before - len(df)


def duplicates_expected_by_chance(df: pd.DataFrame,
                                  factor: float = 10.0) -> bool:
    """True if identical rows are expected between *different* cases.

    With few columns that each take few values (e.g. sex, region, a 0-24
    score), different people often give identical answers; such rows are
    real observations and must be kept. Only when the columns allow far more
    combinations than there are rows (at least ``factor`` times as many) is
    an identical row likely to be an accidental copy.
    """
    combos = 1.0
    for c in df.columns:
        combos *= max(1, df[c].nunique(dropna=False))
        if combos >= factor * len(df):
            return False
    return True


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def categorical_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def numeric_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def cast_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Make every non-numeric column plain text (missing values stay missing).

    This is a type conversion only; nothing is learned from the data.
    """
    X = X.copy()
    for c in categorical_columns(X):
        col = X[c].astype(object)
        X[c] = col.where(col.isna(), col.astype(str))
    return X


# --------------------------------------------------------------------------- #
# Target encoding (a fixed label mapping - not a learned statistic)
# --------------------------------------------------------------------------- #

def encode_target(y: pd.Series) -> Tuple[np.ndarray, List[str], LabelEncoder]:
    """Class labels -> 0..k-1. Alphabetical order, except for ordered
    categories (e.g. Low / Medium / High groups), which keep their order."""
    le = LabelEncoder()
    if isinstance(y.dtype, pd.CategoricalDtype) and y.cat.ordered:
        present = [c for c in y.cat.categories if (y == c).any()]
        le.classes_ = np.array([str(c) for c in present], dtype=object)
        mapping = {str(c): i for i, c in enumerate(present)}
        y_enc = y.astype(str).map(mapping).to_numpy(dtype=int)
        return y_enc, [str(c) for c in present], le
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, [str(c) for c in le.classes_], le


# --------------------------------------------------------------------------- #
# Learned preprocessing -> goes inside the Pipeline
# --------------------------------------------------------------------------- #

@dataclass
class PrepConfig:
    """User choices for the learned preprocessing steps."""

    impute: str = "median"        # "mean" | "median" | "mode"
    encoding: str = "auto"        # "auto" | "label" | "onehot"
    scale_method: str = "standard"  # "standard" | "minmax"
    scale: Optional[bool] = None  # True / False / None = decide per classifier
    onehot_max_categories: int = 10


def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _ordinal():
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def build_preprocessor(X: pd.DataFrame, cfg: PrepConfig,
                       scale) -> ColumnTransformer:
    """Build an *unfitted* transformer for the columns of ``X``.

    ``scale``: False/None (no scaling), True (``cfg.scale_method``), or the
    method itself, "standard" or "minmax".

    Only column names and types are read from ``X`` here; all statistics are
    learned later, when the pipeline is fitted on training data.
    """
    num_cols = numeric_columns(X)
    cat_cols = categorical_columns(X)
    num_strategy = "most_frequent" if cfg.impute == "mode" else cfg.impute
    method = scale if isinstance(scale, str) else (
        cfg.scale_method if scale else None)

    transformers = []
    if num_cols:
        steps = [("impute", SimpleImputer(strategy=num_strategy))]
        if method:
            scaler = (MinMaxScaler() if method == "minmax"
                      else StandardScaler())
            steps.append(("scale", scaler))
        transformers.append(("num", Pipeline(steps), num_cols))

    if cat_cols:
        if cfg.encoding == "label":
            onehot_cols, ordinal_cols = [], cat_cols
        elif cfg.encoding == "onehot":
            onehot_cols, ordinal_cols = cat_cols, []
        else:  # auto: one-hot for few categories, ordinal for many
            onehot_cols = [c for c in cat_cols
                           if X[c].nunique(dropna=True)
                           <= cfg.onehot_max_categories]
            ordinal_cols = [c for c in cat_cols if c not in onehot_cols]
        impute = ("impute", SimpleImputer(strategy="most_frequent"))
        if onehot_cols:
            transformers.append(("cat_onehot", Pipeline(
                [impute, ("encode", _onehot())]), onehot_cols))
        if ordinal_cols:
            transformers.append(("cat_ordinal", Pipeline(
                [("impute", SimpleImputer(strategy="most_frequent")),
                 ("encode", _ordinal())]), ordinal_cols))

    return ColumnTransformer(transformers, remainder="drop")


def build_pipeline(X: pd.DataFrame, cfg: PrepConfig, classifier,
                   scale: bool) -> Pipeline:
    """Preprocessing + classifier, fitted together on training data only."""
    return Pipeline([
        ("prep", build_preprocessor(X, cfg, scale)),
        ("model", classifier),
    ])


def describe_transformed(X: pd.DataFrame, cfg: PrepConfig,
                         scale: bool) -> np.ndarray:
    """Transform the full data for *description only* (feature counts,
    whether negative values occur). Never used for scoring."""
    return build_preprocessor(X, cfg, scale).fit_transform(X)


# --------------------------------------------------------------------------- #
# Class balance
# --------------------------------------------------------------------------- #

def class_distribution(y) -> Dict[str, int]:
    """Count rows per class, ignoring missing labels and mixed types."""
    counts = pd.Series(np.asarray(y, dtype=object)).dropna().astype(str) \
        .value_counts(sort=False)
    return {str(k): int(v) for k, v in counts.items()}


def imbalance_ratio(y) -> float:
    """Ratio of the largest class to the smallest. 1.0 means balanced."""
    counts = list(class_distribution(y).values())
    if not counts or min(counts) == 0:
        return float("inf")
    return max(counts) / min(counts)
