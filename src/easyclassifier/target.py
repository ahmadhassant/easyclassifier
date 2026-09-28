"""Choosing and checking the column to predict (the target / class column).

Everything here is plain logic with no user interaction, so it can be tested
directly. The wizard uses it to

* describe every column in plain words so the user can pick the right one,
* suggest the most likely class column,
* catch unsuitable choices (a single value, an ID column, a measurement
  instead of a category, classes with too few rows), and
* turn a measurement into categories when the user wants that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Kinds of column
CATEGORY = "category"          # a sensible class column
MEASUREMENT = "measurement"    # numbers with many different values
ID_LIKE = "id"                 # different in every row (an identifier)
CONSTANT = "constant"          # a single value, nothing to predict
MANY_TEXT = "many_text"        # text with very many different values
EMPTY = "empty"                # no values at all

COMMON_TARGET_NAMES = [
    "target", "label", "class", "outcome", "result", "disease",
    "diagnosis", "purchased", "churn", "y", "category", "species",
    "status", "response", "grade",
]

MIN_ROWS_PER_CLASS = 2         # fewer than this cannot be tested at all
FEW_ROWS_PER_CLASS = 5         # fewer than this gives unreliable results


def _max_categories(n_rows: int) -> int:
    """How many different values a class column may reasonably have."""
    return max(20, int(0.05 * n_rows))


def _is_integer_valued(s: pd.Series) -> bool:
    v = s.dropna().to_numpy(dtype=float)
    return bool(len(v)) and bool(np.all(np.equal(np.mod(v, 1), 0)))


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


@dataclass
class ColumnInfo:
    name: str
    kind: str
    n_unique: int
    n_values: int
    label: str                    # one-line plain-language description
    counts: Dict[str, int] = field(default_factory=dict)


def describe_column(s: pd.Series) -> ColumnInfo:
    """Classify a column and describe it in plain words."""
    name = str(s.name)
    vals = s.dropna()
    n, u = len(vals), int(vals.nunique())
    numeric = (pd.api.types.is_numeric_dtype(s)
               and not pd.api.types.is_bool_dtype(s))

    if n == 0:
        return ColumnInfo(name, EMPTY, 0, 0, f"{name} - empty column")
    if u == 1:
        return ColumnInfo(name, CONSTANT, 1, n,
                          f"{name} - only one value ({_fmt(vals.iloc[0])})")

    looks_like_id = "id" in name.lower().replace("_", " ").split() or \
        name.lower().endswith("id")
    if u == n and n >= 20:
        consecutive = (numeric and _is_integer_valued(vals)
                       and vals.max() - vals.min() + 1 == n)
        if not numeric or consecutive or looks_like_id:
            return ColumnInfo(name, ID_LIKE, u, n,
                              f"{name} - different in every row "
                              "(looks like an ID)")

    if numeric and u > 2:
        integer = _is_integer_valued(vals)
        mostly_distinct = u > 10 and u > n / 2
        if (not integer and u > 10) or mostly_distinct \
                or u > _max_categories(n):
            return ColumnInfo(
                name, MEASUREMENT, u, n,
                f"{name} - numbers from {_fmt(vals.min())} to "
                f"{_fmt(vals.max())} ({u} different)")

    if not numeric and u > _max_categories(n):
        return ColumnInfo(name, MANY_TEXT, u, n,
                          f"{name} - {u} different text values")

    counts = vals.value_counts()
    shown = ", ".join(_fmt(v) for v in counts.index[:3])
    more = ", ..." if u > 3 else ""
    return ColumnInfo(name, CATEGORY, u, n,
                      f"{name} - {u} categories ({shown}{more})",
                      counts={_fmt(k): int(v) for k, v in counts.items()})


def suggest_target(df: pd.DataFrame) -> Optional[str]:
    """Most likely class column, or None if nothing looks suitable."""
    infos = {c: describe_column(df[c]) for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    for name in COMMON_TARGET_NAMES:
        c = lower.get(name)
        if c is not None and infos[c].kind == CATEGORY:
            return c
    # Otherwise the last category-like column (class columns are usually
    # last), preferring few categories.
    cands = [c for c in df.columns if infos[c].kind == CATEGORY]
    if not cands:
        return None
    return min(reversed(cands), key=lambda c: infos[c].n_unique)


# --------------------------------------------------------------------------- #
# Class-size checks
# --------------------------------------------------------------------------- #

def tiny_and_small_classes(s: pd.Series) -> Tuple[List[str], List[str]]:
    """Classes with fewer than 2 rows (unusable) and fewer than 5 (risky)."""
    counts = s.dropna().astype(str).value_counts()
    tiny = [k for k, v in counts.items() if v < MIN_ROWS_PER_CLASS]
    small = [k for k, v in counts.items()
             if MIN_ROWS_PER_CLASS <= v < FEW_ROWS_PER_CLASS]
    return tiny, small


# --------------------------------------------------------------------------- #
# Turning a measurement into categories
# --------------------------------------------------------------------------- #

GROUP_NAMES = {
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
    4: ["Very low", "Low", "High", "Very high"],
}


def group_equal(s: pd.Series, k: int) -> Tuple[pd.Series, List[float], str]:
    """Split numbers into ``k`` groups of (roughly) equal size.

    Returns the new labels, the cut-points, and a description. If many rows
    share the same value, fewer groups may result.
    """
    vals = pd.to_numeric(s, errors="coerce")
    uniq = np.sort(vals.dropna().unique())
    cuts = []
    for q in vals.quantile([i / k for i in range(1, k)]).tolist():
        # A value >= cut goes to the higher group. A cut at (or below) the
        # smallest value would leave the lowest group empty, so move it to
        # the next value up - e.g. many zeros give "0" vs "above 0".
        above = uniq[uniq > q] if q <= uniq[0] else uniq[uniq >= q]
        if len(above):
            cuts.append(float(above[0]))
    cuts = sorted(set(cuts))
    if not cuts:
        raise ValueError("all values are the same; cannot form groups")
    return _cut(vals, cuts)


def group_at(s: pd.Series, threshold: float) -> Tuple[pd.Series, List[float], str]:
    """Two groups: below the threshold, and at or above it."""
    vals = pd.to_numeric(s, errors="coerce")
    return _cut(vals, [float(threshold)])


def _cut(vals: pd.Series, cuts: List[float]) -> Tuple[pd.Series, List[float], str]:
    cuts = sorted(set(float(c) for c in cuts))
    k = len(cuts) + 1
    names = GROUP_NAMES.get(k, [f"Group {i + 1}" for i in range(k)])
    labels = []
    for i in range(k):
        if i == 0:
            rng = f"< {_fmt(cuts[0])}"
        elif i == k - 1:
            rng = f">= {_fmt(cuts[-1])}"
        else:
            rng = f"{_fmt(cuts[i - 1])} to < {_fmt(cuts[i])}"
        labels.append(f"{names[i]} ({rng})")
    codes = np.searchsorted(np.asarray(cuts), vals.to_numpy(dtype=float),
                            side="right")
    # An ordered category keeps Low < Medium < High in tables and figures.
    out = pd.Series(pd.Categorical(
        [labels[c] if not np.isnan(v) else np.nan
         for c, v in zip(codes, vals.to_numpy(dtype=float))],
        categories=labels, ordered=True), index=vals.index, name=vals.name)
    desc = "; ".join(labels)
    return out, cuts, desc
