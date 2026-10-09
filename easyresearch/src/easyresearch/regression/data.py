"""Choosing and checking the numeric column to predict, and the demo data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from easyclassifier import target as tg

# A column with this many different whole numbers or fewer may be a set of
# categories (e.g. a 1-5 rating); regression is allowed but a note is shown.
FEW_VALUES = 10
MIN_ROWS = 20

COMMON_TARGET_NAMES = [
    "target", "y", "price", "value", "amount", "sales", "revenue", "cost",
    "yield", "score", "progression", "output", "response", "income",
    "salary", "weight", "height", "length", "age", "temperature",
    "concentration", "duration", "time", "load", "strength", "rate",
]


@dataclass
class TargetCheck:
    suitable: bool
    note: str          # why it is unsuitable, or a caution; "" if fine
    label: str         # plain description shown in the column list


def numeric_values(s: pd.Series) -> Optional[pd.Series]:
    """The column as numbers, or None if it is not numeric.

    Text columns are accepted when every non-empty value is a number
    (including decimal commas, e.g. "3,5")."""
    if pd.api.types.is_bool_dtype(s):
        return None
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    text = s.dropna().astype(str).str.strip()
    if text.empty:
        return None
    conv = pd.to_numeric(text.str.replace(",", ".", regex=False),
                         errors="coerce")
    if conv.isna().any():
        return None
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[conv.index] = conv.to_numpy(dtype=float)
    return out


def check_target(s: pd.Series) -> TargetCheck:
    """Can this column be predicted with regression?"""
    name = str(s.name)
    info = tg.describe_column(s)
    if info.kind in (tg.EMPTY, tg.CONSTANT):
        return TargetCheck(False, "It has only one value, so there is "
                           "nothing to predict.", info.label)
    values = numeric_values(s)
    if values is None:
        return TargetCheck(False, "It contains text categories; use "
                           "Classification to predict a category.",
                           info.label)
    v = values.dropna()
    u = int(v.nunique())
    label = (f"{name} - numbers from {tg._fmt(v.min())} to "
             f"{tg._fmt(v.max())} ({u} different)")
    if info.kind == tg.ID_LIKE:
        return TargetCheck(False, "It is different in every row and looks "
                           "like an identifier, not a measurement.",
                           info.label)
    if u == 2:
        return TargetCheck(False, "It has only two values (a yes/no "
                           "outcome); use Classification.", label)
    if u <= FEW_VALUES and tg._is_integer_valued(v):
        return TargetCheck(True, f"It has only {u} different whole numbers. "
                           "If these are categories rather than amounts "
                           "(e.g. codes), use Classification.", label)
    return TargetCheck(True, "", label)


def suggest_target(df: pd.DataFrame) -> Optional[str]:
    """Most likely numeric column to predict, or None."""
    checks = {c: check_target(df[c]) for c in df.columns}
    clean = [c for c in df.columns
             if checks[c].suitable and not checks[c].note]
    lower = {str(c).lower(): c for c in clean}
    for name in COMMON_TARGET_NAMES:
        if name in lower:
            return lower[name]
    if clean:
        return clean[-1]           # outcome columns are usually last
    usable = [c for c in df.columns if checks[c].suitable]
    return usable[-1] if usable else None


def load_demo() -> pd.DataFrame:
    """Diabetes progression data (Efron et al., 2004), shipped with
    scikit-learn: 442 patients, ten baseline measurements, and a measure of
    disease progression one year later."""
    from sklearn.datasets import load_diabetes
    df = load_diabetes(scaled=False, as_frame=True).frame.copy()
    return df.rename(columns={
        "bp": "blood_pressure", "s1": "total_cholesterol", "s2": "ldl",
        "s3": "hdl", "s4": "cholesterol_hdl_ratio",
        "s5": "log_triglycerides", "s6": "glucose", "target": "Progression",
    })


DEMO_NAME = "Diabetes progression demonstration"
DEMO_CITATION = ("Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. "
                 "(2004). Least angle regression. Annals of Statistics, "
                 "32(2), 407-499.")
