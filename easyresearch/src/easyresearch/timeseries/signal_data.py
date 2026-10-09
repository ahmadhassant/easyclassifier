"""Reading recordings for signal classification (one label per recording).

Accepted layouts
----------------
* A table (CSV/Excel) with one row per recording: a label column, and the
  signal values in the other numeric columns, in time order (e.g. ECG
  heartbeats, sensor windows). Identifier columns are left out.
* UCR/UEA archive files (``.tsv`` or ``.txt``): no header, the label in the
  first column, the signal in the remaining columns.

Cleaning works inside each recording only (nothing is learned from other
recordings, so it is safe before splitting): gaps inside a recording are
filled by straight lines between neighbouring values, and a shorter
recording is padded with its last value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from easyclassifier import dataset as ds
from easyclassifier import target as tg

MIN_LENGTH = 10


@dataclass
class Recordings:
    X: pd.DataFrame                  # (recordings, time points), raw values
    labels: pd.Series
    file_rows: np.ndarray
    notes: List[str] = field(default_factory=list)


def load_table(path: str) -> pd.DataFrame:
    """CSV/Excel via EasyClassifier; UCR .tsv/.txt with the label first."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tsv", ".txt"):
        df = pd.read_csv(path, sep=r"[\t ,]+", header=None, engine="python")
        df.columns = ["label"] + [f"t{i}" for i in range(1, df.shape[1])]
        return df
    return ds.load_csv(path)


ID_NAMES = ("id", "index", "subject", "patient", "record", "recording")


def signal_columns(df: pd.DataFrame, label: str) -> List[str]:
    """Numeric columns other than the label and identifiers, in file order."""
    return [c for c in df.columns
            if c != label and pd.api.types.is_numeric_dtype(df[c])
            and str(c).lower() not in ID_NAMES
            and tg.describe_column(df[c]).kind != tg.ID_LIKE]


def suggest_label(df: pd.DataFrame):
    names = {str(c).lower(): c for c in df.columns}
    for n in ("label", "class", "target", "diagnosis", "type", "y"):
        if n in names:
            return names[n]
    cats = [c for c in df.columns
            if tg.describe_column(df[c]).kind == tg.CATEGORY]
    return cats[0] if cats else None


def prepare(df: pd.DataFrame, label: str) -> Recordings:
    if label not in df.columns:
        raise ValueError(f"Column '{label}' was not found.")
    info = tg.describe_column(df[label])
    if info.kind != tg.CATEGORY:
        raise ValueError(f"'{label}' does not look like class labels "
                         f"({info.label}). Choose the column that holds the "
                         "class of each recording.")
    cols = signal_columns(df, label)
    if len(cols) < MIN_LENGTH:
        raise ValueError(f"Each recording needs at least {MIN_LENGTH} "
                         f"values (numeric columns); {len(cols)} were found.")
    notes: List[str] = []
    left = [str(c) for c in df.columns if c != label and c not in cols]
    if left:
        notes.append("Columns not used as signal values: " + ", ".join(left)
                     + ".")
    keep = df[label].notna()
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} recording(s) without a label "
                     "were removed.")
    X = df.loc[keep, cols].astype(float)
    rows = np.flatnonzero(keep.to_numpy()) + 2
    empty = X.isna().all(axis=1)
    if empty.any():
        notes.append(f"{int(empty.sum())} empty recording(s) were removed.")
        X, rows = X[~empty], rows[~empty.to_numpy()]
    gaps = int(X.isna().sum().sum())
    if gaps:
        X = X.T.interpolate(limit_direction="both").T
        notes.append(f"{gaps} missing value(s) inside recordings were filled "
                     "from neighbouring values of the same recording.")
    labels = df.loc[X.index, label].astype(str)
    counts = labels.value_counts()
    tiny = counts[counts < 2]
    if len(tiny):
        raise ValueError("Every class needs at least two recordings; too "
                         "few for: " + ", ".join(map(str, tiny.index)) + ".")
    X.columns = [str(c) for c in X.columns]
    return Recordings(X.reset_index(drop=True), labels.reset_index(drop=True),
                      rows, notes)


def load_demo(seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC heartbeats for trying the tool (not real patients): 300
    beats of 140 samples in three shape classes, with noise, baseline
    wander and small timing differences."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 140)

    def g(c, w, a):
        return a * np.exp(-0.5 * ((t - c) / w) ** 2)

    rows = []
    for cls in ("Normal", "Wide QRS", "ST depression"):
        for _ in range(100):
            s = rng.normal(0, 0.012)
            p = g(0.2 + s, 0.025, 0.15)
            q = g(0.37 + s, 0.012, -0.12)
            r_w = 0.016 if cls == "Wide QRS" else 0.011
            r = g(0.40 + s, r_w * rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.2))
            sw = g(0.43 + s, 0.012 if cls != "Wide QRS" else 0.018, -0.2)
            st = (-0.07 * ((t > 0.45 + s) & (t < 0.58 + s))
                  if cls == "ST depression" else 0)
            tw = g(0.68 + s, 0.05, rng.uniform(0.25, 0.35)
                   * (0.6 if cls == "ST depression" else 1))
            base = 0.05 * np.sin(2 * np.pi * (t * rng.uniform(0.3, 1)
                                              + rng.uniform()))
            beat = p + q + r + sw + st + tw + base + rng.normal(0, 0.05,
                                                                 t.size)
            rows.append([cls] + list(np.round(beat, 4)))
    df = pd.DataFrame(rows, columns=["label"] + [f"t{i}" for i in
                                                range(1, 141)])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


DEMO_NAME = "Synthetic heartbeat demonstration (not real patient data)"
