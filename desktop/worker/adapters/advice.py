"""Plain-language advice shown after a file is read (step 1 of the desktop).

Each module adds a few sentences about the data it has just inspected: what
will be left out, what looks unusual, and how the final score will be
obtained. Advice is a help, never a requirement: if anything fails here, the
inspection still succeeds without it.
"""
from __future__ import annotations

import os

import pandas as pd
from easyclassifier import target as tg
from easyclassifier.evaluation import recommend_selection

SHOWN_CLASSES = 6


def names(items):
    items = [f"'{i}'" for i in items]
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def safely(build):
    """Advice must never stop an inspection."""
    try:
        return [line for line in build() if line]
    except Exception:  # noqa: BLE001
        return []


def table(df, info, target):
    """Identifier and constant columns, empty cells, repeated rows."""
    ids = [str(c) for c in df.columns if c != target and tg.describe_column(df[c]).kind == tg.ID_LIKE]
    flat = [str(c) for c in df.columns if c != target and tg.describe_column(df[c]).kind in (tg.CONSTANT, tg.EMPTY)]
    out = []
    if ids:
        out.append(f"{names(ids)} {'is' if len(ids) == 1 else 'are'} different in every row (an identifier or name) and will be "
                   "left out: an identifier cannot predict new cases and can make results look better than they are.")
    if flat:
        out.append(f"{names(flat)} {'has' if len(flat) == 1 else 'have'} a single value (or none) and will be left out.")
    if info.missing_total:
        out.append(f"{info.missing_total:,} cells are empty. They are filled automatically, learning only from the training "
                   "rows, so no information leaks from the rows used for testing.")
    if info.duplicate_rows:
        d = info.duplicate_rows
        out.append(f"{d:,} {'row is an exact copy' if d == 1 else 'rows are exact copies'} of another row. Check that "
                   f"{'it is a real repeated observation' if d == 1 else 'they are real repeated observations'}: a copy in both "
                   "the training and the test part makes the scores look better than they are.")
    return out


EXAMPLES = {'classification_iris.csv': 'Classification', 'regression_diabetes.csv': 'Regression',
            'forecasting_co2_monthly.csv': 'Time-series forecasting',
            'signals_heartbeats_synthetic.csv': 'Signal classification'}


def other_example(dataset, task):
    """A pointer when an example file of another task is opened here."""
    owner = EXAMPLES.get(os.path.basename(str(dataset)).lower())
    if owner is None or owner == task:
        return []
    return [f'This is the example file for {owner}; choose {owner} in the sidebar to use it as intended.']


def label_text(labels):
    """Class labels as people write them: whole numbers without '.0'."""
    s = pd.Series(labels).dropna()
    if pd.api.types.is_numeric_dtype(s) and len(s) and bool((s == s.round()).all()):
        s = s.astype('int64')
    return s.astype(str)


def classes(labels, noun='rows'):
    """Class sizes, imbalance and rare classes for a label column."""
    labels = label_text(labels)
    counts = labels.value_counts()
    if len(counts) < 2:
        return []
    shown = ', '.join(f'{k} ({v:,})' for k, v in counts.head(SHOWN_CLASSES).items())
    more = f' and {len(counts) - SHOWN_CLASSES} more' if len(counts) > SHOWN_CLASSES else ''
    out = [f"'{labels.name}' has {len(counts)} classes: {shown}{more}."]
    if counts.min() < 3:
        rare = [k for k, v in counts.items() if v < 3]
        out.append(f"{names(rare)} {'has' if len(rare) == 1 else 'have'} fewer than three {noun}, which is too few to learn "
                   "and test reliably. Collect more examples or combine rare classes before analysing.")
    elif counts.min() / counts.max() < 0.5:
        out.append(f"The classes are unbalanced (smallest {counts.min():,}, largest {counts.max():,} {noun}). Models are "
                   "chosen by balanced accuracy, so every class counts equally, and the most-frequent-class rule is shown "
                   "for comparison.")
    return out


def final_score(n, method, noun='rows'):
    if method == 'final_test':
        return (f"With {n:,} {noun}, 20% of them will be set aside and used only once, for the final score of the "
                "selected model.")
    return (f"With {n:,} {noun}, the final score will come from nested cross-validation, which tests the whole "
            "procedure of choosing a model, so the score is not inflated by picking the best one.")


def classification(df, info, target):
    out = table(df, info, target)
    if target is not None:
        labels = label_text(df[target])
        out += classes(df[target])
        if labels.nunique() >= 2:
            out.append(final_score(len(labels), recommend_selection(pd.factorize(labels)[0])))
    return out


def regression(df, info, target):
    out = table(df, info, target)
    if target is not None:
        n = int(pd.to_numeric(df[target], errors='coerce').notna().sum())
        out.append(final_score(n, 'nested' if n <= 2000 else 'final_test'))
    return out


def deep_note(models):
    return (f"Deep models ({models}) are not in the fast set: each takes several minutes on a normal computer. "
            "Tick them in step 3 to add them to the comparison.")
