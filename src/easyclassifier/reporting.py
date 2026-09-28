"""Output files: CSV/Excel summaries, predictions, model, importance table.

Figures are drawn by :mod:`easyclassifier.figures`.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd  # noqa: E402
import joblib  # noqa: E402

from .evaluation import Result, METRICS  # noqa: E402


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_feature_importance(table: pd.DataFrame, path: str) -> str:
    table.round(4).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Tabular summaries
# --------------------------------------------------------------------------- #

def results_dataframe(results: List[Result]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"Classifier": r.classifier_name}
        for key, label in METRICS.items():
            if key in r.metrics:
                row[label] = round(r.metrics[key], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def save_summary_csv(results: List[Result], path: str) -> str:
    results_dataframe(results).to_csv(path, index=False)
    return path


def save_excel(results: List[Result], path: str) -> Optional[str]:
    try:
        results_dataframe(results).to_excel(path, index=False,
                                            sheet_name="Results")
        return path
    except Exception:  # noqa: BLE001 (openpyxl missing)
        return None


def save_predictions(result: Result, path: str) -> str:
    names = result.class_names
    df = pd.DataFrame({
        "actual": [names[int(v)] if int(v) < len(names) else v
                   for v in result.y_true],
        "predicted": [names[int(v)] if int(v) < len(names) else v
                      for v in result.y_pred],
    })
    df.to_csv(path, index=False)
    return path


def save_model(model, path: str) -> str:
    joblib.dump(model, path)
    return path
