"""Built-in demo dataset so users can try the tool without their own CSV."""

from __future__ import annotations

import pandas as pd


def load_demo() -> pd.DataFrame:
    """Return a small, friendly classification dataset (the Iris flowers)."""
    try:
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        # Give it human-friendly column names and a text target.
        df = df.rename(columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        })
        df["Species"] = pd.Categorical.from_codes(
            data.target, data.target_names
        ).astype(str)
        df = df.drop(columns=["target"])
        return df
    except Exception:  # noqa: BLE001
        # Extremely small fallback if sklearn datasets are unavailable.
        return pd.DataFrame({
            "x1": [1, 2, 3, 4, 5, 6],
            "x2": [6, 5, 4, 3, 2, 1],
            "Species": ["A", "A", "A", "B", "B", "B"],
        })
