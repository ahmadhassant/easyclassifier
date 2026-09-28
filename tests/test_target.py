"""Tests for choosing and checking the column to predict."""

import numpy as np
import pandas as pd

from easyclassifier import target as tg


def test_column_kinds():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "PatientID": np.arange(1000, 1000 + n),
        "income": rng.normal(50000, 10000, n).round(2),
        "age": rng.integers(18, 90, n),
        "stage": rng.integers(1, 5, n),
        "country": ["Jordan"] * n,
        "name": [f"person{i}" for i in range(n)],
        "Outcome": rng.choice(["Yes", "No"], n),
    })
    kind = {c: tg.describe_column(df[c]).kind for c in df.columns}
    assert kind == {
        "PatientID": tg.ID_LIKE,
        "income": tg.MEASUREMENT,
        "age": tg.MEASUREMENT,      # 70+ different whole numbers
        "stage": tg.CATEGORY,       # 4 coded groups
        "country": tg.CONSTANT,
        "name": tg.ID_LIKE,
        "Outcome": tg.CATEGORY,
    }
    assert tg.suggest_target(df) == "Outcome"
    assert "2 categories" in tg.describe_column(df["Outcome"]).label


def test_suggestion_skips_measurements_even_with_a_target_like_name():
    df = pd.DataFrame({"x": [1, 2, 3, 4] * 10,
                       "y": np.linspace(0, 1, 40),
                       "group": ["a", "b"] * 20})
    assert tg.describe_column(df["y"]).kind == tg.MEASUREMENT
    assert tg.suggest_target(df) == "group"
    assert tg.suggest_target(df[["y"]]) is None


def test_small_file_measurements_are_recognised():
    df = pd.DataFrame({
        "age": [25, 32, 47, 51, 23, 36, 29, 41, 38, 27, 33, 45, 22, 30, 48, 25],
        "city": ["NY", "LA", "SF", "NY"] * 4,
        "Purchased": ["No", "Yes"] * 8,
    })
    assert tg.describe_column(df["age"]).kind == tg.MEASUREMENT
    assert tg.describe_column(df["city"]).kind == tg.CATEGORY
    assert tg.suggest_target(df) == "Purchased"
    ages = pd.Series([1.0, 2.0, 2.0, np.nan], name="kids")
    assert tg.describe_column(ages).label == "kids - 2 categories (2, 1)"


def test_tiny_and_small_classes():
    s = pd.Series(["a"] * 10 + ["b"] * 3 + ["c"])
    tiny, small = tg.tiny_and_small_classes(s)
    assert tiny == ["c"] and small == ["b"]


def test_group_equal_and_threshold():
    s = pd.Series(np.arange(1, 101, dtype=float), name="score")
    s.iloc[5] = np.nan
    g, cuts, desc = tg.group_equal(s, 3)
    counts = g.value_counts()
    assert len(counts) == 3 and counts.min() >= 32
    assert g.isna().sum() == 1                  # missing stays missing
    assert desc.startswith("Low (< ") and "High (>= " in desc

    g2, cuts2, _ = tg.group_at(s, 60)
    assert cuts2 == [60.0]
    assert (g2[s >= 60] == "High (>= 60)").all()
    assert (g2[s < 60] == "Low (< 60)").all()


def test_grouped_classes_keep_low_to_high_order():
    from easyclassifier.preprocessing import encode_target
    s = pd.Series([90, 10, 50, 20, 80, 60, 30, 70, 40, 100], name="score")
    g, _, _ = tg.group_equal(s, 3)
    y, names, _ = encode_target(g)
    assert [n.split(" ")[0] for n in names] == ["Low", "Medium", "High"]
    assert y[1] == 0 and y[9] == 2      # 10 -> Low, 100 -> High


def test_group_equal_with_many_ties_gives_fewer_groups():
    s = pd.Series([0.0] * 80 + list(np.linspace(1, 2, 20)))
    g, cuts, _ = tg.group_equal(s, 4)
    assert 2 <= g.nunique() < 4
