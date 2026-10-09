"""Regression core: predict a numeric measurement from a table.

Usage::

    from easyresearch.regression import Settings, run_analysis
    result = run_analysis(df, "Price", Settings(models=["linear_regression",
                          "random_forest"]), out_root=".",
                          source_name="houses.csv", source_stem="houses")
"""

from .analysis import Settings, run_analysis  # noqa: F401
from .models import build_registry  # noqa: F401
