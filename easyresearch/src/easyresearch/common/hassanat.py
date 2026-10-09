"""Hassanat similarity and the MHSP measure.

Hassanat similarity between two values a and b (per dimension of the
Hassanat distance, which is 1 minus this similarity):

    if min(a, b) >= 0:   S = (1 + min) / (1 + max)
    if min(a, b) <  0:   S = (1 + min + |min|) / (1 + max + |min|)

S is 1 when a == b and approaches 0 as they move apart, so a single large
error cannot dominate an average of similarities.

MHSP (mean Hassanat similarity percentage) of a set of predictions is

    MHSP = 100 * mean_i S(actual_i, predicted_i)

100% means every prediction equals its actual value; higher is better.
It is computed on the values as they are (no scaling), as defined in
Hassanat et al. (2024), Mathematics 12(22), 3623.
"""

from __future__ import annotations

import numpy as np

# The paper that introduced MHSP as an outlier-robust accuracy measure for
# regression; cited wherever MHSP is reported.
MHSP_CITATION = (
    "Hassanat, A. B., Alqaralleh, M. K., Tarawneh, A. S., Almohammadi, K., "
    "Alamri, M., Alzahrani, A., Altarawneh, G. A., & Alhalaseh, R. (2024). "
    "A novel outlier-robust accuracy measure for machine learning regression "
    "using a non-convex distance metric. Mathematics, 12(22), 3623. "
    "https://doi.org/10.3390/math12223623")

# Cited for the Hassanat distance when KNN uses it: the IEEE conference paper
# that reviews its applications, and the arXiv preprint that introduced it
# (the same two as EasyClassifier).
HASSANAT_CITATIONS = [
    "Hassanat, A. B., Alkafaween, E., Tarawneh, A. S., & Elmougy, S. (2022). "
    "Applications review of Hassanat distance metric. In 2022 International "
    "Conference on Emerging Trends in Computing and Engineering Applications "
    "(ETCEA), Karak, Jordan (pp. 1-6). IEEE. "
    "https://doi.org/10.1109/ETCEA57049.2022.10009844",
    "Hassanat, A. B. (2014). Dimensionality invariant similarity measure. "
    "arXiv preprint arXiv:1409.0923. https://arxiv.org/abs/1409.0923",
]


def hassanat_similarity(a, b) -> np.ndarray:
    """Element-wise Hassanat similarity in [0, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    shift = np.where(mn < 0, -mn, 0.0)
    return (1.0 + mn + shift) / (1.0 + mx + shift)


def mhsp(actual, predicted) -> float:
    """Mean Hassanat similarity percentage (0-100, higher is better)."""
    return float(100.0 * np.mean(hassanat_similarity(actual, predicted)))


MHSP_EXPLANATION = (
    "MHSP (mean Hassanat similarity percentage): for each prediction, the "
    "Hassanat similarity (1 + smaller value) / (1 + larger value) between "
    "the actual and predicted value, averaged and shown as a percentage. "
    "100% means perfect predictions. Each prediction contributes at most "
    "its share, so one very large error cannot dominate it. It depends on "
    "the units of the values: errors on large values count as relatively "
    "small.")
