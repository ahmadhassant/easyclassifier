"""Distance metrics for the KNN classifier, including the Hassanat distance.

Hassanat distance (per dimension i):

    if min(a_i, b_i) >= 0   (normal form):
        D = 1 - (1 + min) / (1 + max)

    if min(a_i, b_i) <  0   (signed form, for negative values):
        D = 1 - (1 + min + |min|) / (1 + max + |min|)

    HD(A, B) = sum_i D(a_i, b_i)

Each dimension contributes at most 1, which makes the metric invariant to
data scale, noise and outliers.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


HASSANAT_CITATION = (
    "Hassanat, A. B. (2014). Dimensionality Invariant Similarity Measure. "
    "Journal of American Science, 10(8). arXiv:1409.0923."
)

HASSANAT_CITATIONS: List[str] = [HASSANAT_CITATION]

# key -> (menu label, sklearn metric name or None for custom)
DISTANCES: Dict[str, tuple] = {
    "hassanat": ("Hassanat distance (robust to outliers)",
                 None),
    "euclidean": ("Euclidean distance", "euclidean"),
    "manhattan": ("Manhattan (city-block) distance", "manhattan"),
    "chebyshev": ("Chebyshev distance", "chebyshev"),
    "canberra": ("Canberra distance", "canberra"),
    "cosine": ("Cosine distance", "cosine"),
}

DISTANCE_HELP = {
    "hassanat": "The Hassanat distance compares each feature using the ratio "
                "of its smaller to larger value, so every feature adds at most "
                "1 to the total; one extreme value cannot dominate. "
                "EasyClassifier scales the columns to 0-1 first, which gave "
                "the best results in its benchmarks. Citation: "
                + HASSANAT_CITATION,
    "euclidean": "Euclidean distance is the straight-line distance between "
                 "two points. Sensitive to scale, so scaling is advised.",
    "manhattan": "Manhattan distance adds up the absolute differences of "
                 "each feature, like walking city blocks.",
    "chebyshev": "Chebyshev distance uses only the single largest feature "
                 "difference.",
    "canberra": "Canberra distance is a weighted version of Manhattan that "
                "is sensitive to small values near zero.",
    "cosine": "Cosine distance compares the direction of two rows, ignoring "
              "their size.",
}


# --------------------------------------------------------------------------- #
# Hassanat distance
# --------------------------------------------------------------------------- #

def hassanat_distance(a, b) -> float:
    """Hassanat distance between two 1-D vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    shift = np.where(mn < 0, -mn, 0.0)  # |min| only when min < 0
    return float(np.sum(1.0 - (1.0 + mn + shift) / (1.0 + mx + shift)))


MAX_CHUNK_CELLS = 4_000_000   # ~32 MB per temporary array


def hassanat_matrix(XA, XB, chunk: int = 256) -> np.ndarray:
    """Pairwise Hassanat distances (rows of XA vs rows of XB), vectorised.

    Rows of XA are processed in chunks small enough that the temporary
    arrays stay within a few hundred MB, even for large datasets.
    """
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)
    per_row = max(1, XB.shape[0] * XB.shape[1])
    chunk = max(1, min(chunk, MAX_CHUNK_CELLS // per_row))
    out = np.empty((XA.shape[0], XB.shape[0]))
    for s in range(0, XA.shape[0], chunk):
        A = XA[s:s + chunk, None, :]
        mn = np.minimum(A, XB[None, :, :])
        mx = np.maximum(A, XB[None, :, :])
        shift = np.where(mn < 0, -mn, 0.0)
        out[s:s + chunk] = np.sum(
            1.0 - (1.0 + mn + shift) / (1.0 + mx + shift), axis=2)
    return out


def hassanat_form(X) -> dict:
    """Report whether the normal or signed form of the formula is used."""
    X = np.asarray(X, dtype=float)
    neg_cols = int(np.sum((X < 0).any(axis=0)))
    if neg_cols == 0:
        return {"form": "normal", "negative_features": 0,
                "text": "Normal form used (all feature values are >= 0)."}
    return {"form": "signed", "negative_features": neg_cols,
            "text": (f"Signed form used: {neg_cols} feature(s) contain "
                     "negative values, so the negative-value case "
                     "1 - (1+min+|min|)/(1+max+|min|) was applied to them.")}


class HassanatKNN(BaseEstimator, ClassifierMixin):
    """K-Nearest Neighbours classifier using the Hassanat distance."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        return self

    def _neighbors(self, X):
        D = hassanat_matrix(X, self.X_)
        k = min(self.n_neighbors, self.X_.shape[0])
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        return idx

    def predict_proba(self, X):
        idx = self._neighbors(X)
        votes = self.y_[idx]
        n_cls = len(self.classes_)
        proba = np.zeros((votes.shape[0], n_cls))
        for c in range(n_cls):
            proba[:, c] = (votes == c).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_knn(distance: str = "hassanat", k: int = 5):
    """Factory returning a KNN classifier for the chosen distance."""
    if distance == "hassanat":
        return HassanatKNN(n_neighbors=k)
    metric = DISTANCES[distance][1]
    return KNeighborsClassifier(n_neighbors=k, metric=metric,
                                algorithm="brute")
