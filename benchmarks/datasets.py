"""Benchmark datasets: public, from several fields, installable with pip.

All datasets come from packages on PyPI (scikit-learn and pydataset), so the
benchmark can be reproduced without downloading anything else:

    pip install easyclassifier pydataset scipy

pydataset ships classic datasets from R packages (MASS, car, Zelig). Each
entry lists the original source to cite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


@dataclass
class BenchmarkDataset:
    key: str
    name: str
    field: str
    target: str
    loader: Callable[[], pd.DataFrame]
    source: str
    notes: str = ""
    drop: List[str] = field(default_factory=list)

    def load(self) -> pd.DataFrame:
        df = self.loader()
        return df.drop(columns=self.drop)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _sk(loader, target: str, labels=None) -> Callable[[], pd.DataFrame]:
    def load():
        b = loader(as_frame=True)
        df = b.frame.copy()
        names = labels or list(b.target_names)
        df[target] = [str(names[i]) for i in df.pop("target")]
        return df
    return load


def _r(name: str) -> Callable[[], pd.DataFrame]:
    def load():
        from pydataset import data
        return data(name).reset_index(drop=True)
    return load


def _pima() -> pd.DataFrame:
    from pydataset import data
    return pd.concat([data("Pima.tr"), data("Pima.te")],
                     ignore_index=True)


def _noise() -> pd.DataFrame:
    """Negative control: 10 random columns and random labels. No method
    can do better than chance (50% balanced accuracy)."""
    rng = np.random.default_rng(2026)
    df = pd.DataFrame(rng.normal(size=(300, 10)).round(4),
                      columns=[f"x{i + 1}" for i in range(10)])
    df["label"] = rng.choice(["A", "B"], size=300)
    return df


def _sklearn():
    from sklearn import datasets as d
    return d


DATASETS: Dict[str, BenchmarkDataset] = {d.key: d for d in [
    BenchmarkDataset(
        "breast_cancer", "Breast cancer (Wisconsin diagnostic)", "Medicine",
        "diagnosis", lambda: _sk(_sklearn().load_breast_cancer,
                                 "diagnosis")(),
        "Street, W.N., Wolberg, W.H. and Mangasarian, O.L. (1993). Nuclear "
        "feature extraction for breast tumor diagnosis. IS&T/SPIE "
        "International Symposium on Electronic Imaging, 1905, 861-870."),
    BenchmarkDataset(
        "pima", "Pima Indians diabetes", "Medicine", "type", _pima,
        "Smith, J.W. et al. (1988). Using the ADAP learning algorithm to "
        "forecast the onset of diabetes mellitus. Proc. Symposium on "
        "Computer Applications in Medical Care, 261-265. Via R package MASS "
        "(Venables and Ripley, 2002).",
        "Pima.tr and Pima.te combined."),
    BenchmarkDataset(
        "birthwt", "Low birth weight", "Medicine (obstetrics)", "low",
        _r("birthwt"),
        "Hosmer, D.W. and Lemeshow, S. (1989). Applied Logistic Regression. "
        "Wiley. Via R package MASS (Venables and Ripley, 2002).",
        "Column 'bwt' (the birth weight itself) removed: it defines the "
        "class exactly (low = bwt < 2500 g), so keeping it would leak the "
        "answer.", drop=["bwt"]),
    BenchmarkDataset(
        "iris", "Iris flowers", "Botany", "species",
        lambda: _sk(_sklearn().load_iris, "species")(),
        "Fisher, R.A. (1936). The use of multiple measurements in taxonomic "
        "problems. Annals of Eugenics, 7(2), 179-188."),
    BenchmarkDataset(
        "wine", "Wine cultivars", "Chemistry", "cultivar",
        lambda: _sk(_sklearn().load_wine, "cultivar")(),
        "Aeberhard, S., Coomans, D. and de Vel, O. (1992). Comparison of "
        "classifiers in high dimensional settings. Tech. Rep. 92-02, James "
        "Cook University of North Queensland."),
    BenchmarkDataset(
        "digits", "Handwritten digits (8x8)", "Computer vision", "digit",
        lambda: _sk(_sklearn().load_digits, "digit",
                    labels=[str(i) for i in range(10)])(),
        "Alpaydin, E. and Kaynak, C. (1998). Cascading classifiers. "
        "Kybernetika, 34(4), 369-374."),
    BenchmarkDataset(
        "womenlf", "Women's labour-force participation (Canada)",
        "Sociology / economics", "partic", _r("Womenlf"),
        "Social Change in Canada Project, York Institute for Social "
        "Research. Via R package car (Fox and Weisberg, 2011)."),
    BenchmarkDataset(
        "chile", "Chile 1988 plebiscite voting intention",
        "Political science", "vote", _r("Chile"),
        "FLACSO/Chile survey. Via R package car (Fox and Weisberg, 2011).",
        "Four classes; contains missing values (rows without a vote are "
        "removed by EasyClassifier)."),
    BenchmarkDataset(
        "turnout", "US election turnout (1992 NES)", "Political science",
        "vote", _r("turnout"),
        "King, G., Tomz, M. and Wittenberg, J. (2000). Making the most of "
        "statistical analyses. American Journal of Political Science, 44, "
        "341-355. Via R package Zelig."),
    BenchmarkDataset(
        "cowles", "Volunteering for psychological research", "Psychology",
        "volunteer", _r("Cowles"),
        "Cowles, M. and Davis, C. (1987). The subject matter of psychology: "
        "Volunteers. British Journal of Social Psychology, 26, 97-102. Via "
        "R package car."),
    BenchmarkDataset(
        "noise", "Random labels (negative control)", "Control", "label",
        _noise,
        "Synthetic: 300 rows, 10 random columns, random labels (seed 2026).",
        "True performance is 50% balanced accuracy by construction."),
]}
