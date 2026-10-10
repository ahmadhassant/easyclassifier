"""Every method the tool can use is cited in the report, and the
bibliography lists exactly the cited references."""

import re

from easyclassifier import references as refs
from easyclassifier.distances import DISTANCES
from easyclassifier.evaluation import METRICS, SELECTION, VALIDATION
from easyclassifier.latex_report import build_tex
from easyclassifier.models import build_registry

# Measures with a standard definition that need no source.
STANDARD_MEASURES = {"accuracy", "precision", "recall", "f1", "specificity",
                     "log_loss"}


def test_every_choice_has_its_references():
    assert set(build_registry()) <= set(refs.CLASSIFIER_REFERENCES)
    assert all(refs.CLASSIFIER_REFERENCES[k] for k in build_registry())
    assert set(DISTANCES) == set(refs.DISTANCE_REFERENCES)
    assert set(VALIDATION) == set(refs.VALIDATION_REFERENCES)
    assert set(SELECTION) - {"auto"} == set(refs.SELECTION_REFERENCES)
    assert set(METRICS) == set(refs.METRIC_REFERENCES) | STANDARD_MEASURES
    tables = [refs.CLASSIFIER_REFERENCES, refs.DISTANCE_REFERENCES,
              refs.VALIDATION_REFERENCES, refs.SELECTION_REFERENCES,
              refs.METRIC_REFERENCES, refs.FIGURE_REFERENCES]
    keys = {k for t in tables for v in t.values() for k in v} | set(refs.SOFTWARE)
    assert keys <= set(refs.REFERENCES)
    assert set(refs.REFERENCES) <= set(refs.TOPICS)
    assert not any("Journal of American Science" in r
                   for r in refs.REFERENCES.values())


def test_bibliography_lists_exactly_the_cited_references():
    text = r"A \cite{sklearn,breiman2001} B \cite{own} C \cite{breiman2001}."
    bib = refs.bibliography(text, {"own": "Own, A. (2026). Data."})
    assert re.findall(r"\\bibitem\{([^}]*)\}", bib) == \
        ["sklearn", "breiman2001", "own"]
    try:
        refs.bibliography(r"\cite{nowhere}")
    except KeyError as exc:
        assert "nowhere" in str(exc)
    else:
        raise AssertionError("an unknown citation must not pass silently")
    lines = refs.citation_lines(text, {"own": "Own, A. (2026). Data."})
    assert sum(line.startswith("[") for line in lines) == 3


def test_report_cites_each_classifier_measure_and_evaluation():
    from test_report import _context
    c = _context()
    c.classifiers = ["Decision Tree", "XGBoost", "LightGBM",
                     "KNN (Canberra, k=5)"]
    c.classifier_keys = ["decision_tree", "xgboost", "lightgbm", "knn"]
    c.knn_text, c.knn_distance = "K-nearest neighbours used k = 5", "canberra"
    c.metrics = ["accuracy", "balanced_accuracy", "mcc", "cohen_kappa"]
    c.selection = "nested"
    c.final = c.results[0]
    doc = build_tex(c)
    cited = refs.cited_keys(doc.split(r"\begin{thebibliography}")[0])
    for key in ("breiman1984", "chen2016", "ke2017", "cover1967", "lance1966",
                "kohavi1995", "varma2006", "brodersen2010", "matthews1975",
                "cohen1960", "sklearn", "numpy", "pandas"):
        assert key in cited, key
    assert re.findall(r"\\bibitem\{([^}]*)\}", doc) == cited


def test_signed_hassanat_form_cites_the_sign_symmetric_paper():
    from test_report import _context
    for signed in (False, True):
        c = _context()
        c.classifiers, c.classifier_keys = ["KNN (Hassanat, k=5)"], ["knn"]
        c.knn_text, c.knn_distance = "K-nearest neighbours used k = 5", "hassanat"
        c.hassanat_used, c.hassanat_signed = True, signed
        cited = refs.cited_keys(build_tex(c))
        assert {"hassanat2022", "hassanat2014"} <= set(cited)
        assert ("alaydaa2026" in cited) == signed
