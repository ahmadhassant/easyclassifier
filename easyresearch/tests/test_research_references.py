"""Every model of the three cores has its references, and every report lists
exactly the references it cites (citations.txt included)."""

import re

from easyresearch.common import references as refs
from easyresearch.regression import Settings, run_analysis
from easyresearch.regression.data import DEMO_CITATION, load_demo
from easyresearch.regression.models import build_registry as regressors
from easyresearch.timeseries.models import build_registry as forecasters
from easyresearch.timeseries.signal_models import build_registry as signals


def test_every_model_has_its_references():
    for registry, table in ((regressors(), refs.REGRESSION_MODEL_REFERENCES),
                            (forecasters(), refs.FORECAST_MODEL_REFERENCES),
                            (signals(), refs.SIGNAL_MODEL_REFERENCES)):
        assert set(registry) == set(table)
        for key, cites in table.items():
            assert cites or key == "majority", key
            assert set(cites) <= set(refs.REFERENCES), key
    assert set(refs.REFERENCES) <= set(refs.TOPICS)


def test_regression_report_cites_every_method_it_used(tmp_path):
    models = ["linear_regression", "ridge", "knn", "neural_network"]
    r = run_analysis(load_demo().head(150), "Progression",
                     Settings(models=models), str(tmp_path), "demo", "demo",
                     DEMO_CITATION, progress=lambda m: None)
    doc = open(f"{r['out']}/report.tex", encoding="utf-8").read()
    body, bib = doc.split(r"\begin{thebibliography}")
    cited = refs.cited_keys(body)
    for key in refs.model_cites(models, refs.REGRESSION_MODEL_REFERENCES) + [
            "sklearn", "numpy", "pandas", "dataset", "mhsp", "kohavi1995",
            "varma2006", "matplotlib"]:
        assert key in cited, key
    assert re.findall(r"\\bibitem\{([^}]*)\}", bib) == cited
    lines = open(f"{r['out']}/citations.txt", encoding="utf-8").read()
    assert len(re.findall(r"^\[\d+\]", lines, re.M)) == len(cited)
    assert "Hoerl" in lines and "Kingma" in lines


def test_forecast_methods_describe_only_the_models_used(tmp_path):
    from easyresearch.timeseries import ForecastSettings, run_forecast
    from easyresearch.timeseries import data as td
    r = run_forecast(td.load_demo(), "CO2_ppm",
                     ForecastSettings(models=["theta"], horizon=6,
                                      figures=False),
                     str(tmp_path), "demo", "demo", progress=lambda m: None)
    doc = open(f"{r['out']}/report.tex", encoding="utf-8").read()
    assert "Theta method on the seasonally adjusted series" in doc
    assert "damped additive trend" not in doc          # ETS was not used
    assert "assimakopoulos2000" in refs.cited_keys(doc)
    assert "hyndman2002" not in refs.cited_keys(doc)
