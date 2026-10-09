"""References for every method the EasyResearch cores can use.

Extends EasyClassifier's reference list (classifiers, distances, evaluation,
measures, software) with the models and methods of regression, forecasting
and signal classification. Each report cites the methods it used; the
bibliography and ``citations.txt`` are generated from those citations (see
``easyclassifier.references``), so a reference is listed exactly when it is
cited.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from easyclassifier.references import (
    CLASSIFIER_REFERENCES,
    FIGURE_REFERENCES,
    HASSANAT_SIGNED_REFERENCES,
    METRIC_REFERENCES,
    REFERENCES as _CLASSIFICATION_REFERENCES,
    SELECTION_REFERENCES,
    TOPICS as _CLASSIFICATION_TOPICS,
    VALIDATION_REFERENCES,
    cite,
    cited_keys,
)
from easyclassifier import references as _base

from .hassanat import MHSP_CITATION

__all__ = ["REFERENCES", "TOPICS", "CLASSIFIER_REFERENCES", "FIGURE_REFERENCES",
           "HASSANAT_SIGNED_REFERENCES",
           "METRIC_REFERENCES", "SELECTION_REFERENCES",
           "VALIDATION_REFERENCES", "REGRESSION_MODEL_REFERENCES",
           "FORECAST_MODEL_REFERENCES", "SIGNAL_MODEL_REFERENCES",
           "bibliography", "citation_lines", "cite", "cited_keys",
           "model_cites"]

REFERENCES: Dict[str, str] = dict(_CLASSIFICATION_REFERENCES)
REFERENCES.update({
    "mhsp": MHSP_CITATION,
    # Software
    "statsmodels": "Seabold, S., & Perktold, J. (2010). Statsmodels: "
                   "Econometric and statistical modeling with Python. "
                   "Proceedings of the 9th Python in Science Conference, "
                   "92-96.",
    "pytorch": "Paszke, A., et al. (2019). PyTorch: An imperative style, "
               "high-performance deep learning library. NeurIPS 2019, "
               "8024-8035.",
    # Regression models
    "hastie2009": "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The "
                  "Elements of Statistical Learning: Data Mining, Inference, "
                  "and Prediction (2nd ed.). Springer. "
                  "https://doi.org/10.1007/978-0-387-84858-7",
    "hoerl1970": "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: "
                 "Biased estimation for nonorthogonal problems. "
                 "Technometrics, 12(1), 55-67. "
                 "https://doi.org/10.1080/00401706.1970.10488634",
    "friedman2001": "Friedman, J. H. (2001). Greedy function approximation: "
                    "A gradient boosting machine. The Annals of Statistics, "
                    "29(5), 1189-1232. "
                    "https://doi.org/10.1214/aos/1013203451",
    "drucker1997": "Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & "
                   "Vapnik, V. (1997). Support vector regression machines. "
                   "In Advances in Neural Information Processing Systems 9 "
                   "(pp. 155-161). MIT Press.",
    "altman1992": "Altman, N. S. (1992). An introduction to kernel and "
                  "nearest-neighbor nonparametric regression. The American "
                  "Statistician, 46(3), 175-185. "
                  "https://doi.org/10.1080/00031305.1992.10475879",
    # Forecasting
    "hyndman2021": "Hyndman, R. J., & Athanasopoulos, G. (2021). "
                   "Forecasting: Principles and Practice (3rd ed.). OTexts. "
                   "https://otexts.com/fpp3/",
    "hyndman2002": "Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, "
                   "S. (2002). A state space framework for automatic "
                   "forecasting using exponential smoothing methods. "
                   "International Journal of Forecasting, 18(3), 439-454.",
    "assimakopoulos2000": "Assimakopoulos, V., & Nikolopoulos, K. (2000). "
                          "The theta model: a decomposition approach to "
                          "forecasting. International Journal of "
                          "Forecasting, 16(4), 521-530.",
    "bontempi2013": "Bontempi, G., Ben Taieb, S., & Le Borgne, Y.-A. (2013). "
                    "Machine learning strategies for time series "
                    "forecasting. In M.-A. Aufaure & E. Zimányi (Eds.), "
                    "Business Intelligence (eBISS 2012), Lecture Notes in "
                    "Business Information Processing 138 (pp. 62-77). "
                    "Springer. https://doi.org/10.1007/978-3-642-36318-4_3",
    "tashman2000": "Tashman, L. J. (2000). Out-of-sample tests of "
                   "forecasting accuracy: an analysis and review. "
                   "International Journal of Forecasting, 16(4), 437-450.",
    "hyndman2006": "Hyndman, R. J., & Koehler, A. B. (2006). Another look at "
                   "measures of forecast accuracy. International Journal of "
                   "Forecasting, 22(4), 679-688.",
    "makridakis1993": "Makridakis, S. (1993). Accuracy measures: theoretical "
                      "and practical concerns. International Journal of "
                      "Forecasting, 9(4), 527-529. "
                      "https://doi.org/10.1016/0169-2070(93)90079-3",
    "cleveland1990": "Cleveland, R. B., Cleveland, W. S., McRae, J. E., & "
                     "Terpenning, I. (1990). STL: A seasonal-trend "
                     "decomposition procedure based on loess. Journal of "
                     "Official Statistics, 6(1), 3-73.",
    # Deep networks
    "hochreiter1997": "Hochreiter, S., & Schmidhuber, J. (1997). Long "
                      "short-term memory. Neural Computation, 9(8), "
                      "1735-1780.",
    "cho2014": "Cho, K., et al. (2014). Learning phrase representations "
               "using RNN encoder-decoder for statistical machine "
               "translation. EMNLP 2014, 1724-1734.",
    "bai2018": "Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical "
               "evaluation of generic convolutional and recurrent networks "
               "for sequence modeling. arXiv:1803.01271.",
    "oreshkin2020": "Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. "
                    "(2020). N-BEATS: Neural basis expansion analysis for "
                    "interpretable time series forecasting. ICLR 2020.",
    "vaswani2017": "Vaswani, A., et al. (2017). Attention is all you need. "
                   "NeurIPS 2017, 5998-6008.",
    # Signal classification
    "dempster2020": "Dempster, A., Petitjean, F., & Webb, G. I. (2020). "
                    "ROCKET: exceptionally fast and accurate time series "
                    "classification using random convolutional kernels. "
                    "Data Mining and Knowledge Discovery, 34(5), 1454-1495.",
    "fulcher2014": "Fulcher, B. D., & Jones, N. S. (2014). Highly "
                   "comparative feature-based time-series classification. "
                   "IEEE Transactions on Knowledge and Data Engineering, "
                   "26(12), 3026-3037. "
                   "https://doi.org/10.1109/TKDE.2014.2316504",
    "wang2017": "Wang, Z., Yan, W., & Oates, T. (2017). Time series "
                "classification from scratch with deep neural networks: a "
                "strong baseline. IJCNN 2017, 1578-1585.",
    "he2016": "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual "
              "learning for image recognition. In Proceedings of the IEEE "
              "Conference on Computer Vision and Pattern Recognition (CVPR) "
              "(pp. 770-778). https://doi.org/10.1109/CVPR.2016.90",
    "ismailfawaz2020": "Ismail Fawaz, H., et al. (2020). InceptionTime: "
                       "finding AlexNet for time series classification. "
                       "Data Mining and Knowledge Discovery, 34(6), "
                       "1936-1962.",
})

TOPICS: Dict[str, str] = dict(_CLASSIFICATION_TOPICS)
TOPICS.update({
    "software": "EasyResearch (this software)",
    "easyclassifier": "EasyClassifier (data preparation and evaluation)",
    "dataset": "The example dataset",
    "mhsp": "Mean Hassanat similarity percentage (MHSP)",
    "statsmodels": "statsmodels (statistical forecasting models)",
    "pytorch": "PyTorch (deep networks)",
    "hastie2009": "Linear regression (least squares)",
    "hoerl1970": "Ridge regression",
    "friedman2001": "Gradient boosting",
    "drucker1997": "Support vector regression",
    "altman1992": "K-nearest-neighbour regression",
    "hyndman2021": "Naive and seasonal naive forecasts; prediction intervals",
    "hyndman2002": "Exponential smoothing (ETS)",
    "assimakopoulos2000": "Theta method",
    "bontempi2013": "Machine-learning forecasting from past values "
                    "(direct multi-step strategy)",
    "tashman2000": "Rolling-origin evaluation",
    "hyndman2006": "MASE",
    "makridakis1993": "sMAPE",
    "cleveland1990": "STL decomposition",
    "hochreiter1997": "LSTM",
    "cho2014": "GRU",
    "bai2018": "Temporal convolutional network (TCN)",
    "oreshkin2020": "N-BEATS",
    "vaswani2017": "Transformer",
    "dempster2020": "ROCKET",
    "fulcher2014": "Feature-based time-series classification",
    "wang2017": "FCN and ResNet for time-series classification",
    "he2016": "Residual networks (ResNet)",
    "ismailfawaz2020": "InceptionTime",
})

HASSANAT = ["hassanat2022", "hassanat2014"]

REGRESSION_MODEL_REFERENCES: Dict[str, List[str]] = {
    "linear_regression": ["hastie2009"],
    "ridge": ["hoerl1970"],
    "decision_tree": ["breiman1984"],
    "random_forest": ["breiman2001"],
    # scikit-learn's histogram-based gradient boosting follows LightGBM.
    "gradient_boosting": ["friedman2001", "ke2017"],
    "svr": ["drucker1997"],
    "knn": ["altman1992"] + HASSANAT,
    "neural_network": ["rumelhart1986", "kingma2015"],
    "xgboost": ["chen2016"],
    "lightgbm": ["ke2017"],
}

FORECAST_MODEL_REFERENCES: Dict[str, List[str]] = {
    "naive": ["hyndman2021"],
    "seasonal_naive": ["hyndman2021"],
    "ets": ["hyndman2002"],
    "theta": ["assimakopoulos2000"],
    "ridge": ["hoerl1970"],
    "random_forest": ["breiman2001"],
    "gradient_boosting": ["friedman2001", "ke2017"],
    "knn": ["altman1992"] + HASSANAT,
    "mlp": ["rumelhart1986"],
    "lstm": ["hochreiter1997"],
    "gru": ["cho2014"],
    "tcn": ["bai2018"],
    "nbeats": ["oreshkin2020"],
    "transformer": ["vaswani2017"],
}

SIGNAL_MODEL_REFERENCES: Dict[str, List[str]] = {
    "majority": [],
    "knn": ["cover1967"] + HASSANAT,
    "rocket": ["dempster2020"],
    "features_rf": ["fulcher2014", "breiman2001"],
    "fcn": ["wang2017"],
    "resnet": ["wang2017", "he2016"],
    "inception": ["ismailfawaz2020"],
    "lstm": ["hochreiter1997"],
}


def model_cites(keys, table: Dict[str, List[str]]) -> List[str]:
    """All references of the given models, in order, without repeats."""
    return list(dict.fromkeys(r for k in keys for r in table.get(k, [])))


def bibliography(text: str, extra: Optional[Dict[str, str]] = None,
                 escape=lambda s: s) -> str:
    """As in EasyClassifier, with the references of all EasyResearch cores
    (``extra`` holds the report's own entries: software, dataset)."""
    return _base.bibliography(text, dict(REFERENCES, **(extra or {})), escape)


def citation_lines(text: str, extra: Optional[Dict[str, str]] = None,
                   topics: Optional[Dict[str, str]] = None) -> List[str]:
    return _base.citation_lines(text, dict(REFERENCES, **(extra or {})),
                                dict(TOPICS, **(topics or {})))
