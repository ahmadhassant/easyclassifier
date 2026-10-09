"""References for every method a report can mention.

Each report cites, in its Methods paragraph, the software and every method
that was actually used: the classifiers (or other models), the KNN distance,
the evaluation design, the measures and the figures that need a source. The
bibliography and ``citations.txt`` are generated from the ``\\cite{...}``
commands in the finished report, so a reference is listed exactly when it is
cited, and nothing that is cited can be missing.

The EasyResearch cores (regression, forecasting, signal classification)
extend this list with the references of their own models.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from .distances import HASSANAT_ARXIV_CITATION, HASSANAT_CITATION

# --------------------------------------------------------------------------- #
# Reference texts
# --------------------------------------------------------------------------- #

REFERENCES: Dict[str, str] = {
    # Software
    "sklearn": "Pedregosa, F., et al. (2011). Scikit-learn: Machine "
               "Learning in Python. Journal of Machine Learning Research, "
               "12, 2825-2830.",
    "numpy": "Harris, C. R., Millman, K. J., van der Walt, S. J., et al. "
             "(2020). Array programming with NumPy. Nature, 585(7825), "
             "357-362. https://doi.org/10.1038/s41586-020-2649-2",
    "pandas": "McKinney, W. (2010). Data structures for statistical "
              "computing in Python. In Proceedings of the 9th Python in "
              "Science Conference (pp. 56-61). "
              "https://doi.org/10.25080/Majora-92bf1922-00a",
    "matplotlib": "Hunter, J. D. (2007). Matplotlib: A 2D graphics "
                  "environment. Computing in Science & Engineering, 9(3), "
                  "90-95. https://doi.org/10.1109/MCSE.2007.55",
    # Classifiers
    "breiman1984": "Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, "
                   "C. J. (1984). Classification and Regression Trees. "
                   "Wadsworth.",
    "breiman2001": "Breiman, L. (2001). Random Forests. Machine Learning, "
                   "45, 5-32. https://doi.org/10.1023/A:1010933404324",
    "cortes1995": "Cortes, C., & Vapnik, V. (1995). Support-vector "
                  "networks. Machine Learning, 20(3), 273-297. "
                  "https://doi.org/10.1007/BF00994018",
    "platt1999": "Platt, J. C. (1999). Probabilistic outputs for support "
                 "vector machines and comparisons to regularized likelihood "
                 "methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & "
                 "D. Schuurmans (Eds.), Advances in Large Margin Classifiers "
                 "(pp. 61-74). MIT Press.",
    "cox1958": "Cox, D. R. (1958). The regression analysis of binary "
               "sequences. Journal of the Royal Statistical Society: Series "
               "B, 20(2), 215-242. "
               "https://doi.org/10.1111/j.2517-6161.1958.tb00292.x",
    "cover1967": "Cover, T., & Hart, P. (1967). Nearest neighbor pattern "
                 "classification. IEEE Transactions on Information Theory, "
                 "13(1), 21-27. https://doi.org/10.1109/TIT.1967.1053964",
    "hand2001": "Hand, D. J., & Yu, K. (2001). Idiot's Bayes - not so "
                "stupid after all? International Statistical Review, 69(3), "
                "385-398. "
                "https://doi.org/10.1111/j.1751-5823.2001.tb00465.x",
    "chen2016": "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree "
                "boosting system. In Proceedings of the 22nd ACM SIGKDD "
                "International Conference on Knowledge Discovery and Data "
                "Mining (pp. 785-794). "
                "https://doi.org/10.1145/2939672.2939785",
    "ke2017": "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., "
              "Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient "
              "gradient boosting decision tree. In Advances in Neural "
              "Information Processing Systems 30 (pp. 3146-3154).",
    "rumelhart1986": "Rumelhart, D. E., Hinton, G. E., & Williams, R. J. "
                     "(1986). Learning representations by back-propagating "
                     "errors. Nature, 323(6088), 533-536. "
                     "https://doi.org/10.1038/323533a0",
    "kingma2015": "Kingma, D. P., & Ba, J. (2015). Adam: A method for "
                  "stochastic optimization. In International Conference on "
                  "Learning Representations (ICLR). arXiv:1412.6980.",
    # Distances
    "hassanat2022": HASSANAT_CITATION,
    "hassanat2014": HASSANAT_ARXIV_CITATION,
    "alaydaa2026": "Alaydaa, M. S., Alotibi, G. N., Tarawneh, A. S., & "
                   "Hassanat, A. B. (2026). A sign-symmetric reformulation "
                   "of the Hassanat distance for data with negative feature "
                   "values. Symmetry, 18(7), 1225. "
                   "https://doi.org/10.3390/sym18071225",
    "lance1966": "Lance, G. N., & Williams, W. T. (1966). Computer programs "
                 "for hierarchical polythetic classification (\"similarity "
                 "analyses\"). The Computer Journal, 9(1), 60-64. "
                 "https://doi.org/10.1093/comjnl/9.1.60",
    "deza2009": "Deza, M. M., & Deza, E. (2009). Encyclopedia of Distances. "
                "Springer. https://doi.org/10.1007/978-3-642-00234-2",
    # Evaluation
    "kohavi1995": "Kohavi, R. (1995). A study of cross-validation and "
                  "bootstrap for accuracy estimation and model selection. In "
                  "Proceedings of the 14th International Joint Conference on "
                  "Artificial Intelligence (IJCAI) (pp. 1137-1143).",
    "lachenbruch1968": "Lachenbruch, P. A., & Mickey, M. R. (1968). "
                       "Estimation of error rates in discriminant analysis. "
                       "Technometrics, 10(1), 1-11. "
                       "https://doi.org/10.1080/00401706.1968.10490530",
    "varma2006": "Varma, S., & Simon, R. (2006). Bias in error estimation "
                 "when using cross-validation for model selection. BMC "
                 "Bioinformatics, 7, 91. "
                 "https://doi.org/10.1186/1471-2105-7-91",
    "cawley2010": "Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting "
                  "in model selection and subsequent selection bias in "
                  "performance evaluation. Journal of Machine Learning "
                  "Research, 11, 2079-2107.",
    # Measures
    "brodersen2010": "Brodersen, K. H., Ong, C. S., Stephan, K. E., & "
                     "Buhmann, J. M. (2010). The balanced accuracy and its "
                     "posterior distribution. In 20th International "
                     "Conference on Pattern Recognition (pp. 3121-3124). "
                     "IEEE. https://doi.org/10.1109/ICPR.2010.764",
    "matthews1975": "Matthews, B. W. (1975). Comparison of the predicted and "
                    "observed secondary structure of T4 phage lysozyme. "
                    "Biochimica et Biophysica Acta - Protein Structure, "
                    "405(2), 442-451. "
                    "https://doi.org/10.1016/0005-2795(75)90109-9",
    "cohen1960": "Cohen, J. (1960). A coefficient of agreement for nominal "
                 "scales. Educational and Psychological Measurement, 20(1), "
                 "37-46. https://doi.org/10.1177/001316446002000104",
    "fawcett2006": "Fawcett, T. (2006). An introduction to ROC analysis. "
                   "Pattern Recognition Letters, 27(8), 861-874. "
                   "https://doi.org/10.1016/j.patrec.2005.10.010",
    "davis2006": "Davis, J., & Goadrich, M. (2006). The relationship between "
                 "precision-recall and ROC curves. In Proceedings of the "
                 "23rd International Conference on Machine Learning "
                 "(pp. 233-240). https://doi.org/10.1145/1143844.1143874",
}

# What each reference is cited for (shown in citations.txt).
TOPICS: Dict[str, str] = {
    "sklearn": "scikit-learn (models, preprocessing and evaluation)",
    "numpy": "NumPy (numerical computing)",
    "pandas": "pandas (data handling)",
    "matplotlib": "Matplotlib (figures)",
    "breiman1984": "Decision tree (CART)",
    "breiman2001": "Random Forest; permutation importance",
    "cortes1995": "Support vector machine",
    "platt1999": "SVM probability estimates (Platt scaling)",
    "cox1958": "Logistic regression",
    "cover1967": "K-nearest neighbours",
    "hand2001": "Naive Bayes",
    "chen2016": "XGBoost",
    "ke2017": "LightGBM; histogram-based gradient boosting",
    "rumelhart1986": "Neural network (multilayer perceptron, backpropagation)",
    "kingma2015": "Adam optimiser (neural network training)",
    "hassanat2022": "Hassanat distance",
    "hassanat2014": "Hassanat distance (original preprint)",
    "alaydaa2026": "Signed form of the Hassanat distance (negative values)",
    "lance1966": "Canberra distance",
    "deza2009": "Distance measures (Euclidean, Manhattan, Chebyshev, cosine)",
    "kohavi1995": "Stratified k-fold cross-validation",
    "lachenbruch1968": "Leave-one-out validation",
    "varma2006": "Nested cross-validation",
    "cawley2010": "Selection bias and separate final evaluation",
    "brodersen2010": "Balanced accuracy",
    "matthews1975": "Matthews correlation coefficient (MCC)",
    "cohen1960": "Cohen's kappa",
    "fawcett2006": "ROC curves and ROC AUC",
    "davis2006": "Precision-recall curves",
}

# --------------------------------------------------------------------------- #
# Which references go with which choice
# --------------------------------------------------------------------------- #

SOFTWARE = ["sklearn", "numpy", "pandas"]

CLASSIFIER_REFERENCES: Dict[str, List[str]] = {
    "decision_tree": ["breiman1984"],
    "random_forest": ["breiman2001"],
    "svm": ["cortes1995", "platt1999"],
    "logistic_regression": ["cox1958"],
    "knn": ["cover1967"],
    "naive_bayes": ["hand2001"],
    "xgboost": ["chen2016"],
    "lightgbm": ["ke2017"],
    "neural_network": ["rumelhart1986", "kingma2015"],
}

DISTANCE_REFERENCES: Dict[str, List[str]] = {
    "hassanat": ["hassanat2022", "hassanat2014"],
    "euclidean": ["deza2009"],
    "manhattan": ["deza2009"],
    "chebyshev": ["deza2009"],
    "canberra": ["lance1966"],
    "cosine": ["deza2009"],
}

# Cited in addition when the signed form of the Hassanat formula was applied
# (some values negative).
HASSANAT_SIGNED_REFERENCES = ["alaydaa2026"]

VALIDATION_REFERENCES: Dict[str, List[str]] = {
    "holdout": [],
    "kfold5": ["kohavi1995"],
    "kfold10": ["kohavi1995"],
    "stratified": ["kohavi1995"],
    "loo": ["lachenbruch1968"],
}

SELECTION_REFERENCES: Dict[str, List[str]] = {
    "nested": ["varma2006", "cawley2010"],
    "final_test": ["cawley2010"],
}

# Measures that need a source (accuracy, precision, recall, F1, specificity
# and log loss are standard definitions).
METRIC_REFERENCES: Dict[str, List[str]] = {
    "balanced_accuracy": ["brodersen2010"],
    "mcc": ["matthews1975"],
    "cohen_kappa": ["cohen1960"],
    "roc_auc": ["fawcett2006"],
}

FIGURE_REFERENCES: Dict[str, List[str]] = {
    "roc_curve": ["fawcett2006"],
    "pr_curve": ["davis2006"],
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_CITE = re.compile(r"\\cite\{([^}]*)\}")


def cite(keys: Iterable[str]) -> str:
    """`` \\cite{a,b}`` for the given keys (in order, without repeats), or an
    empty string when there are none."""
    keys = list(dict.fromkeys(k for k in keys if k))
    return f" \\cite{{{','.join(keys)}}}" if keys else ""


def cited_keys(text: str) -> List[str]:
    """Every key cited in a LaTeX text, in order of first citation."""
    keys: List[str] = []
    for group in _CITE.findall(text):
        for k in group.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def lookup(key: str, extra: Optional[Dict[str, str]] = None) -> str:
    if extra and key in extra:
        return extra[key]
    if key in REFERENCES:
        return REFERENCES[key]
    raise KeyError(f"no reference text for \\cite{{{key}}}")


def bibliography(text: str, extra: Optional[Dict[str, str]] = None,
                 escape=lambda s: s) -> str:
    """The ``thebibliography`` block for every key cited in ``text``, in
    order of first citation. ``extra`` holds report-specific entries (the
    software itself, the dataset). A cited key without a text raises an
    error, so a report can never cite a reference it does not list."""
    keys = cited_keys(text)
    lines = ["\\begin{thebibliography}{99}\n"]
    lines += [f"\\bibitem{{{k}}} {escape(lookup(k, extra))}\n" for k in keys]
    lines.append("\\end{thebibliography}\n")
    return "".join(lines)


def citation_lines(text: str, extra: Optional[Dict[str, str]] = None,
                   topics: Optional[Dict[str, str]] = None) -> List[str]:
    """``citations.txt``: every reference cited in the report, with what it
    is cited for."""
    topics = dict(TOPICS, **(topics or {}))
    lines = ["Please cite the following if you publish these results.",
             "They are the references of the report (report.tex), in the "
             "order they are cited there.", ""]
    for i, k in enumerate(cited_keys(text), 1):
        what = topics.get(k)
        lines.append(f"[{i}] " + (f"{what}:" if what else ""))
        lines.append("    " + lookup(k, extra))
    return lines
