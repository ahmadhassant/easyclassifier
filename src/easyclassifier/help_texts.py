"""Plain-language explanations for metrics, methods, and algorithms.

Every concept the wizard exposes has a short, jargon-free explanation here so
users can ask "what is this?" at any decision point.
"""

GLOSSARY = {
    # Metrics ------------------------------------------------------------- #
    "accuracy": "Accuracy is the share of predictions the model got right, "
                "out of all predictions.",
    "precision": "Precision measures how many of the items the model labeled "
                 "positive are actually positive.",
    "recall": "Recall (sensitivity) measures how many of the real positive "
              "items the model successfully found.",
    "f1": "F1 is a single score that balances precision and recall. It is "
          "high only when both are high.",
    "specificity": "Macro specificity: for each class, the share of items "
                   "from the other classes that the model correctly did not "
                   "put in that class, averaged over all classes. With two "
                   "classes it equals balanced accuracy.",
    "sensitivity": "Sensitivity is another name for recall: how many real "
                   "positives the model found.",
    "mcc": "Matthews Correlation Coefficient summarises the whole confusion "
           "matrix in one number from -1 (worst) to +1 (perfect). It stays "
           "reliable even when classes are imbalanced.",
    "cohen_kappa": "Cohen's Kappa measures agreement between predictions and "
                   "truth, corrected for the agreement you'd expect by chance.",
    "balanced_accuracy": "Balanced accuracy averages the recall of each "
                         "class, so every class counts equally regardless of "
                         "size.",
    "roc_auc": "ROC AUC measures how well the model separates classes across "
               "all thresholds. 1.0 is perfect, 0.5 is random guessing.",
    "pr_auc": "PR AUC summarises the precision-recall curve. It is especially "
              "informative when the positive class is rare.",
    "log_loss": "Log loss penalises confident wrong answers. Lower is better.",

    # Preprocessing ------------------------------------------------------- #
    "missing_values": "Missing values are empty cells. Models cannot train on "
                      "blanks, so we either remove those rows or fill the gaps "
                      "with a sensible value (mean, median, or most common).",
    "duplicates": "Duplicate rows are identical records. They can bias a model "
                  "toward repeated examples, so they are often removed.",
    "label_encoding": "Label encoding turns text categories into whole numbers "
                      "(e.g. red=0, green=1, blue=2). Best for categories with "
                      "a natural order, or for tree-based models.",
    "one_hot_encoding": "One-hot encoding creates a separate yes/no column for "
                        "each category. Best when categories have no natural "
                        "order.",
    "scaling": "Scaling (normalisation) puts all numeric columns on a similar "
               "range. Distance-based models like KNN and SVM need it; "
               "tree-based models do not.",
    "feature_selection": "Feature selection keeps only the most useful columns. "
                         "It can speed up training and reduce noise.",

    # Validation ---------------------------------------------------------- #
    "hold_out": "Hold-out splits the data once into a training part and a "
                "testing part. Fast, but the score depends on the split.",
    "cross_validation": "Cross-validation splits the data into several folds, "
                        "trains and tests multiple times, and averages the "
                        "scores for a more reliable estimate.",
    "stratified": "Stratified splitting keeps the class proportions the same in "
                  "every fold. Recommended when classes are imbalanced.",
    "leave_one_out": "Leave-One-Out tests on one row at a time and trains on "
                     "the rest. Very thorough but slow on large datasets.",

    "selection": "Automatic uses nested cross-validation for datasets of up "
                 "to 2,000 rows - in EasyClassifier's benchmarks it gave the "
                 "most accurate final scores - and a final test set for "
                 "larger datasets, where it is accurate and much faster.",
    "final_test": "20% of the rows are put aside before anything else and "
                  "never looked at while comparing classifiers. The winner is "
                  "then scored once on those rows, like a final exam it has "
                  "never seen. This is the honest number to report.",
    "nested_cv": "Nested cross-validation repeats the whole comparison "
                 "inside each of 5 folds, using only that fold's training "
                 "rows, and scores the winner on the fold's test rows. It "
                 "uses every row for testing once, so it suits small "
                 "datasets. Slower, but honest.",

    # Classifiers --------------------------------------------------------- #
    "decision_tree": "A Decision Tree asks a series of yes/no questions about "
                     "the data to reach a decision. Easy to interpret.",
    "random_forest": "A Random Forest combines many decision trees and lets "
                     "them vote. Accurate and robust with little tuning.",
    "svm": "A Support Vector Machine finds the boundary that best separates "
           "the classes. Works well on clean, scaled data.",
    "logistic_regression": "Logistic Regression estimates the probability of "
                           "each class using a weighted sum of the features. "
                           "Simple and fast.",
    "knn": "K-Nearest Neighbours classifies a row by looking at the most "
           "similar rows around it. You will be asked which distance to use; "
           "EasyClassifier scales the columns automatically for every distance.",
    "naive_bayes": "Naive Bayes uses probability and assumes features are "
                   "independent. Very fast, good for a baseline.",
    "xgboost": "XGBoost builds trees one after another, each fixing the errors "
               "of the last. Often a top performer on tabular data.",
    "lightgbm": "LightGBM is a fast gradient-boosting method similar to "
                "XGBoost, designed to be quick on large datasets.",
    "neural_network": "A Neural Network learns patterns through layers of "
                      "connected 'neurons'. Flexible but needs more data.",

    # Tuning -------------------------------------------------------------- #
    "grid_search": "Grid Search tries every combination of the settings you "
                   "give it and keeps the best. Thorough but slow.",
    "random_search": "Random Search tries random combinations of settings. "
                     "Often finds a good result much faster than grid search.",
}


def explain(key: str) -> str:
    """Return the plain-language explanation for a concept key."""
    return GLOSSARY.get(
        key.lower().replace(" ", "_"),
        "No explanation is available for this item yet.",
    )
