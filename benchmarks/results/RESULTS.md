# EasyClassifier benchmark results

## Datasets

| Dataset                                     | Field                 |   Rows |   Columns |   Classes |   Smallest class |   Missing cells |
|:--------------------------------------------|:----------------------|-------:|----------:|----------:|-----------------:|----------------:|
| Breast cancer (Wisconsin diagnostic)        | Medicine              |    569 |        30 |         2 |              212 |               0 |
| Pima Indians diabetes                       | Medicine              |    532 |         7 |         2 |              177 |               0 |
| Low birth weight                            | Medicine (obstetrics) |    183 |         8 |         2 |               59 |               0 |
| Iris flowers                                | Botany                |    149 |         4 |         3 |               49 |               0 |
| Wine cultivars                              | Chemistry             |    178 |        13 |         3 |               48 |               0 |
| Handwritten digits (8x8)                    | Computer vision       |   1797 |        61 |        10 |              174 |               0 |
| Women's labour-force participation (Canada) | Sociology / economics |    263 |         3 |         3 |               42 |               0 |
| Chile 1988 plebiscite voting intention      | Political science     |   2523 |         7 |         4 |              187 |             107 |
| US election turnout (1992 NES)              | Political science     |   1965 |         4 |         2 |              504 |               0 |
| Volunteering for psychological research     | Psychology            |   1421 |         3 |         2 |              597 |               0 |
| Random labels (negative control)            | Control               |    300 |        10 |         2 |              145 |               0 |

## Benchmark 1: are the reported scores honest?

33 runs (datasets x 3 random 50/50 splits). Balanced accuracy in %. Bias = estimate minus performance on the untouched external half (0 = honest, positive = optimistic).

| Dataset                                     | Final-score method   |   Naive estimate |   Comparison best |   EasyClassifier final |   External (truth) |   Bias naive |   Bias comparison |   Bias final |
|:--------------------------------------------|:---------------------|-----------------:|------------------:|-----------------------:|-------------------:|-------------:|------------------:|-------------:|
| Breast cancer (Wisconsin diagnostic)        | nested CV            |             97.4 |              98.1 |                   97.7 |               96.1 |          0.9 |               2   |          1.6 |
| Pima Indians diabetes                       | nested CV            |             73.8 |              73.8 |                   71.8 |               71.9 |          1.6 |               1.9 |         -0.1 |
| Low birth weight                            | nested CV            |             63.8 |              63.8 |                   57.5 |               55   |          9.9 |               8.8 |          2.5 |
| Iris flowers                                | nested CV            |             97.3 |              96.9 |                   95.5 |               92   |          5.3 |               4.9 |          3.5 |
| Wine cultivars                              | nested CV            |             98   |              98.4 |                   96.8 |               97.9 |          0.5 |               0.5 |         -1   |
| Handwritten digits (8x8)                    | nested CV            |             97.5 |              97.3 |                   96.5 |               97.3 |          0.2 |               0.1 |         -0.8 |
| Women's labour-force participation (Canada) | nested CV            |             53.1 |              51.4 |                   48.8 |               49.4 |          5.5 |               1.9 |         -0.7 |
| Chile 1988 plebiscite voting intention      | nested CV            |             51.2 |              51.3 |                   51.2 |               50.9 |          0.1 |               0.4 |          0.3 |
| US election turnout (1992 NES)              | nested CV            |             58.7 |              58.8 |                   57.8 |               58   |          1.6 |               0.8 |         -0.1 |
| Volunteering for psychological research     | nested CV            |             54.5 |              54.3 |                   52.9 |               50.9 |          3.1 |               3.4 |          2   |
| Random labels (negative control)            | nested CV            |             52.9 |              55.6 |                   48.8 |               47.9 |          9.9 |               7.8 |          0.9 |

Across the real datasets (noise control excluded): mean bias naive 2.9, comparison best 2.5, EasyClassifier final 0.7 percentage points; mean absolute bias 3.0 / 2.9 / 2.3. One-sided Wilcoxon signed-rank test that the naive estimate is more optimistic than the final score: p = 0.000318 (n = 30 runs).

## Benchmark 2: KNN distances

Balanced accuracy (%) of KNN, k = 5, repeated stratified 5-fold CV (2 repeats), with EasyClassifier's automatic preprocessing: Hassanat on data scaled to 0-1, the other distances on standardised data. References: Euclidean and Hassanat on unscaled data, Hassanat on standardised data.

|                                             |   Hassanat |   Euclidean |   Manhattan |   Chebyshev |   Canberra |   Cosine |   Euclidean (unscaled) |   Hassanat (unscaled) |   Hassanat (standardised) |
|:--------------------------------------------|-----------:|------------:|------------:|------------:|-----------:|---------:|-----------------------:|----------------------:|--------------------------:|
| Low birth weight                            |       57.8 |        59   |        59.3 |        56.2 |       57.9 |     59.2 |                   53.1 |                  56.7 |                      56.5 |
| Breast cancer (Wisconsin diagnostic)        |       95.8 |        95.7 |        95.7 |        93.6 |       95.4 |     95.4 |                   92   |                  94.9 |                      95.7 |
| Chile 1988 plebiscite voting intention      |       47.4 |        48.1 |        47.9 |        48.7 |       45   |     47.7 |                   36.3 |                  45   |                      47.6 |
| Volunteering for psychological research     |       52.6 |        53.3 |        53.2 |        53.3 |       54   |     54.3 |                   52.8 |                  52.5 |                      53   |
| Handwritten digits (8x8)                    |       98.1 |        97.5 |        97.6 |        92.9 |       95.8 |     96.5 |                   98.5 |                  97.1 |                      97   |
| Iris flowers                                |       94.6 |        95   |        94.6 |        93.3 |       93.6 |     85.5 |                   96.6 |                  94.9 |                      95   |
| Pima Indians diabetes                       |       71.3 |        70.8 |        70.3 |        68.5 |       71   |     73.5 |                   71.5 |                  69.1 |                      70.5 |
| US election turnout (1992 NES)              |       57.7 |        57.7 |        58.2 |        56.8 |       59   |     58.5 |                   58.2 |                  56   |                      58.1 |
| Wine cultivars                              |       96   |        97.2 |        97.4 |        94.1 |       96.7 |     97   |                   67.4 |                  95.8 |                      96.2 |
| Women's labour-force participation (Canada) |       49.2 |        51.9 |        51   |        50.8 |       50.1 |     51.7 |                   48.1 |                  49.1 |                      50.4 |

Mean rank (1 = best) among the six distances offered: Euclidean 2.70, Manhattan 2.75, Cosine 2.90, Hassanat 3.85, Canberra 3.90, Chebyshev 4.90.

Friedman test across datasets: p = 0.0533.

Hassanat vs each other distance (wins / ties / losses over datasets, mean difference in percentage points, Wilcoxon signed-rank p):

* vs Euclidean: 3/0/7, -0.57, p = 0.105
* vs Manhattan: 3/1/6, -0.48, p = 0.164
* vs Chebyshev: 7/0/3, +1.22, p = 0.084
* vs Canberra: 5/0/5, +0.19, p = 0.846
* vs Cosine: 3/0/7, +0.10, p = 0.375
* vs Euclidean (unscaled): 5/0/5, +4.58, p = 0.275
* vs Hassanat (unscaled): 9/0/1, +0.94, p = 0.0137
* vs Hassanat (standardised): 4/0/6, +0.04, p = 0.922

