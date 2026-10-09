# EasyResearch benchmark results

Produced by `easyresearch_benchmarks.py`. Every analysis ran as a beginner would run it: the fast default set of models and automatic settings. Scores are the honest final scores reported to the user. Times are for the computer below and include every model in the comparison and the final evaluation.

Computer: Python 3.13.16, OS Linux-6.18.44-fc-v80-x86_64-with-glibc2.39, CPUs 2, easyresearch 0.1.0, torch 2.14.1

## Regression

| Dataset | Rows | Selected model | Final score | R² | R² of average | MHSP % | Linear R² (EasyResearch) | Linear R² (scikit-learn) | Seconds | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| Diabetes progression | 442 | Linear Regression | nested | 0.492 | -0.007 | 74.432 | 0.492 | 0.492 | 14.816 | 262.688 |
| Car fuel economy (mtcars) | 32 | Random Forest | nested | 0.840 | -0.022 | 91.807 | 0.691 | 0.691 | 12.831 | 255.871 |
| Occupational prestige (Canada) | 102 | Random Forest | nested | 0.795 | -0.014 | 87.087 | 0.790 | 0.790 | 20.788 | 277.965 |
| New York air quality (ozone) | 116 | Random Forest | nested | 0.708 | -0.003 | 73.125 | 0.596 | 0.596 | 13.569 | 270.754 |
| Friedman #1 (synthetic, noise sd 1) | 500 | Gradient Boosting | nested | 0.852 | -0.008 | 90.719 | 0.640 | 0.640 | 18.291 | 274.207 |

Largest difference from the reference implementation: 0.0000.

Sources: Diabetes progression: Efron et al. (2004), Annals of Statistics 32(2); via scikit-learn; Car fuel economy (mtcars): Henderson & Velleman (1981), Biometrics 37(2); via pydataset (R datasets); Occupational prestige (Canada): Fox & Weisberg (2011), An R Companion to Applied Regression; via pydataset; New York air quality (ozone): Chambers et al. (1983), Graphical Methods for Data Analysis; via pydataset. Has missing values; Friedman #1 (synthetic, noise sd 1): Friedman (1991), Annals of Statistics 19(1); generated with scikit-learn, seed 2026

## Forecasting

| Dataset | Values | Horizon | Selected model | MASE | MASE of naive rule | Beats rule | 80% band coverage | Rule MAE (EasyResearch) | Rule MAE (recomputed) | Seconds | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mauna Loa CO2 (monthly) | 526 | 12 | Linear model on past values (Ridge) | 0.372 | 1.283 | yes | 0.778 | 1.613 | 1.613 | 9.388 | 750.949 |
| Airline passengers (monthly) | 144 | 12 | Theta method | 1.251 | 1.614 | yes | 0.375 | 47.583 | 47.583 | 8.012 | 735.652 |
| Nottingham temperature (monthly) | 240 | 12 | Theta method | 0.593 | 0.837 | yes | 0.812 | 2.329 | 2.329 | 8.742 | 727.238 |
| UK gas consumption (quarterly) | 108 | 4 | Linear model on past values (Ridge) | 1.494 | 1.660 | yes | 0.700 | 42.420 | 42.420 | 7.890 | 735.871 |
| Sunspot activity (yearly) | 309 | 5 | Linear model on past values (Ridge) | 1.125 | 2.431 | yes | 0.820 | 42.924 | 42.924 | 5.400 | 727.316 |
| Nile river flow (yearly) | 100 | 5 | Random Forest on past values | 1.229 | 0.919 | no | 0.350 | 122.600 | 122.600 | 5.813 | 736.980 |

Largest difference from the reference implementation: 0.0000.

Sources: Mauna Loa CO2 (monthly): Keeling & Whorf; via statsmodels; Airline passengers (monthly): Box & Jenkins (1976); via pydataset; Nottingham temperature (monthly): Anderson (1976); via pydataset; UK gas consumption (quarterly): Durbin & Koopman (2001); via pydataset; Sunspot activity (yearly): Yearly sunspot numbers; via statsmodels; Nile river flow (yearly): Cobb (1978), Biometrika 65(2); via statsmodels

## Signals

| Dataset | Recordings | Length | Classes | Selected model | Final score | Balanced accuracy | ROCKET, official split (EasyResearch) | ROCKET, official split (aeon) | Seconds | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| GunPoint (motion) | 200 | 150 | 2 | ROCKET (random convolution kernels) | nested | 1.000 | 1.000 | 1.000 | 70.458 | 916.832 |
| ItalyPowerDemand (electricity) | 1,096 | 24 | 2 | ROCKET (random convolution kernels) | nested | 0.972 | 0.970 | 0.970 | 103.803 | 1481.980 |
| ArrowHead (shape outlines) | 211 | 251 | 3 | ROCKET (random convolution kernels) | nested | 0.938 | 0.806 | 0.823 | 113.029 | 1089.391 |
| OSULeaf (leaf outlines, 6 classes) | 442 | 427 | 6 | ROCKET (random convolution kernels) | nested | 0.967 | 0.942 | 0.942 | 362.050 | 1314.699 |

Largest difference from the reference implementation: 0.0171.

Sources: GunPoint (motion): UCR archive (Dau et al., 2019); via aeon; ItalyPowerDemand (electricity): UCR archive (Dau et al., 2019); via aeon; ArrowHead (shape outlines): UCR archive (Dau et al., 2019); via aeon; OSULeaf (leaf outlines, 6 classes): UCR archive (Dau et al., 2019); via aeon
