EasyResearch Desktop - example files
====================================

One file per task, to try the application and to see how your own data
should be laid out. Open them with "Example files..." in step 1 of each task.
Each file is the same data as that task's example button.

classification_iris.csv
  Task: Classification. Column to predict: Species (3 classes).
  One row per flower; four measurements in centimetres.
  Source: Fisher, R. A. (1936). The use of multiple measurements in
  taxonomic problems. Annals of Eugenics, 7(2), 179-188.

regression_diabetes.csv
  Task: Regression. Column to predict: Progression (disease progression one
  year after baseline). One row per patient; ten baseline measurements in
  their original units (age, sex, BMI, blood pressure, six blood tests).
  Source: Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004).
  Least angle regression. Annals of Statistics, 32(2), 407-499.

forecasting_co2_monthly.csv
  Task: Time-series forecasting. Series: CO2_ppm; date column: month.
  One row per month, March 1958 - December 2001; five months have no value.
  Source: Keeling, C. D., & Whorf, T. P. Atmospheric CO2 records from the
  Mauna Loa Observatory (weekly data distributed with statsmodels, averaged
  per month).

signals_heartbeats_synthetic.csv
  Task: Signal classification. Class column: label (Normal, Wide QRS,
  ST depression); signal values: t1 ... t140.
  One row per heartbeat from a single channel. SYNTHETIC data generated for
  trying the tool - NOT real patient recordings.
