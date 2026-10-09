"""EasyResearch - machine learning without programming.

Analysis cores that share EasyClassifier's rules (leakage-free preparation,
default parameters with fixed seeds, honest final scores, plain-language
reports):

* ``easyresearch.regression``  - predict a number from a table
* ``easyresearch.timeseries``  - forecast a series; classify recordings
  (ECG, EEG, sensor signals)
* classification is provided by the EasyClassifier package (0.8.1)

The cores have no user interaction: settings go in, results and files come
out, so a terminal wizard, the desktop application or a web server can drive
them in the same way.
"""

__version__ = "0.1.0"
__author__ = "Ahmad Hassanat"

CITATION = (
    "Hassanat, A. B. A. (2026). EasyResearch: Machine Learning without "
    "Programming (Version {version}) [Software]. "
    "https://github.com/ahmadhassant/easyclassifier".format(version=__version__)
)
EASYCLASSIFIER_CITATION = (
    "Hassanat, A. B. A. (2026). EasyClassifier: Machine Learning without "
    "Programming (Version 0.8.1) [Software]. Zenodo. "
    "https://doi.org/10.5281/zenodo.23122902")
