"""EasyClassifier - Machine Learning without Programming.

A guided, menu-driven wizard that lets non-programmers build and evaluate
classification models from a CSV file.
"""

__version__ = "0.8.1"
__author__ = "Ahmad Hassanat"

CITATION = (
    "Hassanat, A. (2026). EasyClassifier: Machine Learning without "
    "Programming (Version {version}) [Software].".format(version=__version__)
)

from .wizard import Wizard  # noqa: E402


def run():
    """Launch the interactive wizard."""
    Wizard().run()


__all__ = ["Wizard", "run", "__version__", "CITATION"]
