"""Console entry point for EasyClassifier.

Start the wizard with either of:

    easyclassifier
    python -m easyclassifier

Options:

    --version   show the installed version and exit
    --help      show this help and exit
"""

import sys

HELP = """EasyClassifier - machine learning classification without programming.

Usage:
  easyclassifier            start the step-by-step wizard
  python -m easyclassifier  the same, if the 'easyclassifier' command is
                            not found

Options:
  --version   show the installed version
  --help      show this help

The wizard asks for a data file (.csv or .xlsx). Type 'demo' at that
question to try it with a built-in example. Results are saved in a new
folder inside 'Results', in the folder you started from.
"""


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if any(a in ("-h", "--help", "/?") for a in args):
        print(HELP)
        return 0
    if any(a in ("-V", "--version") for a in args):
        from . import __version__
        print(f"EasyClassifier {__version__} "
              f"(Python {sys.version.split()[0]})")
        return 0
    if args:
        print(f"Unknown option: {' '.join(args)}\n")
        print(HELP)
        return 2

    import warnings
    # Keep the screen clean for non-programmers; details go to the log.
    warnings.filterwarnings("ignore")
    from . import run
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nExiting EasyClassifier. Goodbye!")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
