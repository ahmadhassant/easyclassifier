"""The interactive wizard that ties every phase together.

This is the only stateful orchestrator. Each step calls into a focused module
(ui, dataset, preprocessing, models, evaluation, reporting) so the interface
and the machine-learning logic stay separate and extensible.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import os
from typing import List

import numpy as np
import pandas as pd

from . import __version__, CITATION
from . import ui
from . import dataset as ds
from . import preprocessing as pp
from . import recommend as rec
from . import reporting as rep
from . import target as tg
from . import figures as figs
from .diagnostics import (
    interpret_comparison,
    interpret_learning_curve,
    learning_curve_data,
)
from .importance import honest_importance
from .latex_report import ReportContext, write_and_compile
from .help_texts import explain
from .logbook import LogBook
from .models import build_registry
from .distances import (
    DISTANCES,
    DISTANCE_HELP,
    HASSANAT_CITATION,
    HASSANAT_CITATIONS,
    hassanat_form,
    make_knn,
)
from .evaluation import (
    METRICS,
    SELECTION,
    SELECTION_LABEL,
    VALIDATION,
    evaluate,
    evaluate_on_test,
    fit_final_model,
    nested_cv,
    recommend_selection,
    split_final_test,
)


LARGE_DATA_ROWS = 20_000


def figs_saved(fig_files: dict, out_dir: str) -> List[str]:
    """Absolute paths of all figure files, for the list shown at the end."""
    return [os.path.join(out_dir, p) for v in fig_files.values()
            for p in v.values()]


class Wizard:
    def __init__(self) -> None:
        self.log = LogBook()
        self.registry = build_registry()
        self.mode = "beginner"
        # State collected through the wizard.
        self.df = None
        self.source_name = ""
        self.target = ""
        self.insp = None
        self.knn_distance = None
        self.knn_k = None
        self.hassanat_form = None
        self.selection = None
        self.target_display = ""
        self.target_grouping = ""
        self.fig_theme, self.fig_formats = figs.DEFAULT_THEME, "png"
        self.df_loaded = None
        self.comparison_text = ""
        self.learning_text = ""
        # Facts for the report's Methods section.
        self.rows_loaded = self.cols_loaded = 0
        self.data_notes: List[str] = []
        self.left_out = {}
        self.impute_used = None
        self.encoding_used = None
        self._dev = self._test = None

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.welcome()
        self.choose_mode()
        if not self.load_dataset():
            return
        self.inspect_dataset()
        self.select_target()
        auto = self.choose_auto_or_manual()
        self.drop_useless_columns(auto)
        self.insp = ds.inspect(self.df)  # rows/columns may have changed
        self.show_recommendations()
        X, y, class_names = self.preprocess(auto)
        classifiers = self.choose_classifiers()
        if "knn" in classifiers:
            self.configure_knn()
        validation = self.choose_validation(auto, y)
        self.selection = (self.choose_selection(auto, y)
                          if len(classifiers) > 1 else None)
        metrics = self.choose_metrics(auto)
        figures = self.choose_figures(auto, y)
        if not self.confirm(classifiers, validation, metrics, figures):
            ui.info("No problem - restart any time by running 'easyclassifier'.")
            return
        self.execute(X, y, class_names, classifiers, validation, metrics,
                     figures)

    # ------------------------------------------------------------------ #
    # Honest final score when several classifiers are compared
    # ------------------------------------------------------------------ #

    def choose_selection(self, auto: bool, y) -> str:
        recommended = recommend_selection(y)
        if not auto:
            keys = list(SELECTION.keys())
            ui.blank()
            ui.info("You are comparing several classifiers. The score of the "
                    "winner is slightly too optimistic, because it partly won "
                    "by luck. How should its final score be measured?")
            help_keys = ["selection", "final_test", "nested_cv"]
            idx = ui.menu("Choose one:", list(SELECTION.values()), default=0)
            idx = self._resolve_help(idx, help_keys)
            choice = keys[idx]
            if choice != "auto":
                self.log.add(f"Final score method: {choice} (user)")
                return choice
        ui.note("Final score: " + (
            "20% of rows kept aside as an untouched test set"
            if recommended == "final_test" else
            "nested cross-validation (up to 2,000 rows)")
            + " (auto-selected).")
        self.log.add(f"Final score method: {recommended} (auto)")
        return recommended

    # ------------------------------------------------------------------ #
    # Phase 3: Welcome
    # ------------------------------------------------------------------ #

    def welcome(self) -> None:
        ui.banner("Welcome to EasyClassifier",
                  "Machine Learning without Programming")
        ui.info("This software will guide you through building and evaluating "
                "classification models - one simple question at a time.")
        ui.blank()
        ui.note(f"Version {__version__}")
        ui.note("Estimated time: 3-10 minutes")
        ui.note("Citation: " + CITATION)
        ui.blank()
        ui.info("Tip: at any menu you can type ?N to get a plain-language "
                "explanation of option N.")

    # ------------------------------------------------------------------ #
    # Phase 20: Mode
    # ------------------------------------------------------------------ #

    def choose_mode(self) -> None:
        idx = ui.menu(
            "Select a mode:",
            ["Beginner  (essential choices, sensible defaults)",
             "Advanced  (more control over each step)",
             "Research  (full control, everything shown)"],
            default=0, allow_help=False,
        )
        self.mode = ["beginner", "advanced", "research"][idx]
        self.log.add(f"Mode: {self.mode}")

    @property
    def is_beginner(self) -> bool:
        return self.mode == "beginner"

    # ------------------------------------------------------------------ #
    # Phase 4: Load dataset
    # ------------------------------------------------------------------ #

    def load_dataset(self) -> bool:
        ui.header("Step 1 of 6:  Load your data")
        ui.info("Type the name of your data file (.csv or .xlsx), or drag "
                "the file into this window and press ENTER. Type 'demo' to "
                "try a built-in sample dataset, or 'quit' to exit.")
        ui.note(f"Current folder: {os.getcwd()}")
        while True:
            name = ui.ask_text("What is the name of your data file?")
            low = name.strip().lower()
            if low in ("quit", "exit"):
                return False
            if low == "demo":
                from .demo_data import load_demo
                self.df = load_demo()
                self.source_name = "demo (Iris flowers)"
                self.source_stem = "demo_iris"
                ui.success("Loaded built-in demo dataset (Iris flowers).")
                self.log.add("Dataset: demo (Iris)")
                return True
            path = ds.clean_path(name)
            try:
                self.df = ds.load_csv(path)
                self.source_name = os.path.basename(path)
                self.source_stem = os.path.splitext(self.source_name)[0]
                ui.success(f"Dataset found: {self.source_name} "
                           f"({self.df.shape[0]} rows, "
                           f"{self.df.shape[1]} columns)")
                self.log.add(f"Dataset: {os.path.abspath(path)}")
                return True
            except FileNotFoundError:
                ui.error(f"The file cannot be found: {path}")
                ui.info("Tips: drag the file into this window, or type the "
                        "full path, or copy the file into the current folder "
                        "shown above. Type 'demo' or 'quit' otherwise.")
            except Exception as exc:  # noqa: BLE001
                ui.error(f"Could not read that file: {exc}")

    # ------------------------------------------------------------------ #
    # Phase 5: Inspect
    # ------------------------------------------------------------------ #

    def inspect_dataset(self) -> None:
        self.insp = ds.inspect(self.df)
        insp = self.insp
        self.rows_loaded, self.cols_loaded = insp.n_rows, insp.n_cols
        self.df_loaded = self.df.copy()   # as loaded, for the missing map
        ui.header("Step 2 of 6:  Data inspection")
        ui.note(f"Rows: {insp.n_rows}")
        ui.note(f"Columns: {insp.n_cols}")
        ui.note(f"Numeric columns: {len(insp.numeric_cols)}")
        ui.note(f"Text/categorical columns: {len(insp.categorical_cols)}")
        ui.note(f"Missing values: {insp.missing_total}")
        ui.note(f"Duplicate rows: {insp.duplicate_rows}")
        self.log.section("Inspection")
        self.log.add(f"{insp.n_rows} rows x {insp.n_cols} cols, "
                     f"{insp.missing_total} missing, "
                     f"{insp.duplicate_rows} duplicates")

    # ------------------------------------------------------------------ #
    # Phase 6: Target
    # ------------------------------------------------------------------ #

    def select_target(self) -> None:
        ui.header("Step 3 of 6:  Choose what to predict")
        ui.info("Pick the column with the groups (classes) you want to "
                "predict, for example Yes/No, Healthy/Sick, or Species.")
        original = self.df.copy()
        cols = original.columns.tolist()
        infos = [tg.describe_column(original[c]) for c in cols]
        labels = [i.label for i in infos]
        guess = tg.suggest_target(original)
        default = cols.index(guess) if guess is not None else None
        marks = {default: "   <- suggested"} if default is not None else {}

        while True:
            self.df = original
            idx = ui.pick_from_long_list(
                "Which column do you want to predict?", labels, cols,
                default=default, marks=marks)
            if self._accept_target(cols[idx], infos[idx]):
                break
        ui.success(f"Target: {self.target_display}")
        self.log.add(f"Target column: {self.target_display}")

    def drop_useless_columns(self, auto: bool) -> None:
        """Leave out columns that cannot help predict: IDs/names that differ
        in every row, and columns with a single value."""
        reasons = {}
        for c in self.df.columns:
            if c == self.target:
                continue
            kind = tg.describe_column(self.df[c]).kind
            if kind == tg.ID_LIKE:
                reasons[c] = "different in every row (ID or name)"
            elif kind in (tg.CONSTANT, tg.EMPTY):
                reasons[c] = "only one value"
        if not reasons:
            return
        ui.blank()
        ui.info("These columns cannot help predict and would only add noise "
                "(an ID can even fake good results):")
        for c, why in reasons.items():
            ui.note(f"{c} - {why}")
        leave_out = True
        if not auto:
            leave_out = ui.menu("What would you like to do?",
                                ["Leave them out (recommended)",
                                 "Keep them"],
                                default=0, allow_help=False) == 0
        if leave_out:
            self.df = self.df.drop(columns=list(reasons))
            self.left_out = dict(reasons)
            ui.success(f"Left out {len(reasons)} column(s).")
            self.log.add("Left out columns: " + ", ".join(
                f"{c} ({why})" for c, why in reasons.items()))
        else:
            self.log.add("Kept ID/single-value columns: "
                         + ", ".join(reasons))

    def _accept_target(self, col: str, info) -> bool:
        """Check the chosen column; fix or reject it with the user's help.

        Returns True when the column (possibly grouped) can be used.
        """
        self.target, self.target_display, self.target_grouping = col, col, ""

        if info.kind in (tg.EMPTY, tg.CONSTANT):
            ui.error(f"'{col}' has only one value, so there is nothing to "
                     "predict. Please choose another column.")
            return False

        if info.kind == tg.ID_LIKE:
            ui.warn(f"'{col}' is different in every row, so it looks like "
                    "an ID or a name, not a group. A model cannot learn "
                    "groups from it.")
            return ui.menu("What would you like to do?",
                           ["Choose another column", "Use it anyway"],
                           default=0, allow_help=False) == 1

        if info.kind == tg.MANY_TEXT:
            ui.warn(f"'{col}' has {info.n_unique} different values. Each "
                    "would become its own class, with very few rows each, "
                    "so results would be unreliable.")
            return ui.menu("What would you like to do?",
                           ["Choose another column", "Use it anyway"],
                           default=0, allow_help=False) == 1

        if info.kind == tg.MEASUREMENT and not self._group_measurement(
                col, info):
            return False

        return self._check_class_sizes()

    def _group_measurement(self, col: str, info) -> bool:
        s = self.df[col]
        ui.warn(f"'{col}' looks like a measurement ({info.n_unique} "
                f"different numbers from {tg._fmt(s.min())} to "
                f"{tg._fmt(s.max())}), not a set of groups.")
        ui.info("EasyClassifier predicts groups (classification). Predicting "
                "an exact number is called regression and is not supported "
                "yet. You can turn the numbers into groups instead:")
        choice = ui.menu(
            "What would you like to do?",
            ["Split into 2 equal-sized groups (Low / High)",
             "Split into 3 equal-sized groups (Low / Medium / High)",
             "Split into 4 equal-sized groups",
             "Split at a value I choose (below / at or above it)",
             "Choose another column"],
            allow_help=False)
        if choice == 4:
            return False
        try:
            if choice == 3:
                while True:
                    raw = ui.ask_text(
                        f"Type the value to split at (between "
                        f"{tg._fmt(s.min())} and {tg._fmt(s.max())})")
                    try:
                        t = float(raw)
                    except ValueError:
                        ui.error("Please type a number.")
                        continue
                    if s.min() < t <= s.max():
                        break
                    ui.error("That value would put every row in one group.")
                grouped, cuts, desc = tg.group_at(s, t)
            else:
                grouped, cuts, desc = tg.group_equal(s, choice + 2)
        except ValueError as exc:
            ui.error(f"Could not form groups: {exc}.")
            return False

        df = self.df.copy()
        df[col] = grouped
        self.df = df
        counts = grouped.value_counts()
        ui.success("Groups created:")
        for name in counts.index:
            ui.note(f"{name}: {counts[name]} rows")
        if choice < 3 and grouped.nunique() < choice + 2:
            ui.note("Fewer groups than asked, because many rows share the "
                    "same value.")
        self.target_grouping = desc
        self.target_display = f"{col} (grouped: {desc})"
        self.log.add(f"'{col}' was a measurement; grouped into: {desc}. "
                     "Cut-points were set once on all rows, as part of "
                     "defining the question.")
        return True

    def _check_class_sizes(self) -> bool:
        s = self.df[self.target]
        tiny, small = tg.tiny_and_small_classes(s)
        if tiny:
            n_rows = int(s.astype(str).isin(tiny).sum())
            ui.warn(f"{len(tiny)} class(es) have only one row, so they can "
                    f"never be tested: {', '.join(tiny[:5])}"
                    + (", ..." if len(tiny) > 5 else ""))
            choice = ui.menu(
                "What would you like to do?",
                [f"Remove those {n_rows} row(s) and continue",
                 "Choose another column"],
                default=0, allow_help=False)
            if choice == 1:
                return False
            keep = ~s.astype(str).isin(tiny) | s.isna()
            self.df = self.df[keep].reset_index(drop=True)
            self.data_notes.append(
                f"{n_rows} row(s) belonging to classes with a single row "
                f"({', '.join(tiny)}) were removed, because such classes "
                "cannot be tested.")
            self.log.add(f"Removed {n_rows} rows of single-row classes: "
                         f"{', '.join(tiny)}")
            if self.df[self.target].nunique() < 2:
                ui.error("Fewer than two classes are left. Please choose "
                         "another column.")
                return False
        if small:
            ui.note(f"Some classes have fewer than {tg.FEW_ROWS_PER_CLASS} "
                    f"rows ({', '.join(small[:5])}"
                    + (", ..." if len(small) > 5 else "")
                    + "). Results for them will be unreliable.")
            self.log.add(f"Small classes (<{tg.FEW_ROWS_PER_CLASS} rows): "
                         f"{', '.join(small)}")
        return True

    # ------------------------------------------------------------------ #
    # Phase 5b: auto vs manual
    # ------------------------------------------------------------------ #

    def choose_auto_or_manual(self) -> bool:
        if self.is_beginner:
            idx = ui.menu(
                "How would you like to prepare the data?",
                ["Automatically prepare everything (recommended)",
                 "Review each step manually"],
                default=0, allow_help=False,
            )
            return idx == 0
        return False

    def show_recommendations(self) -> None:
        y_raw = self.df[self.target]
        notes = rec.build_notes(self.insp, self.df, self.target, y_raw)
        if notes:
            ui.header("Smart recommendations")
            for n in notes:
                ui.note(n)
            self.log.section("Recommendations")
            for n in notes:
                self.log.add(n)

    # ------------------------------------------------------------------ #
    # Phase 7-9: Preprocessing
    # ------------------------------------------------------------------ #

    def preprocess(self, auto: bool):
        ui.header("Step 4 of 6:  Prepare the data")
        ui.info("Note: fill values, scaling and encodings are learned from "
                "the training part only, separately for every test round, so "
                "no information from the test data leaks into training.")
        df = self.df
        insp = self.insp
        cfg = pp.PrepConfig()

        # Rows without a class label can't be used at all.
        df, n = pp.drop_missing_target(df, self.target)
        if n:
            ui.success(f"Removed {n} rows with no value in '{self.target}'.")
            self.log.add(f"Removed {n} rows missing the target")
            self.data_notes.append(f"{n} row(s) with no value for "
                                   f"{self.target} were removed.")

        # ---- Missing values ----
        has_missing = bool(df.drop(columns=[self.target]).isna().any().any())
        if has_missing:
            if auto:
                strat = rec.missing_strategy(insp)
                strat = "median" if strat in ("none", "median") else strat
            else:
                choice = ui.menu(
                    f"Missing values detected ({insp.missing_total}). "
                    "How should they be handled?",
                    ["Remove rows with missing values",
                     "Replace with mean",
                     "Replace with median",
                     "Replace with most common value",
                     "Automatic recommendation"],
                    default=4,
                    help_keys=["missing_values"],
                )
                choice = self._resolve_help(choice, ["missing_values"] * 5)
                strat = ["drop", "mean", "median", "mode",
                         rec.missing_strategy(insp)][choice]
                strat = "median" if strat == "none" else strat
            if strat == "drop":
                df, n = pp.drop_missing_rows(df)
                desc = f"Removed {n} rows with missing values."
            else:
                cfg.impute = strat
                label = "most common value" if strat == "mode" else strat
                desc = (f"Missing numeric values will be filled with the "
                        f"{label} (text columns: most common value), "
                        "learned from the training data only.")
            ui.success(desc)
            self.log.add(f"Missing values: {desc}")
            if strat == "drop":
                self.data_notes.append(f"{n} row(s) with missing values "
                                       "were removed.")
            else:
                self.impute_used = strat

        # ---- Duplicates ----
        dup = int(df.duplicated().sum())
        if dup:
            by_chance = pp.duplicates_expected_by_chance(df)
            remove = not by_chance
            if not auto:
                remove = ui.menu(
                    f"{dup} identical rows detected. "
                    + ("With so few possible value combinations, different "
                       "cases can easily be identical, so keeping them is "
                       "recommended." if by_chance else
                       "They are probably accidental copies, so removing "
                       "them is recommended."),
                    ["Remove them", "Keep them"],
                    default=0 if remove else 1, allow_help=False) == 0
            if remove:
                df, n = pp_remove(df)
                ui.success(f"Removed {n} duplicate rows.")
                self.log.add(f"Removed {n} duplicate rows")
                self.data_notes.append(f"{n} duplicate row(s) were "
                                       "removed.")
            else:
                ui.success(f"Kept {dup} identical rows"
                           + (" (expected between different cases with so "
                              "few possible value combinations)."
                              if by_chance else "."))
                self.log.add(f"Kept {dup} identical rows "
                             f"(expected by chance: {by_chance})")
                if by_chance:
                    self.data_notes.append(
                        f"{dup} identical rows were kept, because with so "
                        "few possible value combinations different cases "
                        "are expected to coincide.")

        # Split X / y
        y_raw = df[self.target]
        X = pp.cast_categoricals(df.drop(columns=[self.target]))

        # ---- Encoding ----
        has_cat = any(not pd.api.types.is_numeric_dtype(X[c])
                      for c in X.columns)
        if has_cat:
            if auto:
                method = rec.encoding_method(df, self.target)
            else:
                choice = ui.menu(
                    "Text/categorical columns found. Choose how to encode "
                    "them into numbers:",
                    ["Automatic", "Label Encoding", "One-Hot Encoding"],
                    default=0,
                    help_keys=["label_encoding", "one_hot_encoding"],
                )
                choice = self._resolve_help(
                    choice, ["one_hot_encoding", "label_encoding",
                             "one_hot_encoding"])
                method = ["auto", "label", "onehot"][choice]
            cfg.encoding = method
            self.encoding_used = method
            ui.success(f"Text columns will be encoded ({method}).")
            self.log.add(f"Encoding: {method}")

        y, class_names, _ = pp.encode_target(y_raw)

        # ---- Scaling ----
        do_scale = None  # None = decide per classifier
        method = "standard"
        if not auto:
            choice = ui.menu(
                "Would you like to scale (normalise) the numeric data?",
                ["Yes (standardise)", "Yes (0-1 range)", "No",
                 "Automatic (decide from classifier)"],
                default=3,
                help_keys=["scaling"],
            )
            choice = self._resolve_help(choice, ["scaling"] * 4)
            if choice == 0:
                method, do_scale = "standard", True
            elif choice == 1:
                method, do_scale = "minmax", True
            elif choice == 2:
                method, do_scale = "none", False
            else:
                method, do_scale = "standard", None  # decide later
        cfg.scale_method = method
        cfg.scale = do_scale
        self.prep_cfg = cfg
        self.log.add(f"Scaling choice: {method if do_scale else do_scale}")

        n_feat = pp.describe_transformed(X, cfg, scale=False).shape[1]
        ui.success(f"Data ready: {X.shape[0]} rows, {X.shape[1]} columns "
                   f"({n_feat} features after encoding).")
        self.log.add(f"Data: {X.shape[0]} rows x {X.shape[1]} columns, "
                     f"{n_feat} features after encoding")
        return X, y, class_names

    # ------------------------------------------------------------------ #
    # Phase 10: Classifiers
    # ------------------------------------------------------------------ #

    def choose_classifiers(self) -> List[str]:
        ui.header("Step 5 of 6:  Choose classifier(s)")
        specs = list(self.registry.values())
        labels = []
        for s in specs:
            labels.append(s.name if s.available
                          else f"{s.name}  ({s.reason})")
        options = ["All available classifiers"] + labels
        while True:
            picks = ui.multi_select(
                "Which classifier(s) would you like to try? "
                "(You can pick several.)",
                options, default=[0],   # ENTER = all available
            )
            if not picks:
                ui.error("Please choose at least one classifier.")
                continue
            keys: List[str] = []
            if 0 in picks:
                keys = [s.key for s in specs if s.available]
            else:
                for p in picks:
                    s = specs[p - 1]
                    if s.available:
                        keys.append(s.key)
                    else:
                        ui.warn(f"{s.name} is not installed - skipping.")
            if keys:
                self.log.add(f"Classifiers: {', '.join(keys)}")
                slow = [self.registry[k].name for k in keys
                        if k in ("svm", "knn", "neural_network")]
                if len(self.df) > LARGE_DATA_ROWS and slow:
                    ui.warn(f"Your data has {len(self.df)} rows. "
                            + ", ".join(slow) + " can take a long time "
                            "(many minutes or more) on data this size. "
                            "Random Forest, Logistic Regression, Naive Bayes "
                            "and Decision Tree are much faster.")
                    if not ui.ask_yes_no("Continue with this choice?",
                                         default=True):
                        continue
                return keys
            ui.error("None of the chosen classifiers are available.")

    # ------------------------------------------------------------------ #
    # KNN configuration: distance metric and k
    # ------------------------------------------------------------------ #

    def configure_knn(self) -> None:
        keys = list(DISTANCES.keys())
        ui.header("KNN settings")
        idx = ui.menu(
            "Which distance should KNN use to measure similarity?",
            [DISTANCES[k][0] for k in keys],
            default=0,
        )
        while idx < 0:
            ui.blank()
            ui.info(DISTANCE_HELP[keys[-idx - 1]])
            ui.pause()
            idx = ui.menu(
                "Which distance should KNN use to measure similarity?",
                [DISTANCES[k][0] for k in keys], default=0,
            )
        dist = keys[idx]

        k = 5
        if not self.is_beginner:
            while True:
                raw = ui.ask_text("How many neighbours (k)?", default="5")
                if raw.isdigit() and int(raw) >= 1:
                    k = int(raw)
                    break
                ui.error("Please enter a whole number of 1 or more.")

        self.knn_distance, self.knn_k = dist, k
        spec = self.registry["knn"]
        spec.factory = lambda d=dist, kk=k: make_knn(d, kk)
        spec.name = f"KNN ({DISTANCES[dist][0].split(' (')[0]}, k={k})"
        ui.success(f"KNN will use: {spec.name}")
        self.log.add(f"KNN distance: {dist}, k={k}")

        if dist == "hassanat":
            ui.blank()
            ui.info("Citation for the Hassanat distance - please cite if you "
                    "publish results:")
            for c in HASSANAT_CITATIONS:
                ui.note(c)
            self.log.add("Citation: " + HASSANAT_CITATION)

    # ------------------------------------------------------------------ #
    # Phase 12: Validation
    # ------------------------------------------------------------------ #

    def choose_validation(self, auto: bool, y) -> str:
        if auto:
            method = rec.validation_method(y)
            ui.note(f"Validation: {VALIDATION[method]} (auto-selected).")
            self.log.add(f"Validation: {method}")
            return method
        keys = list(VALIDATION.keys())
        idx = ui.menu(
            "How should the models be evaluated?",
            list(VALIDATION.values()),
            default=1,
            help_keys=["hold_out", "cross_validation", "cross_validation",
                       "stratified", "leave_one_out"],
        )
        idx = self._resolve_help(idx, ["hold_out", "cross_validation",
                                       "cross_validation", "stratified",
                                       "leave_one_out"])
        self.log.add(f"Validation: {keys[idx]}")
        return keys[idx]

    # ------------------------------------------------------------------ #
    # Phase 13: Metrics
    # ------------------------------------------------------------------ #

    def choose_metrics(self, auto: bool) -> List[str]:
        keys = list(METRICS.keys())
        if auto:
            chosen = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            ui.note("Metrics: Accuracy, Precision, Recall, F1, ROC AUC "
                    "(auto-selected).")
            self.log.add("Metrics: auto set")
            return chosen
        picks = ui.multi_select(
            "Which performance metrics would you like? (Select several.)",
            list(METRICS.values()), default_all=True,
        )
        chosen = [keys[p] for p in picks]
        self.log.add(f"Metrics: {', '.join(chosen)}")
        return chosen

    # ------------------------------------------------------------------ #
    # Phase 14: Figures
    # ------------------------------------------------------------------ #

    def choose_figures(self, auto: bool, y) -> List[str]:
        keys = list(figs.FIGURES.keys())
        imbalanced = pp.imbalance_ratio(y) > 1.5
        has_missing = bool(self.df_loaded.isna().values.any())
        defaults = figs.default_figures(imbalanced, has_missing)
        self.fig_theme, self.fig_formats = figs.DEFAULT_THEME, "png"

        if auto:
            chosen = defaults
            extra = [figs.WHEN_NEEDED[k] for k in figs.WHEN_NEEDED
                     if k in chosen]
            ui.note("Figures: the standard set"
                    + (" (plus extra because " + " and ".join(extra) + ")"
                       if extra else "")
                    + f"; {figs.THEMES[self.fig_theme].name.lower()} "
                    "colours; PNG at 300 dpi (auto-selected).")
        else:
            picks = ui.multi_select(
                "Which figures would you like? ENTER keeps the suggested "
                "ones (the most used in papers).",
                list(figs.FIGURES.values()),
                default=[keys.index(k) for k in defaults])
            chosen = [keys[p] for p in picks]
            if ("learning_curve" in chosen
                    and len(self.df) > LARGE_DATA_ROWS):
                ui.warn("The learning curve trains the model 25 more times; "
                        f"with {len(self.df)} rows this can take long.")
            if chosen:
                tkeys = list(figs.THEMES)
                idx = ui.menu(
                    "Colour theme for the figures:",
                    [f"{figs.THEMES[k].name} - {figs.THEMES[k].description}"
                     for k in tkeys], default=0, allow_help=False)
                self.fig_theme = tkeys[idx]
                fkeys = list(figs.FORMATS)
                idx = ui.menu("File format for the figures:",
                              list(figs.FORMATS.values()), default=0,
                              allow_help=False)
                self.fig_formats = fkeys[idx]
        self.log.add(f"Figures: {', '.join(chosen) or 'none'}; theme "
                     f"{self.fig_theme}; format {self.fig_formats}")
        return chosen

    # ------------------------------------------------------------------ #
    # Phase 16: Confirm
    # ------------------------------------------------------------------ #

    def confirm(self, classifiers, validation, metrics, figures) -> bool:
        ui.header("Step 6 of 6:  Review your choices")
        ui.note(f"Dataset:    {self.source_name}")
        ui.note(f"Target:     {self.target_display or self.target}")
        ui.note("Classifiers: " +
                ", ".join(self.registry[k].name for k in classifiers))
        if self.knn_distance:
            ui.note(f"KNN:        {DISTANCES[self.knn_distance][0]}, "
                    f"k={self.knn_k}")
        ui.note(f"Validation: {VALIDATION[validation]}")
        if self.selection:
            ui.note("Final score: " + SELECTION[self.selection])
        ui.note("Metrics:    " + ", ".join(METRICS[m] for m in metrics))
        ui.note("Figures:    " +
                (", ".join(figs.FIGURES[f] for f in figures) or "none")
                + (f" ({figs.THEMES[self.fig_theme].name.lower()}, "
                   f"{self.fig_formats.upper().replace('+', ' + ')})"
                   if figures else ""))
        return ui.ask_yes_no("Start the analysis now?", default=True)

    # ------------------------------------------------------------------ #
    # Phase 17-19: Execute
    # ------------------------------------------------------------------ #

    def execute(self, X, y, class_names, classifiers, validation, metrics,
                figures) -> None:
        cfg = self.prep_cfg

        # Decide scaling separately for each classifier.
        # None, "standard" (mean 0, SD 1) or "minmax" (0-1), per classifier.
        self._scale_for = {k: rec.scaling_for(k, self.knn_distance,
                                              cfg.scale, cfg.scale_method)
                           for k in classifiers}
        label = {"standard": "standardised", "minmax": "scaled to 0-1"}
        scaled = [f"{self.registry[k].name} ({label[m]})"
                  for k, m in self._scale_for.items() if m]
        if scaled:
            ui.note("Scaling (learned on training data only) used for: "
                    + ", ".join(scaled))
        self.log.add("Scaling per classifier: " + ", ".join(
            f"{k}={v or 'none'}" for k, v in self._scale_for.items()))
        if self._scale_for.get("knn") == "minmax" and cfg.scale is None:
            ui.note("KNN with the Hassanat distance uses columns scaled to "
                    "0-1, which worked best in EasyClassifier's "
                    "benchmarks.")

        # Report which form of the Hassanat formula applies to this data.
        if "knn" in classifiers and self.knn_distance == "hassanat":
            Xt = pp.describe_transformed(X, cfg, self._scale_for["knn"])
            self.hassanat_form = hassanat_form(Xt)
            ui.note("Hassanat distance: " + self.hassanat_form["text"])
            self.log.add("Hassanat distance: " + self.hassanat_form["text"])

        ui.header("Running analysis")
        y = np.asarray(y)
        final = None
        self._validation, self._metrics = validation, metrics
        self._classifiers = classifiers
        self._class_names = class_names

        # Rows used to compare classifiers (all rows, unless a final test
        # set is kept aside first).
        if self.selection == "final_test":
            dev, test = split_final_test(X, y)
            self._dev, self._test = dev, test
            X_cmp, y_cmp = X.iloc[dev], y[dev]
            ui.note(f"{len(test)} rows (20%) set aside as a final test set; "
                    f"classifiers are compared on the other {len(dev)} rows.")
            self.log.add(f"Final test set: {len(test)} rows; "
                         f"development: {len(dev)} rows")
        else:
            X_cmp, y_cmp = X, y

        results = []
        for key in classifiers:
            spec = self.registry[key]
            ui.info(f"Training {spec.name} ...")
            try:
                result = evaluate(self._pipeline_spec(key, X), X_cmp, y_cmp,
                                  class_names, validation, metrics)
                results.append(result)
                self.log.add(f"{spec.name}: " + ", ".join(
                    f"{METRICS[m]}={result.metrics[m]:.4f}"
                    for m in metrics if m in result.metrics)
                    + f", {SELECTION_LABEL}={result.selection_score:.4f}")
            except Exception as exc:  # noqa: BLE001
                ui.error(f"{spec.name} failed: {exc}")
                self.log.add(f"{spec.name} FAILED: {exc}")

        if not results:
            ui.error("No models could be trained. Please check your data.")
            return

        results.sort(key=lambda r: r.primary_score, reverse=True)
        best = results[0]
        self.comparison_text = interpret_comparison(results)
        if self.comparison_text:
            self.log.add("Comparison: " + self.comparison_text)
        self.log.add(f"Selected: {best.classifier_name} "
                     f"(highest {SELECTION_LABEL.lower()})")

        if self.selection and len(results) > 1:
            try:
                if self.selection == "final_test":
                    ui.info(f"Scoring {best.classifier_name} once on the "
                            "untouched test set ...")
                    final = evaluate_on_test(
                        self._pipeline_spec(best.classifier_key, X),
                        X.iloc[dev], y[dev], X.iloc[test], y[test],
                        class_names, metrics)
                    final.note = (f"{best.classifier_name}, trained on "
                                  f"{len(dev)} rows, tested once on "
                                  f"{len(test)} unseen rows")
                else:
                    ui.info("Running nested cross-validation (repeats the "
                            "whole comparison inside each of 5 folds) ...")
                    ok = [r.classifier_key for r in results]
                    final = nested_cv(
                        [self._pipeline_spec(k, X) for k in ok], X, y,
                        class_names, validation, metrics,
                        progress=lambda f, n: ui.note(
                            f"fold {f}: {n} selected"))
                self.log.add("Final estimate: " + ", ".join(
                    f"{METRICS[m]}={final.metrics[m]:.4f}"
                    for m in metrics if m in final.metrics)
                    + f" ({final.note})")
            except Exception as exc:  # noqa: BLE001
                ui.error(f"Final scoring failed: {exc}")
                self.log.add(f"Final scoring FAILED: {exc}")
                final = None

        self.show_results(best, results, metrics, final)
        self.save_outputs(best, results, y, figures, X, class_names, final)

    @staticmethod
    def _show_metrics(result, metrics) -> None:
        for m in metrics:
            if m in result.metrics:
                val = result.metrics[m]
                shown = (f"{val*100:.2f}%"
                         if m in ("accuracy", "precision", "recall", "f1",
                                  "balanced_accuracy", "specificity")
                         else f"{val:.4f}")
                ui.note(f"{METRICS[m]:<18} {shown}")

    def show_results(self, best, results, metrics, final=None) -> None:
        ui.header("Results")
        ui.info(f"Predicting: {self.target_display or self.target}")
        ui.blank()
        if final is None:
            ui.info(f"Best classifier: {best.classifier_name}")
            ui.blank()
            self._show_metrics(best, metrics)
        else:
            ui.info(f"Selected classifier: {best.classifier_name}")
            ui.blank()
            if self.selection == "final_test":
                ui.info("Final score on the untouched 20% test set - "
                        "these are the numbers to report:")
            else:
                ui.info("Nested cross-validation estimate - these are the "
                        "numbers to report:")
            self._show_metrics(final, metrics)
            ui.note(final.note)
        if len(results) > 1:
            ui.blank()
            ui.info(f"Comparison of all classifiers ({SELECTION_LABEL.lower()}"
                    ", ranked)" + (" - used only to choose the winner; "
                                   "slightly optimistic, do not report as "
                                   "the final result:" if final else ":"))
            for r in results:
                score = r.primary_score
                ui.note(f"{r.classifier_name:<34} {score*100:6.2f}%")
            if self.comparison_text:
                ui.blank()
                ui.info(self.comparison_text)
        if self.hassanat_form:
            ui.blank()
            ui.info("Hassanat distance - " + self.hassanat_form["text"])
            ui.info("Please cite:")
            for c in HASSANAT_CITATIONS:
                ui.note(c)

    def save_outputs(self, best, results, y, figures, X, class_names,
                     final=None) -> None:
        # A new folder for every run, so earlier results are never
        # overwritten: Results/<data file>_<date>_<time>/
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                       for ch in getattr(self, "source_stem", "data"))[:40]
        out_dir = rep.ensure_dir(os.path.join(os.getcwd(), "Results",
                                              f"{safe}_{stamp}"))
        fig_dir = rep.ensure_dir(os.path.join(out_dir, "figures"))
        saved: List[str] = []

        table = list(results)
        if final is not None:
            label = ("FINAL - untouched 20% test set"
                     if self.selection == "final_test"
                     else "FINAL - nested cross-validation")
            table = [dataclasses.replace(
                final, classifier_name=f"{label} ({best.classifier_name})")
            ] + table
        saved.append(rep.save_summary_csv(
            table, os.path.join(out_dir, "summary.csv")))
        xl = rep.save_excel(table, os.path.join(out_dir, "results.xlsx"))
        if xl:
            saved.append(xl)

        # Figures and predictions come from the honest estimate when present.
        shown = final if final is not None else best

        # Which columns mattered (measured on held-out rows).
        importance = None
        if {"feature_importance", "columns_by_class"} & set(figures):
            ui.info("Measuring which columns matter ...")
            try:
                importance = honest_importance(
                    self._pipeline_spec(best.classifier_key, X), X, y,
                    self._validation, self._dev, self._test)
                saved.append(rep.save_feature_importance(
                    importance, os.path.join(out_dir,
                                             "feature_importance.csv")))
                top = ", ".join(str(c) for c in importance["column"].head(3))
                ui.note(f"Most important columns: {top}")
                self.log.add("Permutation importance (top 5): " + ", ".join(
                    f"{r.column}={r.importance:.4f}"
                    for r in importance.head(5).itertuples()))
            except Exception as exc:  # noqa: BLE001
                ui.warn(f"Could not measure column importance: {exc}")
                self.log.add(f"Importance FAILED: {exc}")
                importance = None

        # Would more data help? (development rows only, if a final test set
        # was kept aside, so the test rows stay untouched.)
        curve = None
        if "learning_curve" in figures:
            ui.info("Computing the learning curve ...")
            rows = self._dev if self._dev is not None else np.arange(len(y))
            try:
                curve = learning_curve_data(
                    self._pipeline_spec(best.classifier_key, X),
                    X.iloc[rows], y[rows])
                if curve is not None:
                    self.learning_text = interpret_learning_curve(curve)
                    ui.note(self.learning_text)
                    self.log.add("Learning curve: " + self.learning_text)
                else:
                    ui.note("Learning curve skipped: too few rows (it needs "
                            "at least 20, with 2 or more in every class).")
                    self.log.add("Learning curve skipped: too few rows")
            except Exception as exc:  # noqa: BLE001
                ui.warn(f"Could not compute the learning curve: {exc}")
                self.log.add(f"Learning curve FAILED: {exc}")

        fig_files = self._draw_figures(figures, fig_dir, out_dir, best,
                                       results, final, shown, X, y,
                                       importance, curve)
        for paths in figs_saved(fig_files, out_dir):
            saved.append(paths)

        saved.append(rep.save_predictions(
            shown, os.path.join(out_dir, "predictions.csv")))

        # Saved model = full pipeline (cleaning + encoding + scaling + model),
        # so it can be applied directly to new raw data with the same columns.
        model = fit_final_model(self._pipeline_spec(best.classifier_key, X),
                                X, y)
        self.log.add(f"Saved model: {best.classifier_name}, refitted on all "
                     f"{len(y)} rows")
        saved.append(rep.save_model(
            model, os.path.join(out_dir, "trained_model.pkl")))

        saved.append(self.save_citations(os.path.join(out_dir,
                                                      "citations.txt")))

        # Report: report.tex, compiled to report.pdf if LaTeX is installed.
        ui.info("Writing the report ...")
        try:
            ctx = self._report_context(best, results, final, X, y,
                                       fig_files, importance)
            tex_path, pdf_path, msg = write_and_compile(ctx, out_dir)
            saved.append(tex_path)
            if pdf_path:
                saved.append(pdf_path)
                ui.success("Report: " + msg)
            else:
                ui.warn(msg)
            self.log.add("Report: " + msg)
        except Exception as exc:  # noqa: BLE001
            ui.warn(f"Could not write the report: {exc}")
            self.log.add(f"Report FAILED: {exc}")

        saved.append(self.log.save(os.path.join(out_dir, "log.txt")))

        ui.header("Saved output")
        ui.success("All results were saved in this folder:")
        print(f"  {out_dir}")
        ui.blank()
        n_fig = sum(1 for s in saved
                    if os.path.dirname(os.path.relpath(s, out_dir)))
        for s in saved:
            rel = os.path.relpath(s, out_dir)
            if not os.path.dirname(rel):
                ui.note(rel)
        if n_fig:
            ui.note(f"figures{os.sep}  ({n_fig} files)")
        ui.blank()
        main_file = ("report.pdf" if any(s.endswith("report.pdf")
                                         for s in saved) else "report.tex")
        ui.info(f"Finished. Start with {main_file}: it explains the results "
                "in plain words and includes a Methods paragraph you can "
                "adapt for a paper. Run EasyClassifier again at any time; "
                "each run gets its own new folder.")

    def _draw_figures(self, figures, fig_dir, out_dir, best, results, final,
                      shown, X, y, importance, curve) -> dict:
        """Draw the chosen figures. One failing figure never stops the run.

        Returns {figure key: {format: path relative to out_dir}}.
        """
        fm = figs.FigureMaker(fig_dir, theme=self.fig_theme,
                              formats=self.fig_formats,
                              class_names=self._class_names)
        top_cols = (list(importance["column"]) if importance is not None
                    else list(X.columns))
        final_label = ("Final score (untouched test set)"
                       if self.selection == "final_test"
                       else "Final score (nested CV)")
        jobs = {
            "class_distribution": lambda: fm.class_distribution(y),
            "missing_values": lambda: fm.missing_values(self.df_loaded),
            "correlation": lambda: fm.correlation(X, prefer=top_cols),
            "comparison": lambda: (fm.comparison(
                results, best.classifier_name,
                final.selection_score if final is not None else None,
                final_label) if len(results) > 1 else None),
            "confusion_matrix": lambda: fm.confusion(shown),
            "roc_curve": lambda: fm.roc(shown),
            "pr_curve": lambda: fm.pr(shown),
            "feature_importance": lambda: (fm.importance(importance)
                                           if importance is not None
                                           else None),
            "columns_by_class": lambda: (fm.columns_by_class(
                X, y, top_cols) if importance is not None else None),
            "learning_curve": lambda: (fm.learning_curve(curve)
                                       if curve is not None else None),
        }
        if figures:
            ui.info("Drawing figures ...")
        for key in figures:
            try:
                jobs[key]()
            except Exception as exc:  # noqa: BLE001
                ui.warn(f"Figure '{figs.FIGURES[key]}' could not be drawn: "
                        f"{exc}")
                self.log.add(f"Figure {key} FAILED: {exc}")
        return {k: {ext: os.path.relpath(p, out_dir) for ext, p in v.items()}
                for k, v in fm.files.items()}

    def _report_context(self, best, results, final, X, y, fig_files,
                        importance) -> ReportContext:
        names = {k: self.registry[k].name for k in self._classifiers}
        y_arr = np.asarray(y)
        counts = {name: int((y_arr == i).sum())
                  for i, name in enumerate(self._class_names)}
        knn_text = ""
        if self.knn_distance:
            knn_text = (f"K-nearest neighbours used k = {self.knn_k} and the "
                        f"{DISTANCES[self.knn_distance][0].split(' (')[0]}")
            if self.hassanat_form:
                form = ("the signed form of the formula, for negative values"
                        if self.hassanat_form["form"] == "signed"
                        else "the standard form of the formula (all values "
                             "non-negative)")
                knn_text += f", applied in {form}"
        return ReportContext(
            software_citation=CITATION,
            version=__version__,
            dataset=self.source_name,
            rows_loaded=self.rows_loaded,
            cols_loaded=self.cols_loaded,
            target=self.target,
            target_grouping=self.target_grouping,
            class_counts={str(k): int(v) for k, v in counts.items()},
            rows_used=len(y),
            predictor_columns=[str(c) for c in X.columns],
            left_out=self.left_out,
            data_notes=self.data_notes,
            impute=self.impute_used,
            encoding=self.encoding_used,
            scaled={m: [names[k] for k, v in self._scale_for.items()
                        if v == m] for m in ("standard", "minmax")
                    if m in self._scale_for.values()},
            classifiers=[names[k] for k in self._classifiers],
            knn_text=knn_text,
            hassanat_used=bool(self.hassanat_form),
            validation=self._validation,
            selection=self.selection if final is not None else None,
            n_dev=len(self._dev) if self._dev is not None else 0,
            n_test=len(self._test) if self._test is not None else 0,
            best_name=best.classifier_name,
            final=final,
            results=results,
            metrics=self._metrics,
            figures=fig_files,
            comparison_text=self.comparison_text,
            learning_text=self.learning_text,
            importance=importance,
        )

    def _pipeline_spec(self, key: str, X):
        """Classifier spec whose factory builds preprocessing + model."""
        spec = self.registry[key]
        cfg, scale = self.prep_cfg, self._scale_for[key]
        return dataclasses.replace(
            spec,
            factory=lambda: pp.build_pipeline(X, cfg, spec.factory(), scale),
        )

    def save_citations(self, path: str) -> str:
        lines = ["Please cite the following if you publish these results:",
                 "", "Software:", "  " + CITATION]
        if self.hassanat_form:
            lines += ["", "Hassanat distance (used by KNN):"]
            lines += ["  " + c for c in HASSANAT_CITATIONS]
            lines += ["", "Formula form applied: " + self.hassanat_form["text"]]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    # ------------------------------------------------------------------ #
    # Help resolution
    # ------------------------------------------------------------------ #

    def _resolve_help(self, choice: int, help_keys: List[str]) -> int:
        """If a menu returned a help sentinel (negative), show help and re-ask.

        Because ``ui.menu`` returns a negative sentinel for ?N, we loop here
        until the user makes a real choice. The caller passes the same prompt
        again implicitly by re-invoking; to keep it simple we just show the
        explanation and ask the user to choose again via a follow-up prompt.
        """
        while choice < 0:
            key_index = (-choice) - 1
            key = help_keys[key_index] if key_index < len(help_keys) else ""
            ui.blank()
            ui.info(explain(key))
            ui.pause()
            # Ask again with a minimal numeric prompt.
            raw = ui.ask_text("Enter your choice number")
            if raw.isdigit():
                choice = int(raw) - 1
            else:
                choice = -1
        return choice


def pp_remove(df):
    """Thin wrapper kept for readability in the wizard."""
    return pp.remove_duplicates(df)
