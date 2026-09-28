"""Dataset loading and inspection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Inspection:
    """Summary of a dataset's structure and quality."""

    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    missing_total: int
    missing_by_col: dict
    duplicate_rows: int
    columns_with_missing: List[str] = field(default_factory=list)

    @property
    def has_missing(self) -> bool:
        return self.missing_total > 0

    @property
    def has_duplicates(self) -> bool:
        return self.duplicate_rows > 0


def clean_path(raw: str) -> str:
    """Turn what the user typed or dragged into the window into a path.

    * Dragging a file into a terminal adds quotes (Windows) or escapes
      spaces with a backslash (macOS / Linux); PowerShell may add "& ".
    * ``~`` means the home folder.
    * If the name has no extension, .csv / .xlsx are tried.
    """
    p = raw.strip()
    if p.startswith("& "):
        p = p[2:].strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'":
        p = p[1:-1]
    if os.sep == "/" and "\\ " in p:
        p = p.replace("\\ ", " ")
    p = os.path.expanduser(p)
    if not os.path.splitext(p)[1] and not os.path.isfile(p):
        for ext in (".csv", ".xlsx"):
            if os.path.isfile(p + ext):
                return p + ext
    return p


EXCEL_EXTENSIONS = (".xlsx", ".xlsm")


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file (or the first sheet of an Excel .xlsx file),
    tolerating common separators and encodings."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in EXCEL_EXTENSIONS:
        return pd.read_excel(path)
    if ext == ".xls":
        raise ValueError("old .xls Excel files are not supported; please "
                         "save the sheet as .xlsx or .csv in Excel "
                         "(File > Save As)")
    # Try a few sensible fallbacks for real-world CSVs.
    attempts = [
        dict(),
        dict(sep=";"),
        dict(encoding="latin-1"),
        dict(sep=";", encoding="latin-1"),
    ]
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
        except Exception:  # noqa: BLE001
            continue
        if df.shape[1] > 1:
            if kwargs.get("sep") == ";":
                df = fix_decimal_commas(df)
            return df
    # Fall back to the plain read and let its error (if any) surface.
    return pd.read_csv(path)


def fix_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    """Semicolon-separated files (Excel in many countries) write decimals
    with a comma: 3,5. Turn such text columns into numbers when *every*
    value converts; anything else is left unchanged."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].dropna().astype(str).str.strip()
        if s.empty or not s.str.contains(",", regex=False).any():
            continue
        conv = pd.to_numeric(s.str.replace(",", ".", regex=False),
                             errors="coerce")
        if conv.notna().all():
            df[c] = pd.to_numeric(
                df[c].astype(str).str.strip().str.replace(",", ".",
                                                         regex=False),
                errors="coerce")
    return df


def inspect(df: pd.DataFrame) -> Inspection:
    """Produce an :class:`Inspection` summary of a dataframe."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    missing_by_col = {
        c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()
    }
    return Inspection(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=df.columns.tolist(),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        missing_total=int(df.isna().sum().sum()),
        missing_by_col=missing_by_col,
        duplicate_rows=int(df.duplicated().sum()),
        columns_with_missing=list(missing_by_col.keys()),
    )


def guess_target(df: pd.DataFrame) -> Optional[str]:
    """Best guess at which column holds the class labels.

    Delegates to :func:`easyclassifier.target.suggest_target`, which only
    suggests columns that look like groups (never IDs or measurements).
    """
    from .target import suggest_target
    return suggest_target(df)
