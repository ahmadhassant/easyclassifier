"""Reading one time series from a table and checking it.

The series is put on a regular time grid (one value per hour, day, week,
month, quarter or year). Missing time points are filled with the last known
value - never with later values, so no information from the future is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from easyclassifier import target as tg

from ..regression.data import numeric_values

TIME_NAMES = ["date", "time", "datetime", "timestamp", "ds", "day", "week",
              "month", "quarter", "year", "period"]
MAX_MISSING_SHARE = 0.2
MIN_POINTS = 30

# unit -> (label, season length, pandas period alias)
UNITS = {
    "h": ("hourly", 24, "h"),
    "D": ("daily", 7, "D"),
    "B": ("business-daily", 5, "B"),
    "W": ("weekly", 52, "W"),
    "M": ("monthly", 12, "M"),
    "Q": ("quarterly", 4, "Q"),
    "Y": ("yearly", 1, "Y"),
}
DEFAULT_HORIZON = {"h": 24, "D": 7, "B": 5, "W": 13, "M": 12, "Q": 4, "Y": 5,
                   None: 10}


@dataclass
class Series:
    values: np.ndarray
    index: pd.Index
    name: str
    unit: Optional[str]                 # key of UNITS, None = row order
    label: str                          # "monthly", "in row order", ...
    season: int                         # 1 = no seasonal pattern assumed
    time_column: Optional[str]
    notes: List[str] = field(default_factory=list)
    filled: int = 0


def _parse_times(s: pd.Series) -> Optional[pd.Series]:
    """The column as timestamps, or None if it is not a time column."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    name = str(s.name).lower()
    vals = s.dropna()
    if vals.empty:
        return None
    if pd.api.types.is_numeric_dtype(s):
        # Whole years such as 1990, 1991, ...
        if name in ("year", "yr") and tg._is_integer_valued(vals) and \
                vals.between(1000, 3000).all():
            return pd.to_datetime(s.astype("Int64").astype(str), format="%Y",
                                  errors="coerce")
        return None
    text = vals.astype(str).str.strip()
    if text.str.fullmatch(r"-?\d+(\.\d+)?").all():
        return None                     # plain numbers, not dates
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    if parsed.notna().mean() < 0.95:
        return None
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[parsed.index] = parsed
    return out


def time_columns(df: pd.DataFrame) -> List[str]:
    found = [str(c) for c in df.columns if _parse_times(df[c]) is not None]
    # Prefer columns named like dates.
    return sorted(found, key=lambda c: (str(c).lower() not in TIME_NAMES,
                                        found.index(c)))


def check_value_column(s: pd.Series, time_cols: List[str]) -> tuple:
    """(suitable, note, label) for a column to forecast."""
    name = str(s.name)
    if name in time_cols:
        return False, "This is the time column.", f"{name} - dates/times"
    values = numeric_values(s)
    info = tg.describe_column(s)
    if values is None:
        return False, "It is not numeric; only numbers can be forecast.", \
            info.label
    v = values.dropna()
    if v.nunique() < 3:
        return False, "It has fewer than three different values.", info.label
    return True, "", (f"{name} - numbers from {tg._fmt(v.min())} to "
                      f"{tg._fmt(v.max())}")


def suggest_value(df: pd.DataFrame) -> Optional[str]:
    tcols = time_columns(df)
    ok = [str(c) for c in df.columns if check_value_column(df[c], tcols)[0]]
    return ok[-1] if ok else None


def _unit_from_spacing(times: pd.Series) -> Optional[str]:
    days = times.sort_values().diff().dropna().dt.total_seconds() / 86400
    if days.empty:
        return None
    d = float(days.median())
    if abs(d - 1 / 24) < 0.2 / 24:
        return "h"
    if 0.8 <= d <= 1.2:
        weekend = times.dt.dayofweek.isin([5, 6]).any()
        return "D" if weekend else "B"
    if 6 <= d <= 8:
        return "W"
    if 27 <= d <= 32:
        return "M"
    if 85 <= d <= 95:
        return "Q"
    if 360 <= d <= 370:
        return "Y"
    return None


def prepare_series(df: pd.DataFrame, value: str,
                   time_column: Optional[str] = None) -> Series:
    if value not in df.columns:
        raise ValueError(f"Column '{value}' was not found.")
    tcols = time_columns(df)
    ok, note, _ = check_value_column(df[value], tcols)
    if not ok:
        raise ValueError(f"'{value}' cannot be forecast. {note}")
    y = numeric_values(df[value])
    notes: List[str] = []
    if time_column is None and tcols:
        time_column = tcols[0]
    if time_column is None:
        vals = y.dropna().to_numpy(dtype=float)
        dropped = int(y.isna().sum())
        if dropped:
            notes.append(f"{dropped} empty value(s) were skipped.")
        notes.append("No date or time column was found, so the rows are "
                     "taken as equally spaced, in file order, and no "
                     "seasonal pattern is assumed.")
        if len(vals) < MIN_POINTS:
            raise ValueError(f"Forecasting needs at least {MIN_POINTS} "
                             f"values; {len(vals)} were found.")
        return Series(vals, pd.RangeIndex(len(vals)), value, None,
                      "in row order", 1, None, notes)

    times = _parse_times(df[time_column])
    if times is None:
        raise ValueError(f"'{time_column}' could not be read as dates.")
    frame = pd.DataFrame({"t": times, "y": y}).dropna(subset=["t"])
    if len(frame) < len(df):
        notes.append(f"{len(df) - len(frame)} row(s) without a readable date "
                     "were skipped.")
    unit = _unit_from_spacing(frame["t"])
    if unit is None:
        raise ValueError("The dates are not evenly spaced (hourly, daily, "
                         "weekly, monthly, quarterly or yearly), so they "
                         "cannot be used for forecasting.")
    label, season, alias = UNITS[unit]
    period = frame["t"].dt.to_period("D" if unit == "B" else alias)
    frame["t"] = period.dt.to_timestamp()
    dup = int(frame["t"].duplicated().sum())
    if dup:
        notes.append(f"{dup} repeated time point(s) were combined by "
                     "averaging their values.")
    s = frame.groupby("t")["y"].mean().sort_index()
    if unit == "B":
        full = pd.bdate_range(s.index.min(), s.index.max())
    else:
        full = pd.period_range(s.index.min(), s.index.max(),
                               freq=alias).to_timestamp()
    s = s.reindex(full)
    missing = int(s.isna().sum())
    if missing / len(s) > MAX_MISSING_SHARE:
        raise ValueError(f"{missing} of {len(s)} time points have no value "
                         "(more than 20%); the series is too incomplete to "
                         "forecast.")
    if s.iloc[0] != s.iloc[0]:          # leading gap: start at first value
        first = s.first_valid_index()
        lead = int(s.index.get_loc(first))
        s = s.loc[first:]
        missing -= lead
        notes.append(f"{lead} time point(s) before the first value were "
                     "left out.")
    if missing:
        s = s.ffill()
        notes.append(f"{missing} missing time point(s) were filled with the "
                     "last known value (no later values are used).")
    if len(s) < MIN_POINTS:
        raise ValueError(f"Forecasting needs at least {MIN_POINTS} time "
                         f"points; {len(s)} were found.")
    if season > 1 and len(s) < 3 * season:
        notes.append(f"The series is shorter than three {label} cycles of "
                     f"{season}, so no seasonal pattern is assumed.")
        season = 1
    return Series(s.to_numpy(dtype=float), s.index, value, unit, label,
                  season, time_column, notes, missing)


def default_horizon(series: Series) -> int:
    h = DEFAULT_HORIZON.get(series.unit, 10)
    return int(max(1, min(h, len(series.values) // 10)))


def load_demo() -> pd.DataFrame:
    """Monthly mean CO2 at Mauna Loa, 1958-2001 (weekly data shipped with
    statsmodels, averaged per month; 5 months have no measurement)."""
    import statsmodels.api as sm
    d = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean()
    return pd.DataFrame({"month": d.index.strftime("%Y-%m"),
                         "CO2_ppm": d.to_numpy()})


DEMO_NAME = "Mauna Loa CO2 demonstration (monthly)"
DEMO_CITATION = ("Keeling, C. D., & Whorf, T. P. Atmospheric CO2 records from "
                 "sites in the SIO air sampling network, Mauna Loa "
                 "Observatory, 1958-2001 (as distributed with statsmodels).")
