"""Things a first-time user runs into: file paths, Excel files, options."""

import os

import numpy as np
import pandas as pd
import pytest

from easyclassifier import __version__
from easyclassifier.__main__ import main
from easyclassifier.dataset import clean_path, load_csv


@pytest.fixture
def data_file(tmp_path):
    folder = tmp_path / "My Data"
    folder.mkdir()
    path = folder / "survey results.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]}).to_csv(path,
                                                                 index=False)
    return str(path)


def test_paths_dragged_into_the_window(data_file):
    assert clean_path(f'"{data_file}"') == data_file          # Windows
    assert clean_path(f"'{data_file}'") == data_file
    assert clean_path(f"& '{data_file}'") == data_file        # PowerShell
    assert clean_path(f"  {data_file}  ") == data_file
    if os.sep == "/":                                          # macOS/Linux
        assert clean_path(data_file.replace(" ", "\\ ")) == data_file


def test_extension_can_be_left_out(data_file):
    assert clean_path(data_file[:-4]) == data_file


def test_excel_files_load(tmp_path):
    path = tmp_path / "data.xlsx"
    df = pd.DataFrame({"a": [1, 2, 3], "Outcome": ["Yes", "No", "Yes"]})
    df.to_excel(path, index=False)
    loaded = load_csv(str(path))
    assert loaded.equals(df)


def test_semicolon_csv_with_decimal_commas(tmp_path):
    path = tmp_path / "europe.csv"
    path.write_text("weight;height;city;Outcome\n"
                    "72,5;1,80;Amman;Yes\n"
                    "64;1,65;Irbid;No\n"
                    ";1,70;Zarqa;Yes\n", encoding="utf-8")
    df = load_csv(str(path))
    assert list(df.columns) == ["weight", "height", "city", "Outcome"]
    assert df["weight"].tolist()[:2] == [72.5, 64.0]
    assert df["weight"].isna().iloc[2]
    assert df["height"].tolist() == [1.80, 1.65, 1.70]
    assert df["city"].tolist() == ["Amman", "Irbid", "Zarqa"]   # unchanged


def test_old_xls_gets_a_clear_message(tmp_path):
    path = tmp_path / "old.xls"
    path.write_bytes(b"not really excel")
    with pytest.raises(ValueError, match="save the sheet as .xlsx or .csv"):
        load_csv(str(path))


def test_version_and_help(capsys):
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
    assert main(["--help"]) == 0
    assert "python -m easyclassifier" in capsys.readouterr().out
    assert main(["--nonsense"]) == 2


def test_identical_rows_kept_when_expected_by_chance():
    from easyclassifier.preprocessing import duplicates_expected_by_chance
    rng = np.random.default_rng(0)
    survey = pd.DataFrame({"sex": rng.choice(["F", "M"], 500),
                           "score": rng.integers(0, 10, 500),
                           "answer": rng.choice(["yes", "no"], 500)})
    assert duplicates_expected_by_chance(survey)          # 40 combinations
    lab = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    assert not duplicates_expected_by_chance(lab)         # copies = errors
