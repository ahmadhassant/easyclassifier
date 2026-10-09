"""Plain-language messages for errors shown in the desktop's status line.

The technical traceback is always sent as well (it goes to the activity log),
so nothing is lost; this only decides the one sentence a beginner reads.
Messages written by the analysis code itself are already plain and are kept.
"""
from __future__ import annotations

import os
import zipfile

# Code whose messages are written for users (kept as they are).
OWN_CODE = tuple(os.sep + name + os.sep for name in ('easyresearch', 'easyclassifier', 'adapters', 'worker'))


def _path(exc):
    if getattr(exc, 'filename', None):
        return str(exc.filename)
    if len(getattr(exc, 'args', ())) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    return ''


def _raised_by_own_code(exc):
    tb = exc.__traceback__
    last = None
    while tb is not None:
        last = tb.tb_frame.f_code.co_filename
        tb = tb.tb_next
    last = (last or '').replace('/', os.sep).lower()
    return any(part in last for part in OWN_CODE)


def plain_message(exc: BaseException) -> str:
    try:
        import pandas as pd
        empty, parser = pd.errors.EmptyDataError, pd.errors.ParserError
    except Exception:  # noqa: BLE001
        empty = parser = ()
    text = str(exc)
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        import sys
        return (f"The Python package '{exc.name.split('.')[0]}' is not installed for the Python the application "
                f'uses ({sys.executable}). Reinstall the complete application folder, or for a development build '
                'run desktop\\build.ps1 with a Python that has it.')
    if isinstance(exc, FileNotFoundError):
        return (f'The file or folder was not found: {_path(exc)}. It may have been moved, renamed or deleted; '
                'choose it again.')
    if isinstance(exc, PermissionError):
        return (f'Windows did not allow access to {_path(exc)}. If the file is open in Excel or another program, '
                'close it and try again; for results, choose a folder you can save to (such as Documents).')
    if isinstance(exc, IsADirectoryError):
        return f'{_path(exc)} is a folder. Choose a file inside it.'
    if empty and isinstance(exc, empty):
        return 'The file is empty. Choose a file with a header row and at least one row of data.'
    if isinstance(exc, UnicodeDecodeError):
        return ('The file could not be read as a text table. If it is a spreadsheet, open it in Excel and save it '
                "as 'CSV UTF-8' or as an Excel workbook (.xlsx), then choose that file.")
    if parser and isinstance(exc, parser):
        return ('The file could not be read as a table: its rows do not have the same number of columns. '
                'Open it in Excel, check for extra separators or notes below the data, and save it again.')
    if isinstance(exc, zipfile.BadZipFile) or 'Excel file format cannot be determined' in text \
            or type(exc).__name__ == 'InvalidFileException':
        return ('The file is not a valid Excel workbook. Open it in Excel and save it again as .xlsx, '
                'or save it as CSV.')
    if isinstance(exc, MemoryError):
        return ('The computer ran out of memory. Try fewer models (deep models need the most), '
                'or a smaller file.')
    if isinstance(exc, OSError):
        reason = exc.strerror or text
        where = f' ({_path(exc)})' if _path(exc) else ''
        return f'Windows reported a problem with a file or folder{where}: {reason}.'
    if isinstance(exc, (ValueError, RuntimeError)) and _raised_by_own_code(exc):
        return text
    return (f'An unexpected problem stopped the analysis ({type(exc).__name__}: {text}). '
            'The technical details are in the activity log.')
