"""Terminal user-interface helpers for EasyClassifier.

All user interaction goes through these functions so the rest of the code
never touches ``input``/``print`` directly. This keeps the wizard logic clean
and makes the interface easy to swap out later (e.g. for a GUI).

The helpers are intentionally plain-text and dependency-free so they work in
any terminal.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Basic output
# --------------------------------------------------------------------------- #

WIDTH = 66


def _line(char: str = "-") -> str:
    return char * WIDTH


def rule(char: str = "-") -> None:
    print(_line(char))


def banner(title: str, subtitle: str = "") -> None:
    """Print a boxed banner."""
    print()
    rule("=")
    print(title.center(WIDTH))
    if subtitle:
        print(subtitle.center(WIDTH))
    rule("=")


def header(text: str) -> None:
    """Section header."""
    print()
    rule()
    print(text)
    rule()


def info(text: str) -> None:
    for line in textwrap.wrap(text, WIDTH) or [""]:
        print(line)


def note(text: str) -> None:
    print(f"  * {text}")


def success(text: str) -> None:
    print(f"[OK] {text}")


def warn(text: str) -> None:
    print(f"[!]  {text}")


def error(text: str) -> None:
    print(f"[X]  {text}")


def blank() -> None:
    print()


def pause(msg: str = "Press ENTER to continue...") -> None:
    try:
        input("\n" + msg)
    except EOFError:
        pass


# --------------------------------------------------------------------------- #
# Input primitives
# --------------------------------------------------------------------------- #

def _read(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        # Non-interactive / piped input exhausted.
        print()
        raise KeyboardInterrupt


def ask_text(prompt: str, default: Optional[str] = None,
             allow_empty: bool = False) -> str:
    """Free-text input with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = _read(f"{prompt}{suffix}\n>> ").strip()
        if not value:
            if default is not None:
                return default
            if allow_empty:
                return ""
            error("Please enter a value.")
            continue
        return value


def ask_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    """Yes/No question. Returns True for yes."""
    if default is True:
        hint = " [Y/n]"
    elif default is False:
        hint = " [y/N]"
    else:
        hint = " [y/n]"
    while True:
        value = _read(f"{prompt}{hint}\n>> ").strip().lower()
        if not value and default is not None:
            return default
        if value in ("y", "yes", "1"):
            return True
        if value in ("n", "no", "2"):
            return False
        error("Please answer y or n.")


def menu(prompt: str, options: Sequence[str],
         default: Optional[int] = None,
         allow_help: bool = True,
         help_keys: Optional[Sequence[str]] = None) -> int:
    """Numbered single-choice menu.

    Returns the zero-based index of the chosen option.
    Options are displayed starting at 1. If ``help_keys`` is provided the user
    may type ``?N`` to see help for option N (handled by the caller via the
    returned sentinel).
    """
    print()
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    hint = ""
    if default is not None:
        hint = f" (default {default + 1})"
    if allow_help:
        hint += "  |  type ?N for help on option N"
    while True:
        raw = _read(f"Choose 1-{len(options)}{hint}\n>> ").strip()
        if not raw and default is not None:
            return default
        if allow_help and raw.startswith("?"):
            num = raw[1:].strip()
            if num.isdigit() and 1 <= int(num) <= len(options):
                return -int(num)  # sentinel: caller shows help
            error("Type ? followed by a valid option number, e.g. ?2")
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        error(f"Please enter a number between 1 and {len(options)}.")


def multi_select(prompt: str, options: Sequence[str],
                 default_all: bool = True,
                 default: Optional[List[int]] = None) -> List[int]:
    """Multiple-choice checkbox menu.

    User enters comma-separated numbers, 'all', or 'none'. ENTER gives
    ``default`` if provided, otherwise all (``default_all``) or none.
    Returns a list of zero-based indices.
    """
    if default is None:
        default = list(range(len(options))) if default_all else []
    print()
    print(prompt)
    for i, opt in enumerate(options, start=1):
        mark = "x" if i - 1 in default else " "
        print(f"  [{mark}] {i}. {opt}")
    if len(default) == len(options):
        default_hint = "ENTER = all"
    elif not default:
        default_hint = "ENTER = none"
    else:
        default_hint = "ENTER = " + ",".join(str(i + 1) for i in default)
    while True:
        raw = _read(
            f"Enter numbers separated by commas, or 'all' / 'none' "
            f"[{default_hint}]\n>> "
        ).strip().lower()
        if not raw:
            return list(default)
        if raw == "all":
            return list(range(len(options)))
        if raw == "none":
            return []
        try:
            picks = sorted({
                int(p) - 1 for p in raw.replace(" ", "").split(",") if p != ""
            })
        except ValueError:
            error("Please enter numbers separated by commas.")
            continue
        if all(0 <= p < len(options) for p in picks) and picks:
            return picks
        error(f"Please enter numbers between 1 and {len(options)}.")


def choose_from_list(prompt: str, items: Sequence[str],
                     extra_options: Sequence[str] = ()) -> Tuple[int, bool]:
    """Present a list of items plus optional extra menu entries.

    Returns ``(index, is_extra)``. If ``is_extra`` is True the index refers to
    ``extra_options``; otherwise it refers to ``items``.
    """
    combined = list(items) + list(extra_options)
    idx = menu(prompt, combined, allow_help=False)
    if idx >= len(items):
        return idx - len(items), True
    return idx, False


def pick_from_long_list(prompt: str, labels: Sequence[str],
                        names: Sequence[str],
                        default: Optional[int] = None,
                        marks: Optional[dict] = None,
                        page_size: int = 20) -> int:
    """Choose one item from a possibly very long list.

    Numbers stay the same while paging or searching, so the user can always
    type the number they saw. Typing text shows only the items whose name
    contains that text. Returns the zero-based index into ``labels``.
    """
    marks = marks or {}
    shown = list(range(len(labels)))
    page = 0
    while True:
        pages = max(1, (len(shown) + page_size - 1) // page_size)
        page = min(page, pages - 1)
        print()
        print(prompt)
        visible = shown[page * page_size:(page + 1) * page_size]
        for i in visible:
            print(f"  {i + 1:>3}. {labels[i]}{marks.get(i, '')}")
        if default is not None and default not in visible:
            print(f"  Suggested: {default + 1}. {labels[default]}")
        hints = []
        if default is not None:
            hints.append(f"ENTER = {default + 1}, the suggested one")
        if pages > 1:
            hints.append(f"page {page + 1} of {pages}: 'n' next, 'p' back")
        if len(labels) > page_size or len(shown) < len(labels):
            hints.append("type part of a name to search"
                         + (", 'all' to list everything"
                            if len(shown) < len(labels) else ""))
        if hints:
            print("  (" + "; ".join(hints) + ")")
        raw = _read("Choose a number\n>> ").strip()
        low = raw.lower()
        if not raw:
            if default is not None:
                return default
            error("Please type a number.")
        elif raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(labels):
                return n - 1
            error(f"Please enter a number between 1 and {len(labels)}.")
        elif low in ("n", "next"):
            page = page + 1 if page + 1 < pages else page
        elif low in ("p", "back", "prev"):
            page = max(0, page - 1)
        elif low == "all":
            shown, page = list(range(len(labels))), 0
        else:
            found = [i for i, nm in enumerate(names) if low in str(nm).lower()]
            if found:
                shown, page = found, 0
            else:
                error(f"No column name contains '{raw}'.")
