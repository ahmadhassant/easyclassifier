"""Advanced options offered by every task in the desktop's "Advanced options"
section (closed by default, so beginners never need it).

A module descriptor may list:

* ``advanced_choices`` - drop-down choices, the same shape as the
  ``parameters`` returned by inspect (id, label, hint, options, default);
* ``advanced_lists``   - tick lists: id, label, hint, options and ``default``
  (the list of option ids ticked at first).

The run request then holds, in ``settings``, each choice's option id and each
list's ticked ids. Anything left out keeps the usual behaviour.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from easyclassifier.figures import DEFAULT_THEME, FORMATS, THEMES


def choice(id: str, label: str, options, default: str, hint: str = '') -> dict:
    return dict(id=id, label=label, hint=hint, default=default,
                options=[dict(id=k, name=v) for k, v in options])


def tick_list(id: str, label: str, options, default: Iterable[str], hint: str = '') -> dict:
    return dict(id=id, label=label, hint=hint, default=list(default),
                options=[dict(id=k, name=v) for k, v in options])


def figure_options(titles: Dict[str, str], usual: Iterable[str], hint: str = '') -> Tuple[List[dict], List[dict]]:
    """The figure list plus the colour theme and file format choices."""
    lists = [tick_list('figure_keys', 'Figures', titles.items(), usual,
                       hint or 'The usual figures are ticked. Extra figures take a little longer.')]
    choices = [
        choice('figure_theme', 'Figure colours', [(k, f'{t.name} - {t.description}') for k, t in THEMES.items()],
               DEFAULT_THEME, 'Colour-blind safe colours suit most papers; greyscale suits printed journals.'),
        choice('figure_format', 'Figure files', FORMATS.items(), 'png',
               'Vector files (PDF or SVG) stay sharp at any size; journals often ask for them.'),
    ]
    return choices, lists


def read_choice(settings: dict, id: str, allowed: Iterable[str], default: str, what: str) -> str:
    value = settings.get(id, default)
    allowed = list(allowed)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f'Unknown {what}: {value!r}. Choose one of: {", ".join(allowed)}.')
    return value


def read_list(settings: dict, id: str, allowed: Iterable[str], what: str) -> Optional[List[str]]:
    """The ticked ids in their usual order, or None when the list was not sent."""
    if id not in settings or settings[id] is None:
        return None
    value = settings[id]
    allowed = list(allowed)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f'The {what} must be a list of names.')
    unknown = [v for v in value if v not in allowed]
    if unknown:
        raise ValueError(f'Unknown {what}: {", ".join(unknown)}.')
    return [k for k in allowed if k in value]


def read_figures(settings: dict, titles: Dict[str, str], usual: Iterable[str]) -> Tuple[str, str, Optional[List[str]]]:
    """Theme, format and the figures to draw. The usual list (as first ticked)
    means "automatic", so the extras the data call for are still added."""
    theme = read_choice(settings, 'figure_theme', THEMES, DEFAULT_THEME, 'figure colour theme')
    formats = read_choice(settings, 'figure_format', FORMATS, 'png', 'figure file format')
    keys = read_list(settings, 'figure_keys', titles, 'figures')
    if keys is not None and keys == [k for k in titles if k in set(usual)]:
        keys = None
    return theme, formats, keys


def describe_choices(theme: str, formats: str, keys: Optional[List[str]]) -> dict:
    """For settings.json: what was asked for the figures."""
    return dict(figure_theme=theme, figure_format=formats,
                figure_keys='automatic' if keys is None else keys)
