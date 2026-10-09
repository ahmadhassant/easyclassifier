"""How much of an analysis is done, for the desktop's progress bar.

The analysis cores already announce every step in a message ("Training
Random Forest ...", "fold 3: ... selected", "Drawing figures ..."). Each
module's adapter knows which steps to expect from the chosen models and
evaluation method, and describes them as rules:

* ``start`` - the message starts a step of that many units (the step before
  it is then complete);
* ``end``   - the message completes that many units (a nested-CV fold);
* ``expect``- the message announces steps of that size without completing
  anything (it lets the bar move during the first fold).

Between messages, deep-learning epochs move the bar within the current step,
never beyond it, so long trainings visibly progress. The fraction never goes
backwards and stays below 1 until the analysis has finished.
"""
from __future__ import annotations

import math
import re
import time
from typing import Iterable, Optional, Tuple

EPOCH_SCALE = 150          # epochs for the within-step movement to reach ~63%
MIN_INTERVAL = 0.5         # seconds between epoch updates sent to the desktop


class ProgressTracker:
    def __init__(self, rules: Iterable[Tuple[str, float, str]], total: float):
        self.rules = [(re.compile(pattern), float(units), kind) for pattern, units, kind in rules]
        self.total = max(1.0, float(total))
        self.done = 0.0
        self.current = 0.0           # units of the step in progress
        self.open = False            # a 'start' step is in progress
        self.epochs = 0
        self.last = 0.0
        self.sent_at = 0.0

    def _fraction(self) -> float:
        within = self.current * 0.9 * (1 - math.exp(-self.epochs / EPOCH_SCALE))
        value = min(0.99, (self.done + within) / self.total)
        self.last = max(self.last, value)
        return round(self.last, 4)

    def message(self, text: str) -> Optional[float]:
        for pattern, units, kind in self.rules:
            if pattern.search(text):
                if kind == 'start':
                    if self.open:
                        self.done += self.current
                    self.current, self.open = units, True
                elif kind == 'end':
                    if self.open:
                        self.done += self.current
                    self.done += units
                    self.current, self.open = units, False
                else:                                        # expect
                    if self.open:
                        self.done += self.current
                    self.current, self.open = units, False
                self.done = min(self.done, self.total)
                self.epochs = 0
                return self._fraction()
        return None

    def epoch(self) -> Optional[float]:
        """One training epoch of a deep model; returns a fraction at most
        every MIN_INTERVAL seconds."""
        self.epochs += 1
        now = time.monotonic()
        if now - self.sent_at < MIN_INTERVAL:
            return None
        self.sent_at = now
        return self._fraction()


TAIL = [(r'^Measuring which columns matter', 1, 'start'), (r'^Drawing figures', 1, 'start'),
        (r'^Training the final', 1, 'start'), (r'^Writing the report', 1, 'start')]


def table_rules(models: int, method: Optional[str], references: int = 0, figures: bool = True,
                training=r'^Training (?!the final).+ \.\.\.$', importance: bool = True, final_fit: bool = True):
    """Rules for the comparison-then-final-score analyses (classification,
    regression, signals): one step per model, then nested CV (five folds,
    each repeating the comparison) or one score on the final test set, then
    the tail steps."""
    rules = [(training, 1, 'start')]
    total = models + references
    if method == 'nested':
        rules += [(r'^Running nested', models, 'expect'), (r'fold \d+: .+ selected', models, 'end')]
        total += 5 * models
    elif method == 'final_test':
        rules += [(r'^Scoring .+', 1, 'start')]
        total += 1
    rules += TAIL
    total += (1 if importance else 0) + (1 if figures else 0) + (1 if final_fit else 0) + 1
    return rules, total


def forecast_rules(models: int, references: int = 2, figures: bool = True):
    rules = [(r'^Training (?!the final).+ at \d+ comparison origins', 1, 'start'),
             (r'^Scoring .+ in the final period', 1, 'start')] + TAIL
    total = models + references + 1 + 1 + (1 if figures else 0) + 1
    return rules, total
