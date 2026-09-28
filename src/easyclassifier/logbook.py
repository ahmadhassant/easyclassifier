"""Simple session log that records every choice and step for reproducibility."""

from __future__ import annotations

import datetime as _dt
from typing import List


class LogBook:
    def __init__(self) -> None:
        self._lines: List[str] = []
        self.add(f"EasyClassifier session started {self._now()}")

    @staticmethod
    def _now() -> str:
        return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add(self, text: str) -> None:
        self._lines.append(f"[{self._now()}] {text}")

    def section(self, title: str) -> None:
        self._lines.append("")
        self._lines.append(f"== {title} ==")

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.text())
        return path
