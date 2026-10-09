"""Versioned JSON-lines protocol; stdout is reserved for events."""
from __future__ import annotations
import json
import sys

VERSION = 1

class Emitter:
    def __init__(self, request_id, stream=None):
        self.request_id = request_id
        self.stream = stream or sys.stdout
        self.tracker = None          # progress.ProgressTracker, set by a module's run()

    def progress(self, message, **payload):
        """A progress message; with a tracker, also the fraction of the work done."""
        fraction = self.tracker.message(message) if self.tracker is not None else None
        if fraction is not None:
            payload['fraction'] = fraction
        self.send('progress', message=message, **payload)

    def epoch(self):
        """Called after each deep-learning epoch: moves the progress bar (no message)."""
        fraction = self.tracker.epoch() if self.tracker is not None else None
        if fraction is not None:
            self.send('progress', message='', fraction=fraction)

    def send(self, kind, **payload):
        event = dict(protocol_version=VERSION, request_id=self.request_id, type=kind, **payload)
        self.stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()

class ProgressStream:
    """Keep the package's terminal output away from the protocol stream."""
    def __init__(self, emitter):
        self.emitter = emitter
        self.pending = ''

    def write(self, text):
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            line = line.strip()
            if line:
                self.emitter.progress(line)
        return len(text)

    def flush(self):
        if self.pending.strip():
            self.emitter.progress(self.pending.strip())
        self.pending = ''

    def isatty(self):
        return False
