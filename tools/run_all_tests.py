"""Run every automated check of the project with one command.

    python tools/run_all_tests.py            # all suites
    python tools/run_all_tests.py --quick    # skip the slow desktop suite

Suites
------
1. EasyClassifier            pytest tests
2. EasyResearch cores        pytest easyresearch/tests
3. Desktop adapters          the desktop's Python tests (the same code path
                             as the application); on Windows with the .NET
                             SDK installed, desktop/build.ps1 is run instead,
                             which also builds the C# application and runs
                             the C# integration checks.

Use the Python environment that has both packages installed in editable mode
(see docs/DEVELOPMENT.md). The script stops nothing early: every suite runs,
and a summary at the end lists what passed and failed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def suites(quick: bool):
    py = sys.executable
    yield 'EasyClassifier', [py, '-m', 'pytest', '-q', 'tests'], ROOT
    yield 'EasyResearch cores', [py, '-m', 'pytest', '-q', 'easyresearch/tests'], ROOT
    if quick:
        return
    if os.name == 'nt' and shutil.which('dotnet'):
        yield ('Desktop (C# build, adapters, integration)',
               ['powershell', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT / 'desktop' / 'build.ps1'),
                '-Python', py], ROOT)
    else:
        yield 'Desktop adapters', [py, '-m', 'unittest', 'discover', '-s', 'tests/python'], ROOT / 'desktop'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--quick', action='store_true', help='skip the slow desktop suite')
    args = parser.parse_args(argv)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    results = []
    for name, command, cwd in suites(args.quick):
        print(f'\n=== {name} ===', flush=True)
        start = time.time()
        code = subprocess.call(command, cwd=cwd, env=env)
        results.append((name, code == 0, time.time() - start))
    print('\nSummary')
    for name, ok, seconds in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({seconds / 60:.1f} min)")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
