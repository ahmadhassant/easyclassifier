"""One request per process: catalog, inspection, or a cancellable analysis job."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys
import traceback
import warnings

# The embedded interpreter has an isolated search path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from errors import plain_message
from protocol import Emitter, ProgressStream, VERSION
from registry import Registry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-dir', required=True)
    args = parser.parse_args()
    # The protocol is UTF-8 whatever the Windows code page, so labels such as
    # R² and CO₂ and file paths in Arabic travel intact in both directions.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    request = None
    emitter = Emitter('invalid-request')
    try:
        line = sys.stdin.readline()
        if len(line) > 1_000_000:
            raise ValueError('The request is too large.')
        request = json.loads(line)
        emitter = Emitter(str(request.get('request_id', 'invalid-request')))
        if request.get('protocol_version') != VERSION:
            raise ValueError('Unsupported desktop protocol version.')
        os.environ.setdefault('MPLBACKEND', 'Agg')
        warnings.filterwarnings('ignore')
        sink = ProgressStream(emitter)
        with redirect_stdout(sink), redirect_stderr(sink):
            registry = Registry(args.manifest_dir)
            action = request.get('action')
            if action == 'catalog':
                result = dict(modules=registry.describe())
            else:
                module = registry.load(request.get('module_id'))
                if action == 'inspect':
                    result = module.inspect(request.get('dataset', ''))
                elif action == 'run':
                    result = module.run(request, emitter)
                else:
                    raise ValueError(f'Unknown action: {action}')
            sink.flush()
        emitter.send('completed', result=result)
        return 0
    except Exception as exc:
        emitter.send('error', message=plain_message(exc), detail=traceback.format_exc())
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
