"""Discover locally shipped modules from manifests, without network access."""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path

class Registry:
    def __init__(self, folder):
        self.manifests = {}
        for path in sorted(Path(folder).glob('*.json')):
            manifest = json.loads(path.read_text(encoding='utf-8'))
            if manifest.get('protocol_version') != 1:
                raise ValueError(f'Unsupported module protocol in {path.name}')
            module_id = manifest['id']
            if module_id in self.manifests:
                raise ValueError(f'Duplicate module id: {module_id}')
            self.manifests[module_id] = manifest
        if not self.manifests:
            raise ValueError('No analysis modules were found.')

    def load(self, module_id):
        if module_id not in self.manifests:
            raise ValueError(f'Unknown analysis module: {module_id}')
        manifest = self.manifests[module_id]
        module_name, class_name = manifest['adapter'].split(':', 1)
        return getattr(importlib.import_module(module_name), class_name)(manifest)

    def describe(self):
        # One module that cannot load (for example a missing or damaged
        # dependency) must not hide the others. It is still listed, with the
        # reason in plain words, so the desktop can show why it is unavailable.
        modules = []
        for key, manifest in self.manifests.items():
            try:
                modules.append(self.load(key).describe())
            except Exception as exc:
                reason = unavailable_reason(manifest, exc)
                print(f"The {manifest.get('title', key)} module could not be loaded: {reason}")
                modules.append(dict(id=key, title=manifest.get('title', key),
                                    description=manifest.get('description', ''),
                                    input_kind=manifest.get('input_kind', ''),
                                    package=manifest.get('package', ''),
                                    package_version=manifest.get('package_version', ''),
                                    unavailable_reason=reason))
        return modules


def unavailable_reason(manifest, exc):
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        package = exc.name.split('.')[0]
        return (f"the Python package '{package}' is not installed for the Python the application uses "
                f"({sys.executable}). Install it there, or point EASYRESEARCH_PYTHON to a Python that has it.")
    return str(exc)
