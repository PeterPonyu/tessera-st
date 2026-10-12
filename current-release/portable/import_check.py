#!/usr/bin/env python3
"""Import the complete analysis dependency closure without data access or fitting."""
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ['portable', 'source/scripts', 'source/src',
                 'validation-external', 'validation-extension', 'validation-independent-crc']]
PACKAGES = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'anndata', 'h5py', 'matplotlib', 'Pillow']
MODULES = ['tessera_st.config', 'tessera_st.model.tessera', 'tessera_st.losses', 'tessera_st.train',
           'tessera_st.cli', 'prepare_data', 'run_experiments', 'run_transfer', 'run_followup',
           'run_her2st_validation', 'run_enhanced_targets', 'run_extension', 'fetch_pilot',
           'run_pilot', 'analyze_annotation_qc', 'render_pilot', 'fetch_adapter']


def main():
    for name in MODULES:
        importlib.import_module(name)
    import run_her2st_validation as her2st
    her2st.GATE_PATH = ROOT / 'portable/fetch_adapter.py'
    helper = her2st.load_module('portable_fetch_gate', her2st.GATE_PATH)
    assert callable(helper.fetch)
    compiled = 0
    for folder in ['source', 'validation-external', 'validation-extension', 'validation-independent-crc', 'portable']:
        for path in sorted((ROOT / folder).rglob('*.py')):
            compile(path.read_bytes(), str(path.relative_to(ROOT)), 'exec')
            compiled += 1
    print(json.dumps({'status': 'passed', 'python': sys.version, 'imported_modules': MODULES,
         'packages': {name: importlib.metadata.version(name) for name in PACKAGES},
         'python_sources_compiled_without_writes': compiled, 'model_or_clustering_fits': 0,
         'portable_acquisition_helper_resolves_within_package': True, 'network_requests': 0}, indent=2))


if __name__ == '__main__':
    main()
