#!/usr/bin/env python3
"""Run preserved producers with explicit portable inputs and a fresh output root.

--check performs only a local dependency/input preflight. Running without --check
executes the selected experiment and may fit models. The wrapper changes paths
and the acquisition adapter only; original producer/protocol bytes are preserved.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ['portable', 'source/scripts', 'source/src',
                 'validation-external', 'validation-extension', 'validation-independent-crc']]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('analysis', choices=['primary', 'transfer', 'her2st', 'enhanced', 'hardening', 'dev', 'breadth', 'crc', 'prepare'])
    parser.add_argument('--output-root', type=Path, required=True, help='Fresh directory outside this extracted package')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='Checksum-bound primary derived input tree')
    parser.add_argument('--raw-dir', type=Path, help='Checksum-bound manually acquired HER2ST files')
    parser.add_argument('--lab-root', type=Path, help='Portable directory containing raw sources in the manifest lab_relative_path layout')
    parser.add_argument('--proxy', help='Optional per-request HTTP proxy for the package-only fetch adapter')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--check', action='store_true', help='No fitting, network access, output creation or source mutation')
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('--output-root must be outside the extracted archive')
    if output.exists():
        parser.error('--output-root must be fresh and must not already exist')
    import common
    import run_experiments as rx
    import run_followup as fu
    import run_transfer as rt
    import run_her2st_validation as hv
    import run_enhanced_targets as enhanced
    import run_extension as extension
    import run_pilot as pilot
    import prepare_data as prepare
    import torch
    from acquire import check, sha
    manifest = json.loads((ROOT / 'provenance/upstream-inputs.json').read_text())
    for path in [ROOT / 'results/run-lock.json', ROOT / 'results/transfer/run-lock.json']:
        for name, digest in common.read(path)['code_sha256'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Frozen source changed: ' + name)
    data = args.data_dir.resolve()
    if args.analysis in ['primary', 'transfer', 'dev']:
        for name in common.OBJECTS:
            meta = common.read(data / name / 'input.json')
            if sha(data / name / 'input.npz') != meta['matrix_sha256']:
                raise ValueError('Derived input changed: ' + name)
    if args.analysis in ['her2st', 'enhanced', 'hardening']:
        if args.raw_dir is None:
            parser.error('This analysis requires --raw-dir, verified by portable/acquire.py')
        for row in manifest['acquisition']['her2st']:
            check(args.raw_dir / row['filename'], row)
        if (args.raw_dir / 'J1.tsv.gz').exists():
            raise ValueError('Historical J1 counts were unavailable; no replacement is checksum-bound')
    if args.analysis in ['breadth', 'prepare']:
        if args.lab_root is None:
            parser.error('This analysis requires --lab-root matching the recorded relative source layout')
        rows = manifest['breadth_raw_sources'] if args.analysis == 'breadth' else manifest['primary_raw_sources']
        for row in rows:
            path = args.lab_root / row['lab_relative_path']
            if sha(path) != row['sha256']:
                raise ValueError('Raw source changed: ' + row['lab_relative_path'])
    if args.analysis == 'crc':
        for section in pilot.SECTIONS:
            base = ROOT / 'validation-independent-crc/inputs' / section
            if sha(base / 'input.npz') != common.read(base / 'input.json')['input_sha256']:
                raise ValueError('CRC derived input changed')
    if args.check:
        print(json.dumps({'status': 'passed', 'analysis': args.analysis, 'fresh_output': str(output),
                         'checks': 'source hashes, dependency imports and input hashes', 'fits': 0,
                         'network_requests': 0, 'outputs_created': 0}, indent=2))
        return
    output.mkdir(parents=True)
    start = {'analysis': args.analysis, 'wrapper_sha256': sha(Path(__file__)),
             'data_dir': str(data), 'raw_dir': str(args.raw_dir) if args.raw_dir else None,
             'lab_root': str(args.lab_root) if args.lab_root else None, 'device': args.device,
             'environment': rx.environment(), 'scope': 'New execution in fresh output directory; original research bytes remain unchanged'}
    common.dump(output / 'portable-execution-start.json', start)
    if args.proxy:
        os.environ['TESSERA_FETCH_PROXY'] = args.proxy
    # These imported module attributes configure I/O only. Hash-bound original
    # source, numerical estimator, graph, K, target definitions and protocols stay intact.
    common.DATA = rx.DATA = rt.DATA = fu.DATA = data
    common.RESULTS = rx.RESULTS = rt.RESULTS = output / 'results'
    hv.RAW = args.raw_dir.resolve() if args.raw_dir else ROOT / 'validation-external/raw'
    hv.GATE_PATH = ROOT / 'portable/fetch_adapter.py'
    hv.OUT = output / 'validation-external/results'
    enhanced.OUT = hv.OUT / 'enhanced'
    extension.OUT = output / 'validation-extension/results'
    sys.argv = [str(Path(__file__))]
    if args.analysis == 'primary':
        sys.argv += ['--device', args.device]
        rx.main()
    elif args.analysis == 'transfer':
        sys.argv += ['--device', args.device]
        rt.main()
    elif args.analysis == 'her2st':
        hv.main()
    elif args.analysis == 'enhanced':
        enhanced.main()
    elif args.analysis in ['hardening', 'dev', 'breadth']:
        if args.analysis == 'breadth':
            extension.BREADTH = {row['object']: (args.lab_root / row['lab_relative_path'], row['label_column'])
                                 for row in manifest['breadth_raw_sources']}
        sys.argv += [args.analysis]
        extension.main()
    elif args.analysis == 'prepare':
        prepare.LAB = args.lab_root.resolve()
        prepare.DATA = output / 'data'
        prepare.main()
    elif args.analysis == 'crc':
        # Refit from frozen included PCA inputs. Raw assay preprocessing is a
        # separate step documented in REPRODUCE.md, not silently regenerated.
        original = ROOT / 'validation-independent-crc'
        pilot.HERE = output / 'validation-independent-crc'
        pilot.HERE.mkdir()
        pilot.OUT = pilot.HERE / 'results'
        pilot.OUT.mkdir()
        for name in ['PROTOCOL.md', 'protocol-lock.json', 'ANNOTATION_QC_PROTOCOL.md',
                     'ANNOTATION_QC_PROTOCOL.lock.json', 'runtime-decision.json']:
            shutil.copy2(original / name, pilot.HERE / name)
        shutil.copytree(original / 'inputs', pilot.HERE / 'inputs')
        torch.set_num_threads(1)
        for section in pilot.SECTIONS:
            pilot.run(section)
        pilot.summarize()
    common.dump(output / 'portable-execution-completed.json', {**start, 'status': 'completed'})


if __name__ == '__main__':
    main()
