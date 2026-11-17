"""Verify a pulled WP4 tree before the boxes are stopped.

Checks every fit record: status ok, its embedding file present, and the embedding's sha256 equal to the one the
record holds. With --stage2, also lists every job that stage2_r3.py plans from selected_configs.json and reports
the ones that have no record. Exit code 0 only if nothing is missing or wrong.

Usage: python verify_w4.py <W4 folder> [--stage2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'external'))
sys.path.insert(0, str(HERE))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


ARMS = {'graphst': ('released', 'no_contrast', 'identity_encoder', 'permuted_xy'),
        'stagate': ('released', 'uniform_attention', 'identity_graph', 'permuted_xy'),
        'tessera': ('released', 'no_gate', 'no_contrast', 'self_only', 'permuted_xy_no_contrast', 'contrast_no_mp')}


def planned_stage2(w4: Path) -> set[tuple]:
    """(dataset, encoder, input_mode, arm, config, seed) for every job stage2_r3.py plans, expanded with the
    runner's own rules (run_external_r3.plan) without importing the encoder adapters."""
    import argparse as ap_
    import common_external as ce
    import stage2_r3 as s2
    p = ap_.ArgumentParser()
    p.add_argument('--encoders', nargs='+'); p.add_argument('--datasets', nargs='+'); p.add_argument('--arms', nargs='+')
    p.add_argument('--configs', nargs='+'); p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--input-modes', nargs='+', default=['common', 'native']); p.add_argument('--native-arms', nargs='+', default=['released'])
    p.add_argument('--device'); p.add_argument('--out')
    selected = json.loads((w4 / 'selected_configs.json').read_text())
    jobs = set()
    for cmd in s2.commands(selected, sys.executable, str(w4)):
        a = p.parse_args(cmd[2:])
        for enc in a.encoders:
            for ds in a.datasets:
                for mode in a.input_modes:
                    if mode == 'native' and not ce.has_native(ds):
                        continue
                    for arm in (a.arms or ARMS[enc]):
                        if mode == 'native' and arm not in a.native_arms:
                            continue
                        for config in a.configs:
                            for seed in a.seeds:
                                jobs.add((ds, enc, mode, arm, config, int(seed)))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('w4', type=Path)
    ap.add_argument('--stage2', action='store_true')
    args = ap.parse_args()
    problems, have = [], set()
    records = sorted((args.w4 / 'fits').glob('*/*.json'))
    for r in records:
        rec = json.loads(r.read_text())
        key = (rec['dataset'], rec['encoder'], rec['input_mode'], rec['arm'], rec['config'], int(rec['seed']))
        have.add(key)
        if rec.get('status') != 'ok':
            problems.append(f'not ok: {r.name} ({rec.get("status")})')
            continue
        emb = args.w4 / rec['embedding_file']
        if not emb.exists():
            problems.append(f'missing embedding: {rec["embedding_file"]}')
        elif sha(emb) != rec['embedding_sha256']:
            problems.append(f'hash mismatch: {rec["embedding_file"]}')
    print(f'{len(records)} records checked, {len(problems)} problems')
    if args.stage2:
        missing = sorted(planned_stage2(args.w4) - have)
        print(f'stage-2 jobs without a record: {len(missing)}')
        problems += [f'no record: {m}' for m in missing]
    for p in problems[:30]:
        print(' ', p)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
