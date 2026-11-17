"""Stage 2 of WP4 on the box: every arm at each unit's selected configurations, seeds 1-5.

For each dataset and encoder in selected_configs.json: all arms of the encoder at the
selected configurations; the released arm at the default configuration; and, where a
native input exists (NSCLC, HER2ST), the released arm on native input. Seeds 1-5. The
runner's per-job locks and records make this resumable and safe to run in parallel.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / 'external'
DEFAULT = {'graphst': 'k3-e600', 'stagate': 'knn6-e1000', 'tessera': 'k6-w0.5-t0.2-e120'}
NATIVE = {'graphst', 'stagate'}
# GraphST memory accumulates across fits within one process (two stage-1 runners were OOM-killed), so its
# stage-2 work is split into one process per arm, at most 15 fits each.
GRAPHST_ARMS = ('released', 'no_contrast', 'identity_encoder', 'permuted_xy')


def commands(selected: dict, py: str, out: str) -> list[list[str]]:
    sys.path.insert(0, str(EXT))
    import common_external as ce
    cmds = []
    base = [py, str(EXT / 'run_external_r3.py'), '--device', 'cuda', '--out', out, '--seeds', '1', '2', '3', '4', '5']
    for dataset, by_encoder in selected.items():
        for encoder, configs in by_encoder.items():
            if encoder == 'graphst':
                for arm in GRAPHST_ARMS:
                    cmds.append(base + ['--encoders', encoder, '--datasets', dataset, '--configs', *configs,
                                        '--arms', arm, '--input-modes', 'common'])
            else:
                cmds.append(base + ['--encoders', encoder, '--datasets', dataset, '--configs', *configs,
                                    '--input-modes', 'common'])
            cmds.append(base + ['--encoders', encoder, '--datasets', dataset, '--configs', DEFAULT[encoder],
                                '--arms', 'released', '--input-modes', 'common'])
            if encoder in NATIVE and ce.has_native(dataset):
                cmds.append(base + ['--encoders', encoder, '--datasets', dataset, '--configs', DEFAULT[encoder],
                                    '--input-modes', 'native', '--native-arms', 'released'])
    return cmds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()
    selected = json.loads((Path(args.out) / 'selected_configs.json').read_text())
    cmds = commands(selected, args.python, args.out)
    logs = Path(args.out) / 'logs'
    logs.mkdir(exist_ok=True)

    def run(i_cmd):
        i, cmd = i_cmd
        with open(logs / f'stage2-{i:04d}.log', 'w') as log:
            return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(run, enumerate(cmds)))
    print(f'{len(cmds)} runner calls, {sum(c != 0 for c in codes)} non-zero exits')


if __name__ == '__main__':
    main()
