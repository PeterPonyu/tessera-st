#!/usr/bin/env python3
"""Portable checksum + stored-prediction verification, no raw assay or model fit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/generated/field-identified-fixed-backend-20260923-v1'


def main():
    manifest = ROOT / 'RELEASE-MANIFEST.json'
    if manifest.exists():
        for item in json.loads(manifest.read_text())['files']:
            path = ROOT / item['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Release identity changed: ' + item['path'])
    protocol = json.loads((RUN / 'protocol.json').read_text())
    count = 0
    for name in protocol['retained_inputs']:
        folder = RUN / name
        rows = np.load(folder / 'rows.npz', allow_pickle=True)
        # Trusted, checksum-bound author-generated NPZ contains string arrays.
        truth, fields = rows['reference_label'], rows['field']
        src, dst = rows['spatial_src'], rows['spatial_dst']
        assert (src != dst).all() and (fields[src] == fields[dst]).all(), name
        good = truth >= 0
        for method in protocol['methods']:
            for seed in protocol['seeds']:
                row = json.loads((folder / f'{method}-seed{seed}.json').read_text())
                path = folder / f'{method}-seed{seed}.npz'
                assert row['status'] == 'complete' and row['seed'] == seed and row['method'] == method
                assert hashlib.sha256(path.read_bytes()).hexdigest() == row['predictions_sha256']
                prediction = np.load(path)
                for key, metric in [('raw', 'raw_ari'), ('refined', 'refined_ari'),
                                    ('diagnostic_only_refined_control', 'diagnostic_only_refined_control_ari'),
                                    ('diagnostic_only_pooled_coordinates', 'diagnostic_only_pooled_coordinate_ari')]:
                    if metric in row:
                        assert abs(adjusted_rand_score(truth[good], prediction[key][good]) - row[metric]) < 1e-12
                if method == 'expression-only':
                    assert not row['spatial_refinement'] and 'refined' not in prediction
                count += 1
    assert count == 147
    subprocess.run([sys.executable, str(ROOT / 'manuscript/verify_committed_claims.py')], check=True)
    print('PUBLIC RELEASE PASS: 147 saved prediction records rescored; within-field edges checked; scientific HOLD remains.')


if __name__ == '__main__':
    main()
