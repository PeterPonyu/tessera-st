"""WP1b: construction-scale dose-response on the NSCLC cells (PROTOCOL-R3, section 4).

For each lung patient and construction scale k_c, rebuild a niche target the way the
source regions were made: per-cell frequencies of the supplied cell types among the
k_c nearest other cells in the frame, clustered by K-means (10 initializations,
random state 0) at the patient's region class count. On every rebuilt target the
label diagnostics and smoothing outcomes of WP1 are computed, with Tessera-ST
scored from its saved embeddings. Run only after the protocol freeze.

Outputs (results/revision-r3/W1b/): targets.csv, diagnostics.csv, run-lock.json.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_r3 as c3  # noqa: E402
import diagnostics_r3 as dg  # noqa: E402
import run_w1  # noqa: E402

OUT = c3.OUT / 'W1b'
SCALES = (10, 25, 50, 100, 200, 400)


def rebuilt_target(unit, k_c: int) -> np.ndarray:
    types, type_classes = unit['targets']['celltype']
    K = len(unit['targets']['primary'][1])
    idx, _, size = dg.frame_knn(unit['coords'], unit['frames'], k_c)
    onehot = np.eye(len(type_classes))[np.maximum(types, 0)] * (types >= 0)[:, None]
    valid = idx >= 0
    freq = (onehot[np.maximum(idx, 0)] * valid[..., None]).sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
    return KMeans(K, n_init=10, random_state=0).fit(freq).labels_.astype(np.int64)


def one(unit, k_c):
    y = rebuilt_target(unit, k_c)
    K = len(np.unique(y))
    rebuilt = dict(unit, targets={'rebuilt': (y, [str(c) for c in range(K)])})
    c3.PROVENANCE[(unit['cohort'], 'rebuilt')] = 'computed-niche'
    row, prof, _ = run_w1.one(rebuilt, 'rebuilt')
    row.update(k_c=k_c, ari_rebuilt_vs_source=float(adjusted_rand_score(unit['targets']['primary'][0], y)))
    prof.insert(0, 'k_c', k_c)
    return row, prof


def main():
    c3.write_lock(OUT, 'W1b construction-scale dose-response', frozen_protocol=True)
    lung = [u for u in c3.bounded_units() if u['cohort'] == 'NSCLC']
    out = Parallel(n_jobs=int(os.environ.get('R3_JOBS', '12')), verbose=5)(
        delayed(one)(u, k) for u in lung for k in SCALES)
    pd.DataFrame([r for r, _ in out]).to_csv(OUT / 'targets.csv', index=False)
    pd.concat([p for _, p in out], ignore_index=True).to_csv(OUT / 'diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
