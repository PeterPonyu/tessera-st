"""W8: self-inclusive propagation over Tessera-ST's own graph (PROTOCOL-R3-DEVIATIONS item 18;
secondary, descriptive; specified before any P3 result).

Tessera-ST's graph carries no self-loops, whereas the GraphST and STAGATE graphs do, and Tessera-ST's
layers keep each observation's own state through a residual update. For every WP4 dataset and every
Tessera-ST neighbour setting (6, 15, 30), a self-loop is added to every node, the common input is
propagated L in {1, 2, 3, 4, 8} times with the row-normalized operator, and each target is scored by
K-means at its class count, seeds 1-5. The analysis joins these scores with the configuration
selected for each unit on the box (analyze_r3.p3_descriptive).

Output (results/revision-r3/W8/): selfloop.csv, run-lock.json.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'external'))
import common_r3 as c3  # noqa: E402

OUT = c3.OUT / 'W8'
LS = (1, 2, 3, 4, 8)
SEEDS = (1, 2, 3, 4, 5)


def one(dataset_id: str, k: int) -> list[dict]:
    import common_external as ce
    import graph_ops as go
    import run_tessera_r3 as rt
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    ds = ce.load_dataset(dataset_id)
    n = len(ds['x_common'])
    src, dst = rt.encoder_edges(ds['coords'], ds['frames'], 'released', f'k{k}-w0.5-t0.2-e120', 1)
    src, dst = np.asarray(src), np.asarray(dst)
    assert not (src == dst).any(), 'Tessera-ST graph already has self-loops'
    src, dst = np.concatenate([src, np.arange(n)]), np.concatenate([dst, np.arange(n)])
    go.assert_within_frames(src, dst, ds['frames'])
    rows = []
    for L in LS:
        rep = go.propagate(ds['x_common'], src, dst, L)
        for target, y in ds['targets'].items():
            K, ok = ds['K'][target], y >= 0
            for s in SEEDS:
                pred = KMeans(K, n_init=10, random_state=s).fit(rep).labels_
                rows.append(dict(dataset=dataset_id, cohort=ds['cohort'], k_graph=k, L=L, target=target, seed=s,
                                 rule='kmeans', ari=float(adjusted_rand_score(y[ok], pred[ok]))))
    return rows


def main():
    import common_external as ce
    import run_tessera_r3 as rt
    OUT.mkdir(parents=True, exist_ok=True)
    c3.write_lock(OUT, 'W8: self-inclusive propagation over the Tessera-ST graph (deviation 18)', frozen_protocol=True)
    jobs = [(d, k) for d in ce.dataset_ids('core') for k in rt.NEIGHBORS]
    print(f'{len(jobs)} dataset-graph pairs', flush=True)
    blocks = Parallel(n_jobs=int(os.environ.get('R3_JOBS', '12')), verbose=5)(delayed(one)(d, k) for d, k in jobs)
    pd.DataFrame([r for b in blocks for r in b]).to_csv(OUT / 'selfloop.csv', index=False)


if __name__ == '__main__':
    main()
