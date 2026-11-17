"""WP3: spatial positives without message passing, and depth-matched propagation.

For the 24 existing datasets: the ContrastNoMP arm at seeds 1-3 (one CPU thread,
bit-reproducible) and P^L x on the Tessera-ST graph for L in {1, 2, 3, 4, 8}, each
scored with K-means at the reference class count (seeds 1-3) and with Leiden
(seed 1, resolution bisected to K). Saved Tessera-ST embeddings are scored with
Leiden here as well, so every arm has both rules.

Outputs (results/revision-r3/W3/): scores.csv, embeddings/*.npz, run-lock.json.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_r3 as c3  # noqa: E402

OUT = c3.OUT / 'W3'
DEPTHS = (1, 2, 3, 4, 8)


def head_k(unit) -> int:
    """Untrained head size of the reference fit: class count for bounded objects, two otherwise."""
    return len(unit['targets']['primary'][1]) if unit['tessera'][0] == 'primary' else 2


def score(unit, rep, arm, L, seed_rep, kmeans_seeds, leiden_seed=1):
    import run_k_sweep as rk  # revision-r1 Leiden rule
    fu = c3.c1.fu
    rows = []
    for target, (y, classes) in unit['targets'].items():
        K = len(classes)
        for s in kmeans_seeds:
            ari, _ = fu.kmeans_ari(rep, y, K, s)
            rows.append(dict(dataset=unit['name'], cohort=unit['cohort'], unit=unit['unit'], target=target, arm=arm,
                             L=L, rep_seed=seed_rep, cluster_seed=s, rule='kmeans', K=K, ari=ari))
        labels, k_got, _ = rk.leiden_partition(np.asarray(rep, np.float32), K, leiden_seed)
        ok = y >= 0
        from sklearn.metrics import adjusted_rand_score
        rows.append(dict(dataset=unit['name'], cohort=unit['cohort'], unit=unit['unit'], target=target, arm=arm, L=L,
                         rep_seed=seed_rep, cluster_seed=leiden_seed, rule='leiden', K=K, K_got=int(k_got),
                         ari=float(adjusted_rand_score(y[ok], labels[ok]))))
    return rows


def job_contrast(unit, seed):
    import torch
    torch.set_num_threads(1)
    import arms_r3 as ar
    fit = ar.train_contrast_no_mp(unit['x'].astype(np.float32), unit['coords'], unit['frames'], head_k(unit), seed,
                                  'cpu')
    np.savez_compressed(OUT / 'embeddings' / f'{unit["name"]}-ContrastNoMP-seed{seed}.npz', z=fit['z'])
    rows = score(unit, fit['z'], 'ContrastNoMP', None, seed, [seed])
    for r in rows:
        r['fit_seconds'] = fit['seconds']
    return rows


def job_propagate(unit, L):
    import arms_r3 as ar
    import train_core as tc
    _, edge = tc.same_graph_mean(unit['x'], unit['coords'], unit['frames'])
    return score(unit, ar.propagate(unit['x'], edge, L), 'Propagate', L, None, c3.SEEDS)


def job_tessera_leiden(unit):
    rows = []
    for s, emb in zip(c3.SEEDS, c3.tessera_embeddings(unit)):
        rows += [r for r in score(unit, emb, 'Tessera', None, s, []) if r['rule'] == 'leiden']
    return rows


def main():
    (OUT / 'embeddings').mkdir(parents=True, exist_ok=True)
    c3.write_lock(OUT, 'W3 mechanism arms (calibration, draft protocol)')
    units = c3.existing_units()
    jobs = [delayed(job_contrast)(u, s) for u in units for s in c3.SEEDS]
    jobs += [delayed(job_propagate)(u, L) for u in units for L in DEPTHS]
    jobs += [delayed(job_tessera_leiden)(u) for u in units]
    started = time.time()
    out = Parallel(n_jobs=int(os.environ.get('R3_JOBS', '12')), verbose=5)(jobs)
    rows = [r for block in out for r in block]
    pd.DataFrame(rows).to_csv(OUT / 'scores.csv', index=False)
    print(f'{len(rows)} rows in {time.time() - started:.0f} s')


if __name__ == '__main__':
    main()
