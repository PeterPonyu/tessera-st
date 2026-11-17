"""WP2: control selection under each clustering rule, and seed noise of Tessera-ST.

leiden   All 30 control configurations (PCA50, k-neighbour means, BANKSY-style,
         composition), the same-graph mean and saved Tessera-ST embeddings are
         clustered with K-means and with Leiden (15-neighbour graph, resolution
         bisected to K) at seeds 1-3, for NSCLC regions, BCC cell types and both
         HER2ST pathologist targets. Patient-excluded selection is repeated under each
         rule, giving Tessera-ST minus the selected control for every pairing of
         selection rule and scoring rule.
seeds    Tessera-ST refitted on one CPU thread at seeds 1-10 for the 13 bounded
         objects; seeds 1-3 must reproduce the saved primary fits.

Outputs (results/revision-r3/W2/): grid.csv, selection.csv, seeds.csv, run-lock.json.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_r3 as c3  # noqa: E402

OUT = c3.OUT / 'W2'
TARGETS = {'NSCLC': ['primary'], 'BCC': ['primary'], 'HER2ST': ['all', 'boundary_T1']}


def grid_job(unit, config):
    import run_k_sweep as rk
    import train_core as tc
    fu = c3.c1.fu
    x, xy, fr = unit['x'], unit['coords'], unit['frames']
    if config == 'SameGraphMean':
        reps = {s: tc.same_graph_mean(x, xy, fr)[0] for s in c3.SEEDS}
    elif config == 'Tessera':
        reps = dict(zip(c3.SEEDS, c3.tessera_embeddings(unit)))
    else:
        kind, k, lam = rk.parse_control(config)
        feat = np.asarray(fu.features(kind, k, lam, x, xy, fr, {}, 1), np.float32)
        reps = {s: feat for s in c3.SEEDS}
    rows = []
    for target in TARGETS[unit['cohort']]:
        y, classes = unit['targets'][target]
        K, ok = len(classes), y >= 0
        for s, rep in reps.items():
            ari_km, _ = fu.kmeans_ari(rep, y, K, s)
            labels, k_got, _ = rk.leiden_partition(np.asarray(rep, np.float32), K, s)
            rows.append(dict(cohort=unit['cohort'], unit=unit['unit'], target=target, config=config, seed=s, K=K,
                             kmeans=ari_km, leiden=float(adjusted_rand_score(y[ok], labels[ok])),
                             leiden_K=int(k_got)))
    return rows


def configs():
    fu = c3.c1.fu
    return [f'{kind}|{k}|{lam}' for kind, k, lam in fu.configs()] + ['SameGraphMean', 'Tessera']


def selection(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, target), g in grid.groupby(['cohort', 'target']):
        means = {rule: g.groupby(['unit', 'config'])[rule].mean().unstack() for rule in ('kmeans', 'leiden')}
        candidates = [c for c in means['kmeans'].columns
                      if c not in ('SameGraphMean', 'Tessera') and not c.startswith('PCA50')]
        for unit in means['kmeans'].index:
            for select_rule in ('kmeans', 'leiden'):
                chosen = means[select_rule].drop(index=unit)[candidates].mean().idxmax()
                for score_rule in ('kmeans', 'leiden'):
                    m = means[score_rule]
                    rows.append(dict(cohort=cohort, target=target, unit=unit, select_rule=select_rule,
                                     score_rule=score_rule, selected=chosen, control=m.loc[unit, chosen],
                                     tessera=m.loc[unit, 'Tessera'], same_graph_mean=m.loc[unit, 'SameGraphMean'],
                                     pca50=m.loc[unit, [c for c in m.columns if c.startswith('PCA50')][0]],
                                     tessera_minus_control=m.loc[unit, 'Tessera'] - m.loc[unit, chosen],
                                     tessera_minus_sgm=m.loc[unit, 'Tessera'] - m.loc[unit, 'SameGraphMean']))
    return pd.DataFrame(rows)


def seed_job(unit, seed):
    import torch
    torch.set_num_threads(1)
    import train_core as tc
    fu = c3.c1.fu
    y, classes = unit['targets']['primary']
    fit = tc.train_core(unit['x'].astype(np.float32), unit['coords'], unit['frames'], len(classes), 'Tessera', seed,
                        'cpu', epochs=120)
    z = fit['checkpoints'][120]['z']
    np.savez_compressed(OUT / 'embeddings' / f'{unit["name"]}-Tessera-seed{seed}.npz', z=z)
    reproduced = None
    if seed in c3.SEEDS:
        saved = c3.tessera_embeddings(unit)[seed - 1]
        reproduced = float(np.abs(saved - z).max())
    return dict(dataset=unit['name'], cohort=unit['cohort'], unit=unit['unit'], seed=seed,
                ari=fu.kmeans_ari(z, y, len(classes), seed)[0], max_abs_vs_saved=reproduced,
                fit_seconds=fit['seconds'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['leiden', 'seeds'])
    args = ap.parse_args()
    (OUT / 'embeddings').mkdir(parents=True, exist_ok=True)
    c3.write_lock(OUT, f'W2 {args.step} (calibration, draft protocol)')
    jobs_n = int(os.environ.get('R3_JOBS', '12'))
    if args.step == 'leiden':
        units = [u for u in c3.bounded_units() + c3.her2st_units() if u['cohort'] in TARGETS]
        out = Parallel(n_jobs=jobs_n, verbose=5)(delayed(grid_job)(u, c) for u in units for c in configs())
        grid = pd.DataFrame([r for block in out for r in block])
        grid.to_csv(OUT / 'grid.csv', index=False)
        selection(grid).to_csv(OUT / 'selection.csv', index=False)
    else:
        units = c3.bounded_units()
        out = Parallel(n_jobs=jobs_n, verbose=5)(delayed(seed_job)(u, s) for u in units for s in range(1, 11))
        pd.DataFrame(out).to_csv(OUT / 'seeds.csv', index=False)


if __name__ == '__main__':
    main()
