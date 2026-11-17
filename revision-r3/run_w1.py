"""WP1: label diagnostics and smoothing outcomes for the 24 existing datasets.

Diagnostics come from labels, coordinates and expression only (diagnostics_r3).
Outcomes re-cluster the same representations the round-1 controls used, with the
frozen neighbour rule and K-means scorer, and every recomputed value that has a
saved counterpart is checked against it before anything is written.

Outputs (results/revision-r3/W1/): diagnostics.csv (one row per dataset, target and
scale), targets.csv (one row per dataset and target), checks.json, run-lock.json.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_r3 as c3  # noqa: E402
import diagnostics_r3 as dg  # noqa: E402

c1 = c3.c1
fu = c1.fu
import train_core as tc  # noqa: E402  (revision-r1; same-graph mean on the Tessera-ST graph)

OUT = c3.OUT / 'W1'
PITCH_UM = {'Breast_Visium': 100.0, 'DLPFC_151673': 100.0, 'HER2ST': 200.0}   # spot centre-to-centre


def scale_um(unit, d1):
    pitch = PITCH_UM.get(unit['cohort'])
    return pitch / d1 if pitch else None


def outcomes(unit, y, K):
    x, xy, fr = unit['x'], unit['coords'], unit['frames']
    res = {'pca50': [fu.kmeans_ari(x, y, K, s)[0] for s in c3.SEEDS]}
    for k in dg.KS:
        rows = fu.neighbours(xy, fr, k)
        feat = x[rows].mean(1)
        res[f'mean{k}'] = [fu.kmeans_ari(feat, y, K, s)[0] for s in c3.SEEDS]
    sgm, _ = tc.same_graph_mean(x, xy, fr)
    res['same_graph_mean'] = [fu.kmeans_ari(sgm, y, K, s)[0] for s in c3.SEEDS]
    res['tessera'] = [fu.kmeans_ari(e, y, K, s)[0] for e, s in zip(c3.tessera_embeddings(unit), c3.SEEDS)]
    return res


def one(unit, target):
    started = time.perf_counter()
    y, classes = unit['targets'][target]
    K = len(classes)
    prof = dg.homophily_profile(unit['coords'], unit['frames'], y)
    _, dist, _ = dg.frame_knn(unit['coords'], unit['frames'], 1)
    d1 = float(np.nanmedian(dist[y >= 0, 0]))
    ell = dg.correlation_length(unit['coords'], unit['frames'], y, scale_um(unit, d1))
    pred = dg.predictability(unit['x'], unit['coords'], unit['frames'], y)
    res = outcomes(unit, y, K)
    means = {k: float(np.mean(v)) for k, v in res.items()}
    by_k = {k: means[f'mean{k}'] for k in dg.KS}
    k_star = max(by_k, key=by_k.get)
    row = dict(dataset=unit['name'], cohort=unit['cohort'], unit=unit['unit'], target=target,
               provenance=c3.provenance(unit, target), development=unit['cohort'] in c3.DEVELOPMENT
               or unit['name'] in c3.DEVELOPMENT, K=K, n_labelled=int((y >= 0).sum()),
               n_frames=int(len(np.unique(unit['frames']))),
               h_adj6=float(prof.loc[prof.k == 6, 'h_adj'].iloc[0]),
               li6=float(prof.loc[prof.k == 6, 'li'].iloc[0]),
               ell_k=ell['ell_k'], ell_d1=ell['ell_d1'], ell_um=ell['ell_um'], ell_censored=ell['censored'],
               first_excess=ell['first_excess'],
               **{key: pred[key] for key in ('ba_self', 'kappa_self', 'chance_ba', 'ba_self_random',
                                              'kappa_self_random', 'info_gain_max', 'info_gain_k')},
               **{f'ari_{k}': v for k, v in means.items()},
               g6=means['same_graph_mean'] - means['pca50'], g_star=by_k[k_star] - means['pca50'],
               k_star=int(k_star), tessera_minus_sgm=means['tessera'] - means['same_graph_mean'],
               seconds=time.perf_counter() - started)
    prof.insert(0, 'target', target)
    prof.insert(0, 'dataset', unit['name'])
    return row, prof, {f'{unit["name"]}|{target}|{k}': v for k, v in res.items()}


def checks(per_seed):
    """Recomputed per-seed ARIs against saved round-1 values (absolute differences)."""
    out = {}
    sc = pd.read_csv(c1.RESULTS / 'followup-20261010' / 'scale_controls.csv')
    diffs = []
    for name in c1.OBJECTS:
        for s_i, s in enumerate(c3.SEEDS):
            saved = sc[(sc.object == name) & (sc.kind == 'PCA50') & (sc.seed == s)].ari.iloc[0]
            diffs.append(abs(per_seed[f'{name}|primary|pca50'][s_i] - saved))
            for k in dg.KS:
                saved = sc[(sc.object == name) & (sc.kind == 'NeighborMean') & (sc.k == k) & (sc.seed == s)].ari.iloc[0]
                diffs.append(abs(per_seed[f'{name}|primary|mean{k}'][s_i] - saved))
            saved = json.loads((c1.RESULTS / name / f'Tessera-seed{s}.json').read_text())['scores']['ari']
            diffs.append(abs(per_seed[f'{name}|primary|tessera'][s_i] - saved))
            saved = json.loads((c1.RESULTS / name / f'NeighborMean-seed{s}.json').read_text())['scores']['ari']
            diffs.append(abs(per_seed[f'{name}|primary|same_graph_mean'][s_i] - saved))
    out['bounded_primary_max_abs_diff'] = float(max(diffs))
    out['bounded_primary_values'] = len(diffs)
    rs = pd.read_csv(c1.RESULTS / 'followup-20261010' / 'rescore.csv')
    diffs = []
    for name in c1.LUNG:
        for s_i, s in enumerate(c3.SEEDS):
            saved = rs[(rs.object == name) & (rs.arm == 'Tessera') & (rs.seed == s)].ari_celltype_at_celltype_K.iloc[0]
            diffs.append(abs(per_seed[f'{name}|celltype|tessera'][s_i] - saved))
            saved = rs[(rs.object == name) & (rs.arm == 'PCA50') & (rs.seed == s)].ari_celltype_at_celltype_K.iloc[0]
            diffs.append(abs(per_seed[f'{name}|celltype|pca50'][s_i] - saved))
    out['lung_celltype_max_abs_diff'] = float(max(diffs))
    out['lung_celltype_values'] = len(diffs)
    return out


def main():
    c3.write_lock(OUT, 'W1 label diagnostics (calibration, draft protocol)')
    units = c3.existing_units()
    jobs = [(u, t) for u in units for t in u['targets']]
    print(f'{len(jobs)} dataset-target pairs', flush=True)
    results = Parallel(n_jobs=int(os.environ.get('R3_JOBS', '12')), verbose=5)(delayed(one)(u, t) for u, t in jobs)
    rows = [r for r, _, _ in results]
    profiles = pd.concat([p for _, p, _ in results], ignore_index=True)
    per_seed = {}
    for _, _, s in results:
        per_seed.update(s)
    pd.DataFrame(rows).to_csv(OUT / 'targets.csv', index=False)
    profiles.to_csv(OUT / 'diagnostics.csv', index=False)
    (OUT / 'per_seed.json').write_text(json.dumps(per_seed, indent=1) + '\n')
    c3.dump(OUT / 'checks.json', checks(per_seed))
    print(json.dumps(json.loads((OUT / 'checks.json').read_text()), indent=1))


if __name__ == '__main__':
    main()
