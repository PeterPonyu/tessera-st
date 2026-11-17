"""WP6 (local part): label diagnostics, the 30-configuration control grid and the same-graph mean
on the cohorts prepared after the freeze (PROTOCOL-R3, sections 2, 3.2, 6 and 7).

Encoder fits come from the box (W4); this script needs only the prepared inputs. Every
quantity uses the definitions frozen in PROTOCOL-R3 and the code of diagnostics_r3.

Outputs (results/revision-r3/W6/): targets.csv, diagnostics.csv, grid.csv, run-lock.json.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'external'))
import common_r3 as c3  # noqa: E402
import diagnostics_r3 as dg  # noqa: E402

OUT = c3.OUT / 'W6'


def units():
    import common_external as ce
    out = []
    for dataset_id in ce.r3_ids():
        ds = ce.load_dataset(dataset_id)
        cohort, name = dataset_id.split('/')
        info = json.loads((ce.DATA_R3 / cohort / name / 'input.json').read_text())
        prov = {t: (info.get('targets', {}).get(t, {}) or {}).get('provenance') for t in ds['targets']}
        out.append(dict(dataset=dataset_id, cohort=cohort, unit=info.get('unit', name), x=np.asarray(ds['x_common'],
                        np.float64), coords=np.asarray(ds['coords'], float), frames=np.asarray(ds['frames']).astype(str),
                        targets=ds['targets'], K=ds['K'], provenance=prov, um_per_unit=info.get('um_per_coordinate_unit')))
    return out


def one(u, target):
    import train_core as tc
    fu = c3.c1.fu
    y, K = np.asarray(u['targets'][target], np.int64), int(u['K'][target])
    x, xy, fr = u['x'], u['coords'], u['frames']
    prof = dg.homophily_profile(xy, fr, y)
    ell = dg.correlation_length(xy, fr, y, u['um_per_unit'])
    pred = dg.predictability(x, xy, fr, y)
    grid, cache = [], {}
    for kind, k, lam in fu.configs():
        feat = fu.features(kind, k, lam, x, xy, fr, cache, 1)
        for s in c3.SEEDS:
            grid.append(dict(dataset=u['dataset'], target=target, config=f'{kind}|{k}|{lam}', kind=kind, k=k, lam=lam,
                             seed=s, ari=fu.kmeans_ari(feat, y, K, s)[0]))
    sgm, _ = tc.same_graph_mean(x, xy, fr)
    for s in c3.SEEDS:
        grid.append(dict(dataset=u['dataset'], target=target, config='SameGraphMean', kind='SameGraphMean', k=6,
                         lam=None, seed=s, ari=fu.kmeans_ari(sgm, y, K, s)[0]))
    g = pd.DataFrame(grid)
    means = g.groupby('config').ari.mean()
    pca = means['PCA50|None|None']
    by_k = {k: means[f'NeighborMean|{k}|None'] for k in dg.KS}
    k_star = max(by_k, key=by_k.get)
    controls = means.drop(['PCA50|None|None', 'SameGraphMean'])
    row = dict(dataset=u['dataset'], cohort=u['cohort'], unit=u['unit'], target=target,
               provenance=u['provenance'].get(target), K=K, n_labelled=int((y >= 0).sum()),
               n_frames=int(len(np.unique(fr))), h_adj6=float(prof.loc[prof.k == 6, 'h_adj'].iloc[0]),
               li6=float(prof.loc[prof.k == 6, 'li'].iloc[0]), ell_k=ell['ell_k'], ell_d1=ell['ell_d1'],
               ell_um=ell['ell_um'], ell_censored=ell['censored'], first_excess=ell['first_excess'],
               **{k: pred[k] for k in ('ba_self', 'kappa_self', 'chance_ba', 'ba_self_random', 'kappa_self_random',
                                        'info_gain_max', 'info_gain_k')},
               ari_pca50=float(pca), ari_same_graph_mean=float(means['SameGraphMean']),
               **{f'ari_mean{k}': float(v) for k, v in by_k.items()},
               g6=float(means['SameGraphMean'] - pca), g_star=float(by_k[k_star] - pca), k_star=int(k_star),
               best_control=controls.idxmax(), headroom=float(controls.max() - pca))
    prof.insert(0, 'target', target)
    prof.insert(0, 'dataset', u['dataset'])
    return row, prof, g


def main():
    c3.write_lock(OUT, 'W6 local: diagnostics and control grid on the new cohorts', frozen_protocol=True)
    us = units()
    jobs = [(u, t) for u in us for t in u['targets']]
    print(f'{len(us)} datasets, {len(jobs)} dataset-target pairs', flush=True)
    out = Parallel(n_jobs=int(os.environ.get('R3_JOBS', '12')), verbose=5)(delayed(one)(u, t) for u, t in jobs)
    pd.DataFrame([r for r, _, _ in out]).to_csv(OUT / 'targets.csv', index=False)
    pd.concat([p for _, p, _ in out], ignore_index=True).to_csv(OUT / 'diagnostics.csv', index=False)
    pd.concat([g for _, _, g in out], ignore_index=True).to_csv(OUT / 'grid.csv', index=False)


if __name__ == '__main__':
    main()
