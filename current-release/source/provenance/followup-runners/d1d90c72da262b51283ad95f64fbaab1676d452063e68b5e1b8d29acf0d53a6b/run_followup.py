#!/usr/bin/env python3
"""Reviewer-requested follow-up analyses (2026-10-10).

The frozen estimator code (run_experiments.py, tessera_st) is imported, never
modified, so the locked primary run identity is unchanged. Outputs go to
results/followup-20261010/. Every analysis uses the same fixed inputs, the
same reference K convention as the primary run unless stated, and reports
all configurations; nested selection never sees the held-out patient.

  scale     tuned neighbour-mean, BANKSY-style and composition controls with
            leave-one-patient-out selection inside the lung and BCC cohorts
  rescore   saved embeddings re-clustered at K >= 1% classes and against lung
            cell types at the cell-type K; gate vs trivial edge-score AUROC
  nulls     permuted-graph training without the contrastive term (a geometry
            null that does not turn random cells into positives)
  transfer  patient-held-out transfer of neighbour-mean features (k = 6 and the
            lung-selected k = 200) built on the same source-only PCA transform
"""
from __future__ import annotations
import argparse
import itertools
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from common import ROOT, DATA, RESULTS, OBJECTS, SEEDS, read, dump, sha, encode

OUT = RESULTS / 'followup-20261010'
LUNG, BCC = OBJECTS[:5], OBJECTS[5:9]
KS = [6, 10, 15, 30, 50, 100, 200]
LAMBDAS = [.2, .5, .8]


def load(name):
    d = np.load(DATA / name / 'input.npz', allow_pickle=False)
    meta = read(DATA / name / 'input.json')
    assert sha(DATA / name / 'input.npz') == meta['matrix_sha256']
    return d, meta


def neighbours(xy, fields, k):
    """Self-excluding k nearest neighbours within each coordinate frame."""
    rows = np.empty((len(xy), k), dtype=np.int64)
    for f in np.unique(fields):
        ix = np.flatnonzero(fields == f)
        kk = min(k, len(ix) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(xy[ix])
        idx = nn.kneighbors(xy[ix], return_distance=False)[:, 1:]
        block = ix[idx]
        if kk < k:  # pad small frames by repeating the farthest available neighbour
            block = np.concatenate([block, np.repeat(block[:, -1:], k - kk, axis=1)], axis=1)
        rows[ix] = block
    return rows


def kmeans_ari(features, y, k, seed):
    pred = KMeans(k, n_init=10, random_state=seed).fit(features).labels_
    ok = y >= 0
    return float(adjusted_rand_score(y[ok], pred[ok])), pred


def primary(name, arm):
    return float(np.mean([read(RESULTS / name / f'{arm}-seed{s}.json')['scores']['ari'] for s in SEEDS]))


def exact_sign_flip(values):
    values = np.asarray(values, float)
    observed = abs(values.mean())
    flips = [abs((values * np.array(signs)).mean()) for signs in itertools.product([-1, 1], repeat=len(values))]
    return float(np.mean([f >= observed - 1e-12 for f in flips]))


# ------------------------------------------------------------------ scale
def configs():
    yield ('PCA50', None, None)
    for k in KS:
        yield ('NeighborMean', k, None)
        for lam in LAMBDAS:
            yield ('BANKSY-style', k, lam)
    yield ('Composition', 200, None)


def features(kind, k, lam, x, xy, fields, cache, seed):
    if kind == 'PCA50':
        return x
    if kind == 'Composition':
        if 'comp' not in cache:
            clusters = KMeans(15, n_init=10, random_state=0).fit(x).labels_
            rows = neighbours(xy, fields, 200)
            onehot = np.eye(15)[clusters]
            cache['comp'] = onehot[rows].mean(1)
        return cache['comp']
    if k not in cache:
        rows = neighbours(xy, fields, k)
        cache[k] = x[rows].mean(1)
    mean = cache[k]
    if kind == 'NeighborMean':
        return mean
    return np.hstack([x, np.sqrt(lam) * mean])


def scale_controls():
    rows = []
    for name in LUNG + BCC + OBJECTS[9:]:
        d, meta = load(name)
        x, xy, fields, y, K = d['x'], d['coords'], d['field'], d['ref'], meta['n_classes']
        cache = {}
        for kind, k, lam in configs():
            for seed in SEEDS:
                ari, _ = kmeans_ari(features(kind, k, lam, x, xy, fields, cache, seed), y, K, seed)
                rows.append(dict(object=name, kind=kind, k=k, lam=lam, seed=seed, ari=ari))
        print('scale', name, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'scale_controls.csv', index=False)
    table['config'] = table.kind + '|' + table.k.astype(str) + '|' + table.lam.astype(str)
    means = table.groupby(['object', 'config']).ari.mean().unstack()
    summary = {}
    for cohort, names in [('NSCLC', LUNG), ('BCC', BCC)]:
        held = []
        for name in names:
            others = [n for n in names if n != name]
            # Selection excludes PCA50 so the chosen control always uses geometry.
            candidates = [c for c in means.columns if not c.startswith('PCA50')]
            best = means.loc[others, candidates].mean().idxmax()
            held.append(dict(patient=name, selected=best, nested_control_ari=float(means.loc[name, best]),
                             tessera_ari=primary(name, 'Tessera'), pca50_ari=primary(name, 'PCA50'),
                             neighbor_mean_k6_ari=primary(name, 'NeighborMean')))
        frame = pd.DataFrame(held)
        delta = frame.tessera_ari - frame.nested_control_ari
        summary[cohort] = dict(per_patient=held, tessera_minus_nested_mean=float(delta.mean()),
                               tessera_better=int((delta > 0).sum()), patients=len(delta),
                               exact_sign_flip_p=exact_sign_flip(delta))
    best_overall = means.max(axis=1)
    summary['oracle_best_control_by_object'] = {n: dict(config=means.loc[n].idxmax(), ari=float(best_overall[n]),
        tessera_ari=primary(n, 'Tessera')) for n in means.index}
    dump(OUT / 'scale_controls_summary.json', summary)
    return summary


# ---------------------------------------------------------------- rescore
ARMS_RESCORE = ['PCA50', 'NeighborMean', 'Tessera', 'NoGate', 'NoContrast', 'SelfOnly']


def rescore():
    rows = []
    gate_rows = []
    for name in OBJECTS:
        d, meta = load(name)
        y = d['ref']
        counts = np.bincount(y[y >= 0])
        k_common = int((counts >= .01 * (y >= 0).sum()).sum())
        sec = None
        if name in LUNG and 'secondary_name' in d:
            sec, sec_classes = encode(d['secondary_name'])
        for arm in ARMS_RESCORE:
            for seed in SEEDS:
                emb = np.load(RESULTS / name / f'{arm}-seed{seed}.npz')['embedding']
                ari_k, _ = kmeans_ari(emb, y, k_common, seed)
                rec = dict(object=name, arm=arm, seed=seed, K=meta['n_classes'], K_common=k_common,
                           ari_primary=read(RESULTS / name / f'{arm}-seed{seed}.json')['scores']['ari'],
                           ari_common_K=ari_k)
                if sec is not None:
                    rec['celltype_K'] = len(sec_classes)
                    rec['ari_celltype_at_celltype_K'] = kmeans_ari(emb, sec, len(sec_classes), seed)[0]
                rows.append(rec)
        # Gate AUROC against a trivial expression-difference edge score on the same edges.
        x = d['x']
        for seed in SEEDS:
            p = np.load(RESULTS / name / f'Tessera-seed{seed}.npz')
            edge, gate = p['edge_index'], p['last_gate']
            valid = (y[edge[0]] >= 0) & (y[edge[1]] >= 0)
            seam = y[edge[0]][valid] != y[edge[1]][valid]
            if len(np.unique(seam)) < 2:
                continue
            diff = np.linalg.norm(x[edge[0]] - x[edge[1]], axis=1)[valid]
            gate_rows.append(dict(object=name, seed=seed,
                gate_auroc=float(roc_auc_score(seam, 1 - gate[valid])),
                expression_difference_auroc=float(roc_auc_score(seam, diff)),
                gate_median=float(np.median(gate)), gate_fraction_below_0_05=float((gate < .05).mean()),
                gate_iqr=float(np.subtract(*np.percentile(gate, [75, 25])))))
        print('rescore', name, flush=True)
    table = pd.DataFrame(rows);table.to_csv(OUT / 'rescore.csv', index=False)
    gates = pd.DataFrame(gate_rows);gates.to_csv(OUT / 'gate_baseline.csv', index=False)
    means = table.groupby(['object', 'arm']).mean(numeric_only=True)
    summary = {'k_common': {n: int(table[table.object == n].K_common.iloc[0]) for n in OBJECTS},
               'k_primary': {n: int(table[table.object == n].K.iloc[0]) for n in OBJECTS}}
    for target, column in [('common_K', 'ari_common_K'), ('lung_celltypes', 'ari_celltype_at_celltype_K')]:
        names = OBJECTS if target == 'common_K' else LUNG
        delta = [means.loc[(n, 'Tessera'), column] - means.loc[(n, 'PCA50'), column] for n in names]
        delta_nm = [means.loc[(n, 'Tessera'), column] - means.loc[(n, 'NeighborMean'), column] for n in names]
        summary[target] = dict(objects=names,
            tessera=[float(means.loc[(n, 'Tessera'), column]) for n in names],
            pca50=[float(means.loc[(n, 'PCA50'), column]) for n in names],
            neighbor_mean=[float(means.loc[(n, 'NeighborMean'), column]) for n in names],
            tessera_minus_pca50=[float(v) for v in delta], tessera_minus_neighbor_mean=[float(v) for v in delta_nm])
    g = gates.groupby('object').mean(numeric_only=True)
    summary['gate'] = {n: {k: float(v) for k, v in g.loc[n].items() if k != 'seed'} for n in g.index}
    summary['gate_beaten_by_expression_difference'] = int((g.expression_difference_auroc > g.gate_auroc).sum())
    summary['gate_objects'] = int(len(g))
    dump(OUT / 'rescore_summary.json', summary)
    return summary


# ------------------------------------------------------------------ nulls
def geometry_nulls(device):
    sys.path.insert(0, str(ROOT / 'source/scripts'))
    import run_experiments as rx
    from tessera_st.config import AblationConfig
    original = rx.configuration

    def no_contrast_on_permuted(arm):
        return AblationConfig(edge_gating=True, multi_scale=True, boundary_contrastive=False,
                              calibrated_uncertainty=False) if arm == 'PermutedXY' else original(arm)
    rx.configuration = no_contrast_on_permuted
    rows = []
    try:
        for name in OBJECTS:
            d, meta = load(name)
            x, xy, fields, y, K = d['x'], d['coords'], d['field'], d['ref'], meta['n_classes']
            for seed in SEEDS:
                started = time.perf_counter()
                _, emb, _, _, _, _, _, elapsed, _, _ = rx.train_net(x, xy, fields, K, 'PermutedXY', seed, device)
                ari, _ = kmeans_ari(emb, y, K, seed)
                rows.append(dict(object=name, seed=seed, arm='PermutedXY-NoContrast', ari=ari,
                                 fit_seconds=elapsed, total_seconds=time.perf_counter() - started))
            print('null', name, flush=True)
    finally:
        rx.configuration = original
    table = pd.DataFrame(rows);table.to_csv(OUT / 'geometry_nulls.csv', index=False)
    means = table.groupby('object').ari.mean()
    summary = {n: dict(permuted_no_contrast=float(means[n]), self_only=primary(n, 'SelfOnly'),
                       no_contrast=primary(n, 'NoContrast'), tessera=primary(n, 'Tessera'),
                       permuted_with_contrast=primary(n, 'PermutedXY'), pca50=primary(n, 'PCA50'))
               for n in OBJECTS}
    summary['tessera_minus_self_only_negative_objects'] = int(sum(v['tessera'] < v['self_only'] for v in summary.values()))
    dump(OUT / 'geometry_nulls_summary.json', summary)
    return summary


# --------------------------------------------------------------- transfer
def transfer():
    import run_transfer as rt
    rows = []
    inputs = {n: np.load(DATA / n / 'input.npz', allow_pickle=False) for n in LUNG}
    genes = sorted(set.intersection(*[set(inputs[n]['gene']) for n in LUNG]))
    raw = {n: inputs[n]['gene_log'][:, np.array([np.flatnonzero(inputs[n]['gene'] == g)[0] for g in genes])] for n in LUNG}
    for target in LUNG:
        sources = [n for n in LUNG if n != target]
        sx, tx, *_ = rt.transform_source(np.concatenate([raw[n] for n in sources]), raw[target])
        xy = np.concatenate([inputs[n]['coords'] for n in sources])
        fields = np.concatenate([inputs[n]['field'] for n in sources])
        labels = {key: np.concatenate([inputs[n][key] for n in sources]) for key in ['ref_name', 'secondary_name']}
        tdata = inputs[target]
        for k in [6, 200]:
            se = sx[neighbours(xy, fields, k)].mean(1)
            te = tx[neighbours(tdata['coords'], tdata['field'], k)].mean(1)
            for key, title in [('ref_name', 'region'), ('secondary_name', 'cell_type')]:
                pred, _, classes, _ = rt.centroid_predict(se, te, labels[key])
                score = rt.transfer_score(tdata[key], pred, classes)
                rows.append(dict(patient=target, arm=f'NeighborMean-k{k}', target=title,
                                 accuracy=score['accuracy'], balanced_accuracy=score['balanced_accuracy'],
                                 macro_f1_all_target=score['macro_f1_all_target']))
        print('transfer', target, flush=True)
    table = pd.DataFrame(rows); table.to_csv(OUT / 'transfer_neighbor_mean.csv', index=False)
    primary_rows = pd.read_csv(ROOT / 'source/manuscript/ledgers/all_transfer_scores.csv')
    primary_means = primary_rows.groupby(['patient', 'arm', 'target'])[['accuracy', 'balanced_accuracy']].mean()
    summary = {}
    for title in ['region', 'cell_type']:
        for metric in ['accuracy', 'balanced_accuracy']:
            entry = {}
            for arm in ['PCA50', 'SelfOnly', 'Tessera']:
                entry[arm] = [float(primary_means.loc[(n, arm, title), metric]) for n in LUNG]
            for arm in ['NeighborMean-k6', 'NeighborMean-k200']:
                sub = table[(table.arm == arm) & (table.target == title)].set_index('patient')
                entry[arm] = [float(sub.loc[n, metric]) for n in LUNG]
            entry['tessera_minus_best_neighbor_mean'] = [t - max(a, b) for t, a, b in
                zip(entry['Tessera'], entry['NeighborMean-k6'], entry['NeighborMean-k200'])]
            summary[f'{title}_{metric}'] = entry
    summary['patients'] = LUNG
    dump(OUT / 'transfer_neighbor_mean_summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('analysis', choices=['scale', 'rescore', 'nulls', 'transfer', 'all'])
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    record = {'protocol': 'tessera-followup-20261010', 'script_sha256': sha(__file__),
              'frozen_run_lock_sha256': sha(RESULTS / 'run-lock.json')}
    if args.analysis in ('scale', 'all'): record['scale'] = scale_controls()
    if args.analysis in ('rescore', 'all'): record['rescore'] = rescore()
    if args.analysis in ('nulls', 'all'): record['nulls'] = geometry_nulls(args.device)
    if args.analysis in ('transfer', 'all'): record['transfer'] = transfer()
    dump(OUT / f'run-{args.analysis}.json', record)


if __name__ == '__main__':
    main()
