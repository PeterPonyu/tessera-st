#!/usr/bin/env python3
"""E3: cluster number and clustering geometry with fixed representations.

Modes:
  sweep          K grid x {Euclidean, cosine} K-means, shared label-free K, Leiden (secondary)
  harmonize      HER2ST per-K-head refits on CPU (one thread) for comparison with the fixed head
  refine         matched majority-vote refinement sensitivity at K_ref
  agreement      Amendment A4: partition agreement between representations and between seeds at K_ref
Run after E1 (which supplies the 200-epoch embeddings and the HER2ST fixed-head fits).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

import common_r1 as c
from train_core import same_graph_mean, train_core

E1 = c.OUT / 'E1'
E3 = c.OUT / 'E3'
LEIDEN_TARGETS = {('NSCLC', 'primary'), ('HER2ST', 'all')}


def parse_control(config: str):
    kind, k, lam = config.split('|')
    k = None if k in ('None', 'nan', '') else int(float(k))
    lam = None if lam in ('None', 'nan', '') else float(lam)
    return kind, k, lam


def control_features(config, x, xy, fields):
    kind, k, lam = parse_control(config)
    return np.asarray(c.fu.features(kind, k, lam, x, xy, fields, {}, 1), np.float32)


def saved_selections() -> dict:
    out = {}
    summary = json.loads((c.RESULTS / 'followup-20261010' / 'scale_controls_summary.json').read_text())
    for cohort in ['NSCLC', 'BCC']:
        for r in summary[cohort]['per_patient']:
            out[(r['patient'], 'primary')] = r['selected']
    ext = c.VALIDATION / 'validation-extension' / 'results'
    hard = json.loads((ext / 'run-hardening.json').read_text())['result']['targets']
    for target in ['all', 'compartments', 'boundary_T1']:
        for unit, config in hard[target]['selected_control'].items():
            out[(unit, target)] = config
    breadth = json.loads((ext / 'run-breadth.json').read_text())['result']['selected_control']
    for unit, config in breadth.items():
        out[(unit, 'anatomy')] = config
    return out


def e1_embedding(name, seed, epoch):
    return np.load(E1 / 'embeddings' / f'{name}-Tessera-seed{seed}.npz')[f'z{epoch}']


def build_units() -> list[dict]:
    selected = saved_selections()
    units = []
    for name in c.OBJECTS:
        data, meta = c.load_primary(name)
        targets = {'primary': (data['ref'], meta['classes'])}
        if name in c.LUNG:
            targets['cell_type'] = c.celltype_labels(data)
        saved = {s: np.load(c.RESULTS / name / f'Tessera-seed{s}.npz')['embedding'] for s in c.SEEDS}
        units.append(dict(name=name, cohort=meta['cohort'], x=data['x'], coords=data['coords'], fields=data['field'],
                          targets=targets, tessera120=saved, selected={t: selected.get((name, 'primary'))
                                                                        for t in targets}))
    for stem in c.HER2ST:
        u = c.her2st_unit(stem)
        units.append(dict(u, tessera120={s: e1_embedding(stem, s, 120) for s in c.SEEDS},
                          selected={t: selected[(stem, t)] for t in u['targets']}))
    for name, u in c.tissue_units().items():
        units.append(dict(u, tessera120={s: e1_embedding(name, s, 120) for s in c.SEEDS},
                          selected={'anatomy': selected[(name, 'anatomy')]}))
    return units


def representations(unit, target):
    """Representation instances (name, rep_seed, array); deterministic ones have rep_seed None."""
    x, xy, fields = unit['x'], unit['coords'], unit['fields']
    mean, _ = same_graph_mean(x, xy, fields)
    reps = [('PCA50', None, np.asarray(x, np.float32)), ('SameGraphMean', None, mean)]
    if unit['selected'].get(target):
        reps.append(('TunedControl', None, control_features(unit['selected'][target], x, xy, fields)))
    for s in c.SEEDS:
        reps.append(('Tessera120', s, np.asarray(unit['tessera120'][s], np.float32)))
        reps.append(('Tessera200', s, np.asarray(e1_embedding(unit['name'], s, 200), np.float32)))
    return reps


def k_grid(k_ref):
    return list(range(2, min(40, max(2 * k_ref, 10)) + 1))


def label_free_range(k_ref):
    return list(range(2, 7)) if k_ref <= 2 else list(range(3, min(40, 2 * k_ref) + 1))


def leiden_partition(array, k_target, seed):
    import igraph as ig
    import leidenalg
    nn = NearestNeighbors(n_neighbors=16).fit(array).kneighbors(array, return_distance=False)[:, 1:]
    edges = {(min(i, j), max(i, j)) for i, row in enumerate(nn) for j in row}
    graph = ig.Graph(n=len(array), edges=sorted(edges))
    low, high, best = 1e-3, 20.0, None
    for _ in range(30):
        res = np.sqrt(low * high)
        part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=res, seed=seed)
        labels, k = np.asarray(part.membership), len(set(part.membership))
        if best is None or abs(k - k_target) < abs(best[1] - k_target):
            best = (labels, k, res)
        if k == k_target:
            break
        low, high = (res, high) if k < k_target else (low, res)
    return best


def sweep_job(args):
    index, target = args
    torch.set_num_threads(1)
    unit = UNITS[index]
    y, classes = unit['targets'][target]
    k_ref = len(classes)
    ok = y >= 0
    reps = representations(unit, target)
    digest = {(n, s): hashlib.sha256(a.tobytes()).hexdigest() for n, s, a in reps}
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(y), size=min(3000, len(y)), replace=False))
    rows, sil_rows, leiden_rows = [], [], []
    lf = set(label_free_range(k_ref))
    for name, rep_seed, array in reps:
        geometries = {'euclidean': array,
                      'cosine': array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)}
        seeds = c.SEEDS if rep_seed is None else [rep_seed]
        for k in sorted(set(k_grid(k_ref)) | lf):
            for geometry, matrix in geometries.items():
                for seed in seeds:
                    pred = KMeans(k, n_init=10, random_state=seed).fit(matrix).labels_
                    rows.append(dict(unit=unit['name'], cohort=unit['cohort'], target=target, K_ref=k_ref,
                                     representation=name, rep_seed=rep_seed, kmeans_seed=seed, geometry=geometry,
                                     K=k, ari=float(adjusted_rand_score(y[ok], pred[ok])),
                                     nmi=float(normalized_mutual_info_score(y[ok], pred[ok]))))
                    if geometry == 'euclidean' and k in lf and seed == (rep_seed or 1):
                        sil = float(silhouette_score(matrix[sub], pred[sub])) if len(set(pred[sub])) > 1 else np.nan
                        sil_rows.append(dict(unit=unit['name'], target=target, representation=name,
                                             rep_seed=rep_seed, K=k, silhouette=sil))
        if (unit['cohort'], target) in LEIDEN_TARGETS and rep_seed in (None, 1):
            for k in range(max(2, k_ref - 2), k_ref + 3):
                labels, k_got, res = leiden_partition(array, k, 1)
                leiden_rows.append(dict(unit=unit['name'], cohort=unit['cohort'], target=target, K_ref=k_ref,
                                        representation=name, rep_seed=rep_seed, K_target=k, K_achieved=k_got,
                                        resolution=res, ari=float(adjusted_rand_score(y[ok], labels[ok]))))
    for n, s, a in reps:
        if hashlib.sha256(a.tobytes()).hexdigest() != digest[(n, s)]:
            raise ValueError('representation changed during the sweep')
    return rows, sil_rows, leiden_rows


def run_sweep(processes):
    jobs = [(i, t) for i, u in enumerate(UNITS) for t in u['targets']]
    rows, sil, leiden = [], [], []
    with mp.get_context('fork').Pool(processes) as pool:
        for r, s, l in pool.imap_unordered(sweep_job, jobs):
            rows += r
            sil += s
            leiden += l
            print('E3 sweep', r[0]['unit'] if r else '?', r[0]['target'] if r else '', flush=True)
    pd.DataFrame(rows).to_csv(E3 / 'E3_sweep.csv', index=False)
    pd.DataFrame(leiden).to_csv(E3 / 'E3_leiden.csv', index=False)
    sil = pd.DataFrame(sil)
    sil.to_csv(E3 / 'E3_silhouette.csv', index=False)
    shared = (sil.groupby(['unit', 'target', 'representation', 'K']).silhouette.mean().reset_index()
              .groupby(['unit', 'target', 'K']).silhouette.mean().reset_index())
    pick = shared.loc[shared.groupby(['unit', 'target']).silhouette.idxmax()]
    pick.rename(columns={'K': 'K_label_free', 'silhouette': 'mean_silhouette'}).to_csv(
        E3 / 'E3_label_free_K.csv', index=False)


def _harmonize_job(args):
    stem, seed = args
    torch.set_num_threads(1)
    unit = c.her2st_unit(stem)
    rows = []
    for head_k in sorted({len(cl) for _, cl in unit['targets'].values()}):
        fit = train_core(unit['x'], unit['coords'], unit['fields'], head_k, 'Tessera', seed, 'cpu', epochs=120)
        z = fit['checkpoints'][120]['z']
        for target, (y, classes) in unit['targets'].items():
            if len(classes) != head_k:
                continue
            ok = y >= 0
            pred = KMeans(len(classes), n_init=10, random_state=seed).fit(z).labels_
            fixed = KMeans(len(classes), n_init=10, random_state=seed).fit(e1_embedding(stem, seed, 120)).labels_
            rows.append(dict(unit=stem, target=target, seed=seed, head_k=head_k,
                             ari_per_k_head=float(adjusted_rand_score(y[ok], pred[ok])),
                             ari_fixed_head=float(adjusted_rand_score(y[ok], fixed[ok]))))
    return rows


def run_harmonize(processes):
    jobs = [(s, seed) for s in c.HER2ST for seed in c.SEEDS]
    with mp.get_context('spawn').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(_harmonize_job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(E3 / 'E3_harmonization.csv', index=False)


def refine_job(args):
    index, target = args
    from tessera_st.eval.refine import refine_labels
    unit = UNITS[index]
    y, classes = unit['targets'][target]
    ok = y >= 0
    rows = []
    for name, rep_seed, array in representations(unit, target):
        for seed in (c.SEEDS if rep_seed is None else [rep_seed]):
            pred = KMeans(len(classes), n_init=10, random_state=seed).fit(array).labels_
            refined = refine_labels(pred, unit['coords'], k=6, rounds=2, field_ids=unit['fields'])
            rows.append(dict(unit=unit['name'], cohort=unit['cohort'], target=target, representation=name,
                             rep_seed=rep_seed, kmeans_seed=seed, ari=float(adjusted_rand_score(y[ok], pred[ok])),
                             ari_refined=float(adjusted_rand_score(y[ok], refined[ok]))))
    return rows


def run_refine(processes):
    jobs = [(i, t) for i, u in enumerate(UNITS) for t in u['targets']]
    with mp.get_context('fork').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(refine_job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(E3 / 'E3_refinement.csv', index=False)


AGREEMENT_TARGETS = {('NSCLC', 'primary'), ('BCC', 'primary'), ('HER2ST', 'all'), ('HER2ST', 'boundary_T1')}
AGREEMENT_REPS = ['PCA50', 'SameGraphMean', 'TunedControl', 'Tessera120']


def agreement_job(args):
    """Amendment A4: ARI between partitions at K_ref (Euclidean K-means), across representations and across seeds."""
    index, target = args
    torch.set_num_threads(1)
    unit = UNITS[index]
    k_ref = len(unit['targets'][target][1])
    parts = {}
    for name, rep_seed, array in representations(unit, target):
        if name not in AGREEMENT_REPS:
            continue
        for seed in (c.SEEDS if rep_seed is None else [rep_seed]):
            parts[(name, seed)] = KMeans(k_ref, n_init=10, random_state=seed).fit(array).labels_
    names = [n for n in AGREEMENT_REPS if (n, c.SEEDS[0]) in parts]
    base = dict(unit=unit['name'], cohort=unit['cohort'], target=target, K_ref=k_ref, n_obs=len(parts[(names[0], 1)]))
    rows = []
    for seed in c.SEEDS:
        for a, b in itertools.combinations(names, 2):
            rows.append(dict(base, kind='between', a=a, b=b, seed_a=seed, seed_b=seed,
                             ari=float(adjusted_rand_score(parts[(a, seed)], parts[(b, seed)]))))
    for name in names:
        for s, t in itertools.combinations(c.SEEDS, 2):
            rows.append(dict(base, kind='within', a=name, b=name, seed_a=s, seed_b=t,
                             ari=float(adjusted_rand_score(parts[(name, s)], parts[(name, t)]))))
    return rows


def run_agreement(processes):
    jobs = [(i, t) for i, u in enumerate(UNITS) for t in u['targets'] if (u['cohort'], t) in AGREEMENT_TARGETS]
    with mp.get_context('fork').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(agreement_job, jobs) for r in part]
    pd.DataFrame(rows).sort_values(['cohort', 'target', 'unit', 'kind', 'a', 'b', 'seed_a', 'seed_b']).to_csv(
        E3 / 'E3_agreement.csv', index=False)


UNITS: list[dict] = []


def main():
    global UNITS
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['sweep', 'harmonize', 'refine', 'agreement'])
    p.add_argument('--processes', type=int, default=10)
    args = p.parse_args()
    if args.mode == 'agreement':
        c.write_lock(E3, 'A4 partition agreement', name='run-lock-A4.json')
    else:
        c.write_lock(E3, 'E3 cluster number and geometry')
    started = time.perf_counter()
    if args.mode in ('sweep', 'refine', 'agreement'):
        UNITS = build_units()
    {'sweep': run_sweep, 'harmonize': run_harmonize, 'refine': run_refine,
     'agreement': run_agreement}[args.mode](args.processes)
    print('E3', args.mode, 'done in', round(time.perf_counter() - started), 's', flush=True)


if __name__ == '__main__':
    main()
