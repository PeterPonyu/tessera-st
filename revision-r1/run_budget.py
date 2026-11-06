#!/usr/bin/env python3
"""E1: training budget to 400 epochs for every encoder arm; E1b: transfer at 120 and 200 epochs.

Modes:
  gpu       13 bounded objects and 3 anatomical sections, CUDA
  her2st    8 HER2ST sections, CPU with one thread per process, fixed two-cluster head
  transfer  leave-one-patient-out NSCLC transfer, CUDA
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

import common_r1 as c
from train_core import ARMS, train_core

E1 = c.OUT / 'E1'
SAVE_EPOCHS = [120, 200, 400]
EARLY = [10, 20, 30, 40]          # Amendment A3
EXTRA_SEEDS = list(range(4, 11))  # Amendment A3


def score_partition(emb, y, k, seed):
    pred = KMeans(k, n_init=10, random_state=seed).fit(emb).labels_
    ok = y >= 0
    return float(adjusted_rand_score(y[ok], pred[ok])), float(normalized_mutual_info_score(y[ok], pred[ok]))


def seam_auroc(edge, gate, y):
    ok = (y[edge[0]] >= 0) & (y[edge[1]] >= 0)
    seam = y[edge[0]][ok] != y[edge[1]][ok]
    if len(np.unique(seam)) < 2:
        return None
    return float(roc_auc_score(seam, 1 - gate[ok]))


def fit_unit(unit: dict, arm: str, seed: int, device: str, head_k: int) -> list[dict]:
    fit = train_core(unit['x'], unit['coords'], unit['fields'], head_k, arm, seed, device,
                     epochs=max(c.CHECKPOINTS), checkpoints=c.CHECKPOINTS)
    rows = []
    for epoch, point in fit['checkpoints'].items():
        for target, (y, classes) in unit['targets'].items():
            ari, nmi = score_partition(point['z'], y, len(classes), seed)
            seam = None
            if arm in ('Tessera', 'NoContrast'):
                seam = seam_auroc(fit['edge'], point['gate'], y)
            rows.append(dict(object=unit['name'], cohort=unit['cohort'], arm=arm, seed=seed, epoch=epoch,
                             target=target, K=len(classes), ari=ari, nmi=nmi, seam_auroc=seam,
                             fit_seconds_to_epoch=point['seconds']))
    if arm == 'Tessera':
        np.savez_compressed(E1 / 'embeddings' / f"{unit['name']}-Tessera-seed{seed}.npz",
                            **{f'z{e}': fit['checkpoints'][e]['z'] for e in SAVE_EPOCHS},
                            **{f'gate{e}': fit['checkpoints'][e]['gate'] for e in SAVE_EPOCHS},
                            edge_index=fit['edge'])
    c.dump(E1 / 'records' / f"{unit['name']}-{arm}-seed{seed}.json",
           dict(object=unit['name'], arm=arm, seed=seed, device=device, head_k=head_k,
                epochs=max(c.CHECKPOINTS), seconds=fit['seconds'], gpu_peak_bytes=fit['memory'],
                trainable_parameters=fit['params'], history=fit['history'],
                n_edges=int(fit['edge'].shape[1])))
    return rows


def primary_units():
    units = []
    for name in c.OBJECTS:
        data, meta = c.load_primary(name)
        targets = {'primary': (data['ref'], meta['classes'])}
        if name in c.LUNG:
            targets['cell_type'] = c.celltype_labels(data)
        units.append(dict(name=name, cohort=meta['cohort'], x=data['x'], coords=data['coords'],
                          fields=data['field'], targets=targets, head_k=meta['n_classes']))
    for name, unit in c.tissue_units().items():
        y, classes = unit['targets']['anatomy']
        units.append(dict(name=name, cohort='tissue', x=unit['x'], coords=unit['coords'],
                          fields=unit['fields'], targets={'primary': (y, classes)}, head_k=len(classes)))
    return units


def run_gpu():
    rows = []
    for unit in primary_units():
        for seed in c.SEEDS:
            for arm in ARMS:
                started = time.perf_counter()
                rows += fit_unit(unit, arm, seed, 'cuda', unit['head_k'])
                print('E1', unit['name'], arm, seed, round(time.perf_counter() - started, 1), 's', flush=True)
        pd.DataFrame(rows).to_csv(E1 / 'E1_scores_gpu.csv', index=False)
    return rows


def _her2st_job(args):
    stem, arm, seed = args
    torch.set_num_threads(1)
    unit = c.her2st_unit(stem)
    return fit_unit(unit, arm, seed, 'cpu', 2)


def run_her2st(processes: int):
    jobs = [(stem, arm, seed) for stem in c.HER2ST for arm in ARMS for seed in c.SEEDS]
    with mp.get_context('spawn').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(_her2st_job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(E1 / 'E1_scores_her2st.csv', index=False)
    return rows


def run_transfer():
    import run_transfer as rt
    from tessera_st.model.graph import build_knn_edges
    from train_core import build_graph
    names = c.LUNG
    inputs = {n: np.load(c.DATA / n / 'input.npz', allow_pickle=False) for n in names}
    genes = sorted(set.intersection(*[set(inputs[n]['gene']) for n in names]))
    raw = {n: inputs[n]['gene_log'][:, np.array([np.flatnonzero(inputs[n]['gene'] == g)[0] for g in genes])]
           for n in names}
    rows = []
    for target in names:
        sources = [n for n in names if n != target]
        sx, tx, *_ = rt.transform_source(np.concatenate([raw[n] for n in sources]), raw[target])
        xy = np.concatenate([inputs[n]['coords'] for n in sources])
        fields = np.concatenate([inputs[n]['field'] for n in sources])
        labels = {key: np.concatenate([inputs[n][key] for n in sources]) for key in ['ref_name', 'secondary_name']}
        k = len(set(labels['ref_name'][c.valid(labels['ref_name'])]))
        t = inputs[target]
        target_edge, _ = build_graph(t['coords'], t['field'], 'Tessera', 1)
        _, source_distance = build_knn_edges(xy, k=6, field_ids=fields)
        _, target_distance = build_knn_edges(t['coords'], k=6, field_ids=t['field'])
        target_dist = ((target_distance - source_distance.mean()) / (source_distance.std() + 1e-8)).astype(np.float32)

        def embed_target(epoch, model):
            out = model(torch.as_tensor(tx, dtype=torch.float32, device='cuda'),
                        torch.as_tensor(target_edge, dtype=torch.long, device='cuda'),
                        torch.as_tensor(target_dist, dtype=torch.float32, device='cuda'))
            return out.z.cpu().numpy()
        for seed in c.SEEDS:
            fit = train_core(sx, xy, fields, k, 'Tessera', seed, 'cuda', epochs=200, checkpoints=[120, 200],
                             on_checkpoint=embed_target)
            for epoch in (120, 200):
                se, te = fit['checkpoints'][epoch]['z'], fit['extra'][epoch]
                for key, task in [('ref_name', 'region'), ('secondary_name', 'cell_type')]:
                    pred, _, classes, _ = rt.centroid_predict(se, te, labels[key])
                    s = rt.transfer_score(t[key], pred, classes)
                    rows.append(dict(target=target, seed=seed, epoch=epoch, task=task, accuracy=s['accuracy'],
                                     balanced_accuracy=s['balanced_accuracy'],
                                     macro_f1=s['macro_f1_all_target']))
            print('E1b', target, seed, flush=True)
    pd.DataFrame(rows).to_csv(E1 / 'E1b_transfer.csv', index=False)
    return rows


# ------------------------------------------------------------------ Amendment A3
def fit_scores(unit: dict, seed: int, device: str, head_k: int, epochs: int, checkpoints: list) -> list[dict]:
    """Tessera-ST at the released configuration, scored at each checkpoint; nothing else is saved."""
    fit = train_core(unit['x'], unit['coords'], unit['fields'], head_k, 'Tessera', seed, device, epochs=epochs,
                     checkpoints=checkpoints)
    rows = []
    for epoch, point in fit['checkpoints'].items():
        for target, (y, classes) in unit['targets'].items():
            ari, nmi = score_partition(point['z'], y, len(classes), seed)
            rows.append(dict(object=unit['name'], cohort=unit['cohort'], arm='Tessera', seed=seed, epoch=epoch,
                             target=target, K=len(classes), ari=ari, nmi=nmi,
                             seam_auroc=seam_auroc(fit['edge'], point['gate'], y),
                             fit_seconds_to_epoch=point['seconds']))
    return rows


def _her2st_a3_job(args):
    stem, seed, epochs, checkpoints = args
    torch.set_num_threads(1)
    return fit_scores(c.her2st_unit(stem), seed, 'cpu', 2, epochs, checkpoints)


def run_early(processes: int):
    rows = []
    for unit in primary_units():
        for seed in c.SEEDS:
            rows += fit_scores(unit, seed, 'cuda', unit['head_k'], max(EARLY), EARLY)
        print('A3 early', unit['name'], flush=True)
    pd.DataFrame(rows).to_csv(E1 / 'E1_scores_early_gpu.csv', index=False)
    jobs = [(stem, seed, max(EARLY), EARLY) for stem in c.HER2ST for seed in c.SEEDS]
    with mp.get_context('spawn').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(_her2st_a3_job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(E1 / 'E1_scores_early_her2st.csv', index=False)


def run_her2st_seeds(processes: int):
    jobs = [(stem, seed, 200, [120, 200]) for stem in c.HER2ST for seed in EXTRA_SEEDS]
    with mp.get_context('spawn').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(_her2st_a3_job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(E1 / 'E1_scores_her2st_seeds.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['gpu', 'her2st', 'transfer', 'early', 'her2st-seeds'])
    p.add_argument('--processes', type=int, default=8)
    args = p.parse_args()
    (E1 / 'embeddings').mkdir(parents=True, exist_ok=True)
    (E1 / 'records').mkdir(parents=True, exist_ok=True)
    if args.mode in ('early', 'her2st-seeds'):
        c.write_lock(E1, 'E1 Amendment A3: earlier checkpoints and more breast seeds', name='run-lock-A3.json')
    else:
        c.write_lock(E1, 'E1 training budget')
    started = time.perf_counter()
    {'gpu': run_gpu, 'her2st': lambda: run_her2st(args.processes), 'transfer': run_transfer,
     'early': lambda: run_early(args.processes), 'her2st-seeds': lambda: run_her2st_seeds(args.processes)}[args.mode]()
    print('E1', args.mode, 'done in', round(time.perf_counter() - started), 's', flush=True)


if __name__ == '__main__':
    main()
