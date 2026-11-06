#!/usr/bin/env python3
"""E2: patient-excluded tuning of Tessera over a fixed 24-configuration grid.

Modes:
  grid-gpu       NSCLC regions and BCC cell types on CUDA
  grid-her2st    HER2ST all labels and tumor boundary, CPU with one thread per process
  mechanism      nested selection per held-out patient, then NoGate/NoContrast at the selected
                 configuration and the same-graph mean on the selected graph
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch

import common_r1 as c
from run_budget import score_partition
from train_core import same_graph_mean, train_core

E2 = c.OUT / 'E2'
GRAPH_K = [6, 15, 30]
WEIGHTS = [0.5, 1.0]
TEMPS = [0.2, 0.5]
EPOCHS = [120, 200]
TARGETS = {'NSCLC': ['primary'], 'BCC': ['primary'], 'HER2ST': ['all', 'boundary_T1']}


def config_id(graph_k, weight, temp, epochs) -> str:
    return f'k{graph_k}|w{weight}|t{temp}|e{epochs}'


def cohort_units(cohort: str) -> list[dict]:
    if cohort == 'HER2ST':
        return [c.her2st_unit(stem) for stem in c.HER2ST]
    names = c.LUNG if cohort == 'NSCLC' else c.BCC
    units = []
    for name in names:
        data, meta = c.load_primary(name)
        units.append(dict(name=name, cohort=cohort, x=data['x'], coords=data['coords'], fields=data['field'],
                          targets={'primary': (data['ref'], meta['classes'])}, head_k=meta['n_classes']))
    return units


def fit_grid_point(unit, graph_k, weight, temp, seed, device, head_k, arm='Tessera', epochs=(120, 200)):
    fit = train_core(unit['x'], unit['coords'], unit['fields'], head_k, arm, seed, device, epochs=max(epochs),
                     graph_k=graph_k, contrastive_weight=weight, contrastive_temp=temp, checkpoints=list(epochs))
    rows = []
    for epoch in epochs:
        z = fit['checkpoints'][epoch]['z']
        for target in TARGETS[unit['cohort']]:
            y, classes = unit['targets'][target]
            ari, nmi = score_partition(z, y, len(classes), seed)
            rows.append(dict(cohort=unit['cohort'], patient=unit['name'], target=target, arm=arm,
                             graph_k=graph_k, weight=weight, temp=temp, epochs=epoch,
                             config=config_id(graph_k, weight, temp, epoch), seed=seed, ari=ari, nmi=nmi,
                             fit_seconds=fit['checkpoints'][epoch]['seconds']))
    return rows


def run_grid_gpu():
    rows = []
    for cohort in ['NSCLC', 'BCC']:
        for unit in cohort_units(cohort):
            for graph_k, weight, temp in itertools.product(GRAPH_K, WEIGHTS, TEMPS):
                for seed in c.SEEDS:
                    rows += fit_grid_point(unit, graph_k, weight, temp, seed, 'cuda', unit['head_k'])
            print('E2 grid', unit['name'], flush=True)
            pd.DataFrame(rows).to_csv(E2 / 'E2_grid_gpu.csv', index=False)


def _her2st_job(args):
    stem, graph_k, weight, temp, seed = args
    torch.set_num_threads(1)
    return fit_grid_point(c.her2st_unit(stem), graph_k, weight, temp, seed, 'cpu', 2)


def run_grid_her2st(processes: int):
    jobs = [(s, k, w, t, seed) for s in c.HER2ST for k, w, t in itertools.product(GRAPH_K, WEIGHTS, TEMPS)
            for seed in c.SEEDS]
    with mp.get_context('spawn').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(_her2st_job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(E2 / 'E2_grid_her2st.csv', index=False)


def parse_config(config: str):
    k, w, t, e = config.split('|')
    return int(k[1:]), float(w[1:]), float(t[1:]), int(e[1:])


def selections() -> pd.DataFrame:
    grid = pd.concat([pd.read_csv(E2 / 'E2_grid_gpu.csv'), pd.read_csv(E2 / 'E2_grid_her2st.csv')])
    rows = []
    for (cohort, target), part in grid.groupby(['cohort', 'target']):
        means = part.groupby(['patient', 'config']).ari.mean().unstack()
        for patient in means.index:
            chosen = c.select_nested(means, patient)
            rows.append(dict(cohort=cohort, target=target, patient=patient, selected=chosen,
                             nested_tessera_ari=float(means.loc[patient, chosen]),
                             default_tessera_ari=float(means.loc[patient, config_id(6, 0.5, 0.2, 120)]),
                             oracle_config=str(means.loc[patient].idxmax()),
                             oracle_tessera_ari=float(means.loc[patient].max())))
    table = pd.DataFrame(rows)
    table.to_csv(E2 / 'E2_selection.csv', index=False)
    return table


def _mechanism_job(args):
    cohort, target, patient, selected, device = args
    torch.set_num_threads(1 if device == 'cpu' else 4)
    unit = next(u for u in cohort_units(cohort) if u['name'] == patient)
    graph_k, weight, temp, epochs = parse_config(selected)
    head_k = 2 if cohort == 'HER2ST' else unit['head_k']
    rows = []
    for arm in ['NoGate', 'NoContrast']:
        for seed in c.SEEDS:
            for r in fit_grid_point(unit, graph_k, weight, temp, seed, device, head_k, arm=arm, epochs=(epochs,)):
                if r['target'] == target:
                    rows.append(r)
    y, classes = unit['targets'][target]
    mean, _ = same_graph_mean(unit['x'], unit['coords'], unit['fields'], graph_k)
    for seed in c.SEEDS:
        ari, nmi = score_partition(mean, y, len(classes), seed)
        rows.append(dict(cohort=cohort, patient=patient, target=target, arm=f'SameGraphMean-k{graph_k}',
                         graph_k=graph_k, weight=None, temp=None, epochs=None, config=selected, seed=seed,
                         ari=ari, nmi=nmi, fit_seconds=None))
    return rows


def run_mechanism(processes: int):
    table = selections()
    gpu = [(r.cohort, r.target, r.patient, r.selected, 'cuda') for r in table.itertuples() if r.cohort != 'HER2ST']
    cpu = [(r.cohort, r.target, r.patient, r.selected, 'cpu') for r in table.itertuples() if r.cohort == 'HER2ST']
    rows = [row for job in gpu for row in _mechanism_job(job)]
    with mp.get_context('spawn').Pool(processes) as pool:
        rows += [r for part in pool.imap_unordered(_mechanism_job, cpu) for r in part]
    pd.DataFrame(rows).to_csv(E2 / 'E2_mechanism.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['grid-gpu', 'grid-her2st', 'mechanism'])
    p.add_argument('--processes', type=int, default=8)
    args = p.parse_args()
    c.write_lock(E2, 'E2 nested tuning')
    started = time.perf_counter()
    {'grid-gpu': run_grid_gpu, 'grid-her2st': lambda: run_grid_her2st(args.processes),
     'mechanism': lambda: run_mechanism(args.processes)}[args.mode]()
    print('E2', args.mode, 'done in', round(time.perf_counter() - started), 's', flush=True)


if __name__ == '__main__':
    main()
