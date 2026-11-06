#!/usr/bin/env python3
"""E2: patient-excluded tuning of Tessera over a fixed 24-configuration grid.

Modes:
  grid-gpu       NSCLC regions and BCC cell types on CUDA
  grid-her2st    HER2ST all labels and tumor boundary, CPU with one thread per process
  mechanism      nested selection per held-out patient, then NoGate/NoContrast at the selected
                 configuration and the same-graph mean on the selected graph
  grid-gpu-wide, grid-her2st-wide, mechanism-wide
                 Amendment A1 (PROTOCOL-R1.md): k in {50, 100} for NSCLC regions and HER2ST, selection
                 over the 40-configuration grid, and the mechanism arms at the new selections
  hwcheck-a2, grid-gpu-a2, mechanism-a2
                 Amendment A2: on a rented A100 80 GB, refit the released and A1-selected lung configurations
                 (hardware check), fit k = 200 for NSCLC regions, select over the 48-configuration lung grid and
                 fit NoGate/NoContrast where k = 200 is selected. A2_K and A2_DRY=1 shrink the run for a dry run.
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import os

import torch

import common_r1 as c
from run_budget import score_partition
from train_core import same_graph_mean, train_core

E2 = c.OUT / 'E2'
GRAPH_K = [6, 15, 30]
GRAPH_K_WIDE = [50, 100]  # Amendment A1
WIDE_COHORTS = ['NSCLC', 'HER2ST']
A2_K = int(os.environ.get('A2_K', '200'))      # Amendment A2; a smaller value only for dry runs
A2_DRY = os.environ.get('A2_DRY') == '1'          # one patient, one setting, one seed
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


def run_grid_gpu(cohorts=('NSCLC', 'BCC'), ks=GRAPH_K, out='E2_grid_gpu.csv'):
    rows = []
    for cohort in cohorts:
        for unit in cohort_units(cohort):
            for graph_k, weight, temp in itertools.product(ks, WEIGHTS, TEMPS):
                for seed in c.SEEDS:
                    rows += fit_grid_point(unit, graph_k, weight, temp, seed, 'cuda', unit['head_k'])
                    torch.cuda.empty_cache()
            print('E2 grid', unit['name'], list(ks), flush=True)
            pd.DataFrame(rows).to_csv(E2 / out, index=False)


def _her2st_job(args):
    stem, graph_k, weight, temp, seed = args
    torch.set_num_threads(1)
    return fit_grid_point(c.her2st_unit(stem), graph_k, weight, temp, seed, 'cpu', 2)


def run_grid_her2st(processes: int, ks=GRAPH_K, out='E2_grid_her2st.csv'):
    jobs = [(s, k, w, t, seed) for s in c.HER2ST for k, w, t in itertools.product(ks, WEIGHTS, TEMPS)
            for seed in c.SEEDS]
    jobs.sort(key=lambda j: -j[1])  # longest fits first
    with mp.get_context('spawn').Pool(processes) as pool:
        rows = [r for part in pool.imap_unordered(_her2st_job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(E2 / out, index=False)


def parse_config(config: str):
    k, w, t, e = config.split('|')
    return int(k[1:]), float(w[1:]), float(t[1:]), int(e[1:])


def selections(wide: bool = False) -> pd.DataFrame:
    grid = pd.concat([pd.read_csv(E2 / 'E2_grid_gpu.csv'), pd.read_csv(E2 / 'E2_grid_her2st.csv')])
    if wide:
        grid = pd.concat([grid[grid.cohort.isin(WIDE_COHORTS)], pd.read_csv(E2 / 'E2_grid_gpu_wide.csv'),
                          pd.read_csv(E2 / 'E2_grid_her2st_wide.csv')])
    return select_from(grid, 'E2_selection_wide.csv' if wide else 'E2_selection.csv')


def select_from(grid: pd.DataFrame, out: str) -> pd.DataFrame:
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
    table.to_csv(E2 / out, index=False)
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


def run_mechanism(processes: int, wide: bool = False):
    table = selections(wide)
    reuse = pd.DataFrame()
    if wide:  # Amendment A1: refit only where the selected configuration changed
        done = pd.read_csv(E2 / 'E2_mechanism.csv')
        wanted = set(zip(table.patient, table.target, table.selected))
        reuse = done[np.array([key in wanted for key in zip(done.patient, done.target, done.config)], dtype=bool)]
        have = set(zip(reuse.patient, reuse.target, reuse.config))
        table = table[np.array([key not in have for key in zip(table.patient, table.target, table.selected)], dtype=bool)]
    gpu = [(r.cohort, r.target, r.patient, r.selected, 'cuda') for r in table.itertuples() if r.cohort != 'HER2ST']
    cpu = [(r.cohort, r.target, r.patient, r.selected, 'cpu') for r in table.itertuples() if r.cohort == 'HER2ST']
    rows = [row for job in gpu for row in _mechanism_job(job)]
    with mp.get_context('spawn').Pool(processes) as pool:
        rows += [r for part in pool.imap_unordered(_mechanism_job, cpu) for r in part]
    out = pd.concat([reuse, pd.DataFrame(rows)]) if wide else pd.DataFrame(rows)
    out.to_csv(E2 / ('E2_mechanism_wide.csv' if wide else 'E2_mechanism.csv'), index=False)


# ------------------------------------------------------------------ Amendment A2 (rented GPU)
def _a2_names():
    return c.LUNG[:1] if A2_DRY else c.LUNG


def _a2_seeds():
    return c.SEEDS[:1] if A2_DRY else c.SEEDS


def _gpu_job(args):
    name, graph_k, weight, temp, seed, arm, epochs = args
    torch.set_num_threads(4)
    unit = next(u for u in cohort_units('NSCLC') if u['name'] == name)
    rows = fit_grid_point(unit, graph_k, weight, temp, seed, 'cuda', unit['head_k'], arm=arm, epochs=epochs)
    torch.cuda.empty_cache()
    return rows


def run_gpu_pool(jobs, processes, out):
    """Fit jobs in parallel GPU workers; the table is rewritten after every fit, so a cut-off run keeps its rows."""
    rows = []
    with mp.get_context('spawn').Pool(processes) as pool:
        for part in pool.imap_unordered(_gpu_job, jobs):
            rows += part
            pd.DataFrame(rows).to_csv(E2 / out, index=False)
            print('A2', out, len(rows), 'rows', flush=True)
    return rows


def lung_grid(*extra: str) -> pd.DataFrame:
    base = pd.read_csv(E2 / 'E2_grid_gpu.csv')
    return pd.concat([base[base.cohort == 'NSCLC'], *[pd.read_csv(E2 / f) for f in ('E2_grid_gpu_wide.csv', *extra)]])


def run_hwcheck_a2(processes):
    sel = select_from(lung_grid(), 'E2_selection_a1_lung.csv')  # the A1 lung selections, from the lung grids alone
    configs = {config_id(6, 0.5, 0.2, 120)} | set(sel.selected)
    settings = sorted({parse_config(cfg)[:3] for cfg in configs})
    if A2_DRY:
        settings = settings[:1]
    jobs = [(name, k, w, t, seed, 'Tessera', (120, 200)) for k, w, t in settings for name in _a2_names()
            for seed in _a2_seeds()]
    return run_gpu_pool(jobs, processes, 'E2_hwcheck_a2.csv')


def run_grid_a2(processes):
    settings = list(itertools.product([A2_K], WEIGHTS, TEMPS))[:1 if A2_DRY else None]
    jobs = [(name, k, w, t, seed, 'Tessera', (120, 200)) for k, w, t in settings for name in _a2_names()
            for seed in _a2_seeds()]
    return run_gpu_pool(jobs, processes, 'E2_grid_gpu_a2.csv')


def run_mechanism_a2(processes):
    table = select_from(lung_grid('E2_grid_gpu_a2.csv'), 'E2_selection_a2.csv')
    done_file = E2 / 'E2_mechanism_wide.csv'  # laptop A1 fits, reused where present
    done = pd.read_csv(done_file) if done_file.exists() else pd.DataFrame(columns=['patient', 'target', 'config'])
    wanted = set(zip(table.patient, table.target, table.selected))
    reuse = done[np.array([key in wanted for key in zip(done.patient, done.target, done.config)], dtype=bool)]
    have = set(zip(reuse.patient, reuse.target, reuse.config))
    todo = table[np.array([key not in have for key in zip(table.patient, table.target, table.selected)], dtype=bool)]
    jobs = [(r.patient, *parse_config(r.selected)[:3], seed, arm, (parse_config(r.selected)[3],))
            for r in todo.itertuples() for arm in ['NoGate', 'NoContrast'] for seed in _a2_seeds()]
    rows = run_gpu_pool(jobs, processes, 'E2_mechanism_a2_fits.csv') if jobs else []
    for r in todo.itertuples():
        unit = next(u for u in cohort_units('NSCLC') if u['name'] == r.patient)
        graph_k = parse_config(r.selected)[0]
        y, classes = unit['targets'][r.target]
        mean, _ = same_graph_mean(unit['x'], unit['coords'], unit['fields'], graph_k)
        for seed in _a2_seeds():
            ari, nmi = score_partition(mean, y, len(classes), seed)
            rows.append(dict(cohort='NSCLC', patient=r.patient, target=r.target, arm=f'SameGraphMean-k{graph_k}',
                             graph_k=graph_k, weight=None, temp=None, epochs=None, config=r.selected, seed=seed,
                             ari=ari, nmi=nmi, fit_seconds=None))
    pd.concat([reuse, pd.DataFrame(rows)]).to_csv(E2 / 'E2_mechanism_a2.csv', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['grid-gpu', 'grid-her2st', 'mechanism', 'grid-gpu-wide', 'grid-her2st-wide',
                                    'mechanism-wide', 'hwcheck-a2', 'grid-gpu-a2', 'mechanism-a2'])
    p.add_argument('--processes', type=int, default=8)
    args = p.parse_args()
    if args.mode.endswith('-a2'):
        c.write_lock(E2, 'E2 Amendment A2: 200-neighbour lung graphs on a rented GPU', name='run-lock-A2.json',
                     extra=dict(a2_k=A2_K, dry_run=A2_DRY))
    elif args.mode.endswith('-wide'):
        c.write_lock(E2, f'E2 Amendment A1: wider encoder graphs ({args.mode})',
                     name='run-lock-A1.json' if args.mode.startswith('grid') else 'run-lock-A1-mechanism.json')
    else:
        c.write_lock(E2, 'E2 nested tuning')
    started = time.perf_counter()
    {'grid-gpu': run_grid_gpu, 'grid-her2st': lambda: run_grid_her2st(args.processes),
     'mechanism': lambda: run_mechanism(args.processes),
     'grid-gpu-wide': lambda: run_grid_gpu(('NSCLC',), GRAPH_K_WIDE, 'E2_grid_gpu_wide.csv'),
     'grid-her2st-wide': lambda: run_grid_her2st(args.processes, GRAPH_K_WIDE, 'E2_grid_her2st_wide.csv'),
     'mechanism-wide': lambda: run_mechanism(args.processes, wide=True),
     'hwcheck-a2': lambda: run_hwcheck_a2(args.processes),
     'grid-gpu-a2': lambda: run_grid_a2(args.processes),
     'mechanism-a2': lambda: run_mechanism_a2(args.processes)}[args.mode]()
    print('E2', args.mode, 'done in', round(time.perf_counter() - started), 's', flush=True)


if __name__ == '__main__':
    main()
