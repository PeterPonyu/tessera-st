#!/usr/bin/env python3
"""Frozen-estimator CRC pilot; all writes are confined to this validation tree."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
import torch

HERE = Path(__file__).resolve().parent
REV = HERE.parent
sys.path[:0] = [str(REV / 'source/scripts'), str(REV / 'source/src'), str(REV / 'validation-external')]
import run_experiments as rx
from common import encode, sha, dump, read
from run_her2st_validation import preprocess

SECTIONS = ['SN84_A120838_Rep1', 'SN124_A595688_Rep2', 'SN124_A551763_Rep2']
SEEDS = [1, 2, 3]
ARMS = ['PCA50', 'UniformSameGraph', 'Tessera', 'NoGate']
OUT = HERE / 'results'


def bindings():
    lock = read(HERE / 'protocol-lock.json')
    if sha(HERE / 'PROTOCOL.md') != lock['protocol_sha256']:
        raise ValueError('protocol differs from pre-data lock')
    files = [REV / 'source/scripts/run_experiments.py', REV / 'source/scripts/common.py',
             REV / 'validation-external/run_her2st_validation.py']
    files += sorted((REV / 'source/src').rglob('*.py'))
    return dict(protocol_sha256=lock['protocol_sha256'], script_sha256=sha(Path(__file__)),
                frozen_sha256={str(p.relative_to(REV)): sha(p) for p in files})


def member(z, predicate, description):
    names = [n for n in z.namelist() if not n.startswith('__MACOSX/') and predicate(n)]
    if len(names) != 1:
        raise ValueError(f'{description}: expected one member, found {names}')
    return names[0]


def load_raw(section):
    raw = HERE / 'raw'
    archive = raw / (section + '.zip')
    receipt = read(HERE / 'fetch-receipt.json')
    record = next(r for r in receipt['files'] if r['filename'] == archive.name)
    if sha(archive) != record['sha256']:
        raise ValueError('raw archive binding')
    with zipfile.ZipFile(archive) as z:
        name = member(z, lambda n: n.endswith('/filtered_feature_bc_matrix.h5'), 'filtered count H5')
        with h5py.File(io.BytesIO(z.read(name)), 'r') as f:
            g = f['matrix']
            counts = sparse.csc_matrix((g['data'][:], g['indices'][:], g['indptr'][:]),
                                      shape=tuple(g['shape'][:])).T.tocsr()
            barcodes = np.asarray(g['barcodes'][:], str)
            genes = np.asarray(g['features/name'][:], str)
            if 'feature_type' in g['features']:
                gene_expression = np.asarray(g['features/feature_type'][:], str) == 'Gene Expression'
                counts, genes = counts[:, gene_expression], genes[gene_expression]
        name = member(z, lambda n: n.endswith('/tissue_positions_list.csv') or n.endswith('/tissue_positions.csv'), 'positions')
        text = z.read(name).decode()
        if text.splitlines()[0].startswith('barcode,'):
            pos = pd.read_csv(io.StringIO(text)).set_index('barcode')
        else:
            pos = pd.read_csv(io.StringIO(text), header=None,
                 names=['barcode', 'in_tissue', 'array_row', 'array_col',
                        'pxl_row_in_fullres', 'pxl_col_in_fullres']).set_index('barcode')
        sf_name = member(z, lambda n: n.endswith('/scalefactors_json.json'), 'scalefactors')
        scales = json.loads(z.read(sf_name))
        image_name = member(z, lambda n: n.endswith('/tissue_hires_image.png'), 'H&E hires')
        image = z.read(image_name)
    if len(set(barcodes)) != len(barcodes) or not pos.index.is_unique:
        raise ValueError('duplicate expression/position barcode')
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or not np.equal(counts.data, np.round(counts.data)).all():
        raise ValueError('counts are not finite nonnegative integers')
    if not set(barcodes).issubset(set(pos.index)):
        raise ValueError('expression barcode absent from positions')
    positions = pos.reindex(barcodes)
    keep = (positions.in_tissue.to_numpy() == 1) & (np.asarray(counts.sum(1)).ravel() > 0)
    counts, barcodes, positions = counts[keep], barcodes[keep], positions.loc[keep]
    coords = positions[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(float)
    if not np.isfinite(coords).all() or len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('nonfinite or duplicated tissue positions')
    with zipfile.ZipFile(raw / 'Pathology_SpotAnnotations.zip') as z:
        name = member(z, lambda n: n.endswith('/Pathologist_Annotations_' + section + '.csv'), 'pathology CSV')
        labels = pd.read_csv(io.BytesIO(z.read(name)), dtype=str).set_index('Barcode')
    label_column = [c for c in labels if c in ['Pathologist Annotation', 'Pathologist Annotations']]
    if len(label_column) != 1 or not labels.index.is_unique:
        raise ValueError('ambiguous pathology schema or duplicate barcode')
    label_series = labels[label_column[0]]
    # Denominator is every published annotated spot in the released tissue mask.
    tissue_labels = labels.index.intersection(pos.index[pos.in_tissue == 1])
    matched = tissue_labels.intersection(pd.Index(barcodes))
    rate = len(matched) / max(1, len(tissue_labels))
    if rate < .95:
        raise ValueError(f'annotation-to-filtered-count match {rate:.4f} below .95')
    raw_labels = label_series.reindex(barcodes).fillna('').to_numpy(str)
    y, classes = encode(raw_labels)
    if (y >= 0).sum() < 100 or len(classes) < 2:
        raise ValueError('fewer than 100 labels or two classes')
    meta = dict(section=section, patient=section.split('_')[1], n_tissue=len(barcodes),
        n_annotated=int((y >= 0).sum()), n_published_labels=len(labels),
        n_published_tissue_labels=len(tissue_labels), match_rate=rate, classes=classes,
        class_counts={c: int((raw_labels == c).sum()) for c in classes},
        unmatched_label_barcodes=sorted(set(labels.index) - set(barcodes)),
        n_outside_positions=int((~labels.index.isin(pos.index)).sum()),
        n_genes=counts.shape[1], raw_archive_sha256=sha(archive),
        pathology_archive_sha256=sha(raw / 'Pathology_SpotAnnotations.zip'),
        image_scalefactor=float(scales['tissue_hires_scalef']),
        spot_diameter_fullres=float(scales['spot_diameter_fullres']))
    return counts, genes, barcodes, coords, raw_labels, y, meta, image


def unique_edges(edge):
    pairs = np.sort(edge.T, axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique.T, inverse


def collapse(values, inverse, n):
    return np.bincount(inverse, weights=values, minlength=n) / np.bincount(inverse, minlength=n)


def prepare(section):
    expected = bindings()
    dest = HERE / 'inputs' / section
    if (dest / 'input.json').exists():
        rec = read(dest / 'input.json')
        if rec['bindings'] != expected or sha(dest / 'input.npz') != rec['input_sha256']:
            raise ValueError('prepared input changed')
        return
    counts, genes, bc, xy, raw_labels, y, meta, image = load_raw(section)
    # Reuse the established normalization/PCA implementation; labels are not arguments.
    x, normalized, removed, components, explained = preprocess(counts.toarray().astype(float), genes)
    fields = np.full(len(x), section)
    edge, _ = rx.graph(xy, fields, 'Tessera', 1)
    pair, _ = unique_edges(edge)
    valid = (y[pair[0]] >= 0) & (y[pair[1]] >= 0)
    seam = y[pair[0]][valid] != y[pair[1]][valid]
    if seam.sum() < 20 or (~seam).sum() < 20:
        raise ValueError('fewer than 20 undirected seam/within edges')
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'input.npz', x=x, coords=xy, field=fields,
                        ref=y, ref_name=raw_labels, barcodes=bc, edge_index=edge)
    (dest / 'tissue_hires_image.png').write_bytes(image)
    meta.update(pca_components=components, pca_variance_explained=explained,
        count_normalization=normalized, removed_control_genes=removed,
        n_directed_multiset_edges=edge.shape[1], n_unique_edges=pair.shape[1],
        n_scored_edges=int(valid.sum()), n_seam_edges=int(seam.sum()),
        n_within_edges=int((~seam).sum()), auxiliary_fit_K=2,
        scoring_K=len(meta['classes']), bindings=expected,
        input_sha256=sha(dest / 'input.npz'), image_sha256=sha(dest / 'tissue_hires_image.png'))
    dump(dest / 'input.json', meta)
    print('PREPARED', section, len(x), 'tissue spots;', meta['n_annotated'],
          'labels;', meta['n_seam_edges'], 'unique seams', flush=True)


def run(section):
    prepare(section)
    expected = bindings()
    path = HERE / 'inputs' / section
    d = np.load(path / 'input.npz', allow_pickle=False)
    meta = read(path / 'input.json')
    x, xy, fields, y, edge = (d[k] for k in ['x', 'coords', 'field', 'ref', 'edge_index'])
    if sha(path / 'input.npz') != meta['input_sha256']:
        raise ValueError('input binding')
    same_graph = rx.neighbor_embedding(x, edge).astype(np.float32)
    for arm in ARMS:
        for seed in SEEDS:
            stem = OUT / section / f'{arm}-seed{seed}'
            if stem.with_suffix('.json').exists():
                rec = read(stem.with_suffix('.json'))
                if rec['bindings'] != expected or rec['input_sha256'] != meta['input_sha256'] or sha(stem.with_suffix('.npz')) != rec['prediction_sha256']:
                    raise ValueError('cached fit binding')
                continue
            start = time.perf_counter()
            fit_seconds, history, params = 0., [], 0
            gate = np.empty(0, dtype=np.float32)
            if arm in ['Tessera', 'NoGate']:
                model, emb, boundary, gate, fitted_edge, dist, history, fit_seconds, memory, params = rx.train_net(x, xy, fields, 2, arm, seed, 'cpu')
                if not np.array_equal(fitted_edge, edge):
                    raise ValueError('fitted graph differs from control graph')
                del model
            else:
                emb = x if arm == 'PCA50' else same_graph
            pred = KMeans(len(meta['classes']), n_init=10, random_state=seed).fit_predict(emb)
            if not np.isfinite(emb).all() or not np.isfinite(gate).all():
                raise ValueError('nonfinite predictions')
            stem.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(stem.with_suffix('.npz'), embedding=emb, labels=pred,
                                edge_index=edge, last_gate=gate)
            rec = dict(section=section, patient=meta['patient'], arm=arm, seed=seed,
                device='cpu', auxiliary_fit_K=2, scoring_K=len(meta['classes']),
                n_tissue=len(x), n_annotated=int((y >= 0).sum()),
                ari=float(adjusted_rand_score(y[y >= 0], pred[y >= 0])),
                fit_seconds=fit_seconds, total_seconds=time.perf_counter() - start,
                trainable_parameters=params, history=history, bindings=expected,
                input_sha256=meta['input_sha256'], prediction_sha256=sha(stem.with_suffix('.npz')))
            dump(stem.with_suffix('.json'), rec)
            print('FITTED', section, arm, seed, f'ARI={rec["ari"]:.4f}',
                  f'{rec["total_seconds"]:.1f}s', flush=True)


def bootstrap(values):
    values = np.asarray(values, float)
    rng = np.random.RandomState(20261011)
    means = values[rng.randint(0, len(values), (10000, len(values)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(values.mean()), lower=float(lo), upper=float(hi),
                n_independent_patients=len(values), bootstrap_resamples=10000)


def summarize():
    expected = bindings()
    arms, edges, nulls = [], [], []
    for section in SECTIONS:
        path = HERE / 'inputs' / section
        d = np.load(path / 'input.npz', allow_pickle=False)
        meta = read(path / 'input.json')
        edge, x, y = d['edge_index'], d['x'], d['ref']
        pair, inverse = unique_edges(edge)
        valid = (y[pair[0]] >= 0) & (y[pair[1]] >= 0)
        seam = y[pair[0]][valid] != y[pair[1]][valid]
        scores = dict(RawFeature=np.linalg.norm(x[pair[0]] - x[pair[1]], axis=1),
            UniformMeanFeature=np.linalg.norm(rx.neighbor_embedding(x, edge)[pair[0]] -
                                             rx.neighbor_embedding(x, edge)[pair[1]], axis=1))
        gate_seeds = []
        for arm in ARMS:
            for seed in SEEDS:
                stem = OUT / section / f'{arm}-seed{seed}'
                rec = read(stem.with_suffix('.json'))
                if rec['bindings'] != expected or sha(stem.with_suffix('.npz')) != rec['prediction_sha256']:
                    raise ValueError('fit receipt binding')
                p = np.load(stem.with_suffix('.npz'), allow_pickle=False)
                recomputed = float(adjusted_rand_score(y[y >= 0], p['labels'][y >= 0]))
                if abs(recomputed - rec['ari']) > 1e-12:
                    raise ValueError('saved ARI differs from recomputed ARI')
                arms.append({k: rec[k] for k in ['section', 'patient', 'arm', 'seed', 'ari', 'fit_seconds', 'total_seconds']})
                if arm == 'Tessera':
                    gate_seeds.append(collapse(1 - p['last_gate'], inverse, pair.shape[1]))
        scores['Gate'] = np.mean(gate_seeds, axis=0)
        for name, values in scores.items():
            edges.append(dict(section=section, patient=meta['patient'], score=name,
                auroc=float(roc_auc_score(seam, values[valid])),
                n_unique_edges=pair.shape[1], n_scored_edges=int(valid.sum()),
                n_seam_edges=int(seam.sum()), n_within_edges=int((~seam).sum())))
        rng = np.random.RandomState(20261011)
        for iteration in range(1000):
            shuffled = rng.permutation(y)
            ok = (shuffled[pair[0]] >= 0) & (shuffled[pair[1]] >= 0)
            target = shuffled[pair[0]][ok] != shuffled[pair[1]][ok]
            if len(np.unique(target)) < 2:
                raise ValueError('degenerate label shuffle')
            for name, values in scores.items():
                nulls.append(dict(section=section, score=name, shuffle=iteration,
                                   auroc=float(roc_auc_score(target, values[ok]))))
        np.savez_compressed(OUT / section / 'boundary_scores.npz', unique_edge_index=pair,
                            scored_mask=valid, seam=seam, **scores)
    arms, edges, nulls = pd.DataFrame(arms), pd.DataFrame(edges), pd.DataFrame(nulls)
    for name, frame in [('arms', arms), ('seams', edges), ('label_shuffle', nulls)]:
        frame.to_csv(OUT / (name + '.csv'), index=False)
    patient = arms.groupby(['patient', 'arm']).ari.mean().unstack().reindex([s.split('_')[1] for s in SECTIONS])
    patient['Tessera_minus_UniformSameGraph'] = patient.Tessera - patient.UniformSameGraph
    patient['NoGate_minus_Tessera'] = patient.NoGate - patient.Tessera
    patient.to_csv(OUT / 'patients.csv')
    seam = edges.pivot(index='patient', columns='score', values='auroc').reindex(patient.index)
    seam['Gate_minus_RawFeature'] = seam.Gate - seam.RawFeature
    seam['Gate_minus_UniformMeanFeature'] = seam.Gate - seam.UniformMeanFeature
    seam.to_csv(OUT / 'patient_seams.csv')
    stats = {column: bootstrap(seam[column]) for column in seam}
    stats.update({column: bootstrap(patient[column]) for column in ['Tessera_minus_UniformSameGraph', 'NoGate_minus_Tessera']})
    verdict = dict(gate_discrimination=stats['Gate']['lower'] > .5,
        gate_advantage_over_raw=stats['Gate_minus_RawFeature']['lower'] > 0,
        learned_ari_gain_over_mean_margin_0_02=stats['Tessera_minus_UniformSameGraph']['lower'] > .02)
    receipt = dict(bindings=expected, environment=rx.environment(), device='cpu',
        patients=len(SECTIONS), fit_count=2 * 3 * len(SECTIONS), score_rows=len(arms),
        unique_patient_ids=list(patient.index), auxiliary_fit_K=2,
        statistics=stats, pilot_criteria=verdict, total_fit_seconds=float(arms.fit_seconds.sum()),
        input_sha256={s: sha(HERE / 'inputs' / s / 'input.npz') for s in SECTIONS},
        output_sha256={str(p.relative_to(HERE)): sha(p) for p in sorted(OUT.rglob('*'))
                       if p.is_file() and p.name != 'run-receipt.json'})
    dump(OUT / 'run-receipt.json', receipt)
    print('COMPLETE', json.dumps(dict(statistics=stats, pilot_criteria=verdict), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--section', choices=SECTIONS)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    if args.summarize:
        summarize()
    elif args.section:
        prepare(args.section) if args.prepare else run(args.section)
    else:
        for section in SECTIONS:
            run(section)
        summarize()


if __name__ == '__main__':
    main()
