#!/usr/bin/env python3
"""Read-only verification of retained predictions, receipts and score arithmetic.

No neural fitting, PCA fitting, KMeans fitting, acquisition or document rebuilding.
Score reconstruction is limited to saved predictions actually retained by the
historical producers; other blocks receive table/selection/source checks.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ['source/scripts', 'source/src', 'validation-external', 'validation-extension']]
from common import OBJECTS, SEEDS, ARMS, encode, read, sha
import run_experiments as rx
import run_transfer as rt
import run_followup as fu
import audit_followup as af
import run_her2st_validation as hv
import run_extension as ext


def close(actual, expected, message='', tol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol, err_msg=message)


def compare(actual, expected, label=''):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), label + ' keys'
        for key in expected:
            compare(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label + ' length'
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, label + '/' + str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        close(actual, expected, label)
    else:
        assert actual == expected, label


def frozen_sources():
    bindings = 0
    for path in [ROOT / 'results/run-lock.json', ROOT / 'results/transfer/run-lock.json']:
        for name, digest in read(path)['code_sha256'].items():
            assert sha(ROOT / name) == digest, 'Frozen source: ' + name
            bindings += 1
    return bindings


def primary():
    ledger = pd.read_csv(ROOT / 'source/manuscript/ledgers/all_primary_scores.csv').set_index(['object', 'arm', 'seed'])
    count, embeddings = 0, 0
    for name in OBJECTS:
        base = ROOT / 'data' / name
        meta = read(base / 'input.json')
        assert sha(base / 'input.npz') == meta['matrix_sha256']
        with np.load(base / 'input.npz', allow_pickle=False) as data:
            x, y, fields = data['x'], data['ref'], data['field']
            assert len(x) == meta['n_selected']
            assert hashlib.sha256('\n'.join(data['obs_id']).encode()).hexdigest() == meta['selected_id_sha256']
            for arm in ARMS:
                for seed in SEEDS:
                    stem = ROOT / 'results' / name / f'{arm}-seed{seed}'
                    receipt = read(stem.with_suffix('.json'))
                    assert receipt['input_sha256'] == meta['matrix_sha256']
                    assert sha(stem.with_suffix('.npz')) == receipt['prediction_sha256']
                    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
                        pred, edge = saved['labels'], saved['edge_index']
                        assert np.isfinite(saved['embedding']).all()
                        assert len(saved['embedding']) == len(x) == len(pred)
                        assert (fields[edge[0]] == fields[edge[1]]).all()
                        if arm == 'SelfOnly':
                            assert np.array_equal(edge[0], edge[1])
                        else:
                            assert not (edge[0] == edge[1]).any()
                        result = rx.scores(y, pred, edge, gate=saved['last_gate'] if 'last_gate' in saved else None,
                                           secondary=data['secondary_name'] if 'secondary_name' in data else None)
                        for metric, value in result.items():
                            compare(value, receipt['scores'][metric], name + '/' + metric)
                            if metric in ledger.columns and value is not None:
                                close(value, ledger.loc[(name, arm, seed), metric], 'Primary ledger ' + metric)
                        if 'purity' in data:
                            for threshold, value in receipt['scores']['purity_strata'].items():
                                mask = data['purity'] >= float(threshold)
                                assert int(mask.sum()) == value['n']
                                close(adjusted_rand_score(y[mask], pred[mask]), value['ari'], 'Purity-stratum ARI')
                        if arm == 'PCA50':
                            close(saved['embedding'], x, 'PCA50 feature identity', 0)
                        if arm == 'PCA30':
                            close(saved['embedding'], x[:, :min(30, x.shape[1])], 'PCA30 feature identity', 0)
                        if arm == 'NeighborMean':
                            close(saved['embedding'], rx.neighbor_embedding(x, edge).astype(np.float32), 'Same-graph mean', 2e-6)
                        embeddings += 1
                        count += 1
    assert count == len(ledger) == 351
    return {'saved_score_rows_recomputed': count, 'saved_prediction_and_embedding_files': embeddings,
            'derived_inputs_bound': len(OBJECTS), 'scope': 'ARI/NMI/secondary labels/edge and gate metrics from saved labels and gates; no clustering or training'}


def transfer():
    names = OBJECTS[:5]
    data = {name: np.load(ROOT / 'data' / name / 'input.npz', allow_pickle=False) for name in names}
    ledger = pd.read_csv(ROOT / 'source/manuscript/ledgers/all_transfer_scores.csv').set_index(['patient', 'arm', 'seed', 'target'])
    count = 0
    try:
        for target in names:
            sources = [name for name in names if name != target]
            source_ids = np.concatenate([np.asarray([name + '|' + obs for obs in data[name]['obs_id']]) for name in sources])
            target_ids = np.asarray([target + '|' + obs for obs in data[target]['obs_id']])
            transform = ROOT / 'results/transfer' / f'{target}-transform.npz'
            with np.load(transform, allow_pickle=False) as saved_transform:
                assert np.array_equal(saved_transform['source_obs_id'], source_ids)
                assert np.array_equal(saved_transform['target_obs_id'], target_ids)
                assert not set(source_ids) & set(target_ids)
                assert np.isfinite(saved_transform['pca_components']).all()
            labels = {key: np.concatenate([data[name][key] for name in sources]) for key in ['ref_name', 'secondary_name']}
            for arm in ['PCA50', 'SelfOnly', 'Tessera']:
                for seed in SEEDS:
                    stem = ROOT / 'results/transfer' / f'{target}-{arm}-seed{seed}'
                    receipt = read(stem.with_suffix('.json'))
                    assert receipt['source_patients'] == sources
                    assert receipt['patient_disjoint'] and not receipt['target_transform_fitted']
                    assert receipt['transform_sha256'] == sha(transform)
                    assert receipt['prediction_sha256'] == sha(stem.with_suffix('.npz'))
                    for name, digest in receipt['input_sha256'].items():
                        assert sha(ROOT / 'data' / name / 'input.npz') == digest
                    if receipt['weight_sha256']:
                        assert sha(stem.with_suffix('.pt')) == receipt['weight_sha256']
                    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
                        se, te = saved['source_embedding'], saved['target_embedding']
                        assert np.isfinite(se).all() and np.isfinite(te).all()
                        for key, title in [('ref_name', 'region'), ('secondary_name', 'cell_type')]:
                            pred, cent, classes, margin = rt.centroid_predict(se, te, labels[key])
                            assert np.array_equal(pred, saved[title + '_prediction'])
                            assert np.array_equal(classes, saved[title + '_classes'])
                            close(cent, saved[title + '_centroid'], 'Source-only centroid', 2e-6)
                            close(margin, saved[title + '_margin'], 'Saved margin', 1e-5)
                            scores = rt.transfer_score(data[target][key], pred, classes)
                            compare(scores, receipt['scores'][title], 'Transfer scores')
                            for metric in ledger.columns:
                                close(scores[metric], ledger.loc[(target, arm, seed, title), metric], 'Transfer ledger ' + metric)
                            count += 1
    finally:
        for values in data.values():
            values.close()
    assert count == len(ledger) == 90
    return {'saved_target_score_rows_recomputed': count, 'saved_embedding_files': 45,
            'source_only_transform_files': 5, 'saved_weight_files_bound': 30,
            'scope': 'Metrics and source-label centroid predictions from retained source/target embeddings; no encoder fitting or PCA fitting'}


def followup():
    # The original auditor writes only its report after verifying the existing
    # diagnostic/receipt. Capture that report in memory to keep extraction bytes
    # unchanged. Missing diagnostics are errors, never silently generated here.
    assert af.DIAGNOSTIC.is_file() and af.RECEIPT.is_file()
    captured = {}
    previous = af.dump
    af.dump = lambda path, value: captured.update(value)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            af.main()
    finally:
        af.dump = previous
    assert captured['unrecovered_historical_modes'] == []
    return {'counts': captured['counts'], 'saved_gate_profiles_recomputed': 39,
            'historical_sources_available': True,
            'scope': 'Score/summary arithmetic, nested selection and saved gate weights; whole/null prediction files were not retained by producers'}


def external_and_extension():
    folder = ROOT / 'validation-external'
    record = read(folder / 'results/summary.json')
    assert record['protocol_sha256'] == sha(folder / 'PROTOCOL.md')
    assert record['script_sha256'] == sha(folder / 'run_her2st_validation.py')
    for name, digest in record['frozen_estimator_sha256'].items():
        assert sha(ROOT / 'source/scripts' / name) == digest
    grid = pd.read_csv(folder / 'results/scale_controls.csv')
    fits = pd.read_csv(folder / 'results/tessera_scores.csv')
    patients = pd.read_csv(folder / 'results/primary_patients.csv')
    names = patients.patient.tolist()
    for row in patients.itertuples():
        selected = hv.select_control(grid, names, row.patient)
        assert selected == row.selected_control
        close(fits.loc[fits.patient == row.patient, 'ari'].mean(), row.tessera_ari)
        close(grid.loc[(grid.patient == row.patient) & (grid.config == selected), 'ari'].mean(), row.nested_control_ari)
        close(grid.loc[(grid.patient == row.patient) & (grid.kind == 'PCA50'), 'ari'].mean(), row.pca50_ari)
        close(row.tessera_ari - row.nested_control_ari, row.tessera_minus_nested)
        close(row.tessera_ari - row.pca50_ari, row.tessera_minus_pca50)
    compare(hv.sign_summary(patients.tessera_minus_nested.tolist(), fu.exact_sign_flip), record['primary_tessera_minus_nested'])
    compare(hv.sign_summary(patients.tessera_minus_pca50.tolist(), fu.exact_sign_flip), record['primary_tessera_minus_pca50'])
    assert len(grid) == 720 and len(fits) == 24 and len(patients) == 8
    extension = ROOT / 'validation-extension'
    nested_checks = 0
    score_rows = 0
    for mode in ['hardening', 'dev', 'breadth']:
        receipt = read(extension / 'results' / f'run-{mode}.json')
        assert receipt['protocol_sha256'] == sha(extension / 'PROTOCOL_EXTENSION.md')
        assert receipt['script_sha256'] == sha(extension / 'run_extension.py')
        for name, digest in receipt['frozen_sha256'].items():
            assert sha(ROOT / 'source/scripts' / name) == digest
        for name, digest in receipt['validation_external_sha256'].items():
            assert sha(folder / name) == digest
        arms = pd.read_csv(extension / 'results' / mode / 'arms.csv')
        score_rows += len(arms)
        result = receipt['result']
        if mode == 'dev':
            means = arms.groupby(['object', 'arm']).ari.mean().unstack()
            for name in means.index:
                for arm in means.columns:
                    close(means.loc[name, arm], result['per_object'][name][arm])
            continue
        controls = pd.read_csv(extension / 'results' / mode / 'grid.csv')
        entries = result['targets'] if mode == 'hardening' else {'region': result}
        for target, entry in entries.items():
            sub = controls[controls.target == target]
            for unit in entry['units']:
                chosen = ext.select_config(sub, entry['units'], unit)
                assert chosen == entry['selected_control'][unit]
                nested_checks += 1
            wide = ext.wide_means(arms, controls, entry['selected_control'], target).loc[entry['units']]
            compare({unit: {key: float(wide.loc[unit, key]) for key in wide.columns} for unit in wide.index}, entry['per_unit'])
            for contrast in entry.get('contrasts', {}).values():
                compare(ext.contrast(wide, contrast['a'], contrast['b'], interval='bootstrap95' in contrast), contrast)
    # The hardening runner retained map partitions for seed 1. Those permit
    # direct ARI checks for five arms/controls in three target definitions.
    partition_rows = 0
    arms = pd.read_csv(extension / 'results/hardening/arms.csv')
    controls = pd.read_csv(extension / 'results/hardening/grid.csv')
    for target in ['all', 'compartments', 'boundary_T1']:
        partitions = pd.read_csv(extension / 'results/hardening' / f'partitions_{target}.csv', keep_default_na=False)
        for unit, values in partitions.groupby('unit'):
            y, classes = encode(values.reference.to_numpy())
            valid = y >= 0
            for col, arm in [('tessera', 'Tessera'), ('expression_weighted', 'ExpressionWeighted'), ('uniform_same_graph', 'UniformSameGraph')]:
                row = arms[(arms.unit == unit) & (arms.target == target) & (arms.seed == 1) & (arms.arm == arm)]
                assert len(row) == 1
                close(adjusted_rand_score(y[valid], values[col].to_numpy()[valid]), float(row.ari.iloc[0]), 'Saved partition ARI')
                partition_rows += 1
            for col, config in [('pca50', 'PCA50|None|None'), ('control', values.control_config.iloc[0])]:
                row = controls[(controls.unit == unit) & (controls.target == target) & (controls.seed == 1) & (controls.config == config)]
                assert len(row) == 1
                close(adjusted_rand_score(y[valid], values[col].to_numpy()[valid]), float(row.ari.iloc[0]), 'Saved control partition ARI')
                partition_rows += 1
    return {'her2st_patients': len(patients), 'her2st_grid_rows': len(grid), 'her2st_fit_score_rows': len(fits),
            'extension_arm_score_rows': score_rows, 'patient_or_object_excluded_selections_recomputed': nested_checks,
            'hardening_seed1_partition_ari_rows_recomputed': partition_rows,
            'scope': 'Receipts, patient-excluded controls, table means/contrasts and retained seed-1 partitions; remaining learned embeddings were not retained'}


def crc():
    folder = ROOT / 'validation-independent-crc'
    primary = read(folder / 'results/run-receipt.json')
    qc = read(folder / 'annotation-qc/run-receipt.json')
    expected = primary['bindings']
    assert sha(folder / 'PROTOCOL.md') == expected['protocol_sha256'] == read(folder / 'protocol-lock.json')['protocol_sha256']
    assert sha(folder / 'run_pilot.py') == expected['script_sha256']
    for name, digest in expected['frozen_sha256'].items():
        assert sha(ROOT / name) == digest
    assert qc['primary_receipt_sha256'] == sha(folder / 'results/run-receipt.json')
    assert qc['script_sha256'] == sha(folder / 'analyze_annotation_qc.py')
    assert qc['protocol_sha256'] == sha(folder / 'ANNOTATION_QC_PROTOCOL.md') == read(folder / 'ANNOTATION_QC_PROTOCOL.lock.json')['sha256']
    for receipt in [primary, qc]:
        for name, digest in receipt['output_sha256'].items():
            assert sha(folder / name) == digest
    fetch = read(folder / 'fetch-receipt.json')
    assert fetch['script_sha256'] == sha(folder / 'fetch_pilot.py')
    assert fetch['fetch_gate_sha256'] == sha(ROOT / 'source/provenance/acquisition/fetch_gate.py')
    execution = read(folder / 'results/execution.json')
    assert execution['invocation_script_sha256'] == sha(folder / 'run_singlethread.py')
    assert execution['implementation_sha256'] == sha(folder / 'run_pilot.py')
    assert execution['runtime_decision_sha256'] == sha(folder / 'runtime-decision.json')
    assert execution['torch_num_threads'] == 1 and execution['device'] == 'cpu'
    figures = read(folder / 'figures/figure-receipt.json')
    assert figures['script_sha256'] == sha(folder / 'render_pilot.py')
    for name, digest in figures['evidence_sha256'].items():
        assert sha(folder / name) == digest
    for name, digest in figures['figure_sha256'].items():
        assert sha(folder / 'figures' / name) == digest
    sections = ['SN84_A120838_Rep1', 'SN124_A595688_Rep2', 'SN124_A551763_Rep2']
    ari_rows, edge_rows, null_rows = 0, 0, 0
    for mode in ['results', 'annotation-qc']:
        arms = pd.read_csv(folder / mode / 'arms.csv')
        edges = pd.read_csv(folder / mode / 'seams.csv')
        null = pd.read_csv(folder / mode / 'label_shuffle.csv')
        for section in sections:
            base = folder / 'inputs' / section
            meta = read(base / 'input.json')
            assert sha(base / 'input.npz') == meta['input_sha256'] == primary['input_sha256'][section]
            assert sha(base / 'tissue_hires_image.png') == meta['image_sha256']
            with np.load(base / 'input.npz', allow_pickle=False) as data:
                edge, raw = data['edge_index'], data['ref_name']
                assert not (edge[0] == edge[1]).any()
                assert (data['field'][edge[0]] == data['field'][edge[1]]).all()
                pairs = sorted({tuple(sorted(e)) for e in edge.T})
                pair = np.asarray(pairs).T
                keep = data['ref'] >= 0
                if mode == 'annotation-qc':
                    keep &= np.char.lower(np.char.strip(raw)) != 'exclude'
                names = sorted(set(raw[keep])); lookup = {name: i for i, name in enumerate(names)}
                y = np.asarray([lookup[name] if ok else -1 for name, ok in zip(raw, keep)])
                valid = (y[pair[0]] >= 0) & (y[pair[1]] >= 0)
                seam = y[pair[0]][valid] != y[pair[1]][valid]
                assert keep.sum() >= 100 and seam.sum() >= 20 and (~seam).sum() >= 20
                score_values = {'RawFeature': np.linalg.norm(data['x'][pair[0]] - data['x'][pair[1]], axis=1)}
                mean = np.zeros_like(data['x']); degree = np.zeros(len(mean), int)
                for src, dst in edge.T:
                    mean[dst] += data['x'][src]; degree[dst] += 1
                mean = mean / degree[:, None]
                score_values['UniformMeanFeature'] = np.linalg.norm(mean[pair[0]] - mean[pair[1]], axis=1)
                gates = []
                with np.load(folder / 'results' / section / 'boundary_scores.npz', allow_pickle=False) as saved_scores:
                    assert np.array_equal(pair, saved_scores['unique_edge_index'])
                    for arm in ['PCA50', 'UniformSameGraph', 'Tessera', 'NoGate']:
                        for seed in SEEDS:
                            stem = folder / 'results' / section / f'{arm}-seed{seed}'
                            receipt = read(stem.with_suffix('.json'))
                            assert receipt['bindings'] == expected and receipt['auxiliary_fit_K'] == 2
                            assert sha(stem.with_suffix('.npz')) == receipt['prediction_sha256']
                            with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
                                assert np.array_equal(saved['edge_index'], edge)
                                assert np.isfinite(saved['embedding']).all()
                                with np.load(folder / mode / section / f'{arm}-seed{seed}.npz', allow_pickle=False) as scored:
                                    pred = scored['labels']
                                row = arms[(arms.section == section) & (arms.arm == arm) & (arms.seed == seed)]
                                assert len(row) == 1
                                close(adjusted_rand_score(y[keep], pred[keep]), float(row.ari.iloc[0]), 'CRC saved ARI')
                                ari_rows += 1
                                if arm == 'PCA50':
                                    close(saved['embedding'], data['x'], 'CRC PCA input', 0)
                                if arm == 'UniformSameGraph':
                                    close(saved['embedding'], mean, 'CRC same graph mean', 2e-6)
                                if arm == 'Tessera':
                                    values = {}
                                    for e, score in zip(edge.T, np.subtract(np.float32(1), saved['last_gate'])):
                                        values.setdefault(tuple(sorted(e)), []).append(float(score))
                                    gates.append(np.asarray([np.mean(values[p]) for p in pairs]))
                    score_values['Gate'] = np.mean(gates, axis=0)
                    for name, values in score_values.items():
                        close(values, saved_scores[name], 'CRC edge score ' + name, 2e-7)
                        row = edges[(edges.section == section) & (edges.score == name)]
                        assert len(row) == 1
                        close(roc_auc_score(seam, values[valid]), float(row.auroc.iloc[0]), 'CRC unique-edge AUROC', 1e-8)
                        edge_rows += 1
                    rng = np.random.RandomState(20261011)
                    for iteration in range(10):
                        shuffled = rng.permutation(y)
                        ok = (shuffled[pair[0]] >= 0) & (shuffled[pair[1]] >= 0)
                        label = shuffled[pair[0]][ok] != shuffled[pair[1]][ok]
                        for name, values in score_values.items():
                            row = null[(null.section == section) & (null.score == name) & (null.shuffle == iteration)]
                            assert len(row) == 1
                            close(roc_auc_score(label, values[ok]), float(row.auroc.iloc[0]), 'CRC shuffled-label AUROC', 1e-8)
                            null_rows += 1
                    assert null[null.section == section].groupby('score').auroc.median().between(.45, .55).all()
        patients = arms.groupby(['patient', 'arm']).ari.mean().unstack().reindex([s.split('_')[1] for s in sections])
        seams = edges.pivot(index='patient', columns='score', values='auroc').reindex(patients.index)
        values = {c: seams[c].to_numpy() for c in seams}
        values.update(Gate_minus_RawFeature=(seams.Gate-seams.RawFeature).to_numpy(),
                      Gate_minus_UniformMeanFeature=(seams.Gate-seams.UniformMeanFeature).to_numpy(),
                      Tessera_minus_UniformSameGraph=(patients.Tessera-patients.UniformSameGraph).to_numpy(),
                      NoGate_minus_Tessera=(patients.NoGate-patients.Tessera).to_numpy())
        receipt = qc if mode == 'annotation-qc' else primary
        for name, value in values.items():
            rng = np.random.RandomState(20261011)
            samples = rng.choice(value, size=(10000, 3), replace=True).mean(1)
            lo, hi = np.percentile(samples, [2.5, 97.5])
            close([value.mean(), lo, hi], [receipt['statistics'][name][k] for k in ['mean', 'lower', 'upper']], 'CRC patient bootstrap')
            assert receipt['statistics'][name]['n_independent_patients'] == 3
    return {'independent_patients': 3, 'saved_embedding_files_bound': 36, 'saved_ari_rows_recomputed': ari_rows,
            'unique_edge_auroc_rows_recomputed': edge_rows, 'saved_shuffle_table_rows': 18000,
            'shuffle_rows_independently_recomputed': null_rows, 'new_model_or_clustering_fits': 0,
            'scope': 'Immutable protocols/amendment, saved PCA inputs/predictions/gates, independent edge scoring and patient bootstrap; raw acquisition and historical backup integration are outside this command'}


def forbidden_fit(*args, **kwargs):
    raise RuntimeError('Fitting is prohibited by saved-result verification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['primary', 'transfer', 'followup', 'external', 'crc', 'all'], default='all')
    args = parser.parse_args()
    KMeans.fit = forbidden_fit
    KMeans.fit_predict = forbidden_fit
    rx.train_net = forbidden_fit
    checks = {'frozen_source_hash_bindings': frozen_sources()}
    functions = {'primary': primary, 'transfer': transfer, 'followup': followup,
                 'external': external_and_extension, 'crc': crc}
    for name, function in functions.items():
        if args.only in [name, 'all']:
            checks[name] = function()
    print(json.dumps({'status': 'passed', 'scope': 'saved-result verification', 'checks': checks,
          'model_or_clustering_fits': 0, 'network_requests': 0,
          'runtime': {'python': sys.version, 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch']}},
          'limitations': read(ROOT / 'provenance/verification-scope.json')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
