#!/usr/bin/env python3
"""Read-only recomputation of pilot evidence from saved predictions and labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REV = HERE.parent
SECTIONS = ['SN84_A120838_Rep1', 'SN124_A595688_Rep2', 'SN124_A551763_Rep2']
ARMS = ['PCA50', 'UniformSameGraph', 'Tessera', 'NoGate']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def assert_close(a, b, label, tol=1e-10):
    if not np.allclose(a, b, rtol=0, atol=tol):
        raise AssertionError(label)


def main():
    primary = read(HERE / 'results/run-receipt.json')
    qc = read(HERE / 'annotation-qc/run-receipt.json')
    expected = primary['bindings']
    assert sha(HERE / 'PROTOCOL.md') == expected['protocol_sha256'] == read(HERE / 'protocol-lock.json')['protocol_sha256']
    assert sha(HERE / 'run_pilot.py') == expected['script_sha256']
    for name, digest in expected['frozen_sha256'].items():
        assert sha(REV / name) == digest, name
    assert qc['primary_receipt_sha256'] == sha(HERE / 'results/run-receipt.json')
    assert qc['script_sha256'] == sha(HERE / 'analyze_annotation_qc.py')
    assert qc['protocol_sha256'] == sha(HERE / 'ANNOTATION_QC_PROTOCOL.md') == read(HERE / 'ANNOTATION_QC_PROTOCOL.lock.json')['sha256']
    for receipt in [primary, qc]:
        for name, digest in receipt['output_sha256'].items():
            assert sha(HERE / name) == digest, name
    execution = read(HERE / 'results/execution.json')
    assert execution['torch_num_threads'] == 1 and execution['device'] == 'cpu'
    assert execution['invocation_script_sha256'] == sha(HERE / 'run_singlethread.py')
    assert execution['implementation_sha256'] == sha(HERE / 'run_pilot.py')
    assert execution['runtime_decision_sha256'] == sha(HERE / 'runtime-decision.json')
    audit = read(HERE / 'cohort-audit.json')
    assert audit['untouched_within_audited_scope'] and not audit['hits']
    protected = read(HERE / 'protected-before.json')
    integration = read(HERE / 'integration-manifest.json') if (HERE / 'integration-manifest.json').exists() else {'changes': {}}
    # Existing numerical ledgers, receipts, data, fits and core code stay immutable.
    allowed_changes = {
        'EDITABLE_SOURCE_README.md', 'FIGURE_STYLE_AND_VALIDATION_PLAN.md',
        'FOLLOWUP_PROVENANCE.md', 'REVISION_REPORT.md',
        'output/pdf/tessera-revised-manuscript.pdf',
        'output/pdf/tessera-supporting-information.pdf',
        'source/manuscript/ledgers/reference_audit.json',
        'source/manuscript/ledgers/reference_metadata.json',
        'source/manuscript/paper.tex', 'source/manuscript/supporting.tex',
        'source/scripts/build_revision.py', 'source/scripts/package_revision.py',
        'source/scripts/reference_metadata.py', 'source/scripts/write_visual_review.py',
        'source/tests/test_presentation.py'}
    assert set(integration['changes']) <= allowed_changes
    assert set(integration['changes']) <= set(protected)
    if integration['changes']:
        assert integration['protected_manifest_sha256'] == sha(HERE / 'protected-before.json')
    for name, digest in protected.items():
        if name in integration['changes']:
            change = integration['changes'][name]
            assert change['before_sha256'] == digest
            assert sha(REV / name) == change['after_sha256'], 'integration changed: ' + name
            assert sha(Path(integration['backup']) / name) == digest, 'pre-integration backup changed'
        else:
            assert sha(REV / name) == digest, 'historical file changed: ' + name
    fetch = read(HERE / 'fetch-receipt.json')
    assert len(fetch['files']) == 4 and fetch['script_sha256'] == sha(HERE / 'fetch_pilot.py')
    assert fetch['fetch_gate_sha256'] == sha(REV.parents[2] / 'research/pipelines/fetch_gate.py')
    assert read(HERE / 'source-record.json')['metadata']['license']['id'] == 'cc-by-4.0'
    for record in fetch['files']:
        p = HERE / 'raw' / record['filename']
        assert sha(p) == record['sha256']
        assert 'md5:' + sha(p, 'md5') == record['publisher_md5']
        assert p.stat().st_size == record['bytes']
        with zipfile.ZipFile(p) as archive:
            assert archive.testzip() is None
    figures = read(HERE / 'figures/figure-receipt.json')
    assert figures['script_sha256'] == sha(HERE / 'render_pilot.py')
    for name, digest in figures['evidence_sha256'].items():
        assert sha(HERE / name) == digest
    for name, digest in figures['figure_sha256'].items():
        assert sha(HERE / 'figures' / name) == digest
    assert len(figures['registration_geometry']) == 3
    for record in figures['registration_geometry']:
        assert record['centers_inside_image'] == record['centers']
        assert record['join_rate'] == 1
        assert record['source_image_sha256'] == sha(HERE / 'inputs' / record['section'] / 'tissue_hires_image.png')
    total_fits, tested_null_rows = 0, 0
    for folder, sensitivity in [('results', False), ('annotation-qc', True)]:
        arms = pd.read_csv(HERE / folder / 'arms.csv')
        edges = pd.read_csv(HERE / folder / 'seams.csv')
        null = pd.read_csv(HERE / folder / 'label_shuffle.csv')
        assert len(arms) == 36 and len(edges) == 9 and len(null) == 9000
        assert set(arms.arm) == set(ARMS) and set(arms.seed) == {1, 2, 3}
        for section in SECTIONS:
            base = HERE / 'inputs' / section
            meta = read(base / 'input.json')
            d = np.load(base / 'input.npz', allow_pickle=False)
            assert meta['bindings'] == expected
            assert sha(base / 'input.npz') == meta['input_sha256'] == primary['input_sha256'][section]
            assert len(set(d['barcodes'])) == len(d['barcodes']) == meta['n_tissue']
            assert np.isfinite(d['x']).all() and np.isfinite(d['coords']).all()
            assert meta['match_rate'] >= .95 and meta['n_outside_positions'] == 0
            edge = d['edge_index']
            assert not (edge[0] == edge[1]).any()
            assert (d['field'][edge[0]] == d['field'][edge[1]]).all()
            pair_rows = sorted({tuple(sorted(e)) for e in edge.T})
            pair = np.asarray(pair_rows).T
            saved = np.load(HERE / 'results' / section / 'boundary_scores.npz', allow_pickle=False)
            assert np.array_equal(pair, saved['unique_edge_index'])
            raw = d['ref_name']
            keep = d['ref'] >= 0
            if sensitivity:
                keep &= np.char.lower(np.char.strip(raw)) != 'exclude'
            names = sorted(set(raw[keep]))
            mapping = {name: i for i, name in enumerate(names)}
            y = np.asarray([mapping[name] if ok else -1 for name, ok in zip(raw, keep)])
            valid = (y[pair[0]] >= 0) & (y[pair[1]] >= 0)
            seam = y[pair[0]][valid] != y[pair[1]][valid]
            assert keep.sum() >= 100 and seam.sum() >= 20 and (~seam).sum() >= 20
            score_values = {'RawFeature': np.linalg.norm(d['x'][pair[0]] - d['x'][pair[1]], axis=1)}
            mean = np.zeros_like(d['x'])
            degree = np.zeros(len(mean), int)
            for src, dst in edge.T:
                mean[dst] += d['x'][src]; degree[dst] += 1
            mean = mean / degree[:, None]
            score_values['UniformMeanFeature'] = np.linalg.norm(mean[pair[0]] - mean[pair[1]], axis=1)
            gates = []
            for arm in ARMS:
                for seed in [1, 2, 3]:
                    stem = HERE / 'results' / section / f'{arm}-seed{seed}'
                    fit = read(stem.with_suffix('.json'))
                    pred = np.load(stem.with_suffix('.npz'), allow_pickle=False)
                    assert fit['bindings'] == expected and fit['auxiliary_fit_K'] == 2
                    assert sha(stem.with_suffix('.npz')) == fit['prediction_sha256']
                    assert np.array_equal(pred['edge_index'], edge)
                    if arm in ['Tessera', 'NoGate'] and not sensitivity:
                        assert len(fit['history']) == 11 and fit['history'][-1]['epoch'] == 119
                        assert all(np.isfinite(h['loss']) for h in fit['history'])
                        total_fits += 1
                    scored = np.load(HERE / folder / section / f'{arm}-seed{seed}.npz', allow_pickle=False)['labels']
                    recluster = KMeans(len(names), n_init=10, random_state=seed).fit_predict(pred['embedding'])
                    # Equivalent partitions may use different integer names.
                    assert adjusted_rand_score(scored, recluster) == 1
                    ari = adjusted_rand_score(y[keep], scored[keep])
                    row = arms[(arms.section == section) & (arms.arm == arm) & (arms.seed == seed)]
                    assert len(row) == 1
                    assert_close(float(row.ari.iloc[0]), ari, 'saved ARI')
                    if arm == 'PCA50':
                        assert_close(pred['embedding'], d['x'], 'PCA input identity', 0)
                    if arm == 'UniformSameGraph':
                        assert_close(pred['embedding'], mean, 'same graph mean', 2e-6)
                    if arm == 'Tessera':
                        # Different implementation from the producer's bincount grouping.
                        values = {}
                        gate_scores = np.subtract(np.float32(1.), pred['last_gate'])
                        for e, score in zip(edge.T, gate_scores):
                            values.setdefault(tuple(sorted(e)), []).append(float(score))
                        gates.append(np.asarray([np.mean(values[p]) for p in pair_rows]))
            score_values['Gate'] = np.mean(gates, axis=0)
            for name, values in score_values.items():
                assert_close(values, saved[name], 'edge score ' + name, 2e-7)
                row = edges[(edges.section == section) & (edges.score == name)]
                assert len(row) == 1
                assert_close(float(row.auroc.iloc[0]), roc_auc_score(seam, values[valid]), 'AUROC', 1e-8)
            rng = np.random.RandomState(20261011)
            for iteration in range(1000):
                shuffled = rng.permutation(y)
                if iteration >= 10:
                    continue
                ok = (shuffled[pair[0]] >= 0) & (shuffled[pair[1]] >= 0)
                label = shuffled[pair[0]][ok] != shuffled[pair[1]][ok]
                for name, values in score_values.items():
                    row = null[(null.section == section) & (null.score == name) & (null.shuffle == iteration)]
                    assert len(row) == 1
                    assert_close(float(row.auroc.iloc[0]), roc_auc_score(label, values[ok]), 'shuffle AUROC', 1e-8)
                    tested_null_rows += 1
            null_medians = null[null.section == section].groupby('score').auroc.median()
            assert null_medians.between(.45, .55).all(), null_medians
        patients = arms.groupby(['patient', 'arm']).ari.mean().unstack().reindex([s.split('_')[1] for s in SECTIONS])
        seams = edges.pivot(index='patient', columns='score', values='auroc').reindex(patients.index)
        values = {c: seams[c].to_numpy() for c in seams}
        values.update(Gate_minus_RawFeature=(seams.Gate-seams.RawFeature).to_numpy(),
            Gate_minus_UniformMeanFeature=(seams.Gate-seams.UniformMeanFeature).to_numpy(),
            Tessera_minus_UniformSameGraph=(patients.Tessera-patients.UniformSameGraph).to_numpy(),
            NoGate_minus_Tessera=(patients.NoGate-patients.Tessera).to_numpy())
        receipt = qc if sensitivity else primary
        for name, v in values.items():
            rng = np.random.RandomState(20261011)
            samples = rng.choice(v, size=(10000, 3), replace=True).mean(1)
            lo, hi = np.percentile(samples, [2.5, 97.5])
            assert_close([v.mean(), lo, hi], [receipt['statistics'][name][k] for k in ['mean','lower','upper']], 'patient bootstrap')
            assert receipt['statistics'][name]['n_independent_patients'] == 3
    assert total_fits == primary['fit_count'] == 18
    print(json.dumps(dict(status='passed', historical_files_unchanged=len(protected)-len(integration['changes']),
        backed_up_document_and_verification_changes=len(integration['changes']),
        verifier_sha256=sha(Path(__file__)),
        integration_manifest_sha256=sha(HERE / 'integration-manifest.json') if integration['changes'] else None,
        primary_receipt_sha256=sha(HERE / 'results/run-receipt.json'),
        qc_receipt_sha256=sha(HERE / 'annotation-qc/run-receipt.json'),
        figure_receipt_sha256=sha(HERE / 'figures/figure-receipt.json'),
        independent_patients=3, model_fits=total_fits, saved_ari_rows_recomputed=72,
        unique_edge_auroc_rows_recomputed=18, label_shuffle_rows=18000,
        shuffle_rows_independently_recomputed=tested_null_rows,
        raw_archive_crc_md5_sha256='passed', receipt_bindings='passed',
        all_compartment_null_medians_between_0_45_and_0_55=True), indent=2))


if __name__ == '__main__':
    main()
