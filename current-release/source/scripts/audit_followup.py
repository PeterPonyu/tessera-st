#!/usr/bin/env python3
"""Audit saved follow-ups and quantify relative last-layer gate weights on CPU.

Historical run receipts and saved gate diagnostics are read, never rewritten.
Exact recovered sources establish producer availability; saved-output bindings
remain retrospective consistency checks, not a new execution of old analyses.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from common import ROOT, DATA, RESULTS, OBJECTS, SEEDS, read, dump, sha
from run_followup import configs, exact_sign_flip

F = RESULTS / 'followup-20261010'
DIAGNOSTIC = F / 'gate_weight_diagnostics-20261011.csv'
RECEIPT = F / 'run-gate-weights-20261011.json'
SUMMARIES = {'scale': 'scale_controls', 'rescore': 'rescore',
             'nulls': 'geometry_nulls', 'whole': 'whole_sections',
             'transfer': 'transfer_neighbor_mean'}


def gate_profile(dst, gates, n_nodes):
    """Relative weights over directed multiset entries; isolated nodes excluded."""
    dst = np.asarray(dst, dtype=int)
    gates = np.asarray(gates, dtype=np.float64)
    assert len(dst) == len(gates) and np.isfinite(gates).all() and (gates >= 0).all()
    degree = np.bincount(dst, minlength=n_nodes)
    mass = np.bincount(dst, weights=gates, minlength=n_nodes)
    valid = (degree > 0) & (mass > 0)
    weights = np.divide(gates, mass[dst], out=np.zeros_like(gates), where=mass[dst] > 0)
    uniform = 1. / degree[dst]
    tvd = .5 * np.bincount(dst, weights=np.abs(weights - uniform), minlength=n_nodes)
    squared = np.bincount(dst, weights=weights ** 2, minlength=n_nodes)
    ess_fraction = 1. / (degree[valid] * squared[valid])
    assert valid.any() and (tvd[valid] <= 1 + 1e-12).all()
    return dict(defined_nodes=int(valid.sum()), zero_mass_nodes=int(((degree > 0) & ~valid).sum()),
                isolated_nodes=int((degree == 0).sum()),
                tvd_median=float(np.median(tvd[valid])), tvd_p90=float(np.quantile(tvd[valid], .9)),
                ess_fraction_median=float(np.median(ess_fraction)),
                epsilon_attenuation_max=float(np.max(1e-8 / (mass[valid] + 1e-8))))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def producer_source(recorded, filename):
    current = ROOT / 'source/scripts' / filename
    if sha(current) == recorded:
        return current
    snapshot = ROOT / 'source/provenance/followup-runners' / recorded / filename
    if snapshot.is_file():
        assert sha(snapshot) == recorded, f'Historical source hash mismatch: {snapshot}'
        return snapshot
    return None


def main():
    runner = ROOT / 'source/scripts/run_followup.py'
    receipts = {}
    for mode, stem in SUMMARIES.items():
        receipt = read(F / f'run-{mode}.json')
        assert receipt['frozen_run_lock_sha256'] == sha(RESULTS / 'run-lock.json')
        assert receipt[mode] == read(F / f'{stem}_summary.json')
        recorded = receipt['script_sha256']
        source = producer_source(recorded, 'run_followup.py')
        receipts[mode] = dict(recorded_producer_sha256=recorded, current_runner_sha256=sha(runner),
                              current_runner_matches=recorded == sha(runner),
                              historical_source_available=source is not None,
                              historical_source_path=str(source.relative_to(ROOT)) if source else None)
    # All configurations/seeds must exist, and nested selection excludes the test patient.
    scale = pd.read_csv(F / 'scale_controls.csv')
    scale['config'] = scale.kind + '|' + scale.k.astype(str) + '|' + scale.lam.astype(str)
    assert len(scale) == len(OBJECTS) * len(list(configs())) * len(SEEDS) == 1170
    assert not scale.duplicated(['object', 'config', 'seed']).any()
    means = scale.groupby(['object', 'config']).ari.mean().unstack()
    candidates = [c for c in means if not c.startswith('PCA50')]
    assert len(candidates) == 29
    nested = read(F / 'scale_controls_summary.json')
    for cohort, names in [('NSCLC', OBJECTS[:5]), ('BCC', OBJECTS[5:9])]:
        delta = []
        for row in nested[cohort]['per_patient']:
            patient = row['patient']
            selected = means.loc[[n for n in names if n != patient], candidates].mean().idxmax()
            assert selected == row['selected']
            close(row['nested_control_ari'], means.loc[patient, selected])
            delta.append(row['tessera_ari'] - row['nested_control_ari'])
        close(nested[cohort]['exact_sign_flip_p'], exact_sign_flip(delta))
    # Re-scoring retains all valid reference classes in ARI; only K changes.
    resc = pd.read_csv(F / 'rescore.csv')
    summary = read(F / 'rescore_summary.json')
    rmeans = resc.groupby(['object', 'arm']).mean(numeric_only=True)
    assert len(resc) == 13 * 6 * 3 and not resc.duplicated(['object', 'arm', 'seed']).any()
    for target, column in [('common_K', 'ari_common_K'), ('lung_celltypes', 'ari_celltype_at_celltype_K')]:
        for arm, key in [('PCA50', 'pca50'), ('NeighborMean', 'neighbor_mean'), ('Tessera', 'tessera')]:
            close(summary[target][key], [rmeans.loc[(n, arm), column] for n in summary[target]['objects']])
    gates = pd.read_csv(F / 'gate_baseline.csv').groupby('object').mean(numeric_only=True)
    for n in OBJECTS:
        for key, value in summary['gate'][n].items():
            close(value, gates.loc[n, key])
    nulls = pd.read_csv(F / 'geometry_nulls.csv')
    assert len(nulls) == 39 and not nulls.duplicated(['object', 'seed']).any()
    null_summary = read(F / 'geometry_nulls_summary.json')
    for n, value in nulls.groupby('object').ari.mean().items():
        close(value, null_summary[n]['permuted_no_contrast'])
    whole = pd.read_csv(F / 'whole_sections.csv')
    assert len(whole) == 108 and not whole.duplicated(['object', 'arm', 'seed']).any()
    whole_summary = read(F / 'whole_sections_summary.json')
    for (n, arm), value in whole.groupby(['object', 'arm']).ari.mean().items():
        close(value, whole_summary['mean_ari'][n][arm])
    for n, count in whole.groupby('object').n_cells.first().items():
        assert int(count) == whole_summary['n_cells'][n]
    transfer = pd.read_csv(F / 'transfer_neighbor_mean.csv')
    assert len(transfer) == 20 and not transfer.duplicated(['patient', 'arm', 'target']).any()
    trans_summary = read(F / 'transfer_neighbor_mean_summary.json')
    for target in ['region', 'cell_type']:
        for metric in ['accuracy', 'balanced_accuracy']:
            for arm in ['NeighborMean-k6', 'NeighborMean-k200']:
                sub = transfer[(transfer.target == target) & (transfer.arm == arm)].set_index('patient')
                close(trans_summary[f'{target}_{metric}'][arm], [sub.loc[n, metric] for n in OBJECTS[:5]])
    # New evidence uses only hash-verified saved predictions, no refitting.
    rows, bindings = [], {}
    for n in OBJECTS:
        meta = read(DATA / n / 'input.json')
        for seed in SEEDS:
            path = RESULTS / n / f'Tessera-seed{seed}.npz'
            rec = read(path.with_suffix('.json'))
            assert rec['input_sha256'] == meta['matrix_sha256']
            assert sha(path) == rec['prediction_sha256']
            with np.load(path, allow_pickle=False) as pred:
                rows.append(dict(object=n, seed=seed, **gate_profile(pred['edge_index'][1], pred['last_gate'], meta['n_selected'])))
            bindings[str(path.relative_to(ROOT))] = rec['prediction_sha256']
    calculated = pd.DataFrame(rows)
    if DIAGNOSTIC.exists():
        saved = pd.read_csv(DIAGNOSTIC)
        assert saved[['object', 'seed']].equals(calculated[['object', 'seed']])
        close(saved.drop(columns=['object', 'seed']), calculated.drop(columns=['object', 'seed']))
    else:
        calculated.to_csv(DIAGNOSTIC, index=False)
    if RECEIPT.exists():
        gate_receipt = read(RECEIPT)
        assert gate_receipt['frozen_run_lock_sha256'] == sha(RESULTS / 'run-lock.json')
        assert gate_receipt['prediction_sha256'] == bindings
        assert gate_receipt['output_sha256'] == {DIAGNOSTIC.name: sha(DIAGNOSTIC)}
    else:
        gate_receipt = dict(protocol='saved-last-layer-relative-weights-20261011',
             script_sha256=sha(__file__), frozen_run_lock_sha256=sha(RESULTS / 'run-lock.json'),
             prediction_sha256=bindings, output_sha256={DIAGNOSTIC.name: sha(DIAGNOSTIC)},
             scope='CPU post-processing of saved gates; no new fit or independent cohort.')
        dump(RECEIPT, gate_receipt)
    gate_source = producer_source(gate_receipt['script_sha256'], 'audit_followup.py')
    assert gate_source is not None, 'Saved gate-diagnostic producer source unavailable'
    gap = [mode for mode, rec in receipts.items() if not rec['historical_source_available']]
    dump(ROOT / 'qa/followup_evidence_audit.json', dict(
         status='saved-result consistency verified; historical producer sources available' if not gap else
                'saved-result consistency verified; historical producer source gap disclosed',
         historical_run_receipts=receipts, unrecovered_historical_modes=gap,
         gate_diagnostic_producer=dict(sha256=gate_receipt['script_sha256'],
              path=str(gate_source.relative_to(ROOT)), original_receipt_preserved=True),
         counts=dict(scale_records=1170, spatial_configurations_per_object=29,
                     rescore_records=234, null_records=39, whole_records=108, transfer_records=20),
         selection_scope=dict(bounded='patient-excluded nested configuration selection',
              whole='same patients; fixed cohort-modal controls from bounded analysis, sensitivity only',
              transfer='k=200 informed by the lung bounded analysis; descriptive control, not fold-nested tuning'),
         current_output_sha256={p.name: sha(p) for p in sorted(F.iterdir()) if p.is_file() and p.suffix in {'.csv', '.json'}},
         audit_source_sha256=sha(__file__),
         caveat='Saved-result bindings are retrospective, not a rerun. Recovered sources match original producer hashes; historical receipts and saved gate diagnostics are unchanged.'))
    print(f'FOLLOW-UP AUDIT: saved tables consistent; source gaps {gap}; 39 saved-gate profiles verified')


if __name__ == '__main__':
    main()
