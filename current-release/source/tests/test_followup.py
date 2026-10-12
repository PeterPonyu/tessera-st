"""Evidence-preserving follow-up, display-only rarity and gate-weight contracts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/scripts'))
from common import read, sha
from audit_followup import gate_profile


def test_small_heterogeneous_gates_do_not_become_uniform():
    a = gate_profile([0, 0], [.001, .009], 2)
    b = gate_profile([0, 0], [.01, .09], 2)
    assert a['tvd_median'] == b['tvd_median'] == pytest.approx(.4)
    assert a['ess_fraction_median'] == pytest.approx(1 / (2 * (.1**2 + .9**2)))
    assert a['isolated_nodes'] == 1 and a['zero_mass_nodes'] == 0


def test_uniform_multiset_weights_and_zero_mass_are_counted():
    a = gate_profile([0, 0, 0, 1], [.1, .1, .1, 0], 3)
    assert a['tvd_median'] == pytest.approx(0)
    assert a['ess_fraction_median'] == pytest.approx(1)
    assert a['defined_nodes'] == a['zero_mass_nodes'] == a['isolated_nodes'] == 1


def test_rare_class_coloring_never_mutates_reference_or_predictions():
    from render_figures import display_reference, display_match
    y = np.array([0] * 60 + [1] * 39 + [2] + [-1])
    pred = np.array([10] * 60 + [11] * 20 + [12] * 19 + [12, 13])
    truth_before, pred_before = y.copy(), pred.copy()
    shown = display_reference(y)
    mapped, mapping = display_match(y, pred)
    # At exactly 1% of the 100 valid labels, class 2 is still eligible.
    assert shown[99] == 2
    rare_y = np.r_[y, 0]
    rare_pred = np.r_[pred, 10]
    assert display_reference(rare_y)[99] == -1
    rare_mapped, rare_mapping = display_match(rare_y, rare_pred)
    assert 2 not in rare_mapping.values() and (rare_mapped == -1).any()
    assert np.array_equal(y, truth_before) and np.array_equal(pred, pred_before)


def test_unknown_only_map_is_gray():
    from render_figures import display_match
    shown, mapping = display_match(np.full(4, -1), np.array([0, 1, 0, 2]))
    assert mapping == {} and np.array_equal(shown, np.full(4, -1))


def test_followup_audit_hashes_and_historical_sources_are_verified():
    audit = read(ROOT / 'qa/followup_evidence_audit.json')
    assert audit['audit_source_sha256'] == sha(ROOT / 'source/scripts/audit_followup.py')
    assert audit['unrecovered_historical_modes'] == []
    for mode, binding in audit['historical_run_receipts'].items():
        receipt = read(ROOT / f'results/followup-20261010/run-{mode}.json')
        assert receipt['script_sha256'] == binding['recorded_producer_sha256']
        assert binding['current_runner_sha256'] == sha(ROOT / 'source/scripts/run_followup.py')
        assert binding['current_runner_matches'] == (receipt['script_sha256'] == binding['current_runner_sha256'])
        assert binding['historical_source_available']
        assert sha(ROOT / binding['historical_source_path']) == receipt['script_sha256']
    for filename, expected in audit['current_output_sha256'].items():
        assert sha(ROOT / 'results/followup-20261010' / filename) == expected


def test_recovered_analysis_functions_are_identical_to_the_retained_runner():
    import ast
    from recover_followup_runners import remove_extension
    current = ROOT / 'source/scripts/run_followup.py'
    text = current.read_text()
    recipes = [(['transfer'], remove_extension(text, 'whole'))]
    recipes.append((['scale', 'rescore', 'nulls'], remove_extension(recipes[0][1], 'transfer')))
    current_functions = {node.name: ast.dump(node, include_attributes=False)
                         for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    for modes, recovered in recipes:
        expected = read(ROOT / f'results/followup-20261010/run-{modes[0]}.json')['script_sha256']
        path = ROOT / 'source/provenance/followup-runners' / expected / 'run_followup.py'
        assert sha(path) == expected and path.read_text() == recovered
        for node in ast.parse(recovered).body:
            if isinstance(node, ast.FunctionDef) and node.name != 'main':
                assert ast.dump(node, include_attributes=False) == current_functions[node.name]


def test_saved_gate_diagnostic_is_bound_to_all_original_predictions():
    path = ROOT / 'results/followup-20261010'
    receipt = read(path / 'run-gate-weights-20261011.json')
    producer = read(ROOT / 'qa/followup_evidence_audit.json')['gate_diagnostic_producer']
    assert receipt['script_sha256'] == producer['sha256'] == sha(ROOT / producer['path'])
    assert producer['original_receipt_preserved']
    assert receipt['frozen_run_lock_sha256'] == sha(ROOT / 'results/run-lock.json')
    assert len(receipt['prediction_sha256']) == 39
    for filename, expected in receipt['prediction_sha256'].items():
        assert sha(ROOT / filename) == expected
    table = pd.read_csv(path / 'gate_weight_diagnostics-20261011.csv')
    assert len(table) == 39 and not table.duplicated(['object', 'seed']).any()
    assert not table[['isolated_nodes', 'zero_mass_nodes']].to_numpy().any()
    assert table.tvd_median.between(0, 1).all() and table.ess_fraction_median.between(0, 1).all()


def test_reported_common_k_reversal_is_not_hidden():
    macros = read(ROOT / 'source/manuscript/ledgers/followup_numbers.json')['macros']
    assert macros['LungCommonKBetter'] == '5' and macros['BccCommonKBetter'] == '2'
    paper = (ROOT / 'source/manuscript/paper.tex').read_text()
    assert 'ARI still includes' in paper and 'not uniformly unfavorable' in paper
    assert 'Small gate magnitudes alone' in paper and 'not untouched validation' in paper
    assert '79 additional' not in paper
