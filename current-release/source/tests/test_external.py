"""Regression checks for the independent-cohort, tissue and weighting analyses."""
from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'source/scripts'), str(ROOT / 'source/src'), str(ROOT / 'validation-extension'),
                str(ROOT / 'validation-external')]

import run_experiments as rx  # noqa: E402
import run_extension as ext  # noqa: E402
from common import read, sha  # noqa: E402

EXT = ROOT / 'validation-extension'
RESULTS = EXT / 'results'
LED = ROOT / 'source/manuscript/ledgers'


def macros():
    return {m.group(1): m.group(2) for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', (LED / 'external_numbers.tex').read_text())}


def test_receipts_bind_protocol_and_scripts():
    for mode in ('hardening', 'dev', 'breadth'):
        receipt = read(RESULTS / f'run-{mode}.json')
        assert receipt['protocol_sha256'] == sha(EXT / 'PROTOCOL_EXTENSION.md')
        assert receipt['script_sha256'] == sha(EXT / 'run_extension.py')
        for name, expected in receipt['frozen_sha256'].items():
            assert sha(ROOT / 'source/scripts' / name) == expected
        for name, expected in receipt['validation_external_sha256'].items():
            assert sha(ROOT / 'validation-external' / name) == expected
        assert receipt['environment']['device'] == 'cpu'
    for summary in (ROOT / 'validation-external/results/summary.json', ROOT / 'validation-external/results/enhanced/summary.json'):
        record = read(summary)
        assert record['script_sha256'] == sha(ROOT / 'validation-external' / ('run_her2st_validation.py' if summary.parent.name == 'results' else 'run_enhanced_targets.py'))


def test_tables_have_the_declared_cardinality():
    grid = pd.read_csv(RESULTS / 'hardening/grid.csv')
    arms = pd.read_csv(RESULTS / 'hardening/arms.csv')
    assert len(grid) == 8 * 5 * len(ext.CONFIGS) * 3
    assert len(arms) == 8 * 5 * 5 * 3
    assert set(arms.arm) == {'ExpressionWeighted', 'UniformSameGraph', 'WeightShuffled', 'ShuffledCoords', 'Tessera'}
    assert len(pd.read_csv(RESULTS / 'breadth/grid.csv')) == 3 * len(ext.CONFIGS) * 3
    assert len(pd.read_csv(RESULTS / 'dev/arms.csv')) == 13 * 4 * 3
    assert np.isfinite(arms.ari).all() and np.isfinite(grid.ari).all()


def test_headline_macros_match_the_saved_scores():
    result = read(RESULTS / 'run-hardening.json')['result']
    m = macros()
    arms = pd.read_csv(RESULTS / 'hardening/arms.csv')
    mean = arms[arms.target == 'all'].groupby(['unit', 'arm']).ari.mean().unstack()
    delta = (mean.Tessera - mean.UniformSameGraph).to_numpy()
    assert int((delta < 0).sum()) == int(m['HerAllTessUSixNeg'])
    assert f'{delta.mean():+.3f}' == m['HerAllTessUSixMean']
    boundary = arms[arms.target == 'boundary_T1'].groupby(['unit', 'arm']).ari.mean().unstack()
    grid = pd.read_csv(RESULTS / 'hardening/grid.csv')
    selected = result['targets']['boundary_T1']['selected_control']
    control = pd.Series({u: grid[(grid.target == 'boundary_T1') & (grid.unit == u) & (grid.config == c)].ari.mean() for u, c in selected.items()})
    assert int(((boundary.UniformSameGraph - control) > 0).sum()) == int(m['HerBndUSixNestedPos'])
    # The weighting claim is the one the controls did not support; the text must say so.
    assert m['HerHOneVerdict'] == 'was not met' and not result['H1_weighting']['holds']
    assert int(m['HerBndEUSixPos']) + int(m['HerBndEUSixNeg']) == 8


def test_selected_controls_never_use_the_held_out_patient():
    result = read(RESULTS / 'run-hardening.json')['result']
    grid = pd.read_csv(RESULTS / 'hardening/grid.csv')
    for target, record in result['targets'].items():
        sub = grid[grid.target == target]
        units = record['units']
        for unit, chosen in record['selected_control'].items():
            others = [u for u in units if u != unit]
            means = sub[sub.unit.isin(others)].groupby('config').ari.mean()
            means = means[[c for c in means.index if not c.startswith('PCA50')]]
            assert means.idxmax() == chosen


def test_same_graph_mean_is_the_frozen_neighbor_mean_and_shuffles_change_it():
    rng = np.random.RandomState(0)
    x = rng.randn(80, 6).astype(np.float32)
    xy = rng.rand(80, 2) * 10
    fields = np.full(80, 'f')
    edge, _ = rx.graph(xy, fields, 'Tessera', 1)
    uniform, _ = ext.weighted_features(x, xy, fields, 'uniform')
    assert np.allclose(uniform, rx.neighbor_embedding(x, edge), atol=1e-6)
    weighted, _ = ext.weighted_features(x, xy, fields, 'expression')
    shuffled, _ = ext.weighted_features(x, xy, fields, 'weight_shuffled', 1)
    moved, _ = ext.weighted_features(x, xy, fields, 'shuffled_coords', 1)
    assert not np.allclose(weighted, uniform) and not np.allclose(shuffled, weighted) and not np.allclose(moved, uniform)
    assert np.allclose(weighted.min(), weighted.min()) and np.isfinite(weighted).all()
    # Convex combinations of neighbors cannot leave the feature range.
    assert weighted.max() <= x.max() + 1e-5 and weighted.min() >= x.min() - 1e-5


def test_development_ablation_reproduces_the_frozen_neighbor_mean():
    dev = read(RESULTS / 'run-dev.json')['result']
    assert dev['checks']['uniform_same_graph_vs_frozen_neighbor_mean_max_abs'] < 1e-9


def test_replicates_are_three_runs_of_the_same_protocol():
    rep = pd.read_csv(LED / 'replicates.csv')
    assert len(rep) == 24 and set(rep.target) == {'all', 'compartments', 'boundary_T1'}
    assert (rep['range'] >= 0).all() and rep['range'].max() < 0.1
    assert int(rep.sign_vs_same_graph_changes.sum()) == int(macros()['RepSignChangesSame'])


def test_landscape_covers_every_object_once():
    table = pd.read_csv(LED / 'landscape.csv')
    assert len(table) == 24 and table.object.is_unique
    assert set(table.group) == {'Computed regions', 'Cell-type labels', 'Pathologist regions', 'Anatomical regions'}
    assert int(macros()['LandN']) == 24
    assert int((table.tessera_minus_oracle > 0).sum()) == int(macros()['LandTessOraclePos'])


def test_new_sections_use_macros_and_cite_resolvable_sources():
    text = (ROOT / 'source/manuscript/paper.tex').read_text()
    results = text.split(r'\subsection{An independent pathologist-labelled cohort reproduces the ordering}')[1].split(r'\section{Discussion}')[0]
    assert len(re.findall(r'\\(?:Her|Rep|Br|Land|Dev)[A-Za-z]+\{\}', results)) >= 60
    assert not re.search(r'\b0\.\d{3}\b', re.sub(r'\\caption\{.*?\}\\label', '', results, flags=re.S).replace('0.14', '').replace('0.5', ''))
    for key in ('andersson2021her2st', 'maynard2021dlpfc', 'wang2018starmap', 'moffitt2018merfish'):
        assert key in text
