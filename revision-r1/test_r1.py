"""Equivalence and leakage tests for the revision R1 analyses (run before any fit)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import common_r1 as c
import run_experiments as rx
from tessera_st.config import AblationConfig
from train_core import same_graph_mean, train_core

SMALL = 'Breast_Visium'


@pytest.fixture(scope='module')
def small():
    torch.set_num_threads(1)
    data, meta = c.load_primary(SMALL)
    return data['x'], data['coords'], data['field'], data['ref'], meta['n_classes']


def test_core_matches_frozen(small):
    x, xy, fields, _, k = small
    frozen = rx.train_net(x, xy, fields, k, 'Tessera', 1, 'cpu')
    core = train_core(x, xy, fields, k, 'Tessera', 1, 'cpu', epochs=120)
    point = core['checkpoints'][120]
    assert np.array_equal(frozen[1], point['z'])
    assert np.array_equal(frozen[2], point['boundary'])
    assert np.array_equal(frozen[3], point['gate'])
    assert np.array_equal(frozen[4], core['edge'])


def test_null_matches_frozen(small):
    x, xy, fields, _, k = small
    original = rx.configuration

    def no_contrast_on_permuted(arm):
        if arm == 'PermutedXY':
            return AblationConfig(edge_gating=True, multi_scale=True, boundary_contrastive=False,
                                  calibrated_uncertainty=False)
        return original(arm)
    rx.configuration = no_contrast_on_permuted
    try:
        frozen = rx.train_net(x, xy, fields, k, 'PermutedXY', 2, 'cpu')
    finally:
        rx.configuration = original
    core = train_core(x, xy, fields, k, 'PermutedXY-NoContrast', 2, 'cpu', epochs=120)
    assert np.array_equal(frozen[1], core['checkpoints'][120]['z'])


def test_checkpoint_equals_short_run(small):
    x, xy, fields, _, k = small
    longer = train_core(x, xy, fields, k, 'Tessera', 3, 'cpu', epochs=200, checkpoints=[120])
    short = train_core(x, xy, fields, k, 'Tessera', 3, 'cpu', epochs=120)
    assert np.array_equal(longer['checkpoints'][120]['z'], short['checkpoints'][120]['z'])


def test_same_graph_mean_matches_saved(small):
    x, xy, fields, _, _ = small
    mean, _ = same_graph_mean(x, xy, fields)
    saved = np.load(c.RESULTS / SMALL / 'NeighborMean-seed1.npz')['embedding']
    assert np.array_equal(mean, saved)


def test_pca50_reproduces():
    for name in ['NSCLC_Lung5_Rep2', 'BCC_PatientA']:
        data, meta = c.load_primary(name)
        saved = json.loads((c.RESULTS / name / 'PCA50-seed1.json').read_text())['scores']['ari']
        ari, _ = c.fu.kmeans_ari(data['x'], data['ref'], meta['n_classes'], 1)
        assert abs(ari - saved) < 1e-12


def test_selection_excludes_held_out():
    rng = np.random.default_rng(0)
    means = pd.DataFrame(rng.random((5, 24)), index=list('ABCDE'), columns=[f'c{i}' for i in range(24)])
    chosen = c.select_nested(means, 'C')
    perturbed = means.copy()
    perturbed.loc['C'] = rng.random(24) * 100
    assert c.select_nested(perturbed, 'C') == chosen


def test_head_size_effect_cpu(small):
    x, xy, fields, _, k = small
    a = train_core(x, xy, fields, 2, 'Tessera', 1, 'cpu', epochs=40)['checkpoints'][40]['z']
    b = train_core(x, xy, fields, k, 'Tessera', 1, 'cpu', epochs=40)['checkpoints'][40]['z']
    again = train_core(x, xy, fields, k, 'Tessera', 1, 'cpu', epochs=40)['checkpoints'][40]['z']
    assert np.array_equal(b, again)
    assert not np.array_equal(a, b)
