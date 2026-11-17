"""Tests of the round-3 label diagnostics (run: python -m pytest revision-r3/test_r3.py -q)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostics_r3 as dg  # noqa: E402


def lattice(side):
    a, b = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
    return np.column_stack([a.ravel(), b.ravel()]).astype(float)


def at_k(profile, k, column='h_adj'):
    row = profile.loc[profile.k == k]
    return float(row[column].iloc[0]) if column in row and len(row) else np.nan


def test_frame_knn_stays_within_frames():
    xy = np.vstack([lattice(10), lattice(10)])            # two frames at identical coordinates
    frames = np.repeat(['a', 'b'], 100)
    idx, _, size = dg.frame_knn(xy, frames, 6)
    assert (idx >= 0).all() and (size == 100).all()
    assert (frames[idx] == frames[:, None]).all()
    assert (idx != np.arange(200)[:, None]).all()


def test_k_cap_never_pads():
    xy = lattice(7)[:40]
    idx, dist, _ = dg.frame_knn(xy, np.zeros(40), 60)
    assert (idx[:, 39:] == -1).all() and np.isnan(dist[:, 39:]).all() and (idx[:, :39] >= 0).all()
    prof = dg.homophily_profile(xy, np.zeros(40), np.arange(40) % 2)
    assert np.isnan(at_k(prof, 50)) and np.isfinite(at_k(prof, 30))


def test_hadj_random_labels():
    rng = np.random.default_rng(0)
    xy = lattice(45)
    y = rng.integers(0, 4, len(xy))
    assert abs(at_k(dg.homophily_profile(xy, np.zeros(len(xy)), y), 6)) < 0.02


def test_hadj_blocks():
    xy = lattice(40)
    y = (xy[:, 0] >= 20).astype(int) * 2 + (xy[:, 1] >= 20).astype(int)
    prof = dg.homophily_profile(xy, np.zeros(len(xy)), y)
    assert at_k(prof, 6) > 0.9 and at_k(prof, 6, 'li') > 0.6


def test_hadj_imbalance():
    rng = np.random.default_rng(1)
    xy = lattice(100)
    y = (rng.random(len(xy)) < 0.05).astype(int)
    prof = dg.homophily_profile(xy, np.zeros(len(xy)), y)
    assert at_k(prof, 6, 'h_edge') > 0.85 and abs(at_k(prof, 6)) < 0.03
    assert at_k(prof, 6, 'li') < 0.02


def test_hadj_frame_stratified():
    rng = np.random.default_rng(2)
    xy = np.vstack([lattice(45), lattice(45)])
    n = len(xy) // 2
    frames = np.repeat(['a', 'b'], n)
    y = np.concatenate([(rng.random(n) < 0.1).astype(int), (rng.random(n) < 0.9).astype(int)])
    prof = dg.homophily_profile(xy, frames, y)
    assert abs(at_k(prof, 6)) < 0.03
    # a chance term pooled across frames would report spurious homophily
    p = np.bincount(y) / len(y)
    pooled_chance = float((p ** 2).sum())
    pooled = (at_k(prof, 6, 'h_edge') - pooled_chance) / (1 - pooled_chance)
    assert pooled > 0.3


def test_correlation_length_orders_stripes():
    xy = lattice(60)
    narrow = (xy[:, 0] // 2 % 2).astype(int)
    wide = (xy[:, 0] // 20 % 2).astype(int)
    rng = np.random.default_rng(3)
    random = rng.integers(0, 2, len(xy))
    frames = np.zeros(len(xy))
    a = dg.correlation_length(xy, frames, narrow)
    b = dg.correlation_length(xy, frames, wide)
    c = dg.correlation_length(xy, frames, random)
    assert 0 < a['ell_k'] < b['ell_k']
    assert c['ell_k'] == 0.0 and not c['censored']
    assert dg.correlation_length(xy, frames, wide, scale_um=100.0)['ell_um'] == pytest.approx(b['ell_d1'] * 100.0)


def test_blocked_cv_detects_leak():
    side = 50
    xy = lattice(side)
    rng = np.random.default_rng(4)
    field = np.stack([gaussian_filter(rng.normal(size=(side, side)), 1.5).ravel() for _ in range(6)], 1)
    block = (xy[:, 0] // 10) * 5 + xy[:, 1] // 10            # label blocks aligned with 10-unit folds
    y = rng.permutation(np.arange(25) % 4)[block.astype(int)]
    out = dg.predictability(field, xy, np.zeros(len(xy)), y, ks=(6,),
                            estimator=lambda: KNeighborsClassifier(1))
    assert out['n_blocks_ok']
    assert out['kappa_self_random'] > 0.3
    assert out['kappa_self'] < 0.1


def test_predictability_recovers_informative_features():
    rng = np.random.default_rng(5)
    xy = lattice(40)
    y = rng.integers(0, 3, len(xy))
    x = np.eye(3)[y] * 3 + rng.normal(size=(len(xy), 3))
    out = dg.predictability(x, xy, np.zeros(len(xy)), y, ks=(6,))
    assert out['kappa_self'] > 0.8


def _toy_two_frames(n=150, seed=7):
    rng = np.random.default_rng(seed)
    xy = rng.random((2 * n, 2)) * 10
    frames = np.repeat(['f1', 'f2'], n)
    x = rng.normal(size=(2 * n, 8)).astype(np.float32)
    return x, xy, frames


def test_contrast_no_mp_edges_and_determinism():
    import torch
    import arms_r3 as ar
    torch.set_num_threads(1)                    # bit-reproducible only on one CPU thread, as in round 1
    x, xy, frames = _toy_two_frames()
    a = ar.train_contrast_no_mp(x, xy, frames, head_k=2, seed=1, device='cpu', epochs=3)
    b = ar.train_contrast_no_mp(x, xy, frames, head_k=2, seed=1, device='cpu', epochs=3)
    fwd, con = a['forward_edges'], a['contrastive_edges']
    assert (fwd[0] == fwd[1]).all() and fwd.shape[1] == len(x)                 # self edges only
    assert (fwd[0] != fwd[1]).sum() == 0 and (con[0] != con[1]).all()
    assert (frames[con[0]] == frames[con[1]]).all() and con.shape[1] >= 6 * len(x)  # spatial positives within frames
    assert np.array_equal(a['z'], b['z'])


def test_propagate_matches_same_graph_mean():
    import arms_r3 as ar
    import train_core as tc
    x, xy, frames = _toy_two_frames()
    mean, edge = tc.same_graph_mean(x, xy, frames)
    assert np.allclose(ar.propagate(x, edge, 1), mean, atol=1e-6)
    two = ar.propagate(x, edge, 2)
    assert np.allclose(two, ar.propagate(ar.propagate(x, edge, 1), edge, 1), atol=1e-6)
