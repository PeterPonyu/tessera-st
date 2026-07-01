"""Metric correctness — runs CPU-only, no torch required."""

import numpy as np

from tessera_st.eval import (
    ari,
    boundary_f1,
    chaos_score,
    expected_calibration_error,
    nmi,
    percentage_abnormal_spots,
    small_domain_iou,
)


def _stripe_grid(n=8):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    true = (coords[:, 0] >= n / 2).astype(int)
    return coords, true


def test_perfect_clustering_scores_one():
    true = np.array([0, 0, 1, 1, 2, 2])
    assert ari(true, true.copy()) == 1.0
    assert nmi(true, true.copy()) == 1.0


def test_boundary_f1_on_grid():
    # two vertical stripes -> a known seam in the middle
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    true = (coords[:, 0] >= 3).astype(int)
    f1 = boundary_f1(coords, true, true.copy())
    assert 0.0 <= f1 <= 1.0
    assert f1 > 0.5  # a perfect partition recovers its own seam well


def test_ece_bounds():
    conf = np.array([0.9, 0.8, 0.6, 0.5])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    ece = expected_calibration_error(conf, correct)
    assert 0.0 <= ece <= 1.0


def test_invalid_labels_are_excluded():
    true = np.array([-1, 0, 0, 1, 1])
    pred = np.array([9, 0, 0, 1, 1])  # the -1 spot would hurt ARI if not excluded
    assert ari(true, pred) == 1.0


def test_spatial_metrics_are_orthogonal_to_label_agreement():
    """A spatially shattered labelling and a coherent one can share ARI yet differ on CHAOS/PAS."""
    coords, true = _stripe_grid(8)
    rng = np.random.default_rng(0)
    # a 'checkerboard' relabel: same two labels, but spatially fragmented
    shattered = (coords.sum(axis=1).astype(int) % 2)
    assert chaos_score(coords, true) < chaos_score(coords, shattered)
    assert percentage_abnormal_spots(coords, true) < percentage_abnormal_spots(coords, shattered)
    # the random/shattered partition is highly abnormal spatially
    assert percentage_abnormal_spots(coords, shattered) > 0.4
    _ = rng  # determinism placeholder


def test_small_domain_iou_detects_dissolved_and_split():
    true = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # domain 1 is the small one (2 spots)
    kept = true.copy()
    dissolved = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # small domain absorbed into the big one
    split = np.array([0, 0, 0, 0, 0, 0, 2, 3])  # small domain shattered into two clusters
    assert abs(small_domain_iou(true, kept) - 1.0) < 1e-6
    assert abs(small_domain_iou(true, dissolved) - 0.25) < 1e-6  # absorbed -> low IoU
    assert abs(small_domain_iou(true, split) - 0.5) < 1e-6  # split -> partial IoU
