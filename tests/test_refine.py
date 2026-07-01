"""Spatial label refinement — torch-free."""

import numpy as np

from tessera_st.eval.refine import refine_labels


def _grid(n=5):
    xs, ys = np.meshgrid(range(n), range(n))
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)


def test_refine_fixes_isolated_spot():
    coords = _grid(5)
    labels = np.zeros(25, dtype=int)
    labels[12] = 1  # one wrong label in the centre, surrounded by 0s
    out = refine_labels(labels, coords, k=4)
    assert out[12] == 0  # majority vote flips the isolated spot back


def test_refine_preserves_clean_partition():
    coords = _grid(6)
    labels = (coords[:, 0] >= 3).astype(int)  # clean two-block split
    out = refine_labels(labels, coords, k=4)
    assert (out == labels).mean() > 0.8  # mostly unchanged (boundary may shift a little)


def test_refine_is_noop_on_uniform():
    coords = _grid(4)
    labels = np.zeros(16, dtype=int)
    assert np.array_equal(refine_labels(labels, coords, k=4), labels)
