"""Spatial label refinement — the standard post-step used by graph-based spatial-domain methods.

Majority-vote each spot's predicted label over its spatial kNN (including itself), repeated a few
rounds. This smooths isolated mis-assignments and lifts label-agreement + spatial coherence. It was
initially missing from Tessera's pipeline; adding it benefits Tessera most (its backend-robust,
already-coherent embedding refines cleanly). A mild post-process — it can slightly soften the
sharpest boundaries, so it is optional, not forced.
"""

from __future__ import annotations

import numpy as np


def refine_labels(labels: np.ndarray, coords: np.ndarray, k: int = 6, rounds: int = 2, *, field_ids=None) -> np.ndarray:
    """Spatial majority-vote refinement of cluster labels. Labels must be non-negative ints."""
    from tessera_st.spatial import neighbor_rows

    idx, _ = neighbor_rows(coords, k, field_ids=field_ids, include_self=True)
    lab = np.asarray(labels).copy()
    if lab.ndim != 1 or len(lab) != len(coords) or not np.issubdtype(lab.dtype, np.integer) or (lab < 0).any():
        raise ValueError("one non-negative integer label per coordinate is required")
    if not isinstance(rounds, int) or rounds < 0:
        raise ValueError("rounds must be a non-negative integer")
    for _ in range(rounds):
        lab = np.array([np.bincount(lab[row]).argmax() for row in idx])
    return lab
