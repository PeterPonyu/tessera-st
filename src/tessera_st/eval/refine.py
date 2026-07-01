"""Spatial label refinement — the standard post-step used by graph-based spatial-domain methods.

Majority-vote each spot's predicted label over its spatial kNN (including itself), repeated a few
rounds. This smooths isolated mis-assignments and lifts label-agreement + spatial coherence. It was
initially missing from Tessera's pipeline; adding it benefits Tessera most (its backend-robust,
already-coherent embedding refines cleanly). A mild post-process — it can slightly soften the
sharpest boundaries, so it is optional, not forced.
"""

from __future__ import annotations

import numpy as np


def refine_labels(labels: np.ndarray, coords: np.ndarray, k: int = 6, rounds: int = 2) -> np.ndarray:
    """Spatial majority-vote refinement of cluster labels. Labels must be non-negative ints."""
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords).kneighbors(coords)[1]
    lab = np.asarray(labels).copy()
    for _ in range(rounds):
        block = lab[idx]  # (n, k+1): self + k neighbours
        lab = np.array([np.bincount(row).argmax() for row in block])
    return lab
