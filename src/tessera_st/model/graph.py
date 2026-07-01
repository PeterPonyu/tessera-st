"""Spatial graph construction. Identical kNN convention to the comparison world."""

from __future__ import annotations

import numpy as np


def build_knn_edges(coords: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Return (edge_index, edge_dist).

    edge_index: (2, E) int64 directed edges i<-j (j is a neighbour of i, both directions
    included so message passing is symmetric). edge_dist: (E,) euclidean distance per edge,
    used as an input feature to the edge gate.
    """
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dist, idx = nn.kneighbors(coords)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self

    src = np.repeat(np.arange(n), k)
    dst = idx.ravel()
    d = dist.ravel()
    # symmetrise
    edge_index = np.stack(
        [np.concatenate([dst, src]), np.concatenate([src, dst])], axis=0
    ).astype(np.int64)
    edge_dist = np.concatenate([d, d]).astype(np.float32)
    return edge_index, edge_dist
