"""Brand-neutral reference clusterers for in-package sanity baselines.

These are *generic* algorithms (no proprietary spatial-domain method is reimplemented here).
The named external methods we compare against in the paper are run from their own repositories
behind an adapter and are listed only in BASELINE_REFERENCES.md — never in this source tree.
"""

from __future__ import annotations

import numpy as np


def kmeans_labels(embed: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(embed)


def expression_kmeans(expr: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Non-spatial baseline: cluster raw expression. The floor any spatial method must clear."""
    return kmeans_labels(expr, n_clusters, seed)


def mean_smoothed_kmeans(
    expr: np.ndarray, edge_index: np.ndarray, n_clusters: int, rounds: int = 2, seed: int = 0
) -> np.ndarray:
    """Uniform-smoothing baseline: average features over neighbours, then cluster.

    This is the *over-smoothing* behaviour Tessera's edge gate is designed to beat, expressed
    as a transparent reference so the ablation has an external anchor.
    """
    h = expr.astype(np.float64).copy()
    src, dst = edge_index[0], edge_index[1]
    n = expr.shape[0]
    for _ in range(rounds):
        agg = np.zeros_like(h)
        cnt = np.zeros(n)
        np.add.at(agg, dst, h[src])
        np.add.at(cnt, dst, 1.0)
        h = agg / (cnt[:, None] + 1e-8)
    return kmeans_labels(h, n_clusters, seed)
