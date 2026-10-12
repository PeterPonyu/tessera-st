"""Synthetic *tessellation* generator (numpy only).

Produces a slide of spots tiled into a handful of spatial domains with sharp boundaries,
plus a held-out small domain to stress small-domain recall. Used for offline CI and for
demonstrating that each Tessera component improves boundary-aware metrics — it makes no
biological claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticSlide:
    coords: np.ndarray  # (n, 2) spatial coordinates
    expr: np.ndarray  # (n, g) expression-like features
    labels: np.ndarray  # (n,) ground-truth domain id
    boundary: np.ndarray  # (n,) bool: spot is adjacent to a different domain


def make_tessellation(
    n_side: int = 28,
    n_genes: int = 50,
    n_domains: int = 4,
    domain_signal: float = 0.4,
    iid_noise: float = 1.0,
    small_domain: bool = True,
    seed: int = 0,
) -> SyntheticSlide:
    """Build a grid slide partitioned into contiguous domains where *space is necessary*.

    The difficulty is set by the per-spot signal-to-noise ratio: ``domain_signal`` is the
    std of the (shared, per-domain) mean vectors and ``iid_noise`` is the per-spot Gaussian
    std. With ``domain_signal < iid_noise`` a single spot is too noisy to classify, so a
    **non-spatial** clusterer fails — but because neighbours share a domain mean, averaging
    over a k-neighbourhood recovers the signal while the iid noise shrinks by ~sqrt(k). That
    is exactly the regime where spatial aggregation earns its keep; an over-easy fixture (large
    ``domain_signal``) lets a non-spatial KMeans hit ARI=1.0 and proves nothing.

    A small circular domain is injected to test small-domain recall — the first casualty of
    *uniform* over-smoothing, which the edge gate is meant to spare.
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    n = coords.shape[0]

    # Vertical-stripe partition for the main domains (contiguous, sharp seams).
    stripe = (coords[:, 0] / n_side * n_domains).astype(int)
    labels = np.clip(stripe, 0, n_domains - 1)

    if small_domain:
        center = np.array([n_side * 0.5, n_side * 0.5])
        radius = n_side * 0.12
        in_disc = np.linalg.norm(coords - center, axis=1) < radius
        labels = labels.copy()
        labels[in_disc] = n_domains  # extra small domain id
    n_dom = int(labels.max()) + 1

    # weak per-domain signal + strong per-spot noise => non-spatial classification fails,
    # neighbourhood averaging succeeds.
    means = rng.normal(0.0, domain_signal, size=(n_dom, n_genes))
    expr = means[labels] + rng.normal(0.0, iid_noise, size=(n, n_genes))

    boundary = _boundary_mask(coords, labels)
    return SyntheticSlide(coords=coords, expr=expr.astype(np.float32), labels=labels,
                          boundary=boundary)


def _boundary_mask(coords: np.ndarray, labels: np.ndarray, k: int = 6) -> np.ndarray:
    """A spot is a boundary spot if any of its k nearest neighbours has a different label."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    neigh = idx[:, 1:]
    return (labels[neigh] != labels[:, None]).any(axis=1)
