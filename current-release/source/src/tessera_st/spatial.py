"""Spatial neighbours in independent local coordinate frames.

Coordinates are never translated or rescaled to manufacture separation.
Field identity is a required companion of pooled local coordinates.
"""
from __future__ import annotations

import numpy as np


def neighbor_rows(coords, k=6, *, field_ids=None, include_self=False):
    """Return variable-length index/distance rows, capped within each field.

    A singleton has no neighbours (or itself when explicitly requested).
    The query index, rather than the first zero-distance result, identifies
    self; distinct cells at identical coordinates remain legitimate neighbours.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or not len(coords):
        raise ValueError("non-empty (n, d>=2) coordinates required")
    if not np.isfinite(coords).all():
        raise ValueError("spatial coordinates must be finite")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    n = len(coords)
    if field_ids is None:
        fields = np.zeros(n, dtype=int)
    else:
        fields = np.asarray(field_ids, dtype=object)
        if fields.ndim != 1 or len(fields) != n:
            raise ValueError("field_ids must match coordinate rows")
        for value in fields:
            if value is None or str(value).strip().lower() in {"", "nan", "none", "<na>"}:
                raise ValueError("missing coordinate-frame identifier")
    groups = {}
    for i, field in enumerate(fields):
        groups.setdefault(field, []).append(i)
    indices = [None] * n
    distances = [None] * n
    for members in groups.values():
        members = np.asarray(members, dtype=int)
        xy = coords[members]
        width = min(k+1, len(members))
        dist, idx = NearestNeighbors(n_neighbors=width).fit(xy).kneighbors(xy)
        for local, original in enumerate(members):
            valid = idx[local] != local
            neighbor = members[idx[local][valid][:k]]
            distance = dist[local][valid][:k]
            if include_self:
                neighbor = np.concatenate(([original], neighbor))
                distance = np.concatenate(([0.0], distance))
            indices[original] = neighbor
            distances[original] = distance
    return indices, distances
