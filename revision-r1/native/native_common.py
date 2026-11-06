"""Shared helpers for E4 native-method runs (frame offsets, edge checks, scoring).

Used by run_native.py and test_frames.py in the tessera-native environment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

PROJECT = Path("/home/zeyufu/Desktop/singlecell-genomics-research/projects/tessera-st")
E4 = PROJECT / "results/revision-r1/E4"
INPUTS = E4 / "inputs"
NATIVE = PROJECT / "repository/revision-r1/native"
PROTOCOL = PROJECT / "repository/revision-r1/PROTOCOL-R1.md"
CASTLE_PARENT = E4 / "third_party/CASTLE"  # holds the cloned package directory CASTLE/
SEEDS = [1, 2, 3]
GAP_FACTOR = 50.0  # gap = 50 x median in-frame nearest-neighbour distance (floor)
GAP_K = 30         # and at least 2 x the largest in-frame 30th-neighbour distance


def sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_input(cohort: str, patient: str) -> dict:
    path = INPUTS / f"{cohort}-{patient}.npz"
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out["path"] = str(path)
    out["sha256"] = sha(path)
    return out


def offset_frames(coords, frames, k: int = GAP_K, factor: float = GAP_FACTOR):
    """Translate each frame so that frames sit side by side along x.

    Frames are placed in sorted order; each frame starts at y = 0 and at x equal
    to the previous frame's right edge plus a gap. The gap is the larger of
    `factor` times the median in-frame nearest-neighbour distance and twice the
    largest in-frame k-th neighbour distance, so no k-nearest-neighbour graph
    (k <= `k`) or radius graph with radius below the gap can join two frames.
    Within-frame geometry is unchanged (translation only).
    """
    coords = np.asarray(coords, np.float64)
    frames = np.asarray(frames).astype(str)
    names = sorted(np.unique(frames))
    nn1 = np.empty(len(coords))
    kth = np.empty(len(coords))
    for name in names:
        ix = np.flatnonzero(frames == name)
        if len(ix) <= k:
            raise ValueError(f"frame {name} has {len(ix)} cells, needs more than {k}")
        dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(coords[ix]).kneighbors(coords[ix])
        nn1[ix] = dist[:, 1]
        kth[ix] = dist[:, k]
    median_nn = float(np.median(nn1))
    gap = max(factor * median_nn, 2.0 * float(kth.max()))
    out = np.empty_like(coords)
    cursor = 0.0
    placement = []
    for name in names:
        ix = np.flatnonzero(frames == name)
        lo, hi = coords[ix].min(0), coords[ix].max(0)
        out[ix, 0] = coords[ix, 0] - lo[0] + cursor
        out[ix, 1] = coords[ix, 1] - lo[1]
        placement.append({"frame": name, "x_start": cursor, "width": float(hi[0] - lo[0]), "n": int(len(ix))})
        cursor += float(hi[0] - lo[0]) + gap
    info = {"frames": names, "median_nn": median_nn, "max_kth_nn": float(kth.max()), "k": k,
            "factor": factor, "gap": gap, "placement": placement}
    return out, info


def edge_frame_check(matrix, frames) -> dict:
    """Count edges (nonzero off-diagonal entries) joining cells of different frames."""
    frames = np.asarray(frames).astype(str)
    if sp.issparse(matrix):
        coo = sp.coo_matrix(matrix)
        rows, cols = coo.row, coo.col
        keep = coo.data != 0
        rows, cols = rows[keep], cols[keep]
    else:
        rows, cols = np.nonzero(np.asarray(matrix))
    off = rows != cols
    rows, cols = rows[off], cols[off]
    cross = int((frames[rows] != frames[cols]).sum())
    return {"n_edges": int(len(rows)), "cross_frame_edges": cross}


def knn_frame_check(coords, frames, k: int = GAP_K) -> dict:
    """Generic check: each cell's k nearest neighbours in the (offset) coordinates."""
    frames = np.asarray(frames).astype(str)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    cross = int((frames[idx] != frames[:, None]).sum())
    return {"n_edges": int(idx.size), "cross_frame_edges": cross, "k": k}


def kmeans_matched(embedding, k: int, seed: int) -> np.ndarray:
    return KMeans(k, n_init=10, random_state=seed).fit(embedding).labels_


def score(y, pred) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
    ok = y >= 0
    return {"ari": float(adjusted_rand_score(y[ok], pred[ok])),
            "nmi": float(normalized_mutual_info_score(y[ok], pred[ok])),
            "n_scored": int(ok.sum()), "K_achieved": int(len(np.unique(pred)))}
