"""Graph operations for the WP4 encoders and their parameter-free counterparts (numpy/scipy only).

Edge convention: edge e carries a message from src[e] to dst[e], so node i aggregates over
{src[e] : dst[e] == i} (PyTorch Geometric's source-to-target flow, as in round 1's neighbor_embedding).
A self-loop is an ordinary edge, present exactly when the encoder's own operator has a self term:
  GraphST 1.1.1 ('10X'): operator I + D^-1/2 A D^-1/2 on the symmetrized k-NN graph A  -> self term present;
  STAGATE_pyG: Transfer_pytorch_Data adds the identity to the spatial net               -> self term present.
P is the row-normalized adjacency of that edge multiset: P[i, j] = (edges j -> i) / (edges into i).
same_graph_mean is one step of P; propagate applies it L times (L = the encoder's message-passing depth).
"""
from __future__ import annotations

import importlib

import numpy as np
import scipy.sparse as sp

ADAPTERS = {"graphst": "run_graphst_r3", "stagate": "run_stagate_r3"}


def permute_within_frames(coords, frames, seed: int) -> np.ndarray:
    """Coordinates permuted within each frame, RandomState(1000 + seed), frames in sorted order
    (the PermutedXY rule of run_experiments.graph and revision-r1/train_core.build_graph)."""
    xy = np.asarray(coords, float).copy()
    frames = np.asarray(frames)
    rng = np.random.RandomState(1000 + seed)
    for f in np.unique(frames):
        ix = np.flatnonzero(frames == f)
        xy[ix] = xy[rng.permutation(ix)]
    return xy


def assert_within_frames(src, dst, frames) -> int:
    """Raise if any edge joins two frames; return the number of edges checked."""
    frames = np.asarray(frames)
    cross = int((frames[np.asarray(src)] != frames[np.asarray(dst)]).sum())
    if cross:
        raise AssertionError(f"{cross} of {len(src)} edges join two frames")
    return int(len(src))


def operator(src, dst, n: int) -> sp.csr_matrix:
    """Row-normalized adjacency P (rows = receiving nodes). Every node must receive at least one edge."""
    counts = sp.csr_matrix((np.ones(len(src)), (np.asarray(dst), np.asarray(src))), shape=(n, n))
    indegree = np.asarray(counts.sum(1)).ravel()
    if (indegree == 0).any():
        raise ValueError(f"{int((indegree == 0).sum())} nodes receive no edge; the encoders' graphs carry self-loops")
    return sp.diags(1.0 / indegree) @ counts


def propagate(x, src, dst, L: int) -> np.ndarray:
    """P^L x on the given graph (float64 arithmetic, float32 result); L = 0 returns x."""
    x = np.asarray(x, np.float64)
    P = operator(src, dst, len(x))
    for _ in range(L):
        x = P @ x
    return x.astype(np.float32)


def same_graph_mean(x, src, dst) -> np.ndarray:
    """Unweighted mean of x over each node's incoming edges (its own row included iff the graph has a self-loop)."""
    return propagate(x, src, dst, 1)


def edges_from_encoder(encoder: str, coords, frames, config: str = "default", arm: str = "released", seed: int = 1):
    """The encoder's own message-passing graph (src, dst), built by its adapter exactly as fit() builds it.

    Needs the encoder's environment: GraphST in tessera-native, STAGATE in dl.
    """
    return importlib.import_module(ADAPTERS[encoder]).encoder_edges(coords, frames, arm, config, seed)
