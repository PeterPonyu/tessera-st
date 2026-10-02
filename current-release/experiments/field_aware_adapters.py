"""Field-disjoint adapters for the existing spatial-method implementations.

External source trees are read-only. Spatial positive edges, distance features,
and SEDR reconstruction negatives are restricted to the actual coordinate frame.
Feature-space neighbours may connect fields: those are not spatial distances.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from _roots import external_root
from tessera_st.spatial import neighbor_rows


def directed_graph(coords, fields, k=6):
    rows, distances = neighbor_rows(coords, k=k, field_ids=fields)
    src = np.repeat(np.arange(len(coords)), [len(r) for r in rows])
    dst = np.concatenate(rows)
    distance = np.concatenate(distances)
    fields = np.asarray(fields)
    if np.any(fields[src] != fields[dst]) or np.any(src == dst):
        raise AssertionError("cross-field or self edge in spatial neighbours")
    return src, dst, distance, rows


def neighbor_mean(Z, coords, fields):
    _, _, _, rows = directed_graph(coords, fields)
    # A singleton retains its own features; it has no neighbour agreement term.
    return np.stack([Z[row].mean(0) if len(row) else Z[i]
                     for i, row in enumerate(rows)])


def stagate_input(Z, coords, fields):
    import anndata as ad
    import pandas as pd
    src, dst, dist, _ = directed_graph(coords, fields)
    a = ad.AnnData(X=np.asarray(Z, np.float32))
    a.obsm["spatial"] = np.asarray(coords)
    a.obs["coordinate_frame"] = np.asarray(fields, str)
    names = np.asarray(a.obs_names)
    a.uns["Spatial_Net"] = pd.DataFrame({
        "Cell1": names[src], "Cell2": names[dst], "Distance": dist})
    return a


def sedr_graph(coords, fields, seed):
    """SEDR's documented graph_dict, with explicit within-field negative pairs.

    Upstream negative sampling uses positions in a candidate list as node IDs;
    this adapter samples actual non-neighbour IDs and never treats another
    field as a spatial negative. Tiny complete fields contribute positives only.
    """
    import torch
    src, dst, _, _ = directed_graph(coords, fields)
    n = len(coords)
    adj = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    adj = adj.maximum(adj.T) + sp.eye(n, format="csr")
    adj.data[:] = 1
    degree = np.asarray(adj.sum(1)).ravel()
    normalized = sp.diags(degree**-0.5) @ adj @ sp.diags(degree**-0.5)

    def tensor(matrix):
        coo = matrix.tocoo()
        return torch.sparse_coo_tensor(np.stack([coo.row, coo.col]),
            coo.data.astype(np.float32), coo.shape).coalesce()

    labels = tensor(adj)
    rng = np.random.default_rng(seed)
    fields = np.asarray(fields)
    group = {f: np.flatnonzero(fields == f) for f in np.unique(fields)}
    neg_src, neg_dst = [], []
    for i in range(n):
        positive = adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        candidates = np.setdiff1d(group[fields[i]], positive, assume_unique=True)
        take = min(len(candidates), len(positive))
        if take:
            chosen = rng.choice(candidates, take, replace=False)
            neg_src.extend([i]*take)
            neg_dst.extend(chosen)
    coo = adj.tocoo()
    src_all = np.r_[coo.row, np.asarray(neg_src, int)]
    dst_all = np.r_[coo.col, np.asarray(neg_dst, int)]
    values = np.r_[coo.data, np.zeros(len(neg_src))].astype(np.float32)
    if np.any(fields[src_all] != fields[dst_all]):
        raise AssertionError("SEDR reconstruction mask crosses frames")
    mask = torch.sparse_coo_tensor(np.stack([src_all, dst_all]), values, (n, n)).coalesce()
    possible = sum(len(v)**2 for v in group.values())
    negative_possible = possible - adj.nnz
    if not negative_possible:
        raise ValueError("SEDR requires at least one within-field non-neighbour")
    return {"adj_norm": tensor(normalized), "adj_label": labels, "mask": mask,
            "norm_value": possible / (2*negative_possible)}


def external_embedding(method, Z, coords, fields, seed, *, device="cuda",
                       epochs=None):
    import torch
    sys.path[:0] = [str(external_root()/m) for m in ("STAGATE", "SEDR")]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if method == "STAGATE":
        import STAGATE_pyG as ST
        from STAGATE_pyG.utils import Transfer_pytorch_Data
        a = stagate_input(Z, coords, fields)
        edges = Transfer_pytorch_Data(a).edge_index.numpy()
        f = np.asarray(fields)
        if np.any(f[edges[0]] != f[edges[1]]):
            raise AssertionError("STAGATE runtime input crosses fields")
        fitted = ST.train_STAGATE(a, n_epochs=600 if epochs is None else epochs,
            random_seed=seed, device=torch.device(device), verbose=False)
        return np.asarray(fitted.obsm["STAGATE"])
    if method == "SEDR":
        import SEDR
        from sklearn.decomposition import PCA
        p = PCA(min(200, Z.shape[1]-1, len(Z)-1), random_state=seed).fit_transform(Z)
        graph = sedr_graph(coords, fields, seed)
        net = SEDR.Sedr(p.astype(np.float32), graph, device=device)
        net.train_without_dec(epochs=200 if epochs is None else epochs, N=1)
        return np.asarray(net.process()[0])
    if method == "SpaceFlow":
        # Replacing its graph alone would still pool coordinates in the loss.
        raise RuntimeError("Not run: SpaceFlow global distance penalty lacks a field-aware adapter; gudhi is also absent")
    if method in {"GraphST", "SpaGCN"}:
        raise RuntimeError(f"Not run: {method} spatial distance/refinement adapter is not field-audited")
    raise ValueError(f"Unknown external method: {method}")
