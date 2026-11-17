"""STAGATE_pyG adapter for WP4 (environment dl: torch_geometric). The read-only clone is imported, never edited.

fit(x, coords, frames, arm, config, seed, input_mode, device) -> dict(embedding, src, dst, seconds, peak_gpu_bytes, ...)
  x: x_common for input_mode 'common'; the dataset's native dict for 'native'. Embedding: adata.obsm['STAGATE'],
  the 30-d latent of train_STAGATE at its defaults (hidden [512, 30], lr 1e-3, weight decay 1e-4, gradient
  clipping 5); n_epochs from the config. src/dst: the edge_index STAGATE trains on (Transfer_pytorch_Data).

Graph: the field-aware adapter's rule (labs/capsules/tessera-option-hold/experiments/field_aware_adapters.py,
stagate_input with tessera_st.spatial.neighbor_rows): each cell's k nearest neighbours within its own frame, the
query excluded by index, k capped at the frame size minus 1, written to Spatial_Net as Cell1 = cell,
Cell2 = neighbour. Transfer_pytorch_Data adds self-loops. Messages flow Cell1 -> Cell2 (PyG source-to-target),
so a cell aggregates over itself and the cells that list it among their neighbours.
DEPTH = 1: conv1 propagates; conv2 is called with attention=False and returns its linear map without
propagation, so the latent of a cell depends only on its one-hop in-neighbourhood (the decoder is not used).

Arms:
  released           STAGATE as released.
  uniform_attention  every GATConv built by STAGATE (encoder conv1 and the tied decoder conv3 propagate) is
                     UniformGATConv, whose message() gives each edge score 0 before the softmax, so every
                     edge into a cell gets weight 1 / in-degree. Applied by swapping the GATConv name in the
                     STAGATE_pyG.STAGATE module while the model is built; train_STAGATE runs unchanged.
  identity_graph     empty Spatial_Net: self-loops only.
  permuted_xy        coordinates permuted within each frame (RandomState(1000 + seed)) before the k-NN.
Configs 'knn{k}-e{epochs}': k {6, 15, 30} x epochs {500, 1000}; the default knn6-e1000 is the adapter's rule
(k = 6, which is also the plan's 'kNN 6') with train_STAGATE's n_epochs default. STAGATE ships no default
graph: Cal_Spatial_Net needs a radius or k, and its Visium tutorial radius gives about 6 neighbours per spot.
Inputs: 'common' trains on all columns of x_common (no highly_variable column); 'native' follows the STAGATE
tutorial: counts -> seurat_v3 3,000 HVG, normalize_total 1e4, log1p; the frozen log-normalized NSCLC panel
(no counts) -> all genes marked highly variable.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
STAGATE_ROOT = Path(os.environ.get("TESSERA_STAGATE_ROOT",
                                   "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external/STAGATE"))
if str(STAGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGATE_ROOT))

import common_external  # noqa: E402,F401  (round-2 import paths)
import graph_ops as go  # noqa: E402

ENCODER = "stagate"
ARMS = ("released", "uniform_attention", "identity_graph", "permuted_xy")
GRAPH_K = (6, 15, 30)
EPOCHS = (500, 1000)
CONFIG_FORMAT = "knn{}-e{}"
CONFIGS = tuple(CONFIG_FORMAT.format(k, e) for k in GRAPH_K for e in EPOCHS)
DEFAULT_CONFIG = "knn6-e1000"
DEPTH = 1
SCALING_POWER = 1  # sparse operator: per-epoch cost grows with n (dry-run estimate only)


def parse_config(config) -> tuple[int, int]:
    config = DEFAULT_CONFIG if config in (None, "default") else config
    k, e = config.split("-")
    if not (k.startswith("knn") and e.startswith("e")) or int(k[3:]) < 1 or int(e[1:]) < 1:
        raise ValueError(f"STAGATE config {config!r} is not knn<k>-e<epochs>")
    return int(k[3:]), int(e[1:])


def depth(arm: str) -> int:
    return 0 if arm == "identity_graph" else DEPTH


def knn_within_frames(coords, frames, k: int):
    """Directed k-NN inside each frame (neighbor_rows rule): (src = cell, dst = neighbour, distance)."""
    from sklearn.neighbors import NearestNeighbors
    coords = np.asarray(coords, float)
    frames = np.asarray(frames).astype(str)
    src, dst, dist = [], [], []
    for f in np.unique(frames):
        ix = np.flatnonzero(frames == f)
        d, idx = NearestNeighbors(n_neighbors=min(k + 1, len(ix))).fit(coords[ix]).kneighbors(coords[ix])
        for local in range(len(ix)):
            keep = idx[local] != local
            nb = idx[local][keep][:k]
            src.append(np.full(len(nb), ix[local]))
            dst.append(ix[nb])
            dist.append(d[local][keep][:k])
    src, dst = np.concatenate(src).astype(np.int64), np.concatenate(dst).astype(np.int64)
    if (src == dst).any():
        raise AssertionError("self edge in the spatial neighbours")
    go.assert_within_frames(src, dst, frames)
    return src, dst, np.concatenate(dist)


def spatial_net(coords, frames, arm, config, seed, names):
    """Spatial_Net DataFrame for this arm and config (Cell1 -> Cell2)."""
    import pandas as pd
    if arm not in ARMS:
        raise ValueError(f"unknown STAGATE arm {arm!r}")
    k, _ = parse_config(config)
    if arm == "identity_graph":
        return pd.DataFrame({"Cell1": pd.Series([], dtype=object), "Cell2": pd.Series([], dtype=object),
                             "Distance": pd.Series([], dtype=float)})
    if arm == "permuted_xy":
        coords = go.permute_within_frames(coords, frames, seed)
    src, dst, dist = knn_within_frames(coords, frames, k)
    names = np.asarray(names)
    return pd.DataFrame({"Cell1": names[src], "Cell2": names[dst], "Distance": dist})


def build_adata(x, input_mode):
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    if input_mode == "common":
        x = np.asarray(x, np.float32)
        adata = ad.AnnData(x.copy())
        adata.obs_names = pd.Index([f"c{i}" for i in range(len(x))])
        adata.var_names = pd.Index([f"f{j}" for j in range(x.shape[1])])
        return adata, "x_common; all columns used (no highly_variable column)"
    if input_mode != "native":
        raise ValueError(f"unknown input mode {input_mode!r}")
    matrix = x["counts"] if "counts" in x else x["expr_log"]
    adata = ad.AnnData(np.asarray(matrix, np.float32).copy())
    adata.obs_names = pd.Index(np.asarray(x["obs_id"]).astype(str))
    adata.var_names = pd.Index(np.asarray(x["genes"]).astype(str))
    adata.var_names_make_unique()
    if "counts" in x:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata, "STAGATE tutorial: seurat_v3 3000 HVG on counts, normalize_total 1e4, log1p"
    adata.var["highly_variable"] = True
    return adata, "frozen log-normalized panel (log1p of counts scaled to 1e4); all genes highly variable"


def _edge_index(names, net):
    """STAGATE's own Transfer_pytorch_Data edge_index; it depends only on Spatial_Net and the cell names."""
    import anndata as ad
    from STAGATE_pyG.utils import Transfer_pytorch_Data
    probe = ad.AnnData(np.zeros((len(names), 1), np.float32))
    probe.obs_names = names
    probe.uns["Spatial_Net"] = net
    src, dst = Transfer_pytorch_Data(probe).edge_index.numpy()
    return src.astype(np.int64), dst.astype(np.int64)


def encoder_edges(coords, frames, arm="released", config="default", seed=1):
    """(src, dst) of the edge_index STAGATE trains on for this arm and config, without training."""
    import pandas as pd
    names = pd.Index([f"c{i}" for i in range(len(coords))])
    src, dst = _edge_index(names, spatial_net(coords, frames, arm, config, seed, names))
    go.assert_within_frames(src, dst, frames)
    return src, dst


@functools.lru_cache(maxsize=None)
def uniform_conv_class():
    import torch
    import torch.nn.functional as F
    from STAGATE_pyG.gat_conv import GATConv
    from torch_geometric.utils import softmax

    class UniformGATConv(GATConv):
        """STAGATE's GATConv with every edge score set to 0 before the softmax: weight 1 / in-degree."""

        def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
            alpha = softmax(torch.zeros_like(alpha_j), index, ptr, size_i)
            self._alpha = alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            return x_j * alpha.unsqueeze(-1)

    return UniformGATConv


@contextlib.contextmanager
def conv_class(arm):
    """While active, STAGATE builds its layers from UniformGATConv for the uniform_attention arm."""
    module = importlib.import_module("STAGATE_pyG.STAGATE")
    original = module.GATConv
    if arm == "uniform_attention":
        module.GATConv = uniform_conv_class()
    try:
        yield
    finally:
        module.GATConv = original


def fit(x, coords, frames, arm, config, seed, input_mode="common", device="cpu") -> dict:
    import torch
    import STAGATE_pyG as ST
    frames = np.asarray(frames).astype(str)
    _, epochs = parse_config(config)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adata, prep = build_adata(x, input_mode)
    adata.uns["Spatial_Net"] = spatial_net(coords, frames, arm, config, seed, adata.obs_names)
    src, dst = _edge_index(adata.obs_names, adata.uns["Spatial_Net"])
    checks = {"edge_index": {"n_edges": go.assert_within_frames(src, dst, frames),
                             "n_self_loops": int((src == dst).sum()), "cross_frame_edges": 0}}
    setup_s = time.perf_counter() - started
    with conv_class(arm):
        out = ST.train_STAGATE(adata, n_epochs=epochs, random_seed=seed, device=torch.device(device), verbose=False)
    train_s = time.perf_counter() - started - setup_s
    embedding = np.asarray(out.obsm["STAGATE"], np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite STAGATE embedding")
    n_used = int(out.var["highly_variable"].sum()) if "highly_variable" in out.var.columns else int(out.n_vars)
    return dict(embedding=embedding, src=src, dst=dst, seconds=setup_s + train_s, setup_seconds=setup_s,
                train_seconds=train_s, peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None,
                depth=depth(arm), epochs=epochs, graph_checks=checks, frame_offset=None,
                settings={"preprocessing": prep, "graph_k": parse_config(config)[0], "epochs": epochs,
                          "n_features": n_used,
                          "hidden_dims": inspect.signature(ST.train_STAGATE).parameters["hidden_dims"].default,
                          "graph": "within-frame directed k-NN (field-aware adapter rule) + self-loops",
                          "attention": "uniform" if arm == "uniform_attention" else "released"})
