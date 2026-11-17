"""GraphST 1.1.1 adapter for WP4 (environment tessera-native). GraphST's files are never edited.

fit(x, coords, frames, arm, config, seed, input_mode, device) -> dict(embedding, src, dst, seconds, peak_gpu_bytes, ...)
  x: x_common (n, d) for input_mode 'common'; the dataset's native dict ({'counts' or 'expr_log', 'genes',
  'obs_id'}) for 'native'. Embedding: adata.obsm['emb'] returned by GraphST.train(), the output round 2 clustered.
  Its value is h = A_hat (A_hat X W1) W2 with A_hat = I + D^-1/2 A D^-1/2 (datatype '10X', dropout 0): two
  propagation steps, linear in X, hence DEPTH = 2 (0 for identity_encoder). src/dst: the edges of A_hat, the
  operator the encoder actually used (graph_ops convention).

Arms:
  released          GraphST defaults (alpha 10, beta 1, lr 1e-3, 64 hidden units, datatype '10X').
  no_contrast       beta = 0 (constructor argument): the contrastive loss is off, reconstruction kept.
  identity_encoder  after construction the operator GraphST passes to Encoder.forward (GraphST.adj) is the
                    identity, so no message passing; Encoder.forward uses that one argument in the encoder, the
                    decoder and the corrupted branch. The readout mask (graph_neigh = spatial k-NN + I, given to
                    the Encoder at construction) is unchanged: spatial positives without message passing.
  permuted_xy       coordinates permuted within each frame (RandomState(1000 + seed)) before the graph is built.
Configs 'k{n_neighbors}-e{epochs}': n_neighbors {3, 6, 15, 30} x epochs {600, 1200}; the default k3-e600 is
GraphST's construct_interaction default and epoch default. The graph is built by GraphST's own
construct_interaction before construction (GraphST then skips its own call), identical to the default path.
Inputs: 'common' gives x_common to GraphST as its features (every column marked highly_variable, so GraphST's
preprocess is skipped); 'native' is round 2's run_native.graph_adata (counts: GraphST's preprocess in the
constructor; log-normalized panel: scale(zero_center=False, max_value=10), all genes highly variable).
Frames: coordinates are offset per frame (native_common.offset_frames: gap > 2 x the largest in-frame
30th-neighbour distance), and the adjacency, readout mask and operator are checked for cross-frame edges
before training.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common_external  # noqa: E402,F401  (round-2 import paths)
import graph_ops as go  # noqa: E402
import native_common as nc  # noqa: E402

ENCODER = "graphst"
ARMS = ("released", "no_contrast", "identity_encoder", "permuted_xy")
NEIGHBORS = (3, 6, 15, 30)
EPOCHS = (600, 1200)
CONFIG_FORMAT = "k{}-e{}"
CONFIGS = tuple(CONFIG_FORMAT.format(k, e) for k in NEIGHBORS for e in EPOCHS)
DEFAULT_CONFIG = "k3-e600"
DEPTH = 2
SCALING_POWER = 2  # dense n x n operator: per-epoch cost grows with n^2 (dry-run estimate only)


def parse_config(config) -> tuple[int, int]:
    config = DEFAULT_CONFIG if config in (None, "default") else config
    k, e = config.split("-")
    if not (k.startswith("k") and e.startswith("e")) or int(k[1:]) < 1 or int(e[1:]) < 1:
        raise ValueError(f"GraphST config {config!r} is not k<n_neighbors>-e<epochs>")
    return int(k[1:]), int(e[1:])


def depth(arm: str) -> int:
    return 0 if arm == "identity_encoder" else DEPTH


def build_adata(x, coords, frames, arm, config, seed, input_mode):
    """AnnData with frame-offset coordinates and GraphST's own graph (adj, graph_neigh) for this arm/config."""
    import anndata as ad
    import pandas as pd
    from GraphST.preprocess import construct_interaction
    if arm not in ARMS:
        raise ValueError(f"unknown GraphST arm {arm!r}")
    k, _ = parse_config(config)
    frames = np.asarray(frames).astype(str)
    if arm == "permuted_xy":
        coords = go.permute_within_frames(coords, frames, seed)
    if input_mode == "common":
        x = np.asarray(x, np.float32)
        adata = ad.AnnData(x.copy())
        adata.obs_names = pd.Index([f"c{i}" for i in range(len(x))])
        adata.var_names = pd.Index([f"f{j}" for j in range(x.shape[1])])
        adata.var["highly_variable"] = True
        adata.obsm["spatial"], offset = nc.offset_frames(coords, frames)
        prep = "x_common as GraphST features; all columns highly_variable, GraphST preprocess skipped"
    elif input_mode == "native":
        import run_native as rn  # round-2 E4 preprocessing, imported unchanged
        inp = {"has_counts": "counts" in x, "counts": x.get("counts"), "expr_log": x.get("expr_log"),
               "obs_id": x["obs_id"], "genes": x["genes"], "coords": coords, "frame": frames}
        adata, offset, prep = rn.graph_adata(inp)
    else:
        raise ValueError(f"unknown input mode {input_mode!r}")
    construct_interaction(adata, n_neighbors=k)
    checks = {key: nc.edge_frame_check(adata.obsm[key], frames) for key in ("adj", "graph_neigh")}
    if any(v["cross_frame_edges"] for v in checks.values()) or checks["adj"]["n_edges"] == 0:
        raise AssertionError(f"GraphST graph crosses frames or is empty: {checks}")
    return adata, offset, prep, checks


def encoder_edges(coords, frames, arm="released", config="default", seed=1):
    """(src, dst) of the operator GraphST's encoder uses for this arm and config, without training."""
    from GraphST.preprocess import preprocess_adj
    n = len(coords)
    if arm == "identity_encoder":
        return np.arange(n), np.arange(n)
    adata, *_ = build_adata(np.zeros((n, 1), np.float32), coords, frames, arm, config, seed, "common")
    dst, src = np.nonzero(preprocess_adj(adata.obsm["adj"]))
    go.assert_within_frames(src, dst, frames)
    return src, dst


def prepare(x, coords, frames, arm, config, seed, input_mode="common", device="cpu"):
    """A GraphST model ready for train(), with this arm's change applied, and its graph record."""
    import torch
    from GraphST import GraphST as graphst
    frames = np.asarray(frames).astype(str)
    _, epochs = parse_config(config)
    adata, offset, prep, checks = build_adata(x, coords, frames, arm, config, seed, input_mode)
    kwargs = {"beta": 0.0} if arm == "no_contrast" else {}
    model = graphst.GraphST(adata, device=torch.device(device), random_seed=seed, epochs=epochs, **kwargs)
    if arm == "identity_encoder":
        model.adj = torch.eye(model.adj.shape[0], dtype=model.adj.dtype, device=model.adj.device)
    dst, src = np.nonzero(model.adj.detach().cpu().numpy())
    readout_dst, readout_src = np.nonzero(model.graph_neigh.detach().cpu().numpy())
    checks["operator"] = {"n_edges": go.assert_within_frames(src, dst, frames),
                          "n_self_loops": int((src == dst).sum()), "cross_frame_edges": 0}
    checks["readout_mask"] = {"n_edges": go.assert_within_frames(readout_src, readout_dst, frames),
                              "cross_frame_edges": 0}
    info = {"src": src.astype(np.int64), "dst": dst.astype(np.int64), "graph_checks": checks,
            "frame_offset": offset,
            "settings": {"preprocessing": prep, "n_neighbors": parse_config(config)[0], "epochs": model.epochs,
                         "beta": model.beta, "alpha": model.alpha, "datatype": model.datatype,
                         "n_features": int(model.dim_input), "dim_output": int(model.dim_output),
                         "learning_rate": model.learning_rate, "graph": "GraphST construct_interaction",
                         "operator": "identity" if arm == "identity_encoder" else "I + D^-1/2 A D^-1/2"}}
    return model, info


def fit(x, coords, frames, arm, config, seed, input_mode="common", device="cpu") -> dict:
    import torch
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, info = prepare(x, coords, frames, arm, config, seed, input_mode, device)
    setup_s = time.perf_counter() - started
    out = model.train()
    train_s = time.perf_counter() - started - setup_s
    embedding = np.asarray(out.obsm["emb"], np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite GraphST embedding")
    return dict(embedding=embedding, seconds=setup_s + train_s, setup_seconds=setup_s, train_seconds=train_s,
                peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None, depth=depth(arm),
                epochs=int(model.epochs), **info)
