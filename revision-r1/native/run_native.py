#!/usr/bin/env python3
"""E4 native external methods: GraphST, BANKSY, CASTLE (PROTOCOL-R1.md, E4).

Run in the tessera-native environment, one method at a time, for example
    python -B run_native.py --method GraphST --cohort NSCLC
Inputs come from results/revision-r1/E4/inputs/ (prepare_inputs.py).
For each fit (method, variant, patient, seed) the script writes
    results/revision-r1/E4/embeddings/<method>-<variant>-<patient>-seed<seed>.npz
    results/revision-r1/E4/fits/<method>-<variant>-<patient>-seed<seed>.json
and aggregate.py builds E4_scores.csv from the JSON records.

Every fit offsets coordinates per frame (native_common.offset_frames) and
checks the graph the method actually built: a fit whose graph joins two
frames is stopped before training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
# rpy2 must load the environment's R (R 4.5.3 with r-mclust 6.1.1), not the system R.
os.environ["R_HOME"] = str(Path(sys.prefix) / "lib" / "R")
os.environ["PATH"] = f"{Path(sys.prefix) / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_common as nc  # noqa: E402

PATIENTS = {
    "NSCLC": ["NSCLC_Lung5_Rep2", "NSCLC_Lung6", "NSCLC_Lung9_Rep1", "NSCLC_Lung12", "NSCLC_Lung13"],
    "HER2ST": ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"],
}
# (variant, targets scored) per method and cohort.
PLAN = {
    "GraphST": {"NSCLC": [("default", ["region", "celltype"])],
                "HER2ST": [("default", ["all", "boundary_T1"])]},
    "BANKSY": {"NSCLC": [("lambda0.8", ["region"]), ("lambda0.2", ["celltype"])],
               "HER2ST": [("lambda0.8", ["all", "boundary_T1"])]},
    "CASTLE": {"NSCLC": [("celltype", ["region", "celltype"]), ("expression", ["region", "celltype"])],
               "HER2ST": [("expression", ["all", "boundary_T1"])]},
}
FITS = nc.E4 / "fits"
EMB = nc.E4 / "embeddings"


class CrossFrameError(RuntimeError):
    pass


def require_no_cross(checks: dict) -> None:
    bad = {k: v for k, v in checks.items() if v["cross_frame_edges"]}
    if bad:
        raise CrossFrameError(f"graph edges cross frames: {bad}")


def graph_adata(inp):
    """AnnData for GraphST and CASTLE (identical documented preprocessing).

    HER2ST: raw counts; the method's own preprocess runs in its constructor
    (seurat_v3 HVG 3000 on counts, normalize_total 1e4, log1p,
    scale(zero_center=False, max_value=10)).
    NSCLC: raw counts are not available. The frozen log-normalized matrix
    (log1p of counts scaled to 1e4) equals the method's normalize_total + log1p
    output; seurat_v3 with 3000 top genes would keep all 960 panel genes, so
    all genes are marked highly variable and the remaining documented step,
    scale(zero_center=False, max_value=10), is applied.
    """
    import anndata as ad
    import scanpy as sc
    if bool(inp["has_counts"]):
        adata = ad.AnnData(inp["counts"].astype(np.float32))
        prep = "method preprocess on raw counts"
    else:
        adata = ad.AnnData(inp["expr_log"].astype(np.float32).copy())
        sc.pp.scale(adata, zero_center=False, max_value=10)
        adata.var["highly_variable"] = True
        prep = "log-normalized input, all genes highly_variable, scale(zero_center=False, max_value=10)"
    adata.obs_names = pd.Index(inp["obs_id"].astype(str))
    adata.var_names = pd.Index(inp["genes"].astype(str))
    coords, offset = nc.offset_frames(inp["coords"], inp["frame"])
    adata.obsm["spatial"] = coords
    return adata, offset, prep


def fit_graphst(inp, seed, variant, targets, device):
    import torch
    from GraphST import GraphST as graphst
    from GraphST.utils import clustering
    frames = inp["frame"].astype(str)
    adata, offset, prep = graph_adata(inp)
    started = time.perf_counter()
    model = graphst.GraphST(adata, device=torch.device(device), random_seed=seed)
    checks = {"adj": nc.edge_frame_check(model.adata.obsm["adj"], frames),
              "graph_neigh": nc.edge_frame_check(model.adata.obsm["graph_neigh"], frames)}
    require_no_cross(checks)
    out = model.train()
    embed_s = time.perf_counter() - started
    embedding = np.asarray(out.obsm["emb"], np.float32)
    native, emb_pca = {}, None
    for target in targets:
        k = int(inp[f"K_{target}"])
        t0 = time.perf_counter()
        try:
            clustering(out, n_clusters=k, radius=50, method="mclust", refinement=False)
            labels = np.asarray(out.obs["domain"]).astype(int)
            emb_pca = np.asarray(out.obsm["emb_pca"], np.float32)
            native[target] = {"labels": labels, "seconds": time.perf_counter() - t0,
                              "detail": {"algorithm": "GraphST.utils.clustering mclust (R mclust, EEE, seed 2020) "
                                                      "on PCA(20, random_state=42) of emb; refinement=False"}}
        except Exception as exc:  # recorded, never replaced by another rule
            native[target] = {"error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0}
    extra = {"preprocessing": prep, "datatype": "10X (default)", "epochs": model.epochs,
             "n_features": int(model.dim_input), "graph": "GraphST construct_interaction (3 nearest neighbours)"}
    return embedding, emb_pca, native, embed_s, checks, offset, extra


def banksy_bisect(partitioner, k, seed):
    import leidenalg

    def run(res):
        label, _ = partitioner.partition(resolution=res, partition_metric=leidenalg.RBConfigurationVertexPartition,
                                         n_iterations=-1, seed=seed)
        trace.append({"resolution": float(res), "n_labels": int(label.num_labels)})
        return label.num_labels

    trace = []
    lo, hi = 1e-3, 1.0
    n_lo, n_hi = run(lo), run(hi)
    while n_hi < k and hi < 64:
        lo, n_lo = hi, n_hi
        hi *= 2
        n_hi = run(hi)
    for _ in range(40):
        if n_lo == k or n_hi == k:
            break
        if n_lo > k or n_hi < k:
            break
        mid = float(np.sqrt(lo * hi))
        n_mid = run(mid)
        if n_mid == k:
            break
        if n_mid < k:
            lo, n_lo = mid, n_mid
        else:
            hi, n_hi = mid, n_mid
    best = min(trace, key=lambda r: (abs(r["n_labels"] - k), r["resolution"]))
    return best["resolution"], best["n_labels"], trace


def fit_banksy(inp, seed, variant, targets, device):
    import anndata as ad
    import scanpy as sc
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.main import LeidenPartition
    from banksy.cluster_methods import run_Leiden_partition
    from banksy_utils.umap_pca import pca_umap
    lam = float(variant.replace("lambda", ""))
    frames = inp["frame"].astype(str)
    np.random.seed(seed)
    random.seed(seed)
    adata = ad.AnnData(inp["expr_log"].astype(np.float32).copy())
    adata.obs_names = pd.Index(inp["obs_id"].astype(str))
    adata.var_names = pd.Index(inp["genes"].astype(str))
    if bool(inp["has_counts"]):  # sequencing-based: 2,000 HVGs, scanpy 'seurat' flavor on log data
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
        prep = "log-normalized; 2000 HVG (scanpy seurat flavor)"
    else:
        prep = "log-normalized; all 960 panel genes (imaging platform, no HVG step)"
    coords, offset = nc.offset_frames(inp["coords"], inp["frame"])
    adata.obs["x"], adata.obs["y"] = coords[:, 0], coords[:, 1]
    adata.obsm["spatial"] = coords
    started = time.perf_counter()
    banksy_dict = initialize_banksy(adata, ("x", "y", "spatial"), 15, nbr_weight_decay="scaled_gaussian",
                                    max_m=1, plt_edge_hist=False, plt_nbr_weights=False,
                                    plt_agf_angles=False, plt_theta=False)
    checks = {f"weights_m{m}": nc.edge_frame_check(banksy_dict["scaled_gaussian"]["weights"][m], frames)
              for m in (0, 1)}
    require_no_cross(checks)
    banksy_dict, _ = generate_banksy_matrix(adata, banksy_dict, [lam], max_m=1)
    pca_umap(banksy_dict, pca_dims=[20], plt_remaining_var=False, add_umap=False)
    embedding = np.asarray(banksy_dict["scaled_gaussian"][lam]["adata"].obsm["reduced_pc_20"])
    embed_s = time.perf_counter() - started
    native = {}
    for target in targets:
        k = int(inp[f"K_{target}"])
        t0 = time.perf_counter()
        try:
            partitioner = LeidenPartition(embedding, num_nn=50, nns_have_weights=True,
                                          compute_shared_nn=True, filter_shared_nn=True, shared_nn_max_rank=3,
                                          shared_nn_min_shared_nbrs=5, verbose=False)
            res, n_found, trace = banksy_bisect(partitioner, k, seed)
            results_df, _ = run_Leiden_partition(banksy_dict, [res], num_nn=50, num_iterations=-1,
                                                 partition_seed=seed, match_labels=False)
            label = results_df.iloc[0]["labels"]
            labels = np.asarray(label.dense, int)
            if int(label.num_labels) != n_found:
                raise RuntimeError(f"driver gave {label.num_labels} labels, bisection {n_found}")
            native[target] = {"labels": labels, "seconds": time.perf_counter() - t0,
                              "detail": {"algorithm": "BANKSY Leiden (run_Leiden_partition, num_nn 50, sNN, "
                                                      "RBConfiguration, n_iterations -1, partition_seed = seed)",
                                         "resolution": res, "target_K": k, "achieved_K": int(n_found),
                                         "bisection_trace": trace}}
        except Exception as exc:
            native[target] = {"error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0}
    extra = {"preprocessing": prep, "lambda": lam, "k_geom": 15, "max_m": 1, "nbr_weight_decay": "scaled_gaussian",
             "pca_dims": 20, "n_features": int(adata.n_vars)}
    return embedding, None, native, embed_s, checks, offset, extra


class CastleMemo:
    """Execution-only memoization for CASTLE's training loop.

    CASTLE.train() rebuilds the graph (construct_interaction_celltype) and the
    dense normalized adjacency (preprocess_adj) in every epoch, from inputs that
    never change during training, so every rebuild returns the same matrices.
    Here the first call computes, the second call recomputes and must match the
    first exactly (checked in every fit; a mismatch stops the fit), and later
    calls reuse the stored result. CASTLE's source files are not changed;
    castle_memo_check.py compares memoized and plain runs.
    """
    KEYS = ("graph_neigh", "adj", "graph_neigh_a", "adj_a")

    def __init__(self, module):
        self.module = module
        self.build0, self.prep0 = module.construct_interaction_celltype, module.preprocess_adj
        self.graph, self.kw, self.dense = None, None, {}
        self.stats = {"graph_calls": 0, "graph_computed": 0, "graph_verified": False,
                      "adjacency_calls": 0, "adjacency_computed": 0, "adjacency_verified": 0}

    @staticmethod
    def _same(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and (a != b).nnz == 0

    def build(self, adata, **kw):
        self.stats["graph_calls"] += 1
        if self.stats["graph_verified"]:
            if kw != self.kw:
                raise RuntimeError("CASTLE graph arguments changed during training")
            for key, value in self.graph.items():
                adata.obsm[key] = value
            return
        self.build0(adata, **kw)
        self.stats["graph_computed"] += 1
        fresh = {k: adata.obsm[k] for k in self.KEYS if k in adata.obsm}
        if self.graph is not None:
            if kw != self.kw or fresh.keys() != self.graph.keys() or \
                    not all(self._same(fresh[k], self.graph[k]) for k in fresh):
                raise RuntimeError("CASTLE rebuilt a different graph; memoization is not valid")
            self.stats["graph_verified"] = True
        self.graph, self.kw = fresh, kw

    def prep(self, adj):
        self.stats["adjacency_calls"] += 1
        hit = self.dense.get(id(adj))
        if hit is not None and hit[0] is adj:
            if not hit[2]:
                if not np.array_equal(self.prep0(adj), hit[1]):
                    raise RuntimeError("CASTLE preprocess_adj is not reproducible; memoization is not valid")
                self.stats["adjacency_computed"] += 1
                self.stats["adjacency_verified"] += 1
                self.dense[id(adj)] = (adj, hit[1], True)
            return hit[1]
        out = self.prep0(adj)
        self.stats["adjacency_computed"] += 1
        self.dense[id(adj)] = (adj, out, False)
        return out

    def __enter__(self):
        self.module.construct_interaction_celltype, self.module.preprocess_adj = self.build, self.prep
        return self

    def __exit__(self, *exc):
        self.module.construct_interaction_celltype, self.module.preprocess_adj = self.build0, self.prep0
        return False


def fit_castle(inp, seed, variant, targets, device, epochs=600, memo=True):
    import contextlib
    import torch
    if str(nc.CASTLE_PARENT) not in sys.path:
        sys.path.insert(0, str(nc.CASTLE_PARENT))
    import CASTLE.CASTLE as castle_module
    from CASTLE.CASTLE import CASTLE
    from CASTLE.utils import clustering
    frames = inp["frame"].astype(str)
    adata, offset, prep = graph_adata(inp)
    kwargs = {"device": torch.device(device), "random_seed": seed, "datatype": "cell"}
    if variant == "celltype":
        adata.obs["cell_type"] = pd.Categorical(inp["celltype_name"].astype(str))
        kwargs.update(mode="celltype", cell_type="cell_type", k_celltype=10, weight1=False, weight2=False)
    elif variant == "expression":  # mode and threshold fixed label-free by castle_fallback_density.py
        choice = json.loads((nc.E4 / "castle_fallback_choice.json").read_text())["chosen"]
        kwargs.update(mode=choice["mode"], sim_threshold=choice["threshold"])
    else:
        raise ValueError(variant)
    started = time.perf_counter()
    with (CastleMemo(castle_module) if memo else contextlib.nullcontext()) as memoizer:
        model = CASTLE(adata, epochs=epochs, **kwargs)
        keys = [k for k in ("adj", "graph_neigh", "adj_a", "graph_neigh_a") if k in model.adata.obsm]
        checks = {k: nc.edge_frame_check(model.adata.obsm[k], frames) for k in keys}
        require_no_cross(checks)
        degree = np.asarray((model.adata.obsm["adj"] != 0).sum(1)).ravel()
        out = model.train()
    embed_s = time.perf_counter() - started
    if memo and not (memoizer.stats["graph_verified"] and memoizer.stats["adjacency_verified"] >= 1):
        raise RuntimeError(f"CASTLE memoization was not verified: {memoizer.stats}")
    post = {f"{k}_after_training": nc.edge_frame_check(out.obsm[k], frames) for k in keys}
    require_no_cross(post)
    checks.update(post)
    embedding = np.asarray(out.obsm["emb"], np.float32)
    native, emb_pca = {}, None
    for target in targets:
        k = int(inp[f"K_{target}"])
        t0 = time.perf_counter()
        try:
            clustering(out, n_clusters=k, radius=50, key="emb", method="mclust", refinement=False)
            labels = np.asarray(out.obs["domain"]).astype(int)
            emb_pca = np.asarray(out.obsm["emb_pca"], np.float32)
            native[target] = {"labels": labels, "seconds": time.perf_counter() - t0,
                              "detail": {"algorithm": "CASTLE.utils.clustering mclust (R Mclust, G = K, model by BIC, "
                                                      "seed 2020) on PCA(20, random_state=42) of emb; refinement=False"}}
        except Exception as exc:
            native[target] = {"error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - t0}
    extra = {"preprocessing": prep, "constructor": {k: (str(v) if k == "device" else v) for k, v in kwargs.items()},
             "epochs": model.epochs, "n_features": int(model.dim_input),
             "adj_degree": {"mean": float(degree.mean()), "isolated_fraction": float((degree == 0).mean())},
             "memoization": memoizer.stats if memo else None}
    return embedding, emb_pca, native, embed_s, checks, offset, extra


FIT = {"GraphST": fit_graphst, "BANKSY": fit_banksy, "CASTLE": fit_castle}


def run_one(method, cohort, patient, variant, targets, seed, device, force):
    tag = f"{method}-{variant}-{patient}-seed{seed}"
    record_path = FITS / f"{tag}.json"
    if record_path.exists() and not force:
        print("skip", tag, flush=True)
        return
    inp = nc.load_input(cohort, patient)
    record = {"method": method, "variant": variant, "cohort": cohort, "patient": patient, "seed": seed,
              "targets": targets, "input": inp["path"], "input_sha256": inp["sha256"],
              "script_sha256": nc.sha(Path(__file__).resolve()),
              "common_sha256": nc.sha(Path(nc.__file__).resolve()), "device": device}
    import torch
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        embedding, emb_pca, native, embed_s, checks, offset, extra = FIT[method](inp, seed, variant, targets, device)
        if not np.isfinite(embedding).all():
            raise ValueError("non-finite embedding")
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        FITS.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print("FAILED", tag, record["error"], flush=True)
        return
    rows, arrays = [], {"embedding": embedding, "obs_id": inp["obs_id"], "targets": np.array(targets)}
    if emb_pca is not None:
        arrays["emb_pca"] = emb_pca
    for target in targets:
        y, k = inp[f"y_{target}"], int(inp[f"K_{target}"])
        t0 = time.perf_counter()
        pred = nc.kmeans_matched(embedding, k, seed)
        km_s = time.perf_counter() - t0
        arrays[f"kmeans_{target}"] = pred
        rows.append({"clustering": "kmeans_matched", "target": target, "K_ref": k, **nc.score(y, pred),
                     "runtime_s": embed_s + km_s, "cluster_s": km_s})
        nat = native[target]
        if "labels" in nat:
            arrays[f"native_{target}"] = nat["labels"]
            rows.append({"clustering": "native", "target": target, "K_ref": k, **nc.score(y, nat["labels"]),
                         "runtime_s": embed_s + nat["seconds"], "cluster_s": nat["seconds"]})
    EMB.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB / f"{tag}.npz", **arrays)
    record.update(status="ok", embed_seconds=embed_s, embedding_shape=list(embedding.shape),
                  graph_checks=checks, frame_offset=offset, settings=extra, rows=rows,
                  native={t: {k: v for k, v in n.items() if k != "labels"} for t, n in native.items()},
                  embedding_file=str(EMB / f"{tag}.npz"), embedding_sha256=nc.sha(EMB / f"{tag}.npz"),
                  peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else None)
    FITS.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, default=float) + "\n")
    summary = ", ".join(f"{r['target']}/{r['clustering']} ARI {r['ari']:.3f} K {r['K_achieved']}" for r in rows)
    print("done", tag, f"embed {embed_s:.1f}s", summary, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(FIT))
    ap.add_argument("--cohort", required=True, choices=sorted(PATIENTS))
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=nc.SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.method == "BANKSY" else args.device
    for variant, targets in PLAN[args.method][args.cohort]:
        if args.variants and variant not in args.variants:
            continue
        for patient in args.patients or PATIENTS[args.cohort]:
            for seed in args.seeds:
                run_one(args.method, args.cohort, patient, variant, targets, seed, device, args.force)


if __name__ == "__main__":
    main()
