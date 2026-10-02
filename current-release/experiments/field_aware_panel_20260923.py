#!/usr/bin/env python3
"""Frozen field-identified tissue reanalysis; no historical performance reuse.

The retained set is chosen on documented coordinate identity, before outcome
calculation. This is a seven-object audit panel, not the historical n=11/12/14
panel. No methods, seeds or dataset rows are silently filled or dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from _roots import data_root, external_root
from field_aware_adapters import directed_graph, neighbor_mean, external_embedding
from tessera_st.eval.refine import refine_labels
from tessera_st.config import TrainConfig, AblationConfig
from tessera_st.train import fit_predict

PROTOCOL = "field-identified-fixed-backend-20260923-v1"
OUT = ROOT / "experiments/generated" / PROTOCOL
SEEDS = [1, 2, 3]
METHODS = ["expression-only", "neighbor-mean", "BANKSY-style", "SpatialLeiden",
           "Tessera", "STAGATE", "SEDR"]
UNAVAILABLE = ["SpaceFlow", "GraphST", "SpaGCN"]
# Only recorded frame identity, or a single registered Visium spatial library,
# qualifies. No dataset is selected by label contiguity or model performance.
INPUTS = {
    "DLPFC": ("raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad", "Region", "uns:spatial", "layer"),
    "MERFISH": ("baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad", "domain", "slice_id", "domain"),
    "STARmap": ("processed/starmap_mouse_vcortex_wang2018/anndata.h5ad", "layer_label", "section_id", "region"),
    "MIBI": ("raw/squidpy/mibitof.h5ad", "Cluster", "library_id", "cell cluster"),
    "Open-ST": ("processed/openst_hnscc_sub15k/anndata.h5ad", "ground_truth", "section_id", "annotation"),
    "CODEX": ("processed/codex_spleen_goltsev2018/anndata.h5ad", "niche", "section_id", "cluster-derived niche proxy"),
    "Zhuang": ("processed/niche_abc_zhuang1_brain_slice/anndata.h5ad", "domain", "section_id", "atlas anatomy"),
}
EXCLUDED = {x: "Frame identity not established in this revision; no outcome-based exclusion"
            for x in ["seqFISH", "osmFISH", "BRCA", "IMC", "Slide-seqV2", "st_COAD", "st_LIHC", "st_OV"]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def valid_labels(raw):
    return ~np.isin(np.char.lower(np.char.strip(np.asarray(raw, str))),
                    ["", "nan", "none", "na", "unknown", "<na>"])


def fields_from_data(a, key):
    if key == "uns:spatial":
        keys = list(a.uns.get("spatial", {}))
        if len(keys) != 1:
            raise ValueError("one registered spatial library is required")
        fields = np.repeat(str(keys[0]), a.n_obs)
    else:
        fields = a.obs[key].astype(str).to_numpy()
    if not valid_labels(fields).all():
        raise ValueError("missing coordinate-frame identity")
    return fields


def load_data(name):
    relative, label_key, field_key, annotation = INPUTS[name]
    path = data_root()/relative
    a = ad.read_h5ad(path, backed="r")
    try:
        n_total = a.n_obs
        keep = np.arange(n_total)
        if n_total > 16000:
            keep = np.sort(np.random.RandomState(0).choice(n_total, 16000, replace=False))
        fields = fields_from_data(a, field_key)[keep]
        raw = a.obs[label_key].astype(str).to_numpy()[keep]
        names = np.asarray(a.obs_names, str)[keep]
        xy = np.asarray(a.obsm["spatial"], float)[keep]
        X = a.X[keep]
        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, float)
        features = np.asarray(a.var_names, str)
    finally:
        a.file.close()
    if len(set(names)) != len(names):
        raise ValueError("source observation identifiers are not unique")
    if not np.isfinite(X).all() or not np.isfinite(xy).all():
        raise ValueError("nonfinite input")
    counts = bool(np.allclose(X, np.round(X)) and X.min() >= 0)
    if counts:
        X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    if X.shape[1] > 500:
        gene = np.argsort(-X.var(0), kind="stable")[:3000]
        X, features = X[:, gene], features[gene]
    sd = X.std(0)
    sd[sd == 0] = 1
    Z = np.clip((X-X.mean(0))/sd, -10, 10).astype(np.float32)
    valid = valid_labels(raw)
    classes = {x: i for i, x in enumerate(sorted(set(raw[valid])))}
    truth = np.array([classes[x] if v else -1 for x, v in zip(raw, valid)])
    meta = dict(dataset=name, path=str(path), source_sha256=sha(path), n_total=n_total,
        n_cells=len(Z), n_labelled=int(valid.sum()), n_classes=len(classes), n_features=Z.shape[1],
        n_fields=len(set(fields)), label_key=label_key, field_key=field_key, annotation=annotation,
        count_normalization_applied=counts, classes=classes,
        selection="all rows or RandomState(0) 16000 without replacement, sorted; before label filtering",
        selected_index_sha256=hashlib.sha256(keep.tobytes()).hexdigest())
    return Z, xy, fields, truth, keep, names, features, meta


def contiguity(xy, fields, truth):
    take = truth >= 0
    xy, fields, y = xy[take], fields[take], truth[take]
    src, dst, _, rows = directed_graph(xy, fields)
    active = np.array([len(r) > 0 for r in rows])
    if not active.any():
        raise ValueError("no labelled spatial neighbours")
    fractions = np.array([np.mean(y[r] == y[i]) if len(r) else np.nan
                          for i, r in enumerate(rows)])
    expected = np.zeros(len(y))
    for f in np.unique(fields):
        ix = np.flatnonzero(fields == f)
        if len(ix) > 1:
            counts = np.bincount(y[ix])
            expected[ix] = np.sum(counts*(counts-1))/(len(ix)*(len(ix)-1))
    chance = float(expected[active].mean())
    raw = float(fractions[active].mean())
    pooled_src, pooled_dst, _, _ = directed_graph(xy, np.repeat("pooled", len(y)))
    return dict(contiguity_cell_weighted=raw, contiguity_edge_weighted=float(np.mean(y[src] == y[dst])),
        within_field_permutation_expectation=chance,
        chance_adjusted=(raw-chance)/(1-chance) if chance < 1 else None,
        labelled_directed_edges=len(src), labelled_cross_field_edges=0,
        labelled_singletons_excluded=int((~active).sum()),
        pooled_contiguity=float(np.mean(y[pooled_src] == y[pooled_dst])),
        pooled_cross_field_edges=int(np.sum(fields[pooled_src] != fields[pooled_dst])),
        pooled_directed_edges=len(pooled_src))


def pca(Z, n, seed):
    return PCA(min(n, Z.shape[1], len(Z)-1), random_state=seed).fit_transform(Z)


def expression_only(Z, k, seed):
    # Intentionally no coordinates/fields parameter; true labels supply K only.
    return KMeans(k, n_init=10, random_state=seed).fit_predict(pca(Z, 30, seed))


def spatial_leiden(Z, xy, fields, k, seed):
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import NearestNeighbors
    nbr = neighbor_mean(Z, xy, fields)
    emb = pca(np.c_[np.sqrt(.5)*Z, np.sqrt(.5)*nbr], 30, seed)
    idx = NearestNeighbors(n_neighbors=min(16, len(emb))).fit(emb).kneighbors(emb, return_distance=False)
    edges = [(i, int(j)) for i, row in enumerate(idx) for j in row if j != i]
    graph = ig.Graph(n=len(emb), edges=edges)
    lo, hi = .05, 4.
    for _ in range(12):
        r = (lo+hi)/2
        labels = np.asarray(leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                            resolution_parameter=r, seed=seed).membership)
        nc = len(set(labels))
        if nc == k:
            break
        lo, hi = (r, hi) if nc < k else (lo, r)
    return labels


def fit_method(method, Z, xy, fields, k, seed, device):
    if method == "expression-only":
        return expression_only(Z, k, seed)
    if method == "SpatialLeiden":
        return spatial_leiden(Z, xy, fields, k, seed)
    if method in {"neighbor-mean", "BANKSY-style"}:
        nbr = neighbor_mean(Z, xy, fields)
        X = nbr if method == "neighbor-mean" else np.c_[np.sqrt(.2)*Z, np.sqrt(.8)*nbr]
        embed = pca(X, 30 if method == "neighbor-mean" else 20, seed)
    elif method == "Tessera":
        embed = fit_predict(pca(Z, min(50, k*6), seed).astype(np.float32), xy, k,
            AblationConfig(), TrainConfig(epochs=120, seed=seed, device=device),
            field_ids=fields, refine=False).embed
    else:
        embed = external_embedding(method, Z, xy, fields, seed, device=device)
    if not np.isfinite(embed).all():
        raise ValueError("nonfinite embedding")
    return KMeans(k, n_init=10, random_state=seed).fit_predict(embed)


def score(y, labels):
    mask = y >= 0
    return float(adjusted_rand_score(y[mask], labels[mask]))


def protocol_record():
    source_files = [Path(__file__), ROOT/"experiments/field_aware_adapters.py"]
    source_files += sorted((ROOT/"src/tessera_st").rglob("*.py"))
    source_files += sorted((external_root()/"STAGATE/STAGATE_pyG").glob("*.py"))
    source_files += sorted((external_root()/"SEDR/SEDR").glob("*.py"))
    return dict(protocol_id=PROTOCOL, seeds=SEEDS, methods=METHODS,
        retained_inputs=INPUTS, excluded_inputs=EXCLUDED, unavailable_methods=UNAVAILABLE,
        primary="named method raw KMeans ARI minus raw expression-only PCA30 KMeans ARI; no backend oracle",
        secondary="fixed two-round within-field refinement on spatial methods only; no winner selection",
        graph="k=6 within field; self excluded by row ID; duplicates retained; min(6,n_field-1); singleton no neighbours",
        preprocessing="count heuristic log1p CPM1e4; top variance max3000 if p>500; global z clip[-10,10]; recorded per input",
        epochs={"Tessera":120,"STAGATE":600,"SEDR":200},
        sedr_mask="positive symmetric adjacency plus self; fixed seeded within-field non-neighbours, at most positives per row; within-field pair normalization",
        code_sha256={str(p):sha(p) for p in source_files},
        packages={x:importlib.metadata.version(x) for x in ["numpy","scipy","scikit-learn","torch","torch-geometric","anndata"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(INPUTS), choices=list(INPUTS))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = protocol_record()
    lock = OUT/"protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != json.loads(json.dumps(protocol)):
        raise SystemExit("Implementation/protocol changed; new identity required, no silent cache reuse")
    write_json(lock, protocol)
    for name in args.datasets:
        Z, xy, fields, y, keep, names, features, meta = load_data(name)
        path = OUT/name
        path.mkdir(exist_ok=True)
        if (path/"input.json").exists() and json.loads((path/"input.json").read_text())["source_sha256"] != meta["source_sha256"]:
            raise RuntimeError("source changed")
        src, dst, _, _ = directed_graph(xy, fields)
        meta.update(contiguity(xy, fields, y))
        meta.update(graph_directed_edges=len(src), graph_cross_field_edges=0)
        write_json(path/"input.json", meta)
        if not (path/"rows.npz").exists():
            np.savez_compressed(path/"rows.npz", original_index=keep, obs_id=names,
                coordinates=xy, field=fields, reference_label=y, feature_id=features,
                spatial_src=src, spatial_dst=dst)
        for seed in SEEDS:
            for method in args.methods:
                file = path/f"{method}-seed{seed}.json"
                if file.exists():
                    continue
                started = time.monotonic()
                record = dict(protocol_id=PROTOCOL, dataset=name, method=method, seed=seed,
                    source_sha256=meta["source_sha256"], protocol_sha256=sha(lock),
                    n_cells=len(Z), n_classes=meta["n_classes"], device=args.device)
                try:
                    labels = fit_method(method,Z,xy,fields,meta["n_classes"],seed,args.device)
                    if labels.shape != y.shape:
                        raise ValueError("prediction row count changed")
                    record.update(status="complete", raw_ari=score(y, labels),
                        fitted_n_clusters=len(set(labels)), spatial_refinement=method != "expression-only")
                    arrays = dict(raw=labels)
                    if method != "expression-only":
                        refined = refine_labels(labels,xy,field_ids=fields)
                        record["refined_ari"] = score(y,refined)
                        arrays["refined"] = refined
                    if method == "expression-only":
                        # Clearly separated counterfactual: audit the historical leakage.
                        leaked = refine_labels(labels,xy,field_ids=fields)
                        record["diagnostic_only_refined_control_ari"] = score(y,leaked)
                        arrays["diagnostic_only_refined_control"] = leaked
                    if method == "neighbor-mean" and len(set(fields)) > 1:
                        wrong = neighbor_mean(Z,xy,np.repeat("pooled",len(xy)))
                        pooled = KMeans(meta["n_classes"], n_init=10, random_state=seed).fit_predict(pca(wrong,30,seed))
                        record["diagnostic_only_pooled_coordinate_ari"] = score(y,pooled)
                        arrays["diagnostic_only_pooled_coordinates"] = pooled
                    pred_path = path/f"{method}-seed{seed}.npz"
                    np.savez_compressed(pred_path, **arrays)
                    record["predictions_sha256"] = sha(pred_path)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                record["seconds"] = time.monotonic()-started
                write_json(file, record)
                print(name,method,seed,record["status"],record.get("raw_ari"),flush=True)
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        for method in UNAVAILABLE:
            write_json(path/f"{method}-availability.json",dict(method=method,status="not_run",
                reason="Full field-aware spatial distance/loss/refinement path not validated; no score supplied",
                seeds_not_run=SEEDS))


if __name__ == "__main__":
    main()
