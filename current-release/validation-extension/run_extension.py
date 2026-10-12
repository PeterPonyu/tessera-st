#!/usr/bin/env python3
"""Extension analyses locked in PROTOCOL_EXTENSION.md.

CPU only. The frozen estimator, graph rule and preprocessing are imported and
never modified. Modes: hardening (HER2ST), dev (13 manuscript objects), breadth
(three further tissues).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REV = HERE.parent
sys.path[:0] = [str(REV / "validation-external"), str(REV / "source" / "scripts"), str(REV / "source" / "src")]

import run_enhanced_targets as et  # noqa: E402
import run_experiments as rx  # noqa: E402
import run_followup as fu  # noqa: E402
import run_her2st_validation as hv  # noqa: E402
from common import OBJECTS, RESULTS, encode, read, sha  # noqa: E402

OUT = HERE / "results"
PROTOCOL = HERE / "PROTOCOL_EXTENSION.md"
SEEDS = fu.SEEDS
CONFIGS = {f"{kind}|{k}|{lam}": (kind, k, lam) for kind, k, lam in fu.configs()}
LAB = Path("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/data")
BREADTH = {
    "DLPFC_151673": (LAB / "raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad", "Region"),
    "STARmap_visual_cortex": (LAB / "processed/starmap_mouse_vcortex_wang2018/anndata.h5ad", "Region"),
    "MERFISH_hypothalamus": (LAB / "baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad", "domain"),
}
OTHER_T1 = {"adipose tissue", "connective tissue", "immune infiltrate", "breast glands"}
BOUNDARY = {
    "T1": ({"invasive cancer", "cancer in situ"}, OTHER_T1),
    "T2": ({"invasive cancer"}, OTHER_T1),
    "T3": ({"invasive cancer", "cancer in situ"}, {"adipose tissue", "connective tissue", "immune infiltrate"}),
}
MIN_SIDE = 20
ARMS_ORDER = ["ExpressionWeighted", "UniformSameGraph", "WeightShuffled", "ShuffledCoords"]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def boundary_labels(raw, variant):
    tumor, other = BOUNDARY[variant]
    out = []
    for value in raw:
        label = et.normalize(value)
        if label not in et.ALLOWED:
            raise ValueError(f"unexpected pathologist label {value!r}")
        out.append("tumor" if label in tumor else "other tissue" if label in other else "")
    return np.asarray(out)


def weighted_features(x, coords, fields, mode, seed=0):
    """E6 / U6 / S6 / C6 representation. See PROTOCOL_EXTENSION.md."""
    xy = np.asarray(coords, float)
    if mode == "shuffled_coords":
        rng = np.random.RandomState(2000 + seed)
        xy = xy.copy()
        for frame in np.unique(fields):
            ix = np.flatnonzero(fields == frame)
            xy[ix] = xy[rng.permutation(ix)]
    edge, _ = rx.graph(xy, fields, "Tessera", 1)
    distance = np.linalg.norm(x[edge[0]] - x[edge[1]], axis=1)
    weight = np.exp(-distance / (np.median(distance) + 1e-8)).astype(np.float32)
    if mode == "uniform":
        weight = np.ones_like(weight)
    elif mode == "weight_shuffled":
        weight = np.random.RandomState(3000 + seed).permutation(weight)
    total = np.zeros(x.shape, np.float32)
    np.add.at(total, edge[1], weight[:, None] * x[edge[0]])
    mass = np.zeros(len(x), np.float32)
    np.add.at(mass, edge[1], weight)
    return total / np.maximum(mass[:, None], 1e-8), edge


def bootstrap_interval(values, draws=10000, seed=0):
    values = np.asarray(values, float)
    index = np.random.RandomState(seed).randint(0, len(values), (draws, len(values)))
    means = values[index].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def contrast(wide, a, b, interval=True):
    delta = (wide[a] - wide[b]).to_numpy(float)
    out = {
        "a": a, "b": b, "units": list(wide.index), "deltas": [float(v) for v in delta],
        "mean": float(delta.mean()), "n_positive": int((delta > 0).sum()),
        "n_negative": int((delta < 0).sum()), "n_zero": int((delta == 0).sum()),
        "exact_sign_flip_p": fu.exact_sign_flip(delta),
    }
    if interval:
        out["bootstrap95"] = bootstrap_interval(delta)
    return out


def select_config(grid, units, held_out):
    means = grid.groupby(["unit", "config"]).ari.mean().unstack()
    candidates = [c for c in means.columns if not str(c).startswith("PCA50")]
    others = [u for u in units if u != held_out]
    return str(means.loc[others, candidates].mean().idxmax())


def evaluate_unit(unit, x, coords, fields, targets, tessera):
    """Control grid, weighting arms, optional Tessera fits and seam scores for one unit."""
    cache = {}
    feats = {"ExpressionWeighted": None, "UniformSameGraph": None}
    feats["ExpressionWeighted"], edge = weighted_features(x, coords, fields, "expression")
    feats["UniformSameGraph"], _ = weighted_features(x, coords, fields, "uniform")
    for seed in SEEDS:
        feats[("WeightShuffled", seed)], _ = weighted_features(x, coords, fields, "weight_shuffled", seed)
        feats[("ShuffledCoords", seed)], _ = weighted_features(x, coords, fields, "shuffled_coords", seed)
    grid, arms, seams = [], [], []
    fits = {}
    if tessera:
        for k in sorted({len(classes) for _, classes in targets.values()}):
            for seed in SEEDS:
                started = time.perf_counter()
                _, emb, _, gate, fit_edge, _, _, elapsed, _, _ = rx.train_net(x, coords, fields, k, "Tessera", seed, "cpu")
                if gate.shape[0] != fit_edge.shape[1] or not np.array_equal(fit_edge, edge):
                    raise ValueError(f"{unit}: Tessera graph differs from the canonical graph")
                fits[(k, seed)] = (emb, gate)
                print("tessera", unit, "K", k, "seed", seed, round(elapsed, 1), "s", flush=True)
    for target, (y, classes) in targets.items():
        k = len(classes)
        for kind, kk, lam in fu.configs():
            for seed in SEEDS:
                feature = fu.features(kind, kk, lam, x, coords, fields, cache, seed)
                ari, _ = fu.kmeans_ari(feature, y, k, seed)
                grid.append({"unit": unit, "target": target, "kind": kind, "k": kk, "lam": lam,
                             "config": f"{kind}|{kk}|{lam}", "seed": seed, "ari": ari})
        for seed in SEEDS:
            for arm in ARMS_ORDER:
                feature = feats[arm] if arm in feats else feats[(arm, seed)]
                ari, _ = fu.kmeans_ari(feature, y, k, seed)
                arms.append({"unit": unit, "target": target, "arm": arm, "seed": seed, "ari": ari})
            if tessera:
                emb, _ = fits[(k, seed)]
                ari, _ = fu.kmeans_ari(emb, y, k, seed)
                arms.append({"unit": unit, "target": target, "arm": "Tessera", "seed": seed, "ari": ari})
        valid = (y[edge[0]] >= 0) & (y[edge[1]] >= 0)
        seam = y[edge[0]][valid] != y[edge[1]][valid]
        if tessera and len(np.unique(seam)) == 2:
            raw = np.linalg.norm(x[edge[0]] - x[edge[1]], axis=1)[valid]
            smooth = np.linalg.norm(feats["ExpressionWeighted"][edge[0]] - feats["ExpressionWeighted"][edge[1]], axis=1)[valid]
            plain = np.linalg.norm(feats["UniformSameGraph"][edge[0]] - feats["UniformSameGraph"][edge[1]], axis=1)[valid]
            for seed in SEEDS:
                _, gate = fits[(k, seed)]
                seams.append({
                    "unit": unit, "target": target, "seed": seed, "n_edges": int(valid.sum()),
                    "seam_fraction": float(seam.mean()),
                    "gate": float(roc_auc_score(seam, 1 - gate[valid])),
                    "raw_difference": float(roc_auc_score(seam, raw)),
                    "smoothed_expression_weighted": float(roc_auc_score(seam, smooth)),
                    "smoothed_uniform": float(roc_auc_score(seam, plain)),
                })
    return {"grid": grid, "arms": arms, "seams": seams, "fits": fits, "feats": feats, "cache": cache, "edge": edge}


def wide_means(arms, grid, selected, target):
    sub = arms[arms.target == target].groupby(["unit", "arm"]).ari.mean().unstack()
    g = grid[grid.target == target]
    sub["PCA50"] = g[g.kind == "PCA50"].groupby("unit").ari.mean()
    sub["NestedControl"] = pd.Series({
        u: float(g[(g.unit == u) & (g.config == cfg)].ari.mean()) for u, cfg in selected.items()})
    return sub


def g_join_check():
    fetch = hv.load_module("fetch_gate", hv.GATE_PATH).fetch
    dest = hv.RAW / "G1.tsv.gz"
    if not dest.exists():
        record = fetch(f"{hv.BASE}/data/ST-cnts/G1.tsv.gz", dest, yes_large=False, timeout=120)
        if not record.get("ok"):
            return {"status": "fetch_failed", "detail": jsonable({k: v for k, v in record.items() if k != "head"})}
    labels = pd.read_csv(hv.RAW / "G2_labeled_coordinates.tsv", sep="\t", index_col=0)
    coordinates = labels[["x", "y", "pixel_x", "pixel_y"]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(coordinates.to_numpy(float)).all(1)
    keys = [f"{int(round(a))}x{int(round(b))}" for a, b in zip(labels.x[finite], labels.y[finite])]
    result = {"status": "ok", "n_label_keys": len(keys)}
    ids = {}
    for section in ("G1", "G2"):
        frame = pd.read_csv(hv.RAW / f"{section}.tsv.gz", sep="\t", index_col=0, usecols=[0])
        ids[section] = set(map(str, frame.index))
        hit = sum(key in ids[section] for key in keys)
        result[section] = {"n_count_spots": len(ids[section]), "label_keys_found": hit, "fraction": hit / len(keys)}
    union = len(ids["G1"] | ids["G2"])
    result["count_spot_overlap_jaccard"] = len(ids["G1"] & ids["G2"]) / union
    return result


def reproducibility(arms, grid, selected_by_target):
    earlier_all = pd.read_csv(REV / "validation-external/results/primary_patients.csv").set_index("patient")
    earlier_enh = pd.read_csv(REV / "validation-external/results/enhanced/primary.csv")
    check = {}
    pairs = [("all", earlier_all, "tessera_ari", "Tessera"), ("all", earlier_all, "nested_control_ari", "NestedControl")]
    for target, key in (("compartments", "compartments"), ("boundary_T1", "boundary")):
        frame = earlier_enh[earlier_enh.target == key].set_index("patient")
        pairs += [(target, frame, "tessera_ari", "Tessera"), (target, frame, "expression_gate_ari", "ExpressionWeighted"),
                  (target, frame, "nested_control_ari", "NestedControl")]
    for target, frame, column, arm in pairs:
        wide = wide_means(arms, grid, selected_by_target[target], target)
        wide.index = [str(u)[0] for u in wide.index]
        shared = [u for u in frame.index if u in wide.index]
        diff = float(np.max(np.abs(wide.loc[shared, arm].to_numpy(float) - frame.loc[shared, column].to_numpy(float))))
        check[f"{target}:{arm}"] = {"max_abs_difference": diff, "n_units": len(shared)}
    selection_match = {}
    for target, frame, name in (("all", earlier_all, "all"),):
        for patient, row in frame.iterrows():
            selection_match[f"{target}:{patient}"] = bool(
                selected_by_target[target].get(next(u for u in selected_by_target[target] if u.startswith(patient))) == row.selected_control)
    for target, key in (("compartments", "compartments"), ("boundary_T1", "boundary")):
        frame = earlier_enh[earlier_enh.target == key].set_index("patient")
        for patient, row in frame.iterrows():
            unit = next(u for u in selected_by_target[target] if u.startswith(patient))
            selection_match[f"{target}:{patient}"] = bool(selected_by_target[target][unit] == row.selected_control)
    check["selected_configuration_matches_earlier"] = selection_match
    return check


def write_partitions(store, selected_by_target, out_dir):
    for target in ("all", "compartments", "boundary_T1"):
        rows = []
        for unit, record in store.items():
            y, classes = record["targets"][target]
            k = len(classes)
            cfg = selected_by_target[target][unit]
            kind, kk, lam = CONFIGS[cfg]
            control = fu.features(kind, kk, lam, record["x"], record["coords"], record["fields"], record["cache"], 1)
            parts = {
                "tessera": fu.kmeans_ari(record["fits"][(k, 1)][0], y, k, 1)[1],
                "control": fu.kmeans_ari(control, y, k, 1)[1],
                "expression_weighted": fu.kmeans_ari(record["feats"]["ExpressionWeighted"], y, k, 1)[1],
                "uniform_same_graph": fu.kmeans_ari(record["feats"]["UniformSameGraph"], y, k, 1)[1],
                "pca50": fu.kmeans_ari(record["x"], y, k, 1)[1],
            }
            names = np.asarray(classes)
            frame = pd.DataFrame({"unit": unit, "target": target, "x": record["coords"][:, 0], "y": record["coords"][:, 1],
                                  "reference": np.where(y >= 0, names[np.clip(y, 0, None)], ""), **parts})
            frame["control_config"] = cfg
            rows.append(frame)
        pd.concat(rows, ignore_index=True).to_csv(out_dir / f"partitions_{target}.csv", index=False)


def run_hardening():
    out_dir = OUT / "hardening"
    out_dir.mkdir(parents=True, exist_ok=True)
    store, exclusions = {}, []
    for stem in et.STEMS:
        base, matrix, genes, coords, raw = hv.load_joined(stem)
        pcs, *_ = hv.preprocess(matrix, genes)
        fields = np.full(len(raw), stem)
        if not np.array_equal(boundary_labels(raw, "T1"), et.recode(raw, "boundary")):
            raise ValueError("boundary_T1 differs from the locked enhanced target")
        targets = {"all": encode(raw), "compartments": encode(et.recode(raw, "compartments"))}
        for variant in BOUNDARY:
            y, classes = encode(boundary_labels(raw, variant))
            counts = np.bincount(y[y >= 0], minlength=2) if len(classes) == 2 else np.array([0])
            if len(classes) < 2 or counts.min() < MIN_SIDE:
                exclusions.append({"unit": stem, "target": f"boundary_{variant}", "counts": counts.tolist()})
                continue
            targets[f"boundary_{variant}"] = (y, classes)
        store[stem] = {"base": base, "x": pcs, "coords": coords, "fields": fields, "targets": targets}
    patients = {u: u[0] for u in store}
    for unit, record in store.items():
        result = evaluate_unit(unit, record["x"], record["coords"], record["fields"], record["targets"], True)
        record.update(result)
        print("hardening", unit, flush=True)
    grid = pd.DataFrame([r for rec in store.values() for r in rec["grid"]])
    arms = pd.DataFrame([r for rec in store.values() for r in rec["arms"]])
    seams = pd.DataFrame([r for rec in store.values() for r in rec["seams"]])
    selected = {}
    summary_targets = {}
    for target in sorted(grid.target.unique()):
        units = sorted(grid[grid.target == target].unit.unique())
        selected[target] = {u: select_config(grid[grid.target == target], units, u) for u in units}
        wide = wide_means(arms, grid, selected[target], target).loc[units]
        contrasts = {
            f"{a}_minus_{b}": contrast(wide, a, b)
            for a, b in [("ExpressionWeighted", "UniformSameGraph"), ("ExpressionWeighted", "WeightShuffled"),
                         ("ExpressionWeighted", "ShuffledCoords"), ("ExpressionWeighted", "NestedControl"),
                         ("UniformSameGraph", "NestedControl"), ("Tessera", "NestedControl"),
                         ("Tessera", "ExpressionWeighted"), ("Tessera", "UniformSameGraph"),
                         ("Tessera", "PCA50"), ("ExpressionWeighted", "PCA50")]
        }
        summary_targets[target] = {
            "units": units,
            "per_unit": {u: {col: float(wide.loc[u, col]) for col in wide.columns} for u in units},
            "selected_control": selected[target],
            "contrasts": contrasts,
        }
    t1 = summary_targets["boundary_T1"]["contrasts"]
    h1 = {
        "e6_over_u6": t1["ExpressionWeighted_minus_UniformSameGraph"]["n_positive"],
        "e6_over_s6": t1["ExpressionWeighted_minus_WeightShuffled"]["n_positive"],
        "e6_over_c6": t1["ExpressionWeighted_minus_ShuffledCoords"]["n_positive"],
        "rule": "each count >= 6 of 8",
    }
    h1["holds"] = bool(min(h1["e6_over_u6"], h1["e6_over_s6"], h1["e6_over_c6"]) >= 6)
    seam_summary = {}
    for target in ("compartments", "boundary_T1"):
        sub = seams[seams.target == target].groupby("unit")[
            ["gate", "raw_difference", "smoothed_expression_weighted", "smoothed_uniform"]].mean()
        entry = {"per_unit": {u: row.to_dict() for u, row in sub.iterrows()}, "mean": {}, "differences": {}}
        for col in sub.columns:
            entry["mean"][col] = {"mean": float(sub[col].mean()), "bootstrap95": bootstrap_interval(sub[col].to_numpy())}
        for a, b in [("smoothed_expression_weighted", "gate"), ("raw_difference", "gate"),
                     ("smoothed_expression_weighted", "raw_difference"), ("smoothed_expression_weighted", "smoothed_uniform")]:
            entry["differences"][f"{a}_minus_{b}"] = contrast(sub, a, b)
        entry["gate_exceeds_raw"] = int((sub.gate > sub.raw_difference).sum())
        entry["gate_above_chance"] = int((sub.gate > 0.5).sum())
        seam_summary[target] = entry
    h2 = {"gate_exceeds_raw_difference": seam_summary["boundary_T1"]["gate_exceeds_raw"],
          "rule": "holds when the gate exceeds the raw difference in at most 3 of 8 patients"}
    h2["holds"] = bool(h2["gate_exceeds_raw_difference"] <= 3)
    grid.to_csv(out_dir / "grid.csv", index=False)
    arms.to_csv(out_dir / "arms.csv", index=False)
    seams.to_csv(out_dir / "seams.csv", index=False)
    write_partitions(store, selected, out_dir)
    return {
        "targets": summary_targets, "seams": seam_summary, "H1_weighting": h1, "H2_gate": h2,
        "exclusions": exclusions, "patient_of_unit": patients,
        "reproducibility_against_earlier_runs": reproducibility(arms, grid, selected),
        "patient_G_section_identity": g_join_check(),
        "included": {u: rec["base"] for u, rec in store.items()},
    }


def run_dev():
    out_dir = OUT / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(REV / "source/manuscript/ledgers/all_primary_scores.csv")
    means = primary.groupby(["object", "arm"]).ari.mean().unstack()
    nested_summary = read(RESULTS / "followup-20261010/scale_controls_summary.json")
    nested = {row["patient"]: row["nested_control_ari"] for c in ("NSCLC", "BCC") for row in nested_summary[c]["per_patient"]}
    rows, check = [], {}
    for name in OBJECTS:
        d, meta = fu.load(name)
        x, coords, fields, y, k = d["x"], d["coords"], d["field"], d["ref"], meta["n_classes"]
        uniform, _ = weighted_features(x, coords, fields, "uniform")
        expression, _ = weighted_features(x, coords, fields, "expression")
        for seed in SEEDS:
            rows.append({"object": name, "arm": "ExpressionWeighted", "seed": seed, "ari": fu.kmeans_ari(expression, y, k, seed)[0]})
            rows.append({"object": name, "arm": "WeightShuffled", "seed": seed,
                         "ari": fu.kmeans_ari(weighted_features(x, coords, fields, "weight_shuffled", seed)[0], y, k, seed)[0]})
            rows.append({"object": name, "arm": "ShuffledCoords", "seed": seed,
                         "ari": fu.kmeans_ari(weighted_features(x, coords, fields, "shuffled_coords", seed)[0], y, k, seed)[0]})
            rows.append({"object": name, "arm": "UniformSameGraph", "seed": seed, "ari": fu.kmeans_ari(uniform, y, k, seed)[0]})
        print("dev", name, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "arms.csv", index=False)
    mine = table.groupby(["object", "arm"]).ari.mean().unstack()
    check["uniform_same_graph_vs_frozen_neighbor_mean_max_abs"] = float(
        np.max(np.abs(mine.loc[OBJECTS, "UniformSameGraph"].to_numpy() - means.loc[OBJECTS, "NeighborMean"].to_numpy())))
    wide = mine.loc[OBJECTS].copy()
    wide["PCA50"] = means.loc[OBJECTS, "PCA50"]
    wide["NeighborMean"] = means.loc[OBJECTS, "NeighborMean"]
    wide["Tessera"] = means.loc[OBJECTS, "Tessera"]
    wide["NestedControl"] = pd.Series(nested)
    summary = {"per_object": {u: {c: (None if pd.isna(v) else float(v)) for c, v in wide.loc[u].items()} for u in OBJECTS},
               "checks": check, "cohorts": {}}
    for cohort, names in (("NSCLC", OBJECTS[:5]), ("BCC", OBJECTS[5:9]), ("other", OBJECTS[9:]), ("all13", OBJECTS)):
        sub = wide.loc[names]
        entry = {name: contrast(sub, a, b, interval=False) for name, (a, b) in {
            "E6_minus_NeighborMean": ("ExpressionWeighted", "NeighborMean"),
            "E6_minus_WeightShuffled": ("ExpressionWeighted", "WeightShuffled"),
            "E6_minus_ShuffledCoords": ("ExpressionWeighted", "ShuffledCoords"),
            "E6_minus_PCA50": ("ExpressionWeighted", "PCA50"),
            "E6_minus_Tessera": ("ExpressionWeighted", "Tessera")}.items()}
        if cohort in ("NSCLC", "BCC"):
            entry["E6_minus_NestedControl"] = contrast(sub, "ExpressionWeighted", "NestedControl", interval=False)
        summary["cohorts"][cohort] = entry
    return summary


def load_breadth(name):
    path, column = BREADTH[name]
    adata = ad.read_h5ad(path)
    matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    matrix = np.asarray(matrix, np.float32)
    coords = np.asarray(adata.obsm["spatial"], float)
    raw = adata.obs[column].to_numpy()
    keep = np.isfinite(coords).all(1) & np.isfinite(matrix).all(1) & (np.abs(matrix).sum(1) > 0)
    pcs, counts, n_controls, n_pc, variance = hv.preprocess(matrix[keep], np.asarray(adata.var_names, str))
    info = {"object": name, "source_file_sha256": sha(path), "n_rows": int(adata.n_obs), "n_analysis": int(keep.sum()),
            "count_normalization": counts, "control_features_removed": n_controls, "n_pc": n_pc,
            "pca_explained_variance": variance, "label_column": column}
    return info, pcs, coords[keep], np.full(int(keep.sum()), name), raw[keep]


def run_breadth():
    out_dir = OUT / "breadth"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = {}
    for name in BREADTH:
        info, pcs, coords, fields, raw = load_breadth(name)
        y, classes = encode(raw)
        info.update(n_valid=int((y >= 0).sum()), n_classes=len(classes), classes=classes)
        store[name] = {"base": info, "x": pcs, "coords": coords, "fields": fields, "targets": {"region": (y, classes)}}
    for unit, record in store.items():
        record.update(evaluate_unit(unit, record["x"], record["coords"], record["fields"], record["targets"], True))
        y = record["targets"]["region"][0]
        edge = record["edge"]
        valid = (y[edge[0]] >= 0) & (y[edge[1]] >= 0)
        record["base"]["gt_contiguity"] = float((y[edge[0]][valid] == y[edge[1]][valid]).mean())
        print("breadth", unit, flush=True)
    grid = pd.DataFrame([r for rec in store.values() for r in rec["grid"]])
    arms = pd.DataFrame([r for rec in store.values() for r in rec["arms"]])
    seams = pd.DataFrame([r for rec in store.values() for r in rec["seams"]])
    units = sorted(store)
    selected = {u: select_config(grid, units, u) for u in units}
    wide = wide_means(arms, grid, selected, "region").loc[units]
    means = grid.groupby(["unit", "config"]).ari.mean().unstack()
    oracle = {}
    for unit in units:
        candidates = [c for c in means.columns if not str(c).startswith("PCA50")]
        best = str(means.loc[unit, candidates].idxmax())
        uniform = [c for c in means.columns if str(c).startswith("NeighborMean")]
        best_k = str(means.loc[unit, uniform].idxmax())
        oracle[unit] = {"config": best, "ari": float(means.loc[unit, best]), "use": "not_for_inference",
                        "best_uniform_scale": best_k, "best_uniform_ari": float(means.loc[unit, best_k])}
    grid.to_csv(out_dir / "grid.csv", index=False)
    arms.to_csv(out_dir / "arms.csv", index=False)
    seams.to_csv(out_dir / "seams.csv", index=False)
    return {
        "units": units, "per_unit": {u: {c: float(wide.loc[u, c]) for c in wide.columns} for u in units},
        "selected_control": selected, "oracle_not_for_inference": oracle,
        "seams": {u: seams[seams.unit == u][["gate", "raw_difference", "smoothed_expression_weighted", "smoothed_uniform"]].mean().to_dict()
                  for u in units},
        "included": {u: store[u]["base"] for u in units},
        "contrasts": {name: contrast(wide, a, b, interval=False) for name, (a, b) in {
            "Tessera_minus_NestedControl": ("Tessera", "NestedControl"),
            "ExpressionWeighted_minus_NestedControl": ("ExpressionWeighted", "NestedControl"),
            "ExpressionWeighted_minus_UniformSameGraph": ("ExpressionWeighted", "UniformSameGraph"),
            "Tessera_minus_PCA50": ("Tessera", "PCA50")}.items()},
    }


def environment():
    return {"hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "torch", "pandas")},
            "device": "cpu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["hardening", "dev", "breadth"])
    args = parser.parse_args()
    if not PROTOCOL.exists():
        raise SystemExit("protocol file is missing")
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"hardening": run_hardening, "dev": run_dev, "breadth": run_breadth}[args.mode]()
    receipt = {
        "mode": args.mode, "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
        "frozen_sha256": {name: sha(REV / "source/scripts" / name) for name in
                          ("run_experiments.py", "run_followup.py", "prepare_data.py", "common.py")},
        "validation_external_sha256": {name: sha(REV / "validation-external" / name) for name in
                                       ("run_her2st_validation.py", "run_enhanced_targets.py")},
        "environment": environment(), "result": result,
    }
    (OUT / f"run-{args.mode}.json").write_text(json.dumps(jsonable(receipt), indent=2) + "\n")
    print("wrote", OUT / f"run-{args.mode}.json", flush=True)


if __name__ == "__main__":
    main()
