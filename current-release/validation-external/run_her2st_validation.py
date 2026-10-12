#!/usr/bin/env python3
"""Preregistered HER2ST pathologist-label validation.

Reads validation-external/PROTOCOL.md and does not write into the frozen
manuscript results. Expression-cluster labels are never loaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REV = HERE.parents[0]
REPO = REV.parents[2]
sys.path[:0] = [str(REV / "source" / "scripts"), str(REV / "source" / "src")]

import run_experiments as rx  # noqa: E402
import run_followup as fu  # noqa: E402
from common import encode  # noqa: E402

RAW = HERE / "raw"
OUT = HERE / "results"
PROTOCOL = HERE / "PROTOCOL.md"
GATE_PATH = REPO / "research" / "pipelines" / "fetch_gate.py"
LABEL_STEMS = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1", "J1"]
BASE = "https://raw.githubusercontent.com/almaan/her2st/master"
MIN_SPOTS = 50
MIN_MATCH = 0.90


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patient_id(stem: str) -> str:
    letters = []
    for char in stem:
        if not char.isalpha():
            break
        letters.append(char)
    if not letters:
        raise ValueError(f"section {stem} has no patient prefix")
    return "".join(letters)


def config_id(kind, k, lam) -> str:
    return f"{kind}|{k}|{lam}"


def download(fetch) -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    jobs = []
    for stem in LABEL_STEMS:
        jobs.append((
            f"{BASE}/data/ST-pat/lbl/{stem}_labeled_coordinates.tsv",
            RAW / f"{stem}_labeled_coordinates.tsv",
        ))
        jobs.append((f"{BASE}/data/ST-cnts/{stem}.tsv.gz", RAW / f"{stem}.tsv.gz"))
    for url, dest in jobs:
        if dest.exists() and dest.stat().st_size > 0:
            records.append({"url": url, "path": str(dest), "bytes": dest.stat().st_size,
                            "sha256": sha(dest), "cached": True, "ok": True})
            continue
        rec = fetch(url, dest, yes_large=False, timeout=120)
        records.append(rec)
        if not rec.get("ok"):
            dest.unlink(missing_ok=True)
    return records


def align_counts(counts: pd.DataFrame, label_ids: pd.Index) -> pd.DataFrame:
    wanted = set(map(str, label_ids))
    row_hit = sum(str(i) in wanted for i in counts.index)
    col_hit = sum(str(c) in wanted for c in counts.columns)
    if col_hit > row_hit:
        counts = counts.T
    counts.index = counts.index.map(str)
    if not counts.index.is_unique:
        raise ValueError("duplicate count-matrix spot identifiers")
    return counts.apply(pd.to_numeric, errors="coerce")


def preprocess(values: np.ndarray, genes: np.ndarray):
    """Same numerical steps as prepare_data.finish, without writing frozen inputs."""
    genes = np.asarray(genes, str)
    control = np.array([
        any(token in gene.lower() for token in ["negative", "negprb", "falsecode", "systemcontrol"])
        or gene.lower().startswith("blank")
        for gene in genes
    ])
    values = values[:, ~control]
    genes = genes[~control]
    counts = bool((values >= 0).all() and np.allclose(values, np.round(values), atol=1e-5))
    if counts:
        values = np.log1p(values / np.maximum(values.sum(1, keepdims=True), 1) * 1e4)
    if len(genes) > 3000:
        keep = np.argsort(-values.var(0), kind="stable")[:3000]
        values = values[:, keep]
        genes = genes[keep]
    scale = values.std(0)
    scale[scale == 0] = 1
    standardized = np.clip((values - values.mean(0)) / scale, -10, 10)
    model = PCA(
        n_components=min(50, standardized.shape[1], len(standardized) - 1),
        svd_solver="randomized",
        random_state=0,
    )
    pcs = model.fit_transform(standardized).astype(np.float32)
    return pcs, counts, int(control.sum()), int(pcs.shape[1]), float(model.explained_variance_ratio_.sum())


def load_joined(stem: str):
    """Joined counts and raw pathologist strings. No label recoding."""
    label_path = RAW / f"{stem}_labeled_coordinates.tsv"
    count_path = RAW / f"{stem}.tsv.gz"
    if not count_path.exists():
        return {"section": stem, "patient": patient_id(stem), "status": "excluded_missing_counts"}
    labels = pd.read_csv(label_path, sep="\t", index_col=0)
    required = ["x", "y", "pixel_x", "pixel_y", "label"]
    missing = [col for col in required if col not in labels.columns]
    if missing:
        raise ValueError(f"{stem} label table lacks {missing}")
    n_label_rows = int(len(labels))
    coordinates = labels[["x", "y", "pixel_x", "pixel_y"]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(coordinates.to_numpy(np.float64)).all(1)
    n_finite = int(finite.sum())
    labels = labels.loc[finite].copy()
    labels["spot_id"] = [
        f"{int(round(x))}x{int(round(y))}" for x, y in zip(labels["x"], labels["y"])
    ]
    if labels["spot_id"].duplicated().any():
        raise ValueError(f"{stem} rounded array coordinates are not unique")
    labels = labels.set_index("spot_id")
    counts = align_counts(pd.read_csv(count_path, sep="\t", index_col=0), labels.index)
    shared = labels.index.intersection(counts.index)
    match = len(shared) / max(n_finite, 1)
    if match < MIN_MATCH:
        raise ValueError(f"{stem} exact spot match {match:.3f} is below {MIN_MATCH}")
    labels = labels.loc[shared]
    counts = counts.loc[shared]
    coords = labels[["pixel_x", "pixel_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(coords).all(1)
    matrix = counts.to_numpy(np.float64)
    nonzero = np.isfinite(matrix).all(1) & (np.abs(matrix).sum(1) > 0) & finite
    matrix = matrix[nonzero]
    coords = coords[nonzero]
    raw_labels = labels["label"].to_numpy()[nonzero]
    base = {
        "section": stem,
        "patient": patient_id(stem),
        "n_label_rows": n_label_rows,
        "n_finite_coordinates": n_finite,
        "n_matched": int(len(shared)),
        "match_fraction": float(match),
        "n_analysis": int(len(raw_labels)),
        "status": "joined",
    }
    return base, matrix, counts.columns.to_numpy(), coords, raw_labels


def load_section(stem: str, encode):
    loaded = load_joined(stem)
    if isinstance(loaded, dict):
        return loaded
    base, matrix, genes, coords, raw_labels = loaded
    y, classes = encode(raw_labels)
    valid = int((y >= 0).sum())
    info = dict(base)
    info.update(n_analysis=int(len(y)), n_valid=valid, n_classes=len(classes), classes=classes)
    if valid < MIN_SPOTS or len(classes) < 2:
        info["status"] = "excluded_before_fit"
        return info
    pcs, was_counts, n_controls, n_pc, variance = preprocess(matrix, genes)
    info.update(status="included", count_normalization=was_counts, control_features_removed=n_controls,
                n_pc=n_pc, pca_explained_variance=variance)
    return info, pcs, coords, np.full(len(y), stem), y


def select_control(score_table: pd.DataFrame, patients: list[str], held_out: str) -> str:
    means = score_table.groupby(["patient", "config"]).ari.mean().unstack()
    candidates = [col for col in means.columns if not str(col).startswith("PCA50")]
    others = [patient for patient in patients if patient != held_out]
    return str(means.loc[others, candidates].mean().idxmax())


def sign_summary(deltas: list[float], exact_sign_flip) -> dict:
    values = [float(v) for v in deltas]
    return {
        "deltas": values,
        "mean": float(np.mean(values)),
        "n_positive": int(sum(v > 0 for v in values)),
        "n_negative": int(sum(v < 0 for v in values)),
        "n_zero": int(sum(v == 0 for v in values)),
        "exact_sign_flip_p": exact_sign_flip(values),
    }


def gate_rows_for(section: dict, x, y) -> list[dict]:
    rows = []
    for seed, gate, edge in section["gates"]:
        valid = (y[edge[0]] >= 0) & (y[edge[1]] >= 0)
        seam = y[edge[0]][valid] != y[edge[1]][valid]
        if len(np.unique(seam)) < 2:
            rows.append({"section": section["section"], "seed": seed, "status": "undefined"})
            continue
        difference = np.linalg.norm(x[edge[0]] - x[edge[1]], axis=1)[valid]
        rows.append({
            "section": section["section"],
            "patient": section["patient"],
            "seed": seed,
            "status": "scored",
            "gate_auroc": float(roc_auc_score(seam, 1 - gate[valid])),
            "expression_difference_auroc": float(roc_auc_score(seam, difference)),
            "n_valid_edges": int(valid.sum()),
            "seam_fraction": float(seam.mean()),
        })
    return rows


def self_test(exact_sign_flip) -> None:
    tied = pd.DataFrame([
        {"patient": "A", "config": "NeighborMean|10|None", "ari": 0.2},
        {"patient": "B", "config": "NeighborMean|10|None", "ari": 0.2},
        {"patient": "A", "config": "NeighborMean|6|None", "ari": 0.2},
        {"patient": "B", "config": "NeighborMean|6|None", "ari": 0.2},
        {"patient": "A", "config": "PCA50|None|None", "ari": 0.9},
        {"patient": "B", "config": "PCA50|None|None", "ari": 0.9},
    ])
    chosen = select_control(tied, ["A", "B"], "A")
    if chosen != "NeighborMean|10|None":
        raise AssertionError(chosen)
    if abs(exact_sign_flip([1, 1, 1, 1, 1, 1, 1, 1]) - 2 / 256) > 1e-12:
        raise AssertionError("sign-flip denominator")


def record_url(rec: dict) -> str:
    return rec.get("url") or (rec.get("head") or {}).get("url") or ""


def main() -> None:
    if not PROTOCOL.exists():
        raise SystemExit("protocol file is missing")
    self_test(fu.exact_sign_flip)
    fetch = load_module("fetch_gate", GATE_PATH).fetch
    downloads = download(fetch)
    failed = [rec for rec in downloads if not rec.get("ok") and not record_url(rec).endswith("/J1.tsv.gz")]
    if failed:
        raise SystemExit(json.dumps(failed, indent=2))

    included = []
    exclusions = []
    for stem in LABEL_STEMS:
        loaded = load_section(stem, encode)
        if isinstance(loaded, dict):
            exclusions.append(loaded)
            continue
        info, pcs, coords, fields, y = loaded
        included.append({"info": info, "x": pcs, "coords": coords, "fields": fields, "y": y})

    patients = [item["info"]["patient"] for item in included]
    if len(patients) != len(set(patients)):
        raise SystemExit(f"expected one section per patient, got {patients}")
    if len(patients) < 4:
        raise SystemExit(f"fewer than 4 eligible patients: {patients}")

    control_rows = []
    for item in included:
        info, x, coords, fields, y = item["info"], item["x"], item["coords"], item["fields"], item["y"]
        cache = {}
        k_classes = info["n_classes"]
        for kind, k, lam in fu.configs():
            for seed in fu.SEEDS:
                features = fu.features(kind, k, lam, x, coords, fields, cache, seed)
                ari, _ = fu.kmeans_ari(features, y, k_classes, seed)
                control_rows.append({
                    "section": info["section"], "patient": info["patient"], "kind": kind,
                    "k": k, "lam": lam, "config": config_id(kind, k, lam), "seed": seed, "ari": ari,
                })
        print("controls", info["section"], "K", k_classes, "n", info["n_valid"], flush=True)
    controls = pd.DataFrame(control_rows)

    torch.set_num_threads(4)
    tessera_rows = []
    gate_rows = []
    for item in included:
        info = item["info"]
        gates = []
        for seed in rx.SEEDS:
            started = time.perf_counter()
            _, embedding, _, gate, edge, _, _, elapsed, _, _ = rx.train_net(
                item["x"], item["coords"], item["fields"], info["n_classes"], "Tessera", seed, "cpu")
            if gate.shape[0] != edge.shape[1]:
                raise ValueError(f"{info['section']} gate/edge mismatch")
            ari, _ = fu.kmeans_ari(embedding, item["y"], info["n_classes"], seed)
            tessera_rows.append({
                "section": info["section"], "patient": info["patient"], "arm": "Tessera",
                "seed": seed, "ari": ari, "fit_seconds": elapsed,
                "total_seconds": time.perf_counter() - started, "n_edges": int(edge.shape[1]),
            })
            gates.append((seed, gate, edge))
            print("tessera", info["section"], seed, "ARI", round(ari, 4), "fit_s", round(elapsed, 1), flush=True)
        gate_rows.extend(gate_rows_for(
            {"section": info["section"], "patient": info["patient"], "gates": gates},
            item["x"], item["y"]))

    tessera = pd.DataFrame(tessera_rows)
    per_patient = []
    for patient in patients:
        selected = select_control(controls, patients, patient)
        tessera_ari = float(tessera.loc[tessera.patient == patient, "ari"].mean())
        control_ari = float(controls.loc[(controls.patient == patient) & (controls.config == selected), "ari"].mean())
        pca_ari = float(controls.loc[(controls.patient == patient) & (controls.kind == "PCA50"), "ari"].mean())
        per_patient.append({
            "patient": patient,
            "section": included[patients.index(patient)]["info"]["section"],
            "selected_control": selected,
            "tessera_ari": tessera_ari,
            "nested_control_ari": control_ari,
            "pca50_ari": pca_ari,
            "tessera_minus_nested": tessera_ari - control_ari,
            "tessera_minus_pca50": tessera_ari - pca_ari,
        })
    primary = pd.DataFrame(per_patient)
    without_g = [p for p in patients if p != "G"]
    sensitivity = []
    for patient in without_g:
        selected = select_control(controls[controls.patient != "G"], without_g, patient)
        tessera_ari = float(tessera.loc[tessera.patient == patient, "ari"].mean())
        control_ari = float(controls.loc[(controls.patient == patient) & (controls.config == selected), "ari"].mean())
        sensitivity.append({
            "patient": patient, "selected_control": selected, "tessera_ari": tessera_ari,
            "nested_control_ari": control_ari, "tessera_minus_nested": tessera_ari - control_ari,
        })

    oracle = []
    means = controls.groupby(["patient", "config"]).ari.mean().unstack()
    for patient in patients:
        best = str(means.loc[patient].idxmax())
        oracle.append({"patient": patient, "oracle_config": best, "oracle_ari": float(means.loc[patient, best]),
                       "use": "not_for_inference"})

    gates = pd.DataFrame(gate_rows)
    scored = gates[gates.status == "scored"]
    gate_patient = scored.groupby("patient")[["gate_auroc", "expression_difference_auroc"]].mean()
    gate_patient["gate_minus_expression"] = gate_patient.gate_auroc - gate_patient.expression_difference_auroc

    OUT.mkdir(parents=True, exist_ok=True)
    controls.to_csv(OUT / "scale_controls.csv", index=False)
    tessera.to_csv(OUT / "tessera_scores.csv", index=False)
    primary.to_csv(OUT / "primary_patients.csv", index=False)
    gates.to_csv(OUT / "boundary_edges.csv", index=False)
    summary = {
        "protocol_sha256": sha(PROTOCOL),
        "script_sha256": sha(Path(__file__)),
        "frozen_estimator_sha256": {
            "run_experiments.py": sha(REV / "source/scripts/run_experiments.py"),
            "run_followup.py": sha(REV / "source/scripts/run_followup.py"),
            "prepare_data.py": sha(REV / "source/scripts/prepare_data.py"),
        },
        "device": "cpu",
        "included": [item["info"] for item in included],
        "exclusions": exclusions,
        "primary_tessera_minus_nested": sign_summary(primary.tessera_minus_nested.tolist(), fu.exact_sign_flip),
        "primary_tessera_minus_pca50": sign_summary(primary.tessera_minus_pca50.tolist(), fu.exact_sign_flip),
        "patients": primary.to_dict(orient="records"),
        "sensitivity_without_patient_G": {
            "patients": sensitivity,
            "tessera_minus_nested": sign_summary(
                [row["tessera_minus_nested"] for row in sensitivity], fu.exact_sign_flip),
        },
        "boundary": {
            "patients": {idx: {key: float(val) for key, val in row.items()} for idx, row in gate_patient.iterrows()},
            "gate_higher_than_expression_difference": int((gate_patient.gate_minus_expression > 0).sum()),
            "n_patients": int(len(gate_patient)),
        },
        "oracle_not_for_inference": oracle,
        "downloads": [{k: rec.get(k) for k in ("url", "path", "bytes", "sha256", "ok", "cached", "error")}
                      for rec in downloads],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "patients": summary["primary_tessera_minus_nested"],
        "versus_pca50": summary["primary_tessera_minus_pca50"],
        "boundary_gate_higher": summary["boundary"]["gate_higher_than_expression_difference"],
        "exclusions": exclusions,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
