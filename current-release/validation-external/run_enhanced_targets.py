#!/usr/bin/env python3
"""Named-compartment and tumor-boundary refits. See ENHANCED_PROTOCOL.md."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REV = HERE.parents[0]
sys.path[:0] = [str(HERE), str(REV / "source" / "scripts"), str(REV / "source" / "src")]

import run_experiments as rx  # noqa: E402
import run_followup as fu  # noqa: E402
import run_her2st_validation as hv  # noqa: E402
from common import encode  # noqa: E402

OUT = HERE / "results" / "enhanced"
PROTOCOL = HERE / "ENHANCED_PROTOCOL.md"
STEMS = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]
UNASSIGNED = {"undetermined"}
TUMOR = {"invasive cancer", "cancer in situ"}
OTHER = {"adipose tissue", "connective tissue", "immune infiltrate", "breast glands"}
ALLOWED = UNASSIGNED | TUMOR | OTHER
MIN_SIDE = 20


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(raw) -> str:
    return str(raw).strip().lower()


def recode(raw: np.ndarray, target: str) -> np.ndarray:
    out = []
    for value in raw:
        label = normalize(value)
        if label not in ALLOWED:
            raise ValueError(f"unexpected pathologist label {value!r}")
        if target == "compartments":
            out.append("" if label in UNASSIGNED else str(value).strip())
        elif label in TUMOR:
            out.append("tumor")
        elif label in OTHER:
            out.append("other tissue")
        else:
            out.append("")
    return np.asarray(out)


def expression_gate_mean(x, coords, fields):
    """Fixed k=6 aggregator. High PCA-feature difference lowers the edge weight."""
    edge, _ = rx.graph(coords, fields, "Tessera", 1)
    difference = np.linalg.norm(x[edge[0]] - x[edge[1]], axis=1)
    weight = np.exp(-difference / (np.median(difference) + 1e-8)).astype(np.float32)
    total = np.zeros_like(x)
    mass = np.zeros(len(x), np.float32)
    np.add.at(total, edge[1], weight[:, None] * x[edge[0]])
    np.add.at(mass, edge[1], weight)
    return total / np.maximum(mass[:, None], 1e-8)


def one_target(target, joined):
    prepared = []
    exclusions = []
    for stem, base, matrix, genes, coords, raw in joined:
        labels = recode(raw, target)
        y, classes = encode(labels)
        valid = int((y >= 0).sum())
        counts = {name: int((labels == name).sum()) for name in classes}
        info = dict(base)
        info.update(target=target, n_valid=valid, n_classes=len(classes), classes=classes,
                    class_counts=counts)
        if target == "boundary":
            sides = {name: int((labels == name).sum()) for name in ("tumor", "other tissue")}
            info["side_counts"] = sides
            if min(sides.values()) < MIN_SIDE:
                info["status"] = "excluded_sparse_side"
                exclusions.append(info)
                continue
        if valid < 50 or len(classes) < 2:
            info["status"] = "excluded_before_fit"
            exclusions.append(info)
            continue
        pcs, was_counts, n_controls, n_pc, variance = hv.preprocess(matrix, genes)
        info.update(status="included", count_normalization=was_counts,
                    control_features_removed=n_controls, n_pc=n_pc,
                    pca_explained_variance=variance)
        prepared.append({"info": info, "x": pcs, "coords": coords,
                         "fields": np.full(len(y), stem), "y": y})
    patients = [item["info"]["patient"] for item in prepared]
    if len(patients) < 4:
        raise SystemExit(f"{target} has {len(patients)} patients")

    control_rows = []
    gate_rows = []
    expression_rows = []
    for item in prepared:
        info, x, coords, fields, y = item["info"], item["x"], item["coords"], item["fields"], item["y"]
        cache = {}
        for kind, k, lam in fu.configs():
            for seed in fu.SEEDS:
                features = fu.features(kind, k, lam, x, coords, fields, cache, seed)
                ari, _ = fu.kmeans_ari(features, y, info["n_classes"], seed)
                control_rows.append({
                    "target": target, "section": info["section"], "patient": info["patient"],
                    "kind": kind, "k": k, "lam": lam,
                    "config": f"{kind}|{k}|{lam}", "seed": seed, "ari": ari,
                })
        gated = expression_gate_mean(x, coords, fields)
        for seed in fu.SEEDS:
            ari, _ = fu.kmeans_ari(gated, y, info["n_classes"], seed)
            expression_rows.append({
                "target": target, "section": info["section"], "patient": info["patient"],
                "arm": "ExpressionGateMean", "seed": seed, "ari": ari,
            })
        print("controls", target, info["section"], "K", info["n_classes"], "n", info["n_valid"], flush=True)

    torch.set_num_threads(4)
    tessera_rows = []
    for item in prepared:
        info = item["info"]
        gates = []
        for seed in rx.SEEDS:
            _, embedding, _, gate, edge, _, _, elapsed, _, _ = rx.train_net(
                item["x"], item["coords"], item["fields"], info["n_classes"], "Tessera", seed, "cpu")
            if gate.shape[0] != edge.shape[1]:
                raise ValueError("gate/edge mismatch")
            ari, _ = fu.kmeans_ari(embedding, item["y"], info["n_classes"], seed)
            tessera_rows.append({
                "target": target, "section": info["section"], "patient": info["patient"],
                "arm": "Tessera", "seed": seed, "ari": ari, "fit_seconds": elapsed,
            })
            gates.append((seed, gate, edge))
            print("tessera", target, info["section"], seed, round(ari, 4), round(elapsed, 1), flush=True)
        gate_rows.extend(hv.gate_rows_for(
            {"section": info["section"], "patient": info["patient"], "gates": gates},
            item["x"], item["y"]))
    for row in gate_rows:
        row["target"] = target

    controls = pd.DataFrame(control_rows)
    tessera = pd.DataFrame(tessera_rows)
    expression = pd.DataFrame(expression_rows)
    per_patient = []
    for patient in patients:
        selected = hv.select_control(controls, patients, patient)
        tessera_ari = float(tessera.loc[tessera.patient == patient, "ari"].mean())
        control_ari = float(controls.loc[(controls.patient == patient) & (controls.config == selected), "ari"].mean())
        pca_ari = float(controls.loc[(controls.patient == patient) & (controls.kind == "PCA50"), "ari"].mean())
        expression_ari = float(expression.loc[expression.patient == patient, "ari"].mean())
        per_patient.append({
            "target": target, "patient": patient,
            "section": next(item["info"]["section"] for item in prepared if item["info"]["patient"] == patient),
            "selected_control": selected,
            "tessera_ari": tessera_ari, "nested_control_ari": control_ari, "pca50_ari": pca_ari,
            "expression_gate_ari": expression_ari,
            "tessera_minus_nested": tessera_ari - control_ari,
            "tessera_minus_pca50": tessera_ari - pca_ari,
            "expression_gate_minus_nested": expression_ari - control_ari,
            "tessera_minus_expression_gate": tessera_ari - expression_ari,
        })
    primary = pd.DataFrame(per_patient)
    gates = pd.DataFrame(gate_rows)
    scored = gates[gates.status == "scored"]
    gate_patient = scored.groupby("patient")[["gate_auroc", "expression_difference_auroc"]].mean()
    return {
        "included": [item["info"] for item in prepared],
        "exclusions": exclusions,
        "controls": controls,
        "tessera": tessera,
        "expression": expression,
        "primary": primary,
        "gates": gates,
        "tessera_minus_nested": hv.sign_summary(primary.tessera_minus_nested.tolist(), fu.exact_sign_flip),
        "expression_gate_minus_nested": hv.sign_summary(primary.expression_gate_minus_nested.tolist(), fu.exact_sign_flip),
        "tessera_minus_expression_gate": hv.sign_summary(primary.tessera_minus_expression_gate.tolist(), fu.exact_sign_flip),
        "boundary_gate": {
            patient: {key: float(value) for key, value in row.items()}
            for patient, row in gate_patient.iterrows()
        },
    }


def main() -> None:
    if not PROTOCOL.exists():
        raise SystemExit("enhanced protocol missing")
    joined = []
    for stem in STEMS:
        loaded = hv.load_joined(stem)
        if isinstance(loaded, dict):
            raise SystemExit(f"{stem} did not join")
        joined.append((stem, *loaded))
    results = {}
    tables = {"controls": [], "tessera": [], "expression": [], "primary": [], "gates": []}
    for target in ("compartments", "boundary"):
        pack = one_target(target, joined)
        for key in tables:
            tables[key].append(pack.pop(key))
        results[target] = pack
    OUT.mkdir(parents=True, exist_ok=True)
    for key, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(OUT / f"{key}.csv", index=False)
    summary = {
        "protocol_sha256": sha(PROTOCOL),
        "script_sha256": sha(Path(__file__)),
        "targets": results,
        "reading": (
            "Positive direction is the annotation-matched simple control. "
            "Tessera exceeding that control is not required."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        target: {
            "tessera_minus_nested": results[target]["tessera_minus_nested"],
            "expression_gate_minus_nested": results[target]["expression_gate_minus_nested"],
        }
        for target in results
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
