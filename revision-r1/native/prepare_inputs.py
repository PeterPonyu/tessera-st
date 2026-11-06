#!/usr/bin/env python3
"""E4 input export (PROTOCOL-R1.md, section E4).

Runs in the project interpreter (/home/zeyufu/miniconda3/envs/dl/bin/python).
Writes one npz per patient to results/revision-r1/E4/inputs/ so that the
native external methods can run in their own environment (tessera-native).
Nothing outside that folder is written; frozen modules are only imported.

NSCLC (CosMx): data/<object>/input.npz. The log-normalized matrix gene_log
(all 960 panel genes) is exported as is; raw counts are not part of the
frozen inputs. Region labels: ref (K_ref = n_classes in input.json).
Cell-type labels: common.encode(secondary_name) (K_ref = valid classes).

HER2ST: frozen loader run_her2st_validation.load_joined (raw counts, gene
names, pixel coordinates, raw pathologist strings). Targets: all labels =
encode(raw); tumor boundary = encode(run_extension.boundary_labels(raw, 'T1'))
with K_ref = 2. Log-normalization: log1p(1e4 * counts / rowsum).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

PROJECT = Path("/home/zeyufu/Desktop/singlecell-genomics-research/projects/tessera-st")
sys.path[:0] = [
    str(PROJECT / "repository/current-release/source/scripts"),
    str(PROJECT / "repository/current-release/source/src"),
    str(PROJECT / "validation/validation-external"),
    str(PROJECT / "validation/validation-extension"),
]

import run_her2st_validation as hv  # noqa: E402
import run_extension as rex  # noqa: E402
from common import encode  # noqa: E402

OUT = PROJECT / "results/revision-r1/E4/inputs"
PROTOCOL = PROJECT / "repository/revision-r1/PROTOCOL-R1.md"
NSCLC = ["NSCLC_Lung5_Rep2", "NSCLC_Lung6", "NSCLC_Lung9_Rep1", "NSCLC_Lung12", "NSCLC_Lung13"]
HER2ST = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label_summary(y, classes):
    counts = np.bincount(y[y >= 0], minlength=len(classes))
    return {"K_ref": int(len(classes)), "classes": [str(c) for c in classes],
            "class_counts": [int(c) for c in counts], "n_valid": int((y >= 0).sum()),
            "n_invalid": int((y < 0).sum())}


def export_nsclc(name: str) -> dict:
    src = PROJECT / "data" / name / "input.npz"
    meta = json.loads((PROJECT / "data" / name / "input.json").read_text())
    if sha(src) != meta["matrix_sha256"]:
        raise ValueError(f"{name}: input.npz hash differs from input.json")
    d = np.load(src, allow_pickle=False)
    expr = np.asarray(d["gene_log"], np.float32)
    region = np.asarray(d["ref"], np.int64)
    region_classes = [str(c) for c in meta["classes"]]
    if len(region_classes) != int(meta["n_classes"]) or set(np.unique(region[region >= 0])) - set(range(meta["n_classes"])):
        raise ValueError(f"{name}: region labels disagree with input.json")
    celltype, celltype_classes = encode(d["secondary_name"])
    rowsum = np.expm1(expr.astype(np.float64)).sum(1)
    out = {
        "cohort": np.array("NSCLC"), "patient": np.array(name),
        "expr_log": expr, "has_counts": np.array(False),
        "genes": np.asarray(d["gene"], str), "coords": np.asarray(d["coords"], np.float64),
        "frame": np.asarray(d["field"], str), "obs_id": np.asarray(d["obs_id"], str),
        "targets": np.array(["region", "celltype"]),
        "y_region": region, "K_region": np.array(int(meta["n_classes"])),
        "classes_region": np.array(region_classes),
        "y_celltype": np.asarray(celltype, np.int64), "K_celltype": np.array(len(celltype_classes)),
        "classes_celltype": np.array(celltype_classes, str),
        "celltype_name": np.asarray(d["secondary_name"], str),
    }
    info = {"source": str(src), "source_sha256": meta["matrix_sha256"],
            "expression": "gene_log: log1p(1e4 * counts / rowsum) as frozen; all 960 panel genes; raw counts unavailable",
            "expm1_rowsum_min_max": [float(rowsum.min()), float(rowsum.max())],
            "targets": {"region": label_summary(region, region_classes),
                        "celltype": label_summary(np.asarray(celltype), celltype_classes)}}
    return out, info


def export_her2st(stem: str) -> dict:
    for path in (hv.RAW / f"{stem}.tsv.gz", hv.RAW / f"{stem}_labeled_coordinates.tsv"):
        sidecar = json.loads(Path(f"{path}.sha256.json").read_text())
        if sha(path) != sidecar["sha256"]:
            raise ValueError(f"{path} hash differs from its download record")
    base, matrix, genes, coords, raw = hv.load_joined(stem)
    if not np.allclose(matrix, np.round(matrix)) or matrix.min() < 0:
        raise ValueError(f"{stem}: matrix is not a count matrix")
    counts = np.round(matrix).astype(np.int32)
    rowsum = counts.sum(1, dtype=np.float64)
    if (rowsum <= 0).any():
        raise ValueError(f"{stem}: zero-count spot after the frozen join")
    expr = np.log1p(1e4 * counts / rowsum[:, None]).astype(np.float32)
    y_all, all_classes = encode(raw)
    y_bnd, bnd_classes = encode(rex.boundary_labels(raw, "T1"))
    if len(bnd_classes) != 2:
        raise ValueError(f"{stem}: boundary target does not have two classes")
    out = {
        "cohort": np.array("HER2ST"), "patient": np.array(stem),
        "expr_log": expr, "counts": counts, "has_counts": np.array(True),
        "genes": np.asarray(genes, str), "coords": np.asarray(coords, np.float64),
        "frame": np.full(len(raw), stem), "obs_id": np.array([f"{stem}_{i}" for i in range(len(raw))]),
        "targets": np.array(["all", "boundary_T1"]),
        "y_all": np.asarray(y_all, np.int64), "K_all": np.array(len(all_classes)),
        "classes_all": np.array(all_classes, str),
        "y_boundary_T1": np.asarray(y_bnd, np.int64), "K_boundary_T1": np.array(2),
        "classes_boundary_T1": np.array(bnd_classes, str),
        "raw_label": np.asarray(raw, str),
    }
    info = {"source": base, "raw_sha256": {p.name: sha(p) for p in (hv.RAW / f"{stem}.tsv.gz",
                                                                    hv.RAW / f"{stem}_labeled_coordinates.tsv")},
            "expression": "raw counts kept; log1p(1e4 * counts / rowsum)",
            "n_genes": int(counts.shape[1]), "n_unique_gene_names": int(len(set(map(str, genes)))),
            "targets": {"all": label_summary(np.asarray(y_all), all_classes),
                        "boundary_T1": label_summary(np.asarray(y_bnd), bnd_classes)}}
    return out, info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": str(PROTOCOL), "protocol_sha256": sha(PROTOCOL),
                "script": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__).resolve()),
                "frozen_modules_sha256": {name: sha(Path(mod.__file__)) for name, mod in
                                          (("run_her2st_validation.py", hv), ("run_extension.py", rex),
                                           ("common.py", sys.modules["common"]))},
                "python": sys.version, "platform": platform.platform(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")},
                "patients": {}}
    jobs = [("NSCLC", n, export_nsclc) for n in NSCLC] + [("HER2ST", s, export_her2st) for s in HER2ST]
    for cohort, patient, fn in jobs:
        arrays, info = fn(patient)
        n = len(arrays["coords"])
        for key, value in arrays.items():
            if value.ndim >= 1 and key not in ("genes", "targets") and not key.startswith("classes_") and len(value) != n:
                raise ValueError(f"{patient}: {key} has {len(value)} rows, expected {n}")
        path = OUT / f"{cohort}-{patient}.npz"
        np.savez_compressed(path, **arrays)
        frames, frame_counts = np.unique(arrays["frame"], return_counts=True)
        info.update(file=path.name, sha256=sha(path), n_cells=int(n), n_features=int(arrays["expr_log"].shape[1]),
                    frames={str(f): int(c) for f, c in zip(frames, frame_counts)}, cohort=cohort)
        manifest["patients"][f"{cohort}-{patient}"] = info
        print(cohort, patient, "n", n, "genes", arrays["expr_log"].shape[1], "frames", len(frames),
              {t: info["targets"][t]["K_ref"] for t in info["targets"]}, flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
