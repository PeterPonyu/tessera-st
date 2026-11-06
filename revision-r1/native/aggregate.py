#!/usr/bin/env python3
"""Collect E4 fit records into E4_scores.csv, E4_failures.csv, run-lock.json
and native/VERSIONS.md. Run in the tessera-native environment after the fits.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_common as nc  # noqa: E402

COLUMNS = ["method", "variant", "cohort", "patient", "target", "seed", "clustering", "K_ref", "K_achieved",
           "ari", "nmi", "n_scored", "runtime_s"]
# Target names in the CSV (agreed with the coordinator) versus the input/fit keys.
TARGET_NAMES = {"region": "regions", "celltype": "cell_types", "all": "all", "boundary_T1": "boundary_T1"}
PACKAGES = ["torch", "numpy", "scipy", "pandas", "scikit-learn", "scanpy", "anndata", "leidenalg", "igraph",
            "python-igraph", "POT", "umap-learn", "scikit-misc", "numba", "h5py", "rpy2", "tqdm",
            "GraphST", "pybanksy"]


def git_commit(path: Path) -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True).stdout.strip()
    return {"path": str(path), "commit": run("rev-parse", "HEAD"), "commit_date": run("log", "-1", "--format=%ci"),
            "remote": run("remote", "get-url", "origin"), "dirty": bool(run("status", "--porcelain"))}


def r_versions() -> dict:
    exe = Path(sys.executable).parent / "Rscript"
    out = subprocess.run([str(exe), "-e", 'cat(R.version.string, "\\n"); cat(as.character(packageVersion("mclust")))'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {"R": out[0].strip() if out else None, "mclust": out[1].strip() if len(out) > 1 else None}


def environment() -> dict:
    import torch
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    banksy_dist = importlib.metadata.distribution("pybanksy")
    direct = banksy_dist.read_text("direct_url.json")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": versions, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "r": r_versions(), "pybanksy_direct_url": json.loads(direct) if direct else None}


def mean_table(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "variant", "cohort", "target", "clustering"]
    per_patient = scores.groupby(keys + ["patient"]).ari.mean().reset_index()
    table = per_patient.groupby(keys).ari.agg(mean_ari="mean", n_patients="size").reset_index()
    fits = scores.groupby(keys).size().rename("n_rows").reset_index()
    return table.merge(fits, on=keys)


def main() -> None:
    records = [json.loads(p.read_text()) for p in sorted((nc.E4 / "fits").glob("*.json"))]
    rows, failures, scripts = [], [], set()
    for rec in records:
        base = {k: rec[k] for k in ("method", "variant", "cohort", "patient", "seed")}
        if rec["status"] != "ok":
            failures.append({**base, "stage": "fit", "target": "", "error": rec["error"]})
            continue
        scripts.add(rec["script_sha256"])
        for row in rec["rows"]:
            rows.append({**base, **{k: row[k] for k in COLUMNS if k in row}, "target": TARGET_NAMES[row["target"]]})
        for target, nat in rec["native"].items():
            if "error" in nat:
                failures.append({**base, "stage": "native_clustering", "target": TARGET_NAMES[target],
                                 "error": nat["error"]})
    scores = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["method", "variant", "cohort", "patient", "target", "seed", "clustering"]).reset_index(drop=True)
    scores.to_csv(nc.E4 / "E4_scores.csv", index=False)
    pd.DataFrame(failures, columns=["method", "variant", "cohort", "patient", "seed", "stage", "target",
                                    "error"]).to_csv(nc.E4 / "E4_failures.csv", index=False)
    table = mean_table(scores)
    table.to_csv(nc.E4 / "E4_mean_ari.csv", index=False)
    native = scores[scores.clustering == "native"].assign(K_differs=lambda d: d.K_achieved != d.K_ref)
    native.groupby(["method", "variant", "cohort", "target"]).agg(
        n_partitions=("K_differs", "size"), n_K_achieved_differs=("K_differs", "sum"),
        min_K_achieved=("K_achieved", "min")).reset_index().to_csv(nc.E4 / "E4_native_K.csv", index=False)

    native_dir = Path(__file__).resolve().parent
    third = {"Banksy_py": git_commit(nc.E4 / "third_party/Banksy_py/src"),
             "CASTLE": git_commit(nc.CASTLE_PARENT / "CASTLE"),
             "GraphST": {"source": "PyPI GraphST==1.1.1 (wheel GraphST-1.1.1-py3-none-any.whl, sha256 "
                                   "3d384ac093a20cb928c6e804751b7bf3bc99c545453744098acfa392a36231f3)",
                         "homepage": "https://github.com/JinmiaoChenLab/GraphST"},
             "pins": {"r-mclust": "6.1.1 (conda-forge). mclust >= 6.1.2 names columns from deparse(call$data), "
                                  "which fails when the matrix arrives through rpy2 without column names, as in "
                                  "GraphST.utils.mclust_R",
                      "rpy2": "3.5.17 (PyPI sdist built against the environment's R). rpy2 3.6 raises on "
                              "numpy2ri.activate(), which GraphST.utils.mclust_R calls"}}
    env = environment()
    manifest = nc.INPUTS / "manifest.json"
    lock = {"experiment": "E4 native external methods", "csv_target_names": TARGET_NAMES,
            "protocol": str(nc.PROTOCOL), "protocol_sha256": nc.sha(nc.PROTOCOL),
            "scripts_sha256": {p.name: nc.sha(p) for p in sorted(native_dir.glob("*.py"))},
            "fit_script_sha256_used": sorted(scripts),
            "inputs_manifest_sha256": nc.sha(manifest),
            "inputs_sha256": {k: v["sha256"] for k, v in json.loads(manifest.read_text())["patients"].items()},
            "environment": env, "third_party": third,
            "fits": {"ok": int(sum(r["status"] == "ok" for r in records)),
                     "failed": int(sum(r["status"] != "ok" for r in records)),
                     "score_rows": int(len(scores)), "native_failures": int(sum(f["stage"] == "native_clustering"
                                                                               for f in failures))}}
    test_log = nc.E4 / "logs/test_frames.log"
    if test_log.exists():
        lock["test_frames_log_tail"] = test_log.read_text().strip().splitlines()[-3:]
    (nc.E4 / "run-lock.json").write_text(json.dumps(lock, indent=2) + "\n")

    lines = ["# E4 native-method environment (generated by aggregate.py)", "",
             f"Conda environment `tessera-native`: `{env['executable']}`", "",
             f"- Python: {env['python'].split()[0]}", f"- Platform: {env['platform']}",
             f"- PyTorch CUDA runtime: {env['torch_cuda']}; GPU: {env['gpu']}",
             f"- {env['r']['R']}; R package mclust {env['r']['mclust']} (conda-forge)", "",
             "| Package | Version |", "|---|---|"]
    lines += [f"| {p} | {v} |" for p, v in env["packages"].items()]
    lines += ["", "## Third-party method sources", "",
              f"- GraphST: {third['GraphST']['source']}; {third['GraphST']['homepage']}",
              f"- BANKSY (pybanksy): {third['Banksy_py']['remote']} commit {third['Banksy_py']['commit']} "
              f"({third['Banksy_py']['commit_date']}), installed with pip from the clone at "
              f"results/revision-r1/E4/third_party/Banksy_py/src (sparse checkout: source only, no data)",
              f"- CASTLE: {third['CASTLE']['remote']} commit {third['CASTLE']['commit']} "
              f"({third['CASTLE']['commit_date']}); no installer in the repository, imported as package `CASTLE` "
              f"from results/revision-r1/E4/third_party/CASTLE (GPL-3.0; not copied into this repository)", "",
              "## Pins", "", f"- r-mclust {third['pins']['r-mclust']}", f"- rpy2 {third['pins']['rpy2']}", ""]
    (native_dir / "VERSIONS.md").write_text("\n".join(lines))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("rows", len(scores), "failures", len(failures))


if __name__ == "__main__":
    main()
