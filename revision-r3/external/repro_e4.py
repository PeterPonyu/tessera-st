#!/usr/bin/env python3
"""Reproduce round-2 E4 GraphST with the WP4 adapter (environment tessera-native).

Released GraphST, native input, default config (k3-e600), then matched K-means at K_ref on 'emb'
(native_common.kmeans_matched: KMeans(K, n_init=10, random_state=seed)), as E4 did. Patient means (seed mean
per patient, then the mean over patients) are compared per cohort and target with
results/revision-r1/E4/E4_scores.csv (GraphST, kmeans_matched).
Pass: every cohort x target within 0.03 ARI. That needs all 13 patients x seeds 1-3 (the full check):
    python -B repro_e4.py --device cuda
A subset only proves the path and reports each fit's difference from the same patient and seed in E4:
    python -B repro_e4.py --datasets HER2ST-C1 --seeds 1 --device cpu
Writes <out>/repro_e4/{fits.csv, summary.json}; <out> is $TESSERA_R3_OUT (default results/revision-r3/W4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_external as ce  # noqa: E402
import native_common as nc  # noqa: E402
import run_graphst_r3 as graphst  # noqa: E402
from run_external_r3 import OUT  # noqa: E402

E4_SCORES = ce.c.RESULTS / "revision-r1" / "E4" / "E4_scores.csv"
DATASETS = [*ce.c.LUNG, *ce.HER2ST]
E4_TARGETS = {"primary": "regions", "cell_type": "cell_types", "all": "all", "boundary_T1": "boundary_T1"}
SEEDS = [1, 2, 3]
TOLERANCE = 0.03


def e4_patient(dataset_id: str) -> str:
    return dataset_id.split("-", 1)[1] if dataset_id in ce.HER2ST else dataset_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS, choices=SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    e4 = pd.read_csv(E4_SCORES)
    e4 = e4[(e4.method == "GraphST") & (e4.clustering == "kmeans_matched")]
    rows = []
    for ds_id in args.datasets:
        ds = ce.load_dataset(ds_id)
        for seed in args.seeds:
            fit = graphst.fit(ds["native"], ds["coords"], ds["frames"], "released", "default", seed, "native",
                              args.device)
            for target, name in E4_TARGETS.items():
                if target not in ds["targets"]:
                    continue
                pred = nc.kmeans_matched(fit["embedding"], ds["K"][target], seed)
                saved = e4[(e4.patient == e4_patient(ds_id)) & (e4.target == name) & (e4.seed == seed)].ari
                if len(saved) != 1:
                    raise ValueError(f"E4 has {len(saved)} rows for {ds_id} {name} seed {seed}")
                ari = nc.score(ds["targets"][target], pred)["ari"]
                rows.append(dict(cohort=ds["cohort"], patient=e4_patient(ds_id), target=name, seed=seed, ari=ari,
                                 e4_ari=float(saved.iloc[0]), diff=ari - float(saved.iloc[0]),
                                 seconds=fit["seconds"], device=args.device))
                print(ds_id, name, f"seed {seed}", f"ARI {ari:.4f}", f"E4 {saved.iloc[0]:.4f}",
                      f"fit {fit['seconds']:.1f} s", flush=True)
    fits = pd.DataFrame(rows)
    summary = {"tolerance": TOLERANCE, "device": args.device, "e4_scores": str(E4_SCORES),
               "e4_scores_sha256": ce.sha(E4_SCORES), "groups": []}
    for (cohort, target), part in fits.groupby(["cohort", "target"]):
        mine = part.groupby("patient").ari.mean().mean()
        saved = e4[(e4.cohort == cohort) & (e4.target == target)]
        complete = set(part.patient) == set(saved.patient) and set(part.seed) == set(SEEDS)
        reference = saved.groupby("patient").ari.mean().mean() if complete else \
            part.groupby("patient").e4_ari.mean().mean()
        verdict = ("pass" if abs(mine - reference) <= TOLERANCE else "fail") if complete else "partial"
        summary["groups"].append(dict(cohort=cohort, target=target, n_patients=int(part.patient.nunique()),
                                      seeds=sorted(int(s) for s in part.seed.unique()), mean_ari=float(mine),
                                      e4_mean_ari=float(reference), diff=float(mine - reference), verdict=verdict))
    verdicts = {g["verdict"] for g in summary["groups"]}
    summary["verdict"] = "pass" if verdicts == {"pass"} else ("fail" if "fail" in verdicts else "partial")
    folder = Path(args.out) / "repro_e4"
    folder.mkdir(parents=True, exist_ok=True)
    fits.to_csv(folder / "fits.csv", index=False)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
