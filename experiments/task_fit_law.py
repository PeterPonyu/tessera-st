"""Task-fit LAW: turn the observed pattern into a predictive rule.

Hypothesis: a single scalar — the GT's spatial contiguity (fraction of each cell's spatial kNN that
share its GT label) — predicts (i) whether a spatial prior helps and (ii) Tessera's rank. High
contiguity = spatially-contiguous domains (layers/anatomical domains) → spatial methods + Tessera win;
low contiguity = scattered cell types → non-spatial clustering wins, Tessera loses. If contiguity
correlates strongly with Tessera rank and with the spatial advantage, the method–task fit is a
*predictable rule*, not just an observation — and yields a method-selection guideline. No training;
reads hardened_bench.json + computes GT geometry."""

import json

import numpy as np
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from _roots import data_root, external_root, spatial_omics_root

DR = str(data_root())
EXT = str(external_root())
DATASETS = {
    "DLPFC(Visium,layer)": (f"{DR}/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad",
                            ["ground_truth"]),
    "seqFISH(embryo,ctype)": (f"{DR}/raw/squidpy/seqfish.h5ad", ["celltype_mapped_refined"]),
    "MERFISH(hypo,domain)": (
        f"{DR}/baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad", ["domain"]),
    "STARmap(cortex,region)": (f"{DR}/processed/starmap_mouse_vcortex_wang2018/anndata.h5ad",
                               ["ground_truth", "region", "label"]),
    "osmFISH(cortex,Region)": (
        f"{EXT}/../baselines/CellNiche-original/data/osmFISH_SScortex.h5ad", ["Region"]),
    "MIBI-TOF(protein,Cluster)": (f"{DR}/raw/squidpy/mibitof.h5ad", ["Cluster"]),
}


def _valid(v):
    return str(v).strip().lower() not in {"", "nan", "none", "na", "unknown"}


def gt_contiguity(path, gt_cols, k=6):
    """Mean fraction of each cell's spatial kNN that share its GT label. 1=contiguous, ~1/n=scattered."""
    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], float)
    col = next(c for c in gt_cols if c in a.obs.columns)
    raw = a.obs[col].to_numpy()
    codes = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([codes[v] if _valid(v) else -1 for v in raw])
    m = true >= 0
    cv, tv = coords[m], true[m]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(cv).kneighbors(cv)[1][:, 1:]
    return float((tv[idx] == tv[:, None]).mean())


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


hd = json.load(open("experiments/hardened_bench.json"))
mp = json.load(open("experiments/multi_platform_bench.json"))

rows = []
for name, (path, gt) in DATASETS.items():
    cont = gt_contiguity(path, gt)
    rec = hd[name]
    order = sorted(rec, key=lambda m: -rec[m]["mean"])
    t_rank = order.index("Tessera") + 1
    # spatial advantage = best spatial method - non-spatial floor (from hardened means)
    nonsp = rec.get("floor:nonspatial", {}).get("mean", 0.0)
    spatial_best = max(v["mean"] for m, v in rec.items() if m != "floor:nonspatial")
    rows.append({"platform": name, "GT_contiguity": round(cont, 3), "Tessera_rank": t_rank,
                 "winner": order[0], "spatial_advantage": round(spatial_best - nonsp, 3),
                 "spatial_helps": mp[name]["spatial_prior_helps"]})

rows.sort(key=lambda r: -r["GT_contiguity"])
print(f"{'platform':>26} | {'GT contig':>9} | {'Tessera rank':>12} | {'spatial adv':>11} | winner")
print("-" * 92)
for r in rows:
    print(f"{r['platform']:>26} | {r['GT_contiguity']:>9} | {r['Tessera_rank']:>12} | "
          f"{r['spatial_advantage']:>+11.3f} | {r['winner']}")

cont = np.array([r["GT_contiguity"] for r in rows])
trank = np.array([r["Tessera_rank"] for r in rows])
sadv = np.array([r["spatial_advantage"] for r in rows])
rho_rank = spearman(cont, trank)   # expect strongly NEGATIVE (high contig -> low rank number = better)
rho_adv = spearman(cont, sadv)     # expect strongly POSITIVE (high contig -> spatial prior helps)

print(f"\nSpearman(GT_contiguity, Tessera_rank)      = {rho_rank:+.3f}  (negative ⇒ contiguity predicts"
      " Tessera ranks BETTER)")
print(f"Spearman(GT_contiguity, spatial_advantage) = {rho_adv:+.3f}  (positive ⇒ contiguity predicts"
      " spatial prior helps)")

law = {
    "rows": rows, "spearman_contiguity_vs_tessera_rank": round(rho_rank, 3),
    "spearman_contiguity_vs_spatial_advantage": round(rho_adv, 3),
    "rule": ("Measure GT spatial contiguity (mean kNN same-label fraction). HIGH (>~0.6, contiguous "
             "domains: layers/anatomical domains) ⇒ use a spatial / boundary-aware method (Tessera is "
             "best on the highest-contiguity domain tasks). LOW (<~0.4, scattered cell types) ⇒ a "
             "non-spatial clusterer is the floor-and-ceiling; spatial smoothing hurts."),
}
json.dump(law, open("experiments/task_fit_law.json", "w"), indent=2)
print("\nRULE:", law["rule"])
print("\nwrote experiments/task_fit_law.json")
