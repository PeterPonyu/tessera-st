"""Publication-grade hardening of the 6-platform benchmark.

For each (platform, method) we now: train under MULTIPLE SEEDS, and score each method at ITS OWN best
configuration (best of {KMeans,GMM} × {refine off,on}) — since backend+refinement are part of a
method's recommended pipeline (finding 1 showed backend is a confound, so fixing one is unfair). Report
mean±std over seeds and the per-platform winner. GraphST included (reduced epochs on the 19k seqFISH).
Tests whether the headline (no universal SOTA; Tessera best on MERFISH/osmFISH) survives seed variance
and best-config evaluation."""

import sys
from _roots import data_root, external_root, spatial_omics_root

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

EXT = str(external_root())
DR = str(data_root())
for s in ("STAGATE", "SEDR", "GraphST"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.eval.refine import refine_labels  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1, 2]
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


def load_any(path, gt_cols):
    a = ad.read_h5ad(path)
    from coordinate_guard import require_single_coordinate_frame
    require_single_coordinate_frame(a, path)
    coords = np.asarray(a.obsm["spatial"], float)
    col = next(c for c in gt_cols if c in a.obs.columns)
    raw = a.obs[col].to_numpy()
    uniq = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([uniq[v] if _valid(v) else -1 for v in raw], dtype=np.int64)
    X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float64)
    if np.allclose(X, np.round(X)) and X.min() >= 0:
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
    if X.shape[1] > 500:
        X = X[:, np.argsort(X.var(0))[::-1][:3000]]
    lognorm_hvg = X.copy()  # non-negative log-norm (GraphST needs this, not the scaled Z)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    return np.clip((X - mu) / sd, -10, 10), lognorm_hvg, coords, true


def spca(Z, n, seed=0):
    return PCA(max(2, min(n, Z.shape[1] - 1, Z.shape[0] - 1)), random_state=seed).fit_transform(Z)


def gari(true, lab):
    m = true >= 0
    return float(ari(true[m], lab[m]))


def best_config_ari(emb, true, coords, k, seed):
    """Best ARI of this embedding over {KMeans,GMM} × {refine off,on} — the method's own optimum."""
    best = -1.0
    for lab0 in (KMeans(k, n_init=10, random_state=seed).fit_predict(emb),
                 GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=3,
                                 reg_covar=1e-4).fit_predict(emb)):
        for lab in (lab0, refine_labels(lab0, coords, k=6)):
            best = max(best, gari(true, lab))
    return round(best, 4)


def embeddings(Z, lognorm, coords, k, seed):
    out = {"Tessera": fit_predict(spca(Z, min(50, k * 6)).astype(np.float32), coords, k,
           AblationConfig(), train_cfg=TrainConfig(epochs=120, seed=seed, device=str(dev))).embed}
    try:
        import STAGATE_pyG as ST
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False); torch.manual_seed(seed)
        out["STAGATE"] = np.asarray(ST.train_STAGATE(a, n_epochs=600, random_seed=seed,
                                    device=dev, verbose=False).obsm["STAGATE"])
    except Exception as e:
        print("  STAGATE fail", str(e)[:40])
    try:
        import SEDR
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        a.obsm["X_pca"] = spca(Z, 200).astype(np.float32)
        gd = SEDR.graph_construction(a, 6); torch.manual_seed(seed)
        net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
        out["SEDR"] = np.asarray(net.process()[0])
    except Exception as e:
        print("  SEDR fail", str(e)[:40])
    try:
        from GraphST import GraphST as G
        a = ad.AnnData(X=lognorm.astype(np.float32)); a.obsm["spatial"] = coords
        ep = 150 if Z.shape[0] > 10000 else 400
        out["GraphST"] = np.asarray(G.GraphST(a, device=dev, epochs=ep, random_seed=seed)
                                    .train().obsm["emb"])
    except Exception as e:
        print("  GraphST fail", str(e)[:40])
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    out["BANKSY-style"] = spca(np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1), 20, seed)
    out["floor:nonspatial"] = spca(Z, 30, seed)
    out["floor:smoothed"] = spca(nbr, 30, seed)
    return out


results = {}
for name, (path, gt) in DATASETS.items():
    try:
        Z, lognorm, coords, true = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        print(f"\n=== {name}: {Z.shape[0]} cells, {k} classes ===", flush=True)
        per = {}
        for seed in SEEDS:
            for m, emb in embeddings(Z, lognorm, coords, k, seed).items():
                per.setdefault(m, []).append(best_config_ari(emb, true, coords, k, seed))
        rec = {m: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
               for m, v in per.items()}
        results[name] = rec
        for m in sorted(rec, key=lambda x: -rec[x]["mean"]):
            print(f"  {m:>16}: {rec[m]['mean']}±{rec[m]['std']}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:50]}")

print("\n\n===== HARDENED SUMMARY (best-config ARI, mean over seeds) =====")
print(f"{'platform':>26} | {'winner':>16} | Tessera rank")
for name, rec in results.items():
    order = sorted(rec, key=lambda m: -rec[m]["mean"])
    tr = order.index("Tessera") + 1 if "Tessera" in order else None
    print(f"{name:>26} | {order[0]:>16}({rec[order[0]]['mean']}) | {tr}/{len(order)}")
winners = {sorted(rec, key=lambda m: -rec[m]['mean'])[0] for rec in results.values()}
print(f"\ndistinct winners: {winners}")
json.dump(results, open("experiments/hardened_bench.json", "w"), indent=2)
print("wrote experiments/hardened_bench.json")
