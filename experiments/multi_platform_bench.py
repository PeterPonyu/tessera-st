"""MULTI-PLATFORM benchmark: does method ranking / spatial-prior value / backend-confound direction
hold across 6 platforms? Each is a different technology + tissue + GT type. Per platform run
Tessera + STAGATE/SEDR/BANKSY + 2 floors, KMeans & GMM, report: best method, whether the spatial
prior helps (spatial methods vs non-spatial floor), and the backend-confound direction. If these
differ across platforms, single-dataset benchmarks are proven non-generalisable. GraphST skipped
(too slow under stub). 1 seed — discovery-scale."""

import sys

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

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
DR = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/data"
for s in ("STAGATE", "SEDR"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1

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
    s = str(v).strip().lower()
    return s not in {"", "nan", "none", "na", "unknown"}


def load_any(path, gt_cols):
    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], float)
    col = next((c for c in gt_cols if c in a.obs.columns), None)
    raw = a.obs[col].to_numpy()
    uniq = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([uniq[v] if _valid(v) else -1 for v in raw], dtype=np.int64)
    X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float64)
    if np.allclose(X, np.round(X)) and X.min() >= 0:  # raw counts -> CP10k+log1p
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
    if X.shape[1] > 500:
        X = X[:, np.argsort(X.var(0))[::-1][:3000]]
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = np.clip((X - mu) / sd, -10, 10)
    return Z, coords, true


def gari(true, lab):
    m = true >= 0
    return round(float(ari(true[m], lab[m])), 4)


def cluster(emb, k, backend):
    if backend == "kmeans":
        return KMeans(k, n_init=10, random_state=SEED).fit_predict(emb)
    return GaussianMixture(k, covariance_type="tied", random_state=SEED, n_init=3,
                           reg_covar=1e-4).fit_predict(emb)


def spca(Z, n, seed=0):
    nc = max(2, min(n, Z.shape[1] - 1, Z.shape[0] - 1))
    return PCA(nc, random_state=seed).fit_transform(Z)


def embeddings(Z, coords, k):
    out = {}
    out["Tessera"] = fit_predict(spca(Z, min(50, k * 6)).astype(np.float32),
                                 coords, k, AblationConfig(),
                                 train_cfg=TrainConfig(epochs=120, seed=SEED, device=str(dev))).embed
    try:
        import STAGATE_pyG as ST
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False); torch.manual_seed(SEED)
        o = ST.train_STAGATE(a, n_epochs=600, random_seed=SEED, device=dev, verbose=False)
        out["STAGATE"] = np.asarray(o.obsm["STAGATE"])
    except Exception as e:
        print("  STAGATE fail", str(e)[:50])
    try:
        import SEDR
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        a.obsm["X_pca"] = spca(Z, 200).astype(np.float32)
        gd = SEDR.graph_construction(a, 6); torch.manual_seed(SEED)
        net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
        out["SEDR"] = np.asarray(net.process()[0])
    except Exception as e:
        print("  SEDR fail", str(e)[:50])
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    out["BANKSY-style"] = spca(np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1), 20, SEED)
    out["floor:nonspatial"] = spca(Z, 30, SEED)
    out["floor:smoothed"] = spca(nbr, 30, SEED)
    return out


summary = {}
for name, (path, gt) in DATASETS.items():
    try:
        Z, coords, true = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        print(f"\n=== {name}: {Z.shape[0]} cells, {Z.shape[1]} feat, {k} classes ===", flush=True)
        embs = embeddings(Z, coords, k)
        rec = {}
        for m, emb in embs.items():
            km = gari(true, cluster(emb, k, "kmeans"))
            gm = gari(true, cluster(emb, k, "gmm"))
            rec[m] = {"kmeans": km, "gmm": gm}
            print(f"  {m:>16}: KM {km}  GMM {gm}", flush=True)
        best = max(rec, key=lambda m: rec[m]["gmm"])
        spatial = max(rec[m]["gmm"] for m in rec if m != "floor:nonspatial")
        nonsp = rec["floor:nonspatial"]["gmm"]
        summary[name] = {"per_method": rec, "best_gmm": best,
                         "spatial_prior_helps": spatial > nonsp,
                         "confound_dir_STAGATE": round(
                             rec.get("STAGATE", {}).get("gmm", 0) - rec.get("STAGATE", {}).get("kmeans", 0), 3)}
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:60]}")

print("\n\n===== CROSS-PLATFORM SUMMARY =====")
print(f"{'platform':>26} | {'best method (GMM)':>18} | spatial>nonspatial? | STAGATE Δbackend")
print("-" * 92)
for name, s in summary.items():
    print(f"{name:>26} | {s['best_gmm']:>18} | {str(s['spatial_prior_helps']):>18} | {s['confound_dir_STAGATE']:>+.3f}")
bests = {s["best_gmm"] for s in summary.values()}
helps = {s["spatial_prior_helps"] for s in summary.values()}
print(f"\ndistinct best methods across platforms: {bests}")
print(f"spatial prior helps on some but not others: {len(helps) > 1}")
json.dump(summary, open("experiments/multi_platform_bench.json", "w"), indent=2)
print("wrote experiments/multi_platform_bench.json")
