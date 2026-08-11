"""GT-FREE proxy SEARCH — the naive proxies failed; find a principled one (or report the honest limit).

First pass (gtfree_proxy.py) showed BOTH naive label-free proxies not only fail to track GT contiguity
but INVERT against spatial-prior advantage:
    Moran's I of expression PCs vs advantage   = -0.34   (GT-based reference = +0.72)
    unsupervised kNN same-cluster fraction      = -0.42
Mechanism: raw-expression spatial autocorrelation is dominated by platform/cell-type structure
(Visium spots are smooth; single-cell imaging is noisy), which runs OPPOSITE to where spatial priors
help — imaging anatomical-domain tasks are single-cell-noisy yet contiguous, so smoothing gains most.

So the right label-free signal is not "is expression already smooth" but "does SPATIAL SMOOTHING reveal
structure that raw expression clustering misses". This script computes several principled candidates in
one data pass and reports each one's Spearman vs (a) GT contiguity and (b) spatial-prior advantage,
so we can adopt the best — or document that no simple proxy recovers the trend (a real limitation).

Candidates (all label-free), per platform, KMeans over a fixed k grid x seeds:
  raw_coh     : spatial kNN same-cluster fraction of clusters from RAW expression PCs (= old unsup)
  smooth_coh  : same, but clusters from SPATIALLY-SMOOTHED expression PCs
  coh_gain    : smooth_coh - raw_coh            (extra spatial coherence smoothing buys)
  partition_shift : 1 - ARI(raw clusters, smoothed clusters)  (how much smoothing changes the answer)
  sil_gain    : silhouette(smoothed) - silhouette(raw)  (does smoothing improve cluster separation)
  smoothing_score : smooth_coh * partition_shift  (smoothing both changes the partition AND makes it
                    coherent — the hypothesised signature of "spatial prior adds real domain structure")
"""

import json

import numpy as np
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from _roots import data_root, external_root, spatial_omics_root

DR = str(data_root())
EXT = str(external_root())
PLATFORMS = {
    "DLPFC(Visium,layer)": (f"{DR}/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad",
                            ["layer_label", "Region", "ground_truth"]),
    "seqFISH(embryo,ctype)": (f"{DR}/raw/squidpy/seqfish.h5ad", ["celltype_mapped_refined"]),
    "MERFISH(hypo,domain)": (f"{DR}/baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad",
                             ["domain"]),
    "STARmap(cortex,region)": (f"{DR}/processed/starmap_mouse_vcortex_wang2018/anndata.h5ad",
                               ["layer_label", "Region", "ground_truth"]),
    "osmFISH(cortex,Region)": (f"{EXT}/../baselines/CellNiche-original/data/osmFISH_SScortex.h5ad",
                               ["Region"]),
    "MIBI-TOF(protein,Cluster)": (f"{DR}/raw/squidpy/mibitof.h5ad", ["Cluster"]),
    "BRCA(Visium,tumor)": (f"{DR}/processed/brca1_visium_10x/anndata.h5ad",
                           ["ground_truth", "annot_type", "fine_annot_type"]),
    "IMC(breast,ctype)": (f"{DR}/processed/niche_imc_breast/anndata.h5ad",
                          ["ground_truth", "Region", "cell_type"]),
    "openST(HNSCC,annot)": (f"{DR}/processed/openst_hnscc_sub15k/anndata.h5ad",
                            ["ground_truth", "annotation", "cell_class"]),
    "SlideseqV2(hippo,Region)": (f"{DR}/processed/slideseqv2_hippo/anndata.h5ad",
                                 ["Region", "ground_truth"]),
    "CODEX(spleen,niche)": (f"{DR}/processed/codex_spleen_goltsev2018/anndata.h5ad",
                            ["niche", "cell_type"]),
}
SUBSAMPLE = 16000
KGRID = [8, 15]
SEEDS = [1, 2]
SIL_SUB = 4000  # silhouette is O(n^2); subsample for it only


def _valid(v):
    return str(v).strip().lower() not in {"", "nan", "none", "na", "unknown"}


def load_any(path, gt_cols):
    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], float)
    col = next(c for c in gt_cols if c in a.obs.columns)
    raw = a.obs[col].to_numpy()
    X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float64)
    n = X.shape[0]
    if n > SUBSAMPLE:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, SUBSAMPLE, replace=False))
        X, coords, raw = X[keep], coords[keep], raw[keep]
    codes = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([codes[v] if _valid(v) else -1 for v in raw], dtype=np.int64)
    if np.allclose(X, np.round(X)) and X.min() >= 0:
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
    if X.shape[1] > 500:
        keepg = np.argsort(X.var(0))[::-1][:3000]
        X = X[:, keepg]
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    return np.clip((X - mu) / sd, -10, 10), coords, true


def spca(Z, n, seed=0):
    return PCA(max(2, min(n, Z.shape[1] - 1, Z.shape[0] - 1)), random_state=seed).fit_transform(Z)


def coh(lab, idx):
    return float((lab[idx] == lab[:, None]).mean())


rows = []
adv_pub = {r["platform"]: r for r in json.load(open("experiments/expanded_bench.json"))["rows"]}
for name, (path, gt) in PLATFORMS.items():
    Z, coords, true = load_any(path, gt)
    idx = NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]
    m = true >= 0
    if m.sum() < len(true):
        cv, tv = coords[m], true[m]
        idxm = NearestNeighbors(n_neighbors=7).fit(cv).kneighbors(cv)[1][:, 1:]
        gtc = float((tv[idxm] == tv[:, None]).mean())
    else:
        gtc = float((true[idx] == true[:, None]).mean())
    Zs = Z[idx].mean(1)  # spatial-neighbour smoothing
    P_raw_full, P_s_full = spca(Z, 30), spca(Zs, 30)
    cand = {kk: {} for kk in ("raw_coh", "smooth_coh", "coh_gain", "partition_shift",
                              "sil_gain", "smoothing_score")}
    acc = {k2: [] for k2 in cand}
    sub = np.sort(np.random.RandomState(0).choice(Z.shape[0], min(SIL_SUB, Z.shape[0]), replace=False))
    for seed in SEEDS:
        P_raw, P_s = spca(Z, 30, seed), spca(Zs, 30, seed)
        for kk in KGRID:
            lab_raw = KMeans(kk, n_init=3, random_state=seed).fit_predict(P_raw)
            lab_s = KMeans(kk, n_init=3, random_state=seed).fit_predict(P_s)
            rc, sc = coh(lab_raw, idx), coh(lab_s, idx)
            shift = 1.0 - ari(lab_raw, lab_s)
            sg = (silhouette_score(P_s[sub], lab_s[sub]) - silhouette_score(P_raw[sub], lab_raw[sub]))
            acc["raw_coh"].append(rc); acc["smooth_coh"].append(sc)
            acc["coh_gain"].append(sc - rc); acc["partition_shift"].append(shift)
            acc["sil_gain"].append(float(sg)); acc["smoothing_score"].append(sc * shift)
    rec = {"platform": name, "GT_contiguity": round(gtc, 3),
           "spatial_advantage": adv_pub[name]["spatial_advantage"]}
    for k2, v in acc.items():
        rec[k2] = round(float(np.mean(v)), 3)
    rows.append(rec)
    print(f"{name:>27} | GTc {gtc:.3f} | adv {rec['spatial_advantage']:+.3f} | "
          + " ".join(f"{k2} {rec[k2]:+.3f}" for k2 in ("smooth_coh", "coh_gain", "partition_shift",
                                                       "sil_gain", "smoothing_score")), flush=True)

adv = np.array([r["spatial_advantage"] for r in rows])
gtc = np.array([r["GT_contiguity"] for r in rows])
print("\n===== candidate correlations (n=%d) =====" % len(rows))
print(f"{'candidate':>16} | rho_vs_GTcontig (p) | rho_vs_advantage (p)   [GT-based adv ref = +0.724]")
results = {}
for k2 in ("raw_coh", "smooth_coh", "coh_gain", "partition_shift", "sil_gain", "smoothing_score"):
    v = np.array([r[k2] for r in rows])
    rg = spearmanr(v, gtc); ra = spearmanr(v, adv)
    results[k2] = {"rho_vs_gtcontig": round(float(rg.correlation), 3), "p_vs_gtcontig": round(float(rg.pvalue), 4),
                   "rho_vs_advantage": round(float(ra.correlation), 3), "p_vs_advantage": round(float(ra.pvalue), 4)}
    print(f"{k2:>16} | {rg.correlation:+.3f} ({rg.pvalue:.3f})     | {ra.correlation:+.3f} ({ra.pvalue:.3f})")

json.dump({"rows": rows, "candidates": results, "k_grid": KGRID, "seeds": SEEDS},
          open("experiments/gtfree_proxy_search.json", "w"), indent=2)
print("\nwrote experiments/gtfree_proxy_search.json")
best = max(results, key=lambda c: results[c]["rho_vs_advantage"])
print(f"BEST proxy for advantage: {best} (rho={results[best]['rho_vs_advantage']}, "
      f"p={results[best]['p_vs_advantage']})")
