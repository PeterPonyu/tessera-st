"""SpaGCN with its GENUINE NATIVE pipeline (louvain init + resolution search), run in a dedicated env.

The main panel (spagcn_panel.py) had to run SpaGCN with a kmeans initialisation because numpy 2.5 blocks
real numba (SpaGCN's louvain/resolution path routes through scanpy->umap->pynndescent, which needs
real-numba JIT classes). That understated SpaGCN. Here we run it in a dedicated conda env
(`tessera-spagcn`: python 3.10, numpy 1.26, real numba) so SpaGCN uses its ACTUAL recommended pipeline:
spatial adjacency -> length-scale search -> resolution search (louvain) -> GCN+DEC. This is self-contained
(no tessera_st import, no numba stub) so it runs cleanly in the native env.

Run:  conda run -n tessera-spagcn python experiments/spagcn_native.py
Inserts SpaGCN's native ARI into the published 8-method panel (expanded_bench.json) and asks the same
question as spagcn_panel.py: does the most-cited method, run natively, become a universal winner?
Resumable per-platform cache. Writes experiments/spagcn_native.json.
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import anndata as ad
import SpaGCN as spg
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

DR = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/data"
EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
SUBSAMPLE = 16000
SEEDS = [1]

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
CORE = ["Tessera", "STAGATE", "SEDR", "SpaceFlow", "SpatialLeiden", "BANKSY-style",
        "floor:nonspatial", "floor:smoothed"]


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


def gari(true, lab):
    m = true >= 0
    return float(ari(true[m], lab[m]))


def refine_labels(lab, coords, k=6):
    """Majority vote over spatial kNN (the standard SpaGCN/STAGATE post-step), inline (no tessera_st)."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)[1][:, 1:]
    out = lab.copy()
    for i in range(len(lab)):
        vals, cnts = np.unique(lab[idx[i]], return_counts=True)
        if cnts.max() > k // 2:
            out[i] = vals[cnts.argmax()]
    return out


def run_spagcn_native(Z, coords, k, seed):
    """SpaGCN's recommended native pipeline: adjacency -> search_l -> search_res (louvain) -> GCN+DEC."""
    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    a = ad.AnnData(X=Z.astype(np.float32))
    adj = spg.calculate_adj_matrix(x=list(coords[:, 0]), y=list(coords[:, 1]), histology=False)
    l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    try:
        res = spg.search_res(a, adj, l, target_num=k, start=0.7, step=0.1, tol=5e-3, lr=0.05,
                             max_epochs=20, r_seed=seed, t_seed=seed, n_seed=seed)
    except TypeError:
        res = spg.search_res(a, adj, l, target_num=k)
    clf = spg.SpaGCN(); clf.set_l(l)
    clf.train(a, adj, num_pcs=min(50, Z.shape[1] - 1), init_spa=True, init="louvain", res=res,
              tol=5e-3, lr=0.05, max_epochs=200)
    return np.asarray(clf.predict()[0]).astype(int)


pub = {r["platform"]: r for r in json.load(open("experiments/expanded_bench.json"))["rows"]}
PARTIAL = "experiments/spagcn_native_partial.json"
rows = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else []
done = {r["platform"] for r in rows}

for name, (path, gt) in PLATFORMS.items():
    if name in done:
        print(f"{name}: cached", flush=True); continue
    try:
        Z, coords, true = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        cont = pub[name]["GT_contiguity"]
        scores = []
        for seed in SEEDS:
            try:
                lab = run_spagcn_native(Z, coords, k, seed)
                scores.append(max(gari(true, lab), gari(true, refine_labels(lab, coords))))
            except Exception as e:
                print(f"  {name} seed{seed} fail", str(e)[:70])
        if not scores:
            print(f"{name}: all seeds failed"); continue
        sp_ari = float(np.mean(scores))
        panel = {m: pub[name]["means"][m] for m in CORE if m in pub[name]["means"]}
        panel["SpaGCN-native"] = round(sp_ari, 4)
        order = sorted(panel, key=lambda m: -panel[m])
        rows.append({"platform": name, "GT_contiguity": cont, "k": k,
                     "SpaGCN_native_ari": round(sp_ari, 4),
                     "SpaGCN_native_rank": order.index("SpaGCN-native") + 1,
                     "winner_with_spagcn": order[0], "prev_winner": pub[name]["winner"]})
        json.dump(rows, open(PARTIAL, "w"), indent=2)
        print(f"{name:>27} | contig {cont:.3f} | SpaGCN-native ARI {sp_ari:.3f} "
              f"#{order.index('SpaGCN-native')+1}/9 | winner now {order[0]}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:60]}")

cont = np.array([r["GT_contiguity"] for r in rows])
srank = np.array([r["SpaGCN_native_rank"] for r in rows])
r_sp = spearmanr(cont, srank)
wins = [r["platform"] for r in rows if r["winner_with_spagcn"] == "SpaGCN-native"]
winners_now = sorted(set(r["winner_with_spagcn"] for r in rows))
print(f"\n===== SpaGCN NATIVE (n={len(rows)}, real louvain pipeline) =====")
print(f"SpaGCN-native wins: {wins if wins else 'NONE'}")
print(f"Spearman(contiguity, SpaGCN-native rank) = {r_sp.correlation:+.3f} (p={r_sp.pvalue:.4f})")
print(f"distinct winners with SpaGCN-native: {len(winners_now)} {winners_now}")
json.dump({"rows": rows, "n_platforms": len(rows), "seeds": SEEDS, "pipeline": "native louvain",
           "spagcn_native_wins": wins, "n_spagcn_native_wins": len(wins),
           "spearman_contiguity_vs_spagcn_native_rank": round(float(r_sp.correlation), 3),
           "p_value": round(float(r_sp.pvalue), 4),
           "distinct_winners": winners_now, "n_distinct_winners": len(winners_now)},
          open("experiments/spagcn_native.json", "w"), indent=2)
print("\nwrote experiments/spagcn_native.json")
assert len(rows) >= 6, "too few platforms"
assert len(wins) <= 2, f"SpaGCN-native won {len(wins)} — re-examine no-universal-SOTA"
print("SpaGCN-NATIVE CHECK PASS — even with its genuine louvain pipeline, SpaGCN is not a universal SOTA.")
