"""SOTA EXPANSION — add SpaGCN (the most-cited spatial-domain method) as a 7th real method.

The original panel omitted SpaGCN (jianhuupenn/SpaGCN: graph-conv + spatially-aware DEC clustering). A
referee will ask whether the most popular method overturns "no universal SOTA" or breaks the contiguity
law. We add it. Efficiently: SpaGCN is the ONLY method run here; the other methods' best-config ARIs are
reused from the published expanded_bench.json, and SpaGCN is inserted into each platform's panel to ask:
  (1) Does SpaGCN become a universal winner? (would threaten "no universal SOTA")
  (2) Does SpaGCN's rank track GT contiguity, like the other spatial methods? (method-agnostic law)
  (3) Does adding SpaGCN change the distinct-winner count or the spatial-prior advantage?

SpaGCN outputs labels directly (not an embedding), scored best-of {raw, refined} like SpatialLeiden.
Runs under the numba stub (SpaGCN's louvain init needs it). Writes experiments/spagcn_panel.json.
"""

import sys
from _roots import data_root, external_root, spatial_omics_root

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import anndata as ad  # noqa: E402
import SpaGCN as spg  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, "src")
from tessera_st.eval.refine import refine_labels  # noqa: E402

DR = str(data_root())
EXT = str(external_root())
SUBSAMPLE = 16000
SEEDS = [1]  # SpaGCN.train has no seed arg (global-RNG only); 1 seed, honestly noted

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
    from coordinate_guard import require_single_coordinate_frame
    require_single_coordinate_frame(a, path)
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


def run_spagcn(Z, coords, k, seed):
    """SpaGCN domain detection: spatial adj -> length-scale search -> GCN+DEC.

    We use init="kmeans" with a fixed n_clusters (NOT the default louvain init / resolution search):
    louvain routes through scanpy -> umap -> pynndescent, which needs REAL numba JIT-class machinery
    that the numpy-2.5 numba stub cannot emulate (pynndescent subclasses numba StructRef). kmeans init
    needs only sklearn and bypasses that path entirely. num_pcs is capped to the feature count so the
    internal PCA also works on the low-dim (32-36) protein panels. Seeds are set on the global RNG
    (SpaGCN.train exposes no seed argument)."""
    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    a = ad.AnnData(X=Z.astype(np.float32))
    adj = spg.calculate_adj_matrix(x=list(coords[:, 0]), y=list(coords[:, 1]), histology=False)
    l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    clf = spg.SpaGCN(); clf.set_l(l)
    clf.train(a, adj, num_pcs=min(50, Z.shape[1] - 1), init_spa=True, init="kmeans",
              n_clusters=k, lr=0.05, max_epochs=200, tol=5e-3)
    y_pred, _ = clf.predict()
    return np.asarray(y_pred).astype(int)


pub = {r["platform"]: r for r in json.load(open("experiments/expanded_bench.json"))["rows"]}

# resumable per-platform cache (this box is heavily oversubscribed; never lose completed platforms)
import os  # noqa: E402
PARTIAL = "experiments/spagcn_panel_partial.json"
rows = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else []
done = {r["platform"] for r in rows}
for name, (path, gt) in PLATFORMS.items():
    if name in done:
        print(f"{name}: cached, skipping", flush=True); continue
    try:
        Z, coords, true = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        cont = pub[name]["GT_contiguity"]
        scores = []
        for seed in SEEDS:
            try:
                lab = run_spagcn(Z, coords, k, seed)
                scores.append(max(gari(true, lab), gari(true, refine_labels(lab, coords, k=6))))
            except Exception as e:
                print(f"  {name} SpaGCN seed{seed} fail", str(e)[:60])
        if not scores:
            print(f"{name}: SpaGCN failed all seeds"); continue
        spagcn_ari = float(np.mean(scores))
        # insert SpaGCN into the published core panel for this platform
        panel = {m: pub[name]["means"][m] for m in CORE if m in pub[name]["means"]}
        panel["SpaGCN"] = round(spagcn_ari, 4)
        order = sorted(panel, key=lambda m: -panel[m])
        spagcn_rank = order.index("SpaGCN") + 1
        rows.append({"platform": name, "GT_contiguity": cont, "k": k,
                     "SpaGCN_ari": round(spagcn_ari, 4), "SpaGCN_rank": spagcn_rank,
                     "winner_with_spagcn": order[0], "prev_winner": pub[name]["winner"]})
        json.dump(rows, open(PARTIAL, "w"), indent=2)  # persist immediately (resumable)
        print(f"{name:>27} | contig {cont:.3f} | SpaGCN ARI {spagcn_ari:.3f} #{spagcn_rank}/9 | "
              f"winner now {order[0]} (was {pub[name]['winner']})", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:60]}")

cont = np.array([r["GT_contiguity"] for r in rows])
srank = np.array([r["SpaGCN_rank"] for r in rows])
r_sp = spearmanr(cont, srank)
spagcn_wins = [r["platform"] for r in rows if r["winner_with_spagcn"] == "SpaGCN"]
winners_now = sorted(set(r["winner_with_spagcn"] for r in rows))

print(f"\n===== SpaGCN ADDED (n={len(rows)} platforms, 7th real method) =====")
print(f"SpaGCN wins: {spagcn_wins if spagcn_wins else 'NONE (does not become a universal SOTA)'}")
print(f"Spearman(contiguity, SpaGCN_rank) = {r_sp.correlation:+.3f} (p={r_sp.pvalue:.4f}) "
      f"(negative ⇒ SpaGCN ranks better on contiguous tasks, like the other spatial methods)")
print(f"distinct winners with SpaGCN in panel: {len(winners_now)} {winners_now}")

out = {"rows": rows, "n_platforms": len(rows), "seeds": SEEDS,
       "spagcn_wins": spagcn_wins, "n_spagcn_wins": len(spagcn_wins),
       "spearman_contiguity_vs_spagcn_rank": round(float(r_sp.correlation), 3),
       "p_contiguity_vs_spagcn_rank": round(float(r_sp.pvalue), 4),
       "distinct_winners_with_spagcn": winners_now, "n_distinct_winners": len(winners_now)}
json.dump(out, open("experiments/spagcn_panel.json", "w"), indent=2)
print("\nwrote experiments/spagcn_panel.json")

# verify hook: SpaGCN must actually RUN on enough platforms (guard against a vacuous 0-platform "pass")
# AND must NOT become a universal winner (no-universal-SOTA must survive a 7th method).
assert len(rows) >= 6, f"SpaGCN only ran on {len(rows)} platforms — not enough to conclude anything"
assert len(spagcn_wins) <= 2, f"SpaGCN won {len(spagcn_wins)} platforms — re-examine no-universal-SOTA"
print(f"SpaGCN CHECK PASS — ran on {len(rows)} platforms; a 7th major SOTA does not become universal; "
      "no-universal-SOTA survives.")
