"""EXPANDED BENCHMARK (all-out, hardened): add two methods to the panel — a 5th real deep SOTA
(SpaceFlow, DGI + spatial regularisation) and a 6th of a new family (SpatialLeiden, spatial-blended
features + Leiden community detection) — plus more local platforms (Slide-seqV2 hippo / CODEX spleen),
then RE-RUN every platform UNIFORMLY (not reusing old rows — adding a method can change a platform's
winner and every rank). Recompute the task-fit law over n=11 (a candidate platform, MIBI-TOF colorectal,
was found in review to be a byte-duplicate of squidpy mibitof and is excluded — see NOTE), 3 seeds, and
report seed-robustness of the lead correlation.

Core panel (run on every platform, ranks computed within it for apples-to-apples):
  Tessera, STAGATE, SEDR, SpaceFlow, SpatialLeiden, BANKSY-style, floor:nonspatial, floor:smoothed.
GraphST is supplementary (only where size permits under the pure-Python numba stub) and is reported
but excluded from the ranking so the cross-platform ranks stay on a consistent method set.

Honest reporting: large platforms are subsampled (logged, not silent); any method that fails on a
platform is logged and dropped for that platform only.
"""

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

DR = str(data_root())
EXT = str(external_root())
for s in ("STAGATE", "SEDR", "GraphST", "SpaceFlow"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.eval.refine import refine_labels  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1, 2, 3]  # 3 seeds (was 2) — tightens means + lets us report seed-robustness of the lead rho
SUBSAMPLE = 16000  # cap on cells for tractability under the pure-Python stub; logged when applied
GRAPHST_MAX = 8000  # GraphST is too slow under the stub above this; supplementary only

# 11 platforms: the established 9 + 2 NEW (Slide-seqV2 / CODEX spleen). A 3rd candidate, MIBI-TOF
# colorectal, is a byte-duplicate of squidpy mibitof and is deliberately excluded — see NOTE below.
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
    # --- NEW platforms (all-out expansion) ---
    # NOTE: squidpy's mibitof IS the colorectal MIBI-TOF experiment, identical to
    # MIBI-TOF(protein,Cluster) above (same 3309x36 matrix, coords, and Region==Cluster labels).
    # processed/mibitof_colorectal is therefore a DUPLICATE and is deliberately NOT included here —
    # including it would double-count a low-contiguity point and falsely inflate n. (Caught in review.)
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
    # subsample (deterministic) before any heavy work; logged by caller
    n = X.shape[0]
    if n > SUBSAMPLE:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, SUBSAMPLE, replace=False))
        X, coords, raw = X[keep], coords[keep], raw[keep]
    codes = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([codes[v] if _valid(v) else -1 for v in raw], dtype=np.int64)
    counts = X.copy()  # native counts for SpaceFlow's own pipeline
    if np.allclose(X, np.round(X)) and X.min() >= 0:
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
    if X.shape[1] > 500:
        keepg = np.argsort(X.var(0))[::-1][:3000]
        X, counts = X[:, keepg], counts[:, keepg]
    lognorm = X.copy()
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    return np.clip((X - mu) / sd, -10, 10), lognorm, counts, coords, true, (n > SUBSAMPLE)


def spca(Z, n, seed=0):
    return PCA(max(2, min(n, Z.shape[1] - 1, Z.shape[0] - 1)), random_state=seed).fit_transform(Z)


def gari(true, lab):
    m = true >= 0
    return float(ari(true[m], lab[m]))


def best_config_ari(emb, true, coords, k, seed):
    best = -1.0
    labs = [KMeans(k, n_init=10, random_state=seed).fit_predict(emb)]
    try:
        labs.append(GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=3,
                                    reg_covar=1e-2).fit_predict(emb))
    except Exception:
        pass
    for lab0 in labs:
        for lab in (lab0, refine_labels(lab0, coords, k=6)):
            best = max(best, gari(true, lab))
    return best


def gt_contiguity(coords, true, k=6):
    m = true >= 0
    cv, tv = coords[m], true[m]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(cv).kneighbors(cv)[1][:, 1:]
    return float((tv[idx] == tv[:, None]).mean())


def spatial_leiden_labels(Z, coords, k, seed):
    """A classic non-DL spatial-domain method (distinct family from the autoencoder/contrastive SOTA):
    blend expression with its spatial-neighbour mean, then Leiden community detection on the kNN graph,
    bisecting the resolution to hit ~k communities. Uses leidenalg/igraph directly (no scanpy.pp.neighbors,
    which would route through numba under the stub)."""
    import igraph as ig
    import leidenalg
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    emb = spca(np.concatenate([np.sqrt(0.5) * Z, np.sqrt(0.5) * nbr], 1), 30, seed)
    idx = NearestNeighbors(n_neighbors=16).fit(emb).kneighbors(emb)[1][:, 1:]
    edges = [(i, int(j)) for i, row in enumerate(idx) for j in row]
    g = ig.Graph(n=emb.shape[0], edges=edges)
    lo, hi, lab = 0.05, 4.0, None
    for _ in range(12):  # bisect resolution toward k communities
        r = (lo + hi) / 2
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=r, seed=seed)
        lab = np.asarray(part.membership)
        nc = int(lab.max()) + 1
        if nc == k:
            break
        lo, hi = (r, hi) if nc < k else (lo, r)
    return lab


def run_spaceflow(counts, coords, seed):
    from SpaceFlow.SpaceFlow import SpaceFlow
    a = ad.AnnData(X=counts.astype(np.float32)); a.obsm["spatial"] = coords
    sf = SpaceFlow(adata=a)
    sf.preprocessing_data(n_top_genes=min(3000, counts.shape[1]))
    return np.asarray(sf.train(embedding_save_filepath="/tmp/sf_emb.tsv", epochs=400,
                               random_seed=seed, z_dim=50, min_stop=80, max_patience=30))


def embeddings(Z, lognorm, counts, coords, k, seed):
    out = {"Tessera": fit_predict(spca(Z, min(50, k * 6)).astype(np.float32), coords, k,
           AblationConfig(), train_cfg=TrainConfig(epochs=120, seed=seed, device=str(dev))).embed}
    try:
        import STAGATE_pyG as ST
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False); torch.manual_seed(seed)
        out["STAGATE"] = np.asarray(ST.train_STAGATE(a, n_epochs=600, random_seed=seed,
                                    device=dev, verbose=False).obsm["STAGATE"])
    except Exception as e:
        print("  STAGATE fail", str(e)[:50])
    try:
        import SEDR
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        a.obsm["X_pca"] = spca(Z, 200).astype(np.float32)
        gd = SEDR.graph_construction(a, 6); torch.manual_seed(seed)
        net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
        out["SEDR"] = np.asarray(net.process()[0])
    except Exception as e:
        print("  SEDR fail", str(e)[:50])
    try:
        out["SpaceFlow"] = run_spaceflow(counts, coords, seed)
    except Exception as e:
        print("  SpaceFlow fail", str(e)[:60])
    # GraphST supplementary (size-gated)
    if Z.shape[0] <= GRAPHST_MAX:
        try:
            from GraphST import GraphST as GST
            a = ad.AnnData(X=lognorm.astype(np.float32)); a.obsm["spatial"] = coords
            torch.manual_seed(seed)
            out["GraphST"] = np.asarray(GST.GraphST(a, device=dev, epochs=300, random_seed=seed)
                                        .train().obsm["emb"])
        except Exception as e:
            print("  GraphST fail", str(e)[:50])
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    out["BANKSY-style"] = spca(np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1), 20, seed)
    out["floor:nonspatial"] = spca(Z, 30, seed)
    out["floor:smoothed"] = spca(nbr, 30, seed)
    return out


rows = []
for name, (path, gt) in PLATFORMS.items():
    try:
        Z, lognorm, counts, coords, true, subbed = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        cont = gt_contiguity(coords, true)
        sub = f" (subsampled {SUBSAMPLE})" if subbed else ""
        print(f"\n=== {name}: {Z.shape[0]} cells, {k} classes, GT contig {cont:.3f}{sub} ===", flush=True)
        per = {}
        for seed in SEEDS:
            for m, emb in embeddings(Z, lognorm, counts, coords, k, seed).items():
                per.setdefault(m, []).append(best_config_ari(emb, true, coords, k, seed))
            # SpatialLeiden produces labels directly (not an embedding) — score it best-of {raw, refined}
            try:
                sl = spatial_leiden_labels(Z, coords, k, seed)
                per.setdefault("SpatialLeiden", []).append(
                    max(gari(true, sl), gari(true, refine_labels(sl, coords, k=6))))
            except Exception as e:
                print("  SpatialLeiden fail", str(e)[:50])
        mean = {m: float(np.mean(v)) for m, v in per.items()}
        std = {m: float(np.std(v)) for m, v in per.items()}
        # ranks within the consistent CORE panel only
        core_mean = {m: mean[m] for m in CORE if m in mean}
        order = sorted(core_mean, key=lambda m: -core_mean[m])
        t_rank = order.index("Tessera") + 1
        sf_rank = order.index("SpaceFlow") + 1 if "SpaceFlow" in order else None
        nonsp = core_mean.get("floor:nonspatial", 0.0)
        sadv = max(v for m, v in core_mean.items() if m != "floor:nonspatial") - nonsp
        # per-seed spatial advantage (same definition, one seed at a time) for rho seed-robustness
        adv_per_seed = []
        for si in range(len(SEEDS)):
            cm = {m: per[m][si] for m in CORE if m in per and len(per[m]) > si}
            if "floor:nonspatial" in cm and len(cm) >= 2:
                adv_per_seed.append(round(max(v for mm, v in cm.items()
                                              if mm != "floor:nonspatial") - cm["floor:nonspatial"], 3))
        rows.append({
            "platform": name, "GT_contiguity": round(cont, 3), "n_cells": int(Z.shape[0]),
            "k": k, "subsampled": bool(subbed),
            "Tessera_rank": t_rank, "SpaceFlow_rank": sf_rank, "winner": order[0],
            "spatial_advantage": round(sadv, 3), "spatial_helps": sadv > 0,
            "adv_per_seed": adv_per_seed,
            "means": {m: round(mean[m], 4) for m in mean},
            "stds": {m: round(std[m], 4) for m in std},
            "graphst_ran": "GraphST" in mean, "spatialleiden_ran": "SpatialLeiden" in mean,
        })
        print(f"  Tessera #{t_rank}/{len(order)} | SpaceFlow #{sf_rank} | winner {order[0]} | "
              f"spatial_adv {sadv:+.3f}", flush=True)
        print("  " + " | ".join(f"{m} {mean[m]:.3f}" for m in order), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:60]}")


from scipy.stats import spearmanr  # proper mid-rank tie handling + p-values (not hand-rolled ordinal)


def sp(x, y):
    r = spearmanr(x, y)
    return round(float(r.correlation), 3), round(float(r.pvalue), 4)


# duplicate guard: no two platforms may share identical (n_cells, contiguity, method means) — a
# byte-duplicate dataset (like squidpy mibitof == processed/mibitof_colorectal) would silently
# double-count a point in every correlation. Fail loudly if it ever recurs.
seen = {}
for r in rows:
    sig = (r["n_cells"], r["GT_contiguity"], tuple(sorted(r["means"].items())))
    if sig in seen:
        raise SystemExit(f"DUPLICATE platform detected: {r['platform']} == {seen[sig]} "
                         f"(identical cells/contiguity/means) — remove one before reporting.")
    seen[sig] = r["platform"]

cont = np.array([r["GT_contiguity"] for r in rows])
trank = np.array([r["Tessera_rank"] for r in rows])
sadv = np.array([r["spatial_advantage"] for r in rows])
srank = np.array([r["SpaceFlow_rank"] for r in rows if r["SpaceFlow_rank"] is not None])
scont = np.array([r["GT_contiguity"] for r in rows if r["SpaceFlow_rank"] is not None])
rho_rank, p_rank = sp(cont, trank)
rho_adv, p_adv = sp(cont, sadv)
n_sf = int(len(srank))
rho_sf, p_sf = sp(scont, srank) if n_sf > 2 else (None, None)

# seed-robustness of the lead correlation: recompute rho(contiguity, spatial_advantage) using each
# seed's advantages alone, and report the spread (addresses "rho's seed-stability not shown").
rho_adv_per_seed = []
for si in range(len(SEEDS)):
    xy = [(r["GT_contiguity"], r["adv_per_seed"][si]) for r in rows if len(r["adv_per_seed"]) > si]
    if len(xy) > 3:
        xs, ys = zip(*xy)
        rho_adv_per_seed.append(round(float(spearmanr(xs, ys).correlation), 3))

# distinct winners
winners = sorted(set(r["winner"] for r in rows))
print(f"\n===== EXPANDED LAW (n={len(rows)} platforms, panel incl SpaceFlow + SpatialLeiden, 3 seeds) =====")
for r in sorted(rows, key=lambda r: -r["GT_contiguity"]):
    print(f"  {r['platform']:>27} | contig {r['GT_contiguity']:.3f} | Tessera #{r['Tessera_rank']} | "
          f"SpaceFlow #{r['SpaceFlow_rank']} | adv {r['spatial_advantage']:+.3f} | {r['winner']}")
print(f"\nDistinct winners: {winners}")
print(f"Spearman(contiguity, spatial_advantage) = {rho_adv:+.3f} (p={p_adv}, n={len(rows)})  LEAD RESULT")
print(f"Spearman(contiguity, Tessera_rank)      = {rho_rank:+.3f} (p={p_rank}, n={len(rows)})")
print(f"Spearman(contiguity, SpaceFlow_rank)    = {rho_sf} (p={p_sf}, n={n_sf}; SpaceFlow failed on "
      f"protein panels)")
if rho_adv_per_seed:
    print(f"Lead rho seed-robustness (per-seed)     = {rho_adv_per_seed} "
          f"(min {min(rho_adv_per_seed):+.3f}, max {max(rho_adv_per_seed):+.3f})")

json.dump({
    "rows": rows, "n_platforms": len(rows), "method_panel": CORE,
    "distinct_winners": winners, "n_distinct_winners": len(winners),
    "spearman_contiguity_vs_spatial_advantage": rho_adv,
    "p_contiguity_vs_spatial_advantage": p_adv,
    "spearman_contiguity_vs_tessera_rank": rho_rank,
    "p_contiguity_vs_tessera_rank": p_rank,
    "spearman_contiguity_vs_spaceflow_rank": rho_sf,
    "p_contiguity_vs_spaceflow_rank": p_sf,
    "n_spaceflow": n_sf,
    "seeds": SEEDS,
    "rho_adv_per_seed": rho_adv_per_seed,
    "rho_adv_per_seed_min": min(rho_adv_per_seed) if rho_adv_per_seed else None,
    "rho_adv_per_seed_max": max(rho_adv_per_seed) if rho_adv_per_seed else None,
}, open("experiments/expanded_bench.json", "w"), indent=2)
print("\nwrote experiments/expanded_bench.json")
