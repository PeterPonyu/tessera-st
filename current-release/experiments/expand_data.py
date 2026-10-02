"""DATA EXPANSION — test the §7 law on platforms the original sweep MISSED.

The §7 manuscript claimed local independent domain-GT platforms were "exhausted". A re-audit of the data
tree found that claim overstated: three independent cancer spatial-omics datasets with biological
(cell-type) GT were present and unused — st_COAD, st_LIHC, st_OV (the st_impute cohort; CESC/NSCLC/PRAD
were correctly excludable as they carry only a technical `segmentation_method` label, and Xenium/COAD-
Visium lack any domain GT). They span a wide contiguity range (LIHC 0.24 → OV 0.79), exactly the
low/mid-contiguity regime, so they are an honest test of whether the law survives more data.

This runs the SAME core panel and protocol as expanded_bench.py (best-of {KMeans, GMM-tied} x {refine
off,on}, 3 seeds, 16k subsample), computes each new platform's GT contiguity and spatial-prior advantage,
then re-derives the lead correlation over n=14 (the published 11 from expanded_bench.json + these 3).
Writes a separate audited candidate ledger; negative and nonsignificant results are retained.
"""

import sys
from _roots import data_root, external_root, spatial_omics_root

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402
from protocol_guard import (PROTOCOL_ID, output_path, refinement_candidates, record_seed,
    complete_seed_summary, require_complete_platforms, require_candidate_rows,
    preflight_coordinate_frames, fingerprint, require_cached_input)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

DR = str(data_root())
EXT = str(external_root())
for s in ("STAGATE", "SEDR", "GraphST", "SpaceFlow"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.eval.refine import refine_labels  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1, 2, 3]
SUBSAMPLE = 16000
GRAPHST_MAX = 8000

# NEW independent platforms (cancer spatial-omics, biological cell-type GT) — the st_impute cohort.
NEW_PLATFORMS = {
    "st_COAD(cancer,ctype)": (f"{DR}/baselines/st_impute_ref/processed_data/st_COAD_test.h5ad",
                              ["annotation"]),
    "st_LIHC(cancer,ctype)": (f"{DR}/baselines/st_impute_ref/processed_data/st_LIHC_test.h5ad",
                              ["annotation"]),
    "st_OV(cancer,ctype)": (f"{DR}/baselines/st_impute_ref/processed_data/st_OV_test.h5ad",
                            ["annotation"]),
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
    counts = X.copy()
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


def gt_contiguity(coords, true, k=6):
    m = true >= 0
    cv, tv = coords[m], true[m]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(cv).kneighbors(cv)[1][:, 1:]
    return float((tv[idx] == tv[:, None]).mean())


def best_config_ari(emb, true, coords, k, seed, *, method):
    labs = [KMeans(k, n_init=10, random_state=seed).fit_predict(emb)]
    try:
        labs.append(GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=3,
                                    reg_covar=1e-2).fit_predict(emb))
    except Exception:
        pass
    best = -1.0
    for lab0 in labs:
        for _, lab in refinement_candidates(method, lab0, coords, k=6):
            best = max(best, gari(true, lab))
    return best


def spatial_leiden_labels(Z, coords, k, seed):
    import igraph as ig
    import leidenalg
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    emb = spca(np.concatenate([np.sqrt(0.5) * Z, np.sqrt(0.5) * nbr], 1), 30, seed)
    idx = NearestNeighbors(n_neighbors=16).fit(emb).kneighbors(emb)[1][:, 1:]
    edges = [(i, int(j)) for i, row in enumerate(idx) for j in row]
    g = ig.Graph(n=emb.shape[0], edges=edges)
    lo, hi, lab = 0.05, 4.0, None
    for _ in range(12):
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
    return np.asarray(sf.train(embedding_save_filepath="/tmp/sf_emb_xd.tsv", epochs=400,
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


# Resumable: each platform's row is persisted to a partial cache the moment it finishes, so an
# interruption (this box is heavily oversubscribed) never loses completed platforms — a re-run skips
# whatever is already cached. (Root cause of the earlier lost runs: results only landed at the very end.)
PARTIAL = output_path("expand_data_partial.json")
preflight_coordinate_frames(NEW_PLATFORMS)
import os  # noqa: E402
new_rows = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else []
require_candidate_rows(new_rows)
pub = json.load(open(output_path("expanded_bench.json")))["rows"]
require_candidate_rows(pub)
done = {r["platform"] for r in new_rows}
for name, (path, gt) in NEW_PLATFORMS.items():
    if name in done:
        require_cached_input(next(r for r in new_rows if r["platform"] == name), [path, __file__])
        print(f"=== {name}: cached, skipping ===", flush=True)
        continue
    try:
        Z, lognorm, counts, coords, true, subbed = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        cont = gt_contiguity(coords, true)
        k = int(len(np.unique(true[true >= 0])))
        cont = gt_contiguity(coords, true)
        print(f"\n=== {name}: {Z.shape[0]} cells, {k} classes, GT contig {cont:.3f} ===", flush=True)
        per = {}
        for seed in SEEDS:
            for m, emb in embeddings(Z, lognorm, counts, coords, k, seed).items():
                record_seed(per, m, seed, best_config_ari(emb, true, coords, k, seed, method=m))
            try:
                sl = spatial_leiden_labels(Z, coords, k, seed)
                record_seed(per, "SpatialLeiden", seed,
                    max(gari(true, sl), gari(true, refine_labels(sl, coords, k=6))))
            except Exception as e:
                print("  SpatialLeiden fail", str(e)[:50])
        mean, std, missing_seeds = complete_seed_summary(per, SEEDS)
        core_mean = {m: mean[m] for m in CORE if m in mean}
        order = sorted(core_mean, key=lambda m: -core_mean[m])
        nonsp = core_mean["floor:nonspatial"]
        sadv = max(v for m, v in core_mean.items() if m != "floor:nonspatial") - nonsp
        new_rows.append({"platform": name, "protocol_id": PROTOCOL_ID, "input_fingerprint": fingerprint([path, __file__]), "seeds": list(SEEDS), "per_seed": per, "missing_seeds": missing_seeds, "GT_contiguity": round(cont, 3), "k": k,
                         "winner": order[0], "spatial_advantage": round(sadv, 3),
                         "means": {m: round(mean[m], 4) for m in mean}})
        json.dump(new_rows, open(PARTIAL, "w"), indent=2)  # persist immediately (resumable)
        print(f"  winner {order[0]} | spatial_adv {sadv:+.3f}", flush=True)
        print("  " + " | ".join(f"{m} {mean[m]:.3f}" for m in order), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:60]}")

# combine with the published 11 and re-derive the law at n=14
require_complete_platforms(new_rows, NEW_PLATFORMS)
pub = json.load(open(output_path("expanded_bench.json")))["rows"]
require_candidate_rows(pub)
pub_pts = [(r["GT_contiguity"], r["spatial_advantage"], r["platform"]) for r in pub]
new_pts = [(r["GT_contiguity"], r["spatial_advantage"], r["platform"]) for r in new_rows]
allpts = pub_pts + new_pts
cont = np.array([p[0] for p in allpts]); adv = np.array([p[1] for p in allpts])
r_all = spearmanr(cont, adv)
r_pub = spearmanr([p[0] for p in pub_pts], [p[1] for p in pub_pts])
winners_all = sorted(set([r["winner"] for r in pub] + [r["winner"] for r in new_rows]))

print(f"\n===== LAW AT n={len(allpts)} (published 11 + {len(new_rows)} new) =====")
for p in sorted(allpts, key=lambda x: -x[0]):
    tag = "NEW" if p in new_pts else ""
    print(f"  {p[2]:>27} | contig {p[0]:.3f} | adv {p[1]:+.3f} {tag}")
print(f"\nSpearman(contiguity, advantage): published n=11 {r_pub.correlation:+.3f} (p={r_pub.pvalue:.4f}) "
      f"-> EXPANDED n={len(allpts)} {r_all.correlation:+.3f} (p={r_all.pvalue:.4f})")
print(f"distinct winners across all platforms: {len(winners_all)} {winners_all}")

out = {"new_rows": new_rows, "n_new": len(new_rows), "n_total": len(allpts),
       "spearman_published_n11": round(float(r_pub.correlation), 3),
       "p_published_n11": round(float(r_pub.pvalue), 4),
       "spearman_expanded": round(float(r_all.correlation), 3),
       "p_expanded": round(float(r_all.pvalue), 4),
       "distinct_winners_all": winners_all, "n_distinct_winners_all": len(winners_all)}
json.dump(out, open(output_path("expand_data.json"), "w"), indent=2)
print("\nwrote experiments/expand_data.json")

print("Candidate expansion recorded; no significance-based acceptance gate.")
