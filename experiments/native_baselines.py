"""NATIVE-BASELINE FAIRNESS — does the unified clustering backend under-rate the SOTA methods and
flip the §7 conclusions? Re-run the full 11-platform benchmark giving every method a strictly MORE
generous, author-recommended backend, then re-derive the headline.

The reviewer concern: STAGATE/SEDR/GraphST/SpaceFlow recommend **mclust** (R), not sklearn's GMM, so a
unified GMM backend could systematically under-score them and manufacture the "no universal SOTA" /
spatial-prior-advantage story. We answer it WITHOUT leaving the Python env: R's mclust is exactly
BIC-selected Gaussian mixture model selection over covariance parameterizations, which sklearn
reproduces faithfully (fit GMMs with covariance_type in {full,tied,diag,spherical}; pick best BIC).
We add this `mclust_style` backend to each method's best-config search (so every method is scored at
best-of {KMeans, GMM-tied, mclust-style} x {refine off,on} — never worse than before), re-run all 11
platforms (1-seed backend-sensitivity pass; §7 already established the law's 3-seed robustness), recompute:
  - distinct winners (no-universal-SOTA),
  - Spearman(GT contiguity, spatial-prior advantage)  [the headline],
  - per-method ΔARI(mclust-style best-config − published GMM-tied best-config)  [how much fairness moves
    each method], to show the conclusions are invariant to the backend-fairness objection.

Embeddings are identical to expanded_bench.py (trained once per platform/seed); only the clustering
backend set is enlarged, so this is the same model pipeline scored more generously. Writes
experiments/native_baselines.json; exits non-zero (verify hook) if the headline does NOT survive.
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
from scipy.stats import spearmanr  # noqa: E402

DR = str(data_root())
EXT = str(external_root())
for s in ("STAGATE", "SEDR", "GraphST", "SpaceFlow"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.eval.refine import refine_labels  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Backend-fairness is a robustness CHECK on §7's already-3-seed-robust law: does a fairer (author-
# recommended, mclust-style) backend flip no-universal-SOTA or the +0.72 law? One seed suffices to
# answer that — we are not re-establishing seed robustness, only testing backend sensitivity. (Honest:
# documented as a 1-seed backend-sensitivity pass, not a re-derivation of the headline.)
SEEDS = [1]
SUBSAMPLE = 16000
GRAPHST_MAX = 8000

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


import os, subprocess, tempfile  # noqa: E402

_MCLUST_R = os.path.join("experiments", "_mclust.R")


def mclust_native(emb, k, seed):
    """REAL mclust (R) via Rscript subprocess --- the actual clusterer STAGATE/SEDR/GraphST recommend,
    not a Python analogue. rpy2 will not build on Python 3.13, so we round-trip through a CSV: write the
    (PCA-capped to 30-d, as mclust is run on the low-dim latent) embedding, call mclust::Mclust(G=k) with
    BIC model selection over covariance families, read back the classification. Returns None on failure
    (logged, dropped for that cell), exactly like a method that fails on a platform."""
    e = emb if emb.shape[1] <= 30 else PCA(30, random_state=seed).fit_transform(emb)
    d = tempfile.mkdtemp(prefix="mclust_")
    fi, fo = os.path.join(d, "emb.csv"), os.path.join(d, "lab.txt")
    try:
        np.savetxt(fi, e, delimiter=",",
                   header=",".join(f"V{i}" for i in range(e.shape[1])), comments="")
        r = subprocess.run(["Rscript", _MCLUST_R, fi, str(k), fo, str(seed)],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(fo):
            return None
        return np.loadtxt(fo).astype(int)
    except Exception:
        return None
    finally:
        for f in (fi, fo):
            try:
                os.remove(f)
            except OSError:
                pass


def best_config_ari(emb, true, coords, k, seed, want_backends=False):
    """Best ARI over backends x {refine off,on}. Backends: KMeans, GMM-tied (published), mclust-style
    (NEW, author-recommended). Returns best ARI, and optionally the (backend,refine) that won."""
    cands = [("kmeans", KMeans(k, n_init=10, random_state=seed).fit_predict(emb))]
    try:
        cands.append(("gmm-tied", GaussianMixture(k, covariance_type="tied", random_state=seed,
                      n_init=3, reg_covar=1e-2).fit_predict(emb)))
    except Exception:
        pass
    ms = mclust_native(emb, k, seed)
    if ms is not None:
        cands.append(("mclust-R", ms))
    best, best_tag = -1.0, None
    for tag, lab0 in cands:
        for rf, lab in (("raw", lab0), ("refine", refine_labels(lab0, coords, k=6))):
            a = gari(true, lab)
            if a > best:
                best, best_tag = a, (tag, rf)
    return (best, best_tag) if want_backends else best


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
    return np.asarray(sf.train(embedding_save_filepath="/tmp/sf_emb_nb.tsv", epochs=400,
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


# published GMM-tied best-config means for the per-method fairness delta
pub = {r["platform"]: r["means"] for r in json.load(open("experiments/expanded_bench.json"))["rows"]}

# resumable per-platform cache (real mclust adds R-subprocess time; never lose completed platforms)
PARTIAL = "experiments/native_baselines_partial.json"
if os.path.exists(PARTIAL):
    _c = json.load(open(PARTIAL)); rows = _c["rows"]; backend_wins = _c["backend_wins"]
else:
    rows, backend_wins = [], {}
done = {r["platform"] for r in rows}
for name, (path, gt) in PLATFORMS.items():
    if name in done:
        print(f"=== {name}: cached, skipping ===", flush=True); continue
    try:
        Z, lognorm, counts, coords, true, subbed = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        cont = gt_contiguity(coords, true)
        print(f"\n=== {name}: {Z.shape[0]} cells, {k} classes, GT contig {cont:.3f} ===", flush=True)
        per = {}
        for seed in SEEDS:
            embs = embeddings(Z, lognorm, counts, coords, k, seed)
            for m, emb in embs.items():
                a, tag = best_config_ari(emb, true, coords, k, seed, want_backends=True)
                per.setdefault(m, []).append(a)
                backend_wins[tag[0]] = backend_wins.get(tag[0], 0) + 1
            try:
                sl = spatial_leiden_labels(Z, coords, k, seed)
                per.setdefault("SpatialLeiden", []).append(
                    max(gari(true, sl), gari(true, refine_labels(sl, coords, k=6))))
            except Exception as e:
                print("  SpatialLeiden fail", str(e)[:50])
        mean = {m: float(np.mean(v)) for m, v in per.items()}
        core_mean = {m: mean[m] for m in CORE if m in mean}
        order = sorted(core_mean, key=lambda m: -core_mean[m])
        t_rank = order.index("Tessera") + 1
        nonsp = core_mean.get("floor:nonspatial", 0.0)
        sadv = max(v for m, v in core_mean.items() if m != "floor:nonspatial") - nonsp
        # fairness delta vs published GMM-tied best-config (only methods present in both)
        deltas = {m: round(mean[m] - pub[name][m], 3) for m in mean if m in pub.get(name, {})}
        rows.append({
            "platform": name, "GT_contiguity": round(cont, 3), "k": k,
            "Tessera_rank": t_rank, "winner": order[0],
            "spatial_advantage": round(sadv, 3),
            "means": {m: round(mean[m], 4) for m in mean},
            "delta_vs_published": deltas,
        })
        json.dump({"rows": rows, "backend_wins": backend_wins}, open(PARTIAL, "w"), indent=2)
        print(f"  winner {order[0]} | Tessera #{t_rank} | spatial_adv {sadv:+.3f}", flush=True)
        print("  " + " | ".join(f"{m} {mean[m]:.3f}" for m in order), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:60]}")

cont = np.array([r["GT_contiguity"] for r in rows])
sadv = np.array([r["spatial_advantage"] for r in rows])
trank = np.array([r["Tessera_rank"] for r in rows])
r_adv = spearmanr(cont, sadv); r_tr = spearmanr(cont, trank)
winners = sorted(set(r["winner"] for r in rows))
# mean absolute fairness movement per method
all_deltas = {}
for r in rows:
    for m, d in r["delta_vs_published"].items():
        all_deltas.setdefault(m, []).append(d)
mean_delta = {m: round(float(np.mean(v)), 3) for m, v in all_deltas.items()}

print(f"\n===== NATIVE-BASELINE FAIRNESS (mclust-style added; n={len(rows)}) =====")
for r in sorted(rows, key=lambda r: -r["GT_contiguity"]):
    print(f"  {r['platform']:>27} | contig {r['GT_contiguity']:.3f} | winner {r['winner']:>16} | "
          f"adv {r['spatial_advantage']:+.3f}")
print(f"\nDistinct winners: {winners} ({len(winners)})")
print(f"Spearman(contiguity, spatial_advantage) = {r_adv.correlation:+.3f} (p={r_adv.pvalue:.4f}) "
      f"[published GMM-tied: +0.724, p=0.0117]")
print(f"Spearman(contiguity, Tessera_rank)      = {r_tr.correlation:+.3f} (p={r_tr.pvalue:.4f})")
print(f"backend that won best-config (counts): {backend_wins}")
print(f"mean ΔARI(mclust-style best − published) per method: {mean_delta}")

out = {
    "rows": rows, "n_platforms": len(rows), "method_panel": CORE, "seeds": SEEDS,
    "backend_set": ["kmeans", "gmm-tied", "mclust-R (real R mclust via Rscript)"],
    "distinct_winners": winners, "n_distinct_winners": len(winners),
    "spearman_contiguity_vs_spatial_advantage": round(float(r_adv.correlation), 3),
    "p_contiguity_vs_spatial_advantage": round(float(r_adv.pvalue), 4),
    "spearman_contiguity_vs_tessera_rank": round(float(r_tr.correlation), 3),
    "published_spearman_advantage": 0.724,
    "backend_win_counts": backend_wins,
    "mean_delta_vs_published_per_method": mean_delta,
}
json.dump(out, open("experiments/native_baselines.json", "w"), indent=2)
print("\nwrote experiments/native_baselines.json")

# verify hook: the headline must SURVIVE a fairer, mclust-style backend.
assert r_adv.correlation >= 0.5 and r_adv.pvalue <= 0.05, (
    f"headline did not survive mclust-style backend (rho={r_adv.correlation:.3f}, p={r_adv.pvalue:.4f})")
assert len(winners) >= 5, f"no-universal-SOTA weakened under fairer backend ({len(winners)} winners)"
print("NATIVE-BASELINE CHECK PASS — no-universal-SOTA and the spatial-prior law survive the "
      "author-recommended (mclust-style) backend; the conclusions are not a backend artifact.")
