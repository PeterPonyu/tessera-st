"""G001 RigorousBench: full-factorial DLPFC domain-detection benchmark.

7 methods (Tessera + STAGATE/SEDR/GraphST/BANKSY + 2 floors) x 3 donor sections x 2 clustering
backends (KMeans / GMM) x 11 metrics x seeds. Quantifies (a) the clustering-BACKEND confound
(per-method GMM-minus-KMeans ARI), and (b) cross-section variance. One machine-checked JSON.
numba-stub lets SEDR/GraphST run without changing the env.
"""

import sys

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
BK = f"{EXT}/Banksy/data/DLPFC"
for s in ("STAGATE", "SEDR", "GraphST"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.ablation import _hard_confidence, _metric_row  # noqa: E402
from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.data.dlpfc import _encode_labels  # noqa: E402
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS, compute_layer_scores  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SECTIONS = {"151507": "Br5292", "151669": "Br5595", "151673": "Br8100"}
SEEDS = [1, 2]
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU", "ECE",
        "marker_purity"]


def load_section(s):
    with h5py.File(f"{BK}/{s}/{s}_raw_feature_bc_matrix.h5", "r") as f:
        m = f["matrix"]
        X = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                       shape=tuple(m["shape"][:])).T.tocsr()
        genes = [g.decode() for g in m["features"]["name"][:]]
        bcs = [b.decode() for b in m["barcodes"][:]]
    tp = pd.read_csv(f"{BK}/{s}/tissue_positions_list.txt", header=None).rename(
        columns={0: "bc", 2: "row", 3: "col"}).set_index("bc")
    lm = pd.read_csv(f"{BK}/barcode_level_layer_map.tsv", sep="\t", header=None,
                     names=["bc", "sec", "layer"])
    lm = lm[lm["sec"].astype(str) == s].set_index("bc")["layer"]
    bidx = {b: i for i, b in enumerate(bcs)}
    common = [b for b in bcs if b in tp.index and b in lm.index]
    Xc = np.asarray(X[[bidx[b] for b in common]].todense(), float)
    coords = tp.loc[common, ["row", "col"]].to_numpy(float)
    labels = _encode_labels(lm.loc[common].to_numpy(), len(common))
    lib = Xc.sum(1, keepdims=True); lib[lib == 0] = 1
    ln = np.log1p(Xc / lib * 1e4)
    return ln, genes, coords, labels


def backends(emb, seed, k):
    return {
        "kmeans": KMeans(k, n_init=10, random_state=seed).fit_predict(emb),
        "gmm": GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=5,
                               reg_covar=1e-4).fit_predict(emb),
    }


def embed_methods(ln, genes, coords, k):
    """Return {method: [(seed, embedding), ...]}."""
    mu, sd = ln.mean(0), ln.std(0); sd[sd == 0] = 1
    hvg = np.argsort(ln.var(0))[::-1][:3000]
    Z = np.clip((ln[:, hvg] - mu[hvg]) / sd[hvg], -10, 10)
    out = {}

    # Tessera
    expr = PCA(min(50, k * 8), random_state=0).fit_transform(Z).astype(np.float32)
    out["Tessera"] = [(s, fit_predict(expr, coords, k, AblationConfig(),
                       train_cfg=TrainConfig(epochs=120, seed=s, device=str(dev))).embed)
                      for s in SEEDS]

    # STAGATE
    try:
        import STAGATE_pyG as ST
        a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
        e = []
        for s in SEEDS:
            ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False)
            torch.manual_seed(s)
            o = ST.train_STAGATE(a, n_epochs=1000, random_seed=s, device=dev, verbose=False)
            e.append((s, np.asarray(o.obsm["STAGATE"])))
        out["STAGATE"] = e
    except Exception as ex:
        print("STAGATE fail", s, str(ex)[:60])

    # SEDR
    try:
        import SEDR
        a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
        a.obsm["X_pca"] = PCA(200, random_state=0).fit_transform(Z).astype(np.float32)
        gd = SEDR.graph_construction(a, 6)
        e = []
        for s in SEEDS:
            torch.manual_seed(s)
            net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
            feat, _, _, _ = net.process()
            e.append((s, np.asarray(feat)))
        out["SEDR"] = e
    except Exception as ex:
        print("SEDR fail", str(ex)[:60])

    # GraphST
    try:
        from GraphST import GraphST as G
        e = []
        for s in SEEDS:
            a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
            o = G.GraphST(a, device=dev, epochs=500, random_seed=s).train()
            e.append((s, np.asarray(o.obsm["emb"])))
        out["GraphST"] = e
    except Exception as ex:
        print("GraphST fail", str(ex)[:60])

    # BANKSY-style
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    aug = np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1)
    out["BANKSY-style"] = [(s, PCA(20, random_state=s).fit_transform(aug)) for s in SEEDS]
    out["floor:nonspatial"] = [(s, PCA(30, random_state=s).fit_transform(Z)) for s in SEEDS]
    out["floor:smoothed"] = [(s, PCA(30, random_state=s).fit_transform(nbr)) for s in SEEDS]
    return out


results = {}  # results[method][section][backend] = {metric: mean}
for sec in SECTIONS:
    ln, genes, coords, labels = load_section(sec)
    k = int(len(np.unique(labels[labels >= 0])))
    pres = [g for g in genes if g in {x for v in DLPFC_LAYER_MARKERS.values() for x in v}]
    ls = compute_layer_scores(ln[:, [genes.index(g) for g in pres]], pres, DLPFC_LAYER_MARKERS)
    print(f"\n=== {sec} ({SECTIONS[sec]}): {ln.shape[0]} spots, {k} layers ===", flush=True)
    methods = embed_methods(ln, genes, coords, k)
    for meth, embs in methods.items():
        for backend in ("kmeans", "gmm"):
            rows = []
            for seed, emb in embs:
                lab = backends(emb, seed, k)[backend]
                rows.append(_metric_row("x", coords, labels, lab, emb,
                                        _hard_confidence(emb, lab), kind="m", layer_scores=ls))
            agg = {key: round(float(np.mean([r[key] for r in rows if r.get(key) is not None])), 4)
                   for key in KEYS if any(r.get(key) is not None for r in rows)}
            results.setdefault(meth, {}).setdefault(sec, {})[backend] = agg
        kma = results[meth][sec]["kmeans"].get("ARI")
        gma = results[meth][sec]["gmm"].get("ARI")
        print(f"  {meth:>16}: KMeans ARI={kma}  GMM ARI={gma}  Δ={round(gma - kma, 3)}", flush=True)

# ---- analysis: backend confound + cross-section variance ----
methods = list(results.keys())
print("\n===== BACKEND CONFOUND (mean GMM-KMeans ARI across sections) =====")
confound = {}
for m in methods:
    ds = [results[m][s]["gmm"]["ARI"] - results[m][s]["kmeans"]["ARI"]
          for s in SECTIONS if "ARI" in results[m][s]["gmm"]]
    confound[m] = round(float(np.mean(ds)), 4)
    print(f"  {m:>16}: ΔARI(GMM-KMeans) = {confound[m]}")

print("\n===== CROSS-SECTION ARI (GMM backend): mean ± std over 3 donors =====")
xsec = {}
for m in methods:
    v = [results[m][s]["gmm"]["ARI"] for s in SECTIONS if "ARI" in results[m][s]["gmm"]]
    xsec[m] = {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
    print(f"  {m:>16}: {xsec[m]['mean']} ± {xsec[m]['std']}  per-section={[round(x,3) for x in v]}")

json.dump({"results": results, "backend_confound": confound, "cross_section": xsec,
           "sections": SECTIONS, "seeds": SEEDS},
          open("experiments/rigorous_bench.json", "w"), indent=2)
print("\nwrote experiments/rigorous_bench.json")
