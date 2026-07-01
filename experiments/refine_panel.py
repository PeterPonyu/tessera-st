"""FIX: add the standard spatial label-refinement post-step (used by STAGATE/GraphST/SpaGCN) that the
benchmark was missing. Majority-vote each spot's label over its spatial kNN. Apply to ALL methods
(fair), compare no-refine vs refine on the full panel. Does it lift everyone, and does Tessera
(backend-robust, already-coherent embedding) gain or hold its boundary edge? 151673, GMM, 2 seeds."""

import sys

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
for s in ("STAGATE", "SEDR", "GraphST"):
    sys.path.insert(0, f"{EXT}/{s}")
P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")

from tessera_st.ablation import _hard_confidence, _metric_row, format_table  # noqa: E402
from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.data.dlpfc import _encode_labels  # noqa: E402
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS, compute_layer_scores  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

raw = ad.read_h5ad(P)
X = (raw.X.toarray() if hasattr(raw.X, "toarray") else np.asarray(raw.X)).astype(np.float64)
lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
ln = np.log1p(X / lib * 1e4)
genes = list(map(str, raw.var_names))
coords = np.asarray(raw.obsm["spatial"], float)
true = _encode_labels(raw.obs["ground_truth"].values, ln.shape[0])
k = int(len(np.unique(true[true >= 0])))
pres = [g for g in genes if g in {x for v in DLPFC_LAYER_MARKERS.values() for x in v}]
ls = compute_layer_scores(ln[:, [genes.index(g) for g in pres]], pres, DLPFC_LAYER_MARKERS)
hvg = np.argsort(ln.var(0))[::-1][:3000]
mu, sd = ln[:, hvg].mean(0), ln[:, hvg].std(0); sd[sd == 0] = 1
Z = np.clip((ln[:, hvg] - mu) / sd, -10, 10)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1, 2]
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "boundary_F1", "small_IoU", "ECE", "marker_purity"]

# spatial kNN for refinement (standard SOTA post-step)
ridx = NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1]


def refine(labels, rounds=2):
    lab = labels.copy()
    for _ in range(rounds):
        block = lab[ridx]                          # (n, 7) self + 6 neighbours
        lab = np.array([np.bincount(r).argmax() for r in block])
    return lab


def gmm(emb, seed):
    return GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=5,
                           reg_covar=1e-4).fit_predict(emb)


def embed_methods():
    out = {}
    expr = PCA(min(50, k * 8), random_state=0).fit_transform(Z).astype(np.float32)
    out["Tessera"] = [(s, fit_predict(expr, coords, k, AblationConfig(),
                       train_cfg=TrainConfig(epochs=120, seed=s, device=str(dev))).embed)
                      for s in SEEDS]
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
        print("STAGATE fail", str(ex)[:60])
    try:
        import SEDR
        a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
        a.obsm["X_pca"] = PCA(200, random_state=0).fit_transform(Z).astype(np.float32)
        gd = SEDR.graph_construction(a, 6)
        e = []
        for s in SEEDS:
            torch.manual_seed(s)
            net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
            feat, _, _, _ = net.process(); e.append((s, np.asarray(feat)))
        out["SEDR"] = e
    except Exception as ex:
        print("SEDR fail", str(ex)[:60])
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
    nbr = Z[ridx[:, 1:]].mean(1)
    out["BANKSY-style"] = [(s, PCA(20, random_state=s).fit_transform(
        np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1))) for s in SEEDS]
    out["floor:smoothed"] = [(s, PCA(30, random_state=s).fit_transform(nbr)) for s in SEEDS]
    return out


def score(embs, do_refine):
    rows = []
    for seed, emb in embs:
        lab = gmm(emb, seed)
        if do_refine:
            lab = refine(lab)
        rows.append(_metric_row("x", coords, true, lab, emb, _hard_confidence(emb, lab),
                                kind="m", layer_scores=ls))
    return {key: round(float(np.mean([r[key] for r in rows if r.get(key) is not None])), 4)
            for key in KEYS if any(r.get(key) is not None for r in rows)}


methods = embed_methods()
raw_rows, ref_rows = [], []
print(f"{'method':>15} | {'ARI raw':>8} {'ARI ref':>8} {'ΔARI':>6} | {'bF1 raw':>7} {'bF1 ref':>7}")
print("-" * 62)
for m, embs in methods.items():
    r0 = score(embs, False); r1 = score(embs, True)
    raw_rows.append({"config": m, **r0}); ref_rows.append({"config": f"{m}+refine", **r1})
    print(f"{m:>15} | {r0['ARI']:>8} {r1['ARI']:>8} {round(r1['ARI']-r0['ARI'],3):>6} | "
          f"{r0['boundary_F1']:>7} {r1['boundary_F1']:>7}", flush=True)

print("\n=== NO refine ==="); print(format_table(raw_rows, ["config"] + KEYS))
print("\n=== WITH refine (standard SOTA post-step) ==="); print(format_table(ref_rows, ["config"] + KEYS))
json.dump({"raw": raw_rows, "refined": ref_rows}, open("experiments/refine_panel.json", "w"), indent=2)
print("\nwrote experiments/refine_panel.json")
