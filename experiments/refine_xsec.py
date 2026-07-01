"""Cross-section reliability check for the refinement fix: does Tessera's refinement gain hold across
3 donors (not just 151673)? Tessera + main rivals (STAGATE/SEDR/BANKSY/floor:smoothed), GMM backend,
refine on/off, full key metrics. GraphST skipped (slow under stub + under-adapted; not a main rival).
This is the test that decides whether refinement is a real improvement or another single-section fluke."""

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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
BK = f"{EXT}/Banksy/data/DLPFC"
for s in ("STAGATE", "SEDR"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.ablation import _hard_confidence, _metric_row  # noqa: E402
from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.data.dlpfc import _encode_labels  # noqa: E402
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS, compute_layer_scores  # noqa: E402
from tessera_st.eval.refine import refine_labels  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SECTIONS = ["151507", "151669", "151673"]
SEEDS = [1, 2]
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "boundary_F1", "small_IoU"]


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
    return np.log1p(Xc / lib * 1e4), genes, coords, labels


def gmm(emb, seed, k):
    return GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=5,
                           reg_covar=1e-4).fit_predict(emb)


# accumulate per (method, refine-flag) the metric over (section, seed)
acc = {}


def add(meth, refine_flag, coords, labels, ls, k, emb, seed):
    lab = gmm(emb, seed, k)
    if refine_flag:
        lab = refine_labels(lab, coords, k=6)
    row = _metric_row("x", coords, labels, lab, emb, _hard_confidence(emb, lab),
                      kind="m", layer_scores=ls)
    for key in KEYS:
        if row.get(key) is not None:
            acc.setdefault((meth, refine_flag), {}).setdefault(key, []).append(row[key])


for sec in SECTIONS:
    ln, genes, coords, labels = load_section(sec)
    k = int(len(np.unique(labels[labels >= 0])))
    pres = [g for g in genes if g in {x for v in DLPFC_LAYER_MARKERS.values() for x in v}]
    ls = compute_layer_scores(ln[:, [genes.index(g) for g in pres]], pres, DLPFC_LAYER_MARKERS)
    hvg = np.argsort(ln.var(0))[::-1][:3000]
    mu, sd = ln[:, hvg].mean(0), ln[:, hvg].std(0); sd[sd == 0] = 1
    Z = np.clip((ln[:, hvg] - mu) / sd, -10, 10)
    print(f"=== {sec}: {ln.shape[0]} spots, {k} layers ===", flush=True)

    expr = PCA(min(50, k * 8), random_state=0).fit_transform(Z).astype(np.float32)
    embs = {"Tessera": [(s, fit_predict(expr, coords, k, AblationConfig(),
            train_cfg=TrainConfig(epochs=120, seed=s, device=str(dev))).embed) for s in SEEDS]}
    try:
        import STAGATE_pyG as ST
        a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
        e = []
        for s in SEEDS:
            ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False); torch.manual_seed(s)
            o = ST.train_STAGATE(a, n_epochs=1000, random_seed=s, device=dev, verbose=False)
            e.append((s, np.asarray(o.obsm["STAGATE"])))
        embs["STAGATE"] = e
    except Exception as ex:
        print("STAGATE fail", str(ex)[:50])
    try:
        import SEDR
        a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
        a.obsm["X_pca"] = PCA(200, random_state=0).fit_transform(Z).astype(np.float32)
        gd = SEDR.graph_construction(a, 6); e = []
        for s in SEEDS:
            torch.manual_seed(s)
            net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
            feat, _, _, _ = net.process(); e.append((s, np.asarray(feat)))
        embs["SEDR"] = e
    except Exception as ex:
        print("SEDR fail", str(ex)[:50])
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    embs["BANKSY-style"] = [(s, PCA(20, random_state=s).fit_transform(
        np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1))) for s in SEEDS]
    embs["floor:smoothed"] = [(s, PCA(30, random_state=s).fit_transform(nbr)) for s in SEEDS]

    for meth, elist in embs.items():
        for seed, emb in elist:
            add(meth, False, coords, labels, ls, k, emb, seed)
            add(meth, True, coords, labels, ls, k, emb, seed)

print("\n=== cross-section mean (GMM), raw vs +refine ===")
print(f"{'method':>15} | {'ARI':>6} {'ARI+ref':>8} {'ΔARI':>6} | {'bF1':>6} {'bF1+ref':>8}")
print("-" * 60)
out = {}
methods = sorted({m for m, _ in acc})
for m in methods:
    a0 = float(np.mean(acc[(m, False)]["ARI"]))
    a1 = float(np.mean(acc[(m, True)]["ARI"]))
    b0 = float(np.mean(acc[(m, False)]["boundary_F1"]))
    b1 = float(np.mean(acc[(m, True)]["boundary_F1"]))
    out[m] = {"ARI": round(a0, 4), "ARI_refine": round(a1, 4), "dARI": round(a1 - a0, 4),
              "bF1": round(b0, 4), "bF1_refine": round(b1, 4)}
    print(f"{m:>15} | {a0:.3f} {a1:>8.3f} {a1-a0:>6.3f} | {b0:.3f} {b1:>8.3f}")

best_raw = max(out, key=lambda m: out[m]["ARI"])
best_ref = max(out, key=lambda m: out[m]["ARI_refine"])
print(f"\nbest ARI raw: {best_raw} ({out[best_raw]['ARI']}) | "
      f"best ARI +refine: {best_ref} ({out[best_ref]['ARI_refine']})")
print(f"Tessera ARI rank +refine: "
      f"{sorted(out, key=lambda m: -out[m]['ARI_refine']).index('Tessera')+1}/{len(out)}")
json.dump(out, open("experiments/refine_xsec.json", "w"), indent=2)
print("wrote experiments/refine_xsec.json")
