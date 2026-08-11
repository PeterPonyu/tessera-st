"""Cross-section generalisation: Tessera vs SOTA on multiple DLPFC sections (3 donors), not one.

Single-section results don't generalise — DLPFC has 3 donors of differing difficulty. Here we load
real 10x h5 per section (Banksy data), run Tessera.full / STAGATE / BANKSY-style under the fair GMM
backend on one section per donor (151507=Br5292, 151669=Br5595, 151673=Br8100), and check whether the
'each method wins different axes' picture holds ACROSS sections. 3 seeds each. ARI + boundary_F1 +
ASW (one per family) reported per section + mean across sections.
"""

import sys

import h5py
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy.sparse import csc_matrix
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from _roots import data_root, external_root, spatial_omics_root

EXT = str(external_root())
BK = f"{EXT}/Banksy/data/DLPFC"
sys.path.insert(0, f"{EXT}/STAGATE")

from tessera_st.ablation import _hard_confidence, _metric_row  # noqa: E402
from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.data.dlpfc import _encode_labels  # noqa: E402
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS, compute_layer_scores  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_section(s):
    with h5py.File(f"{BK}/{s}/{s}_raw_feature_bc_matrix.h5", "r") as f:
        m = f["matrix"]
        X = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                       shape=tuple(m["shape"][:])).T.tocsr()
        genes = [g.decode() for g in m["features"]["name"][:]]
        barcodes = [b.decode() for b in m["barcodes"][:]]
    tp = pd.read_csv(f"{BK}/{s}/tissue_positions_list.txt", header=None)
    tp = tp.rename(columns={0: "bc", 2: "row", 3: "col"}).set_index("bc")
    lm = pd.read_csv(f"{BK}/barcode_level_layer_map.tsv", sep="\t", header=None,
                     names=["bc", "sec", "layer"])
    lm = lm[lm["sec"].astype(str) == s].set_index("bc")["layer"]
    bidx = {b: i for i, b in enumerate(barcodes)}
    common = [b for b in barcodes if b in tp.index and b in lm.index]
    rows = [bidx[b] for b in common]
    Xc = np.asarray(X[rows].todense(), dtype=np.float64)
    coords = tp.loc[common, ["row", "col"]].to_numpy(dtype=float)
    labels = _encode_labels(lm.loc[common].to_numpy(), len(common))
    lib = Xc.sum(1, keepdims=True); lib[lib == 0] = 1.0
    lognorm = np.log1p(Xc / lib * 1e4)
    return lognorm, genes, coords, labels


def gmm(emb, seed, k):
    return GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=5).fit_predict(emb)


def agg(rows, name, keys):
    s = {"config": name}
    for key in keys:
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
    return s


KEYS = ["ARI", "NMI", "ASW", "boundary_F1", "small_IoU", "ECE", "marker_purity"]
SECTIONS = {"151507": "Br5292", "151669": "Br5595", "151673": "Br8100"}
per_section = {}

for sec, donor in SECTIONS.items():
    lognorm, genes, coords, labels = load_section(sec)
    k = int(len(np.unique(labels[labels >= 0])))
    pres = [g for g in genes if g in {x for v in DLPFC_LAYER_MARKERS.values() for x in v}]
    ls = compute_layer_scores(lognorm[:, [genes.index(g) for g in pres]], pres, DLPFC_LAYER_MARKERS)
    mu, sd = lognorm.mean(0), lognorm.std(0); sd[sd == 0] = 1
    Z = np.clip((lognorm - mu) / sd, -10, 10)
    hvg = np.argsort(lognorm.var(0))[::-1][:3000]
    tess_expr = PCA(min(50, k * 8), random_state=0).fit_transform(Z[:, hvg]).astype(np.float32)
    print(f"\n=== {sec} ({donor}): {lognorm.shape[0]} spots, {k} layers ===")
    res_rows = {}

    # Tessera.full
    rr = []
    for seed in [1, 2, 3]:
        r = fit_predict(tess_expr, coords, k, AblationConfig(),
                        train_cfg=TrainConfig(epochs=120, seed=seed, device=str(dev)))
        lab = gmm(r.embed, seed, k)
        rr.append(_metric_row("t", coords, labels, lab, r.embed, _hard_confidence(r.embed, lab),
                              kind="t", layer_scores=ls))
    res_rows["Tessera.full"] = agg(rr, "Tessera.full", KEYS)

    # STAGATE
    try:
        import STAGATE_pyG as STAGATE
        adst = ad.AnnData(X=lognorm[:, hvg].astype(np.float32)); adst.obsm["spatial"] = coords
        rr = []
        for seed in [1, 2, 3]:
            STAGATE.Cal_Spatial_Net(adst, k_cutoff=6, model="KNN", verbose=False)
            torch.manual_seed(seed)
            o = STAGATE.train_STAGATE(adst, n_epochs=1000, random_seed=seed, device=dev, verbose=False)
            emb = np.asarray(o.obsm["STAGATE"])
            lab = gmm(emb, seed, k)
            rr.append(_metric_row("s", coords, labels, lab, emb, _hard_confidence(emb, lab),
                                  kind="s", layer_scores=ls))
        res_rows["STAGATE"] = agg(rr, "STAGATE", KEYS)
    except Exception as e:
        print("  STAGATE failed:", str(e)[:80])

    # BANKSY-style
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    aug = np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1)
    rr = []
    for seed in [1, 2, 3]:
        emb = PCA(20, random_state=seed).fit_transform(aug)
        lab = gmm(emb, seed, k)
        rr.append(_metric_row("b", coords, labels, lab, emb, _hard_confidence(emb, lab),
                              kind="b", layer_scores=ls))
    res_rows["BANKSY-style"] = agg(rr, "BANKSY-style", KEYS)

    per_section[sec] = res_rows
    for name, row in res_rows.items():
        print(f"  {name:>14}: ARI={row.get('ARI')}  bF1={row.get('boundary_F1')}  "
              f"ASW={row.get('ASW')}  ECE={row.get('ECE')}  marker={row.get('marker_purity')}")

# cross-section means + per-metric winners
print("\n===== mean across 3 sections =====")
methods = ["Tessera.full", "STAGATE", "BANKSY-style"]
for m in KEYS:
    line = []
    for meth in methods:
        vals = [per_section[s][meth].get(m) for s in SECTIONS if meth in per_section[s]]
        vals = [v for v in vals if v is not None]
        line.append(f"{meth}={round(float(np.mean(vals)), 4)}" if vals else f"{meth}=NA")
    print(f"  {m:>13}: " + "  ".join(line))

import json  # noqa: E402

json.dump(per_section, open("experiments/multisection_panel.json", "w"), indent=2)
print("\nwrote experiments/multisection_panel.json")
