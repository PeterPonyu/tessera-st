"""Multi-SOTA panel on DLPFC 151673, fair GMM backend, FULL metric panel incl. internal ASW/DBI/CAL.

Methods (different winning mechanisms): STAGATE (GAT + pure recon), SEDR (VGAE + self-supervision),
BANKSY-style (non-DL neighbour-augmented features, clean-room reimpl of the idea). GraphST is skipped
(its numba dep is incompatible with this env's numpy — recorded, not hidden). Each method's embedding
is clustered with GMM(tied) and scored on every metric; internal metrics are in each method's own
space (semi-circular, reported as one angle). 3 seeds, mean±std.
"""

import sys

import numpy as np
import torch
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from _roots import data_root, external_root, spatial_omics_root

EXT = str(external_root())
sys.path.insert(0, f"{EXT}/STAGATE")
sys.path.insert(0, f"{EXT}/SEDR")
P = str(spatial_omics_root() / "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")

from tessera_st.ablation import _hard_confidence, _metric_row, format_table  # noqa: E402
from tessera_st.data.dlpfc import _encode_labels  # noqa: E402
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS, compute_layer_scores  # noqa: E402

raw = ad.read_h5ad(P)
X = raw.X.toarray() if hasattr(raw.X, "toarray") else np.asarray(raw.X)
X = X.astype(np.float64)
lib = X.sum(1, keepdims=True); lib[lib == 0] = 1.0
lognorm = np.log1p(X / lib * 1e4)
genes = list(map(str, raw.var_names))
coords = np.asarray(raw.obsm["spatial"], dtype=np.float64)
true = _encode_labels(raw.obs["ground_truth"].values, n=lognorm.shape[0])
k = int(len(np.unique(true[true >= 0])))
pres = [g for g in genes if g in {x for v in DLPFC_LAYER_MARKERS.values() for x in v}]
layer_scores = compute_layer_scores(lognorm[:, [genes.index(g) for g in pres]], pres,
                                    DLPFC_LAYER_MARKERS)
hvg = np.argsort(lognorm.var(0))[::-1][:3000]
hvg_ln = lognorm[:, hvg]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU", "ECE",
        "marker_purity"]


def gmm(emb, seed):
    return GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=5).fit_predict(emb)


def score_method(name, embeds_per_seed):
    rows = []
    for seed, emb in embeds_per_seed:
        lab = gmm(emb, seed)
        rows.append(_metric_row(f"{name}.s{seed}", coords, true, lab, emb,
                                _hard_confidence(emb, lab), kind="sota", layer_scores=layer_scores))
    s = {"config": name, "kind": "sota"}
    for key in KEYS:
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
            s[key + "_std"] = round(float(np.std(v)), 4)
    return s


summaries = []

# ---- STAGATE (GAT + pure reconstruction) ----
try:
    import STAGATE_pyG as STAGATE
    ad_st = ad.AnnData(X=hvg_ln.astype(np.float32)); ad_st.obsm["spatial"] = coords
    embs = []
    for seed in [1, 2, 3]:
        STAGATE.Cal_Spatial_Net(ad_st, k_cutoff=6, model="KNN", verbose=False)
        torch.manual_seed(seed)
        out = STAGATE.train_STAGATE(ad_st, n_epochs=1000, random_seed=seed, device=dev, verbose=False)
        embs.append((seed, np.asarray(out.obsm["STAGATE"])))
    summaries.append(score_method("SOTA:STAGATE", embs))
    print("STAGATE done:", summaries[-1]["ARI"])
except Exception as e:
    print("STAGATE FAILED:", str(e)[:120])

# ---- SEDR (VGAE + graph self-supervision) ----
try:
    import SEDR
    mu, sd = hvg_ln.mean(0), hvg_ln.std(0); sd[sd == 0] = 1
    Xp = PCA(n_components=200, random_state=0).fit_transform(np.clip((hvg_ln - mu) / sd, -10, 10))
    ad_se = ad.AnnData(X=hvg_ln.astype(np.float32)); ad_se.obsm["spatial"] = coords
    ad_se.obsm["X_pca"] = Xp.astype(np.float32)
    gd = SEDR.graph_construction(ad_se, 6)
    embs = []
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        net = SEDR.Sedr(ad_se.obsm["X_pca"], gd, mode="clustering", device=str(dev))
        net.train_without_dec(N=1)
        feat, _, _, _ = net.process()
        embs.append((seed, np.asarray(feat)))
    summaries.append(score_method("SOTA:SEDR", embs))
    print("SEDR done:", summaries[-1]["ARI"])
except Exception as e:
    print("SEDR FAILED:", str(e)[:150])

# ---- BANKSY-style (clean-room: neighbour-mean augmented features, non-DL) ----
try:
    mu, sd = hvg_ln.mean(0), hvg_ln.std(0); sd[sd == 0] = 1
    Z = np.clip((hvg_ln - mu) / sd, -10, 10)
    nn = NearestNeighbors(n_neighbors=7).fit(coords)
    idx = nn.kneighbors(coords)[1][:, 1:]
    nbr_mean = Z[idx].mean(1)
    lam = 0.8
    aug = np.concatenate([np.sqrt(1 - lam) * Z, np.sqrt(lam) * nbr_mean], axis=1)
    embs = []
    for seed in [1, 2, 3]:
        emb = PCA(n_components=20, random_state=seed).fit_transform(aug)
        embs.append((seed, emb))
    summaries.append(score_method("SOTA:BANKSY-style", embs))
    print("BANKSY-style done:", summaries[-1]["ARI"])
except Exception as e:
    print("BANKSY FAILED:", str(e)[:120])

print("\nNOTE: GraphST skipped — its numba dependency is incompatible with this env's numpy.\n")
print(format_table(summaries, ["config", "ARI", "NMI", "ASW", "DBI", "CAL", "boundary_F1",
                               "ECE", "marker_purity"]))
import json  # noqa: E402

json.dump({"summaries": summaries}, open("experiments/sota_panel_151673.json", "w"), indent=2)
print("\nwrote experiments/sota_panel_151673.json")
