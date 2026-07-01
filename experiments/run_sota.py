"""Run a REAL named-SOTA method (STAGATE) on DLPFC 151673 and score it on the SAME full panel.

Fairness rules: STAGATE uses its own recommended input (HVG log-norm + its spatial graph), but the
clustering backend (KMeans) and ALL evaluation metrics are identical to Tessera/baselines — so the
numbers land in the same multi-angle table. STAGATE's paper uses R mclust; we use KMeans to match
our protocol, which can shift its score somewhat (noted honestly). External SOTA repo lives under
external/ and is only referenced here in experiments/, never in tessera src.
"""

import sys

import numpy as np

STAGATE_PATH = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external/STAGATE"
sys.path.insert(0, STAGATE_PATH)
P = (
    "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
    "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad"
)

import anndata as ad  # noqa: E402
import torch  # noqa: E402

import STAGATE_pyG as STAGATE  # noqa: E402
from tessera_st.ablation import _hard_confidence, _metric_row, format_table  # noqa: E402
from tessera_st.data.dlpfc import _encode_labels  # noqa: E402
from tessera_st.eval.baselines import kmeans_labels  # noqa: E402
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS, compute_layer_scores  # noqa: E402

# ---- load + preprocess (numpy, no scanpy) ----
raw = ad.read_h5ad(P)
Xraw = raw.X.toarray() if hasattr(raw.X, "toarray") else np.asarray(raw.X)
Xraw = Xraw.astype(np.float64)
lib = Xraw.sum(1, keepdims=True)
lib[lib == 0] = 1.0
lognorm = np.log1p(Xraw / lib * 1e4)
gene_names = list(map(str, raw.var_names))
coords = np.asarray(raw.obsm["spatial"], dtype=np.float64)
true = _encode_labels(raw.obs["ground_truth"].values, n=lognorm.shape[0])
n_clusters = int(len(np.unique(true[true >= 0])))

# marker-prior scores (same as the rest of the panel)
wanted = {g for gs in DLPFC_LAYER_MARKERS.values() for g in gs}
present = [g for g in gene_names if g in wanted]
idx = [gene_names.index(g) for g in present]
layer_scores = compute_layer_scores(lognorm[:, idx], present, DLPFC_LAYER_MARKERS)

# HVG 3000 for STAGATE input
hvg = np.argsort(lognorm.var(0))[::-1][:3000]
adata = ad.AnnData(X=lognorm[:, hvg].astype(np.float32))
adata.obsm["spatial"] = coords

def gmm_labels(emb, k, seed):
    """mclust's Python cousin: a Gaussian mixture (mclust default EEE ~ tied covariance)."""
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(n_components=k, covariance_type="tied", random_state=seed,
                           n_init=5).fit_predict(emb)


rows = {"kmeans": [], "gmm": []}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for seed in [1, 2, 3]:
    STAGATE.Cal_Spatial_Net(adata, k_cutoff=6, model="KNN", verbose=False)
    torch.manual_seed(seed)
    out = STAGATE.train_STAGATE(adata, n_epochs=1000, random_seed=seed, device=device, verbose=False)
    emb = np.asarray(out.obsm["STAGATE"])
    for backend, fn in [("kmeans", kmeans_labels), ("gmm", gmm_labels)]:
        labels = fn(emb, n_clusters, seed)
        conf = _hard_confidence(emb, labels)
        row = _metric_row(f"SOTA:STAGATE+{backend}.s{seed}", coords, true, labels, emb, conf,
                          kind="sota", layer_scores=layer_scores)
        rows[backend].append(row)
    print(f"seed {seed}: STAGATE+kmeans ARI={rows['kmeans'][-1]['ARI']} | "
          f"STAGATE+gmm ARI={rows['gmm'][-1]['ARI']}")

# aggregate each backend across seeds (names differ per seed, so aggregate manually)
import json  # noqa: E402

keys = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "boundary_F1", "small_IoU", "ECE", "marker_purity"]
summaries = []
for backend in ("kmeans", "gmm"):
    s = {"config": f"SOTA:STAGATE+{backend}", "kind": "sota"}
    for k in keys:
        vals = [r[k] for r in rows[backend] if r.get(k) is not None]
        if vals:
            s[k] = round(float(np.mean(vals)), 4)
            s[k + "_std"] = round(float(np.std(vals)), 4)
    summaries.append(s)
print()
print(format_table(summaries))
json.dump({"rows": rows, "summaries": summaries},
          open("experiments/sota_stagate_151673.json", "w"), indent=2)
print("\nwrote experiments/sota_stagate_151673.json")
