"""Confirm SEDR + GraphST actually TRAIN (not just import) under the numba stub, on 151673."""

import sys

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
sys.path.insert(0, f"{EXT}/SEDR")
sys.path.insert(0, f"{EXT}/GraphST")
P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")
from tessera_st.data.dlpfc import _encode_labels  # noqa: E402
from tessera_st.eval import ari  # noqa: E402

raw = ad.read_h5ad(P)
X = (raw.X.toarray() if hasattr(raw.X, "toarray") else np.asarray(raw.X)).astype(np.float64)
lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
ln = np.log1p(X / lib * 1e4)
coords = np.asarray(raw.obsm["spatial"], float)
true = _encode_labels(raw.obs["ground_truth"].values, ln.shape[0])
k = int(len(np.unique(true[true >= 0])))
hvg = np.argsort(ln.var(0))[::-1][:3000]
mu, sd = ln[:, hvg].mean(0), ln[:, hvg].std(0); sd[sd == 0] = 1
Z = np.clip((ln[:, hvg] - mu) / sd, -10, 10)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gari(emb):
    return round(ari(true, GaussianMixture(k, covariance_type="tied", random_state=1,
                                           n_init=5, reg_covar=1e-4).fit_predict(emb)), 4)


# ---- SEDR ----
try:
    import SEDR
    adata = ad.AnnData(X=ln[:, hvg].astype(np.float32)); adata.obsm["spatial"] = coords
    adata.obsm["X_pca"] = PCA(200, random_state=0).fit_transform(Z).astype(np.float32)
    gd = SEDR.graph_construction(adata, 6)
    net = SEDR.Sedr(adata.obsm["X_pca"], gd, device=str(dev))
    net.train_without_dec(N=1)
    feat, _, _, _ = net.process()
    print("SEDR trained OK, ARI(gmm) =", gari(np.asarray(feat)))
except Exception as e:
    import traceback; traceback.print_exc(); print("SEDR RUN FAIL:", str(e)[:100])

# ---- GraphST ----
try:
    from GraphST import GraphST
    adata = ad.AnnData(X=ln[:, hvg].astype(np.float32)); adata.obsm["spatial"] = coords
    model = GraphST.GraphST(adata, device=dev, epochs=600)
    out = model.train()
    emb = np.asarray(out.obsm["emb"])
    print("GraphST trained OK, ARI(gmm) =", gari(emb))
except Exception as e:
    import traceback; traceback.print_exc(); print("GraphST RUN FAIL:", str(e)[:100])
