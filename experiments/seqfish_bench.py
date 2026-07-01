"""CROSS-PLATFORM test — break out of DLPFC. seqFISH mouse embryo (19416 cells, 351 genes, 22 cell
types), a totally different platform/tissue/GT-type/gene-count from DLPFC Visium. Run the same
benchmark (Tessera + STAGATE/SEDR/BANKSY + 2 floors, 2 backends, full metrics) and ask whether the
DLPFC 'written-in-stone' conclusions FLIP: (a) backend confound magnitude, (b) simple-baseline
competitiveness, (c) Tessera's rank. Hypothesis: cell types are spatially scattered (not contiguous
layers), so spatial-smoothing methods may collapse and detail-preserving ones may rise. GraphST
skipped (too slow under the stub at 19k cells). 1 seed for speed; this is a discovery probe."""

import sys

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
for s in ("STAGATE", "SEDR"):
    sys.path.insert(0, f"{EXT}/{s}")
P = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/data/raw/squidpy/seqfish.h5ad"

from tessera_st.ablation import _hard_confidence, _metric_row, format_table  # noqa: E402
from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU"]
SEED = 1

a = ad.read_h5ad(P)
X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float64)
coords = np.asarray(a.obsm["spatial"], float)
cats = a.obs["celltype_mapped_refined"].astype("category")
true = cats.cat.codes.to_numpy().astype(np.int64)
k = int(true.max() + 1)
lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
ln = np.log1p(X / lib * 1e4)
mu, sd = ln.mean(0), ln.std(0); sd[sd == 0] = 1
Z = np.clip((ln - mu) / sd, -10, 10)   # only 351 genes, use all
print(f"seqFISH: {X.shape[0]} cells, {X.shape[1]} genes, {k} cell types\n", flush=True)


def kmeans(emb):
    return KMeans(k, n_init=10, random_state=SEED).fit_predict(emb)


def gmm(emb):
    return GaussianMixture(k, covariance_type="tied", random_state=SEED, n_init=3,
                           reg_covar=1e-4).fit_predict(emb)


def score(emb, lab):
    return _metric_row("x", coords, true, lab, emb, _hard_confidence(emb, lab), kind="m")


results = {}  # method -> backend -> metrics


def add(name, emb):
    results[name] = {}
    for bk, fn in [("kmeans", kmeans), ("gmm", gmm)]:
        row = score(emb, fn(emb))
        results[name][bk] = {key: round(row[key], 4) for key in KEYS if row.get(key) is not None}
    print(f"  {name:>16}: KMeans ARI={results[name]['kmeans']['ARI']}  "
          f"GMM ARI={results[name]['gmm']['ARI']}", flush=True)


# Tessera
expr = PCA(50, random_state=0).fit_transform(Z).astype(np.float32)
add("Tessera", fit_predict(expr, coords, k, AblationConfig(),
    train_cfg=TrainConfig(epochs=120, seed=SEED, device=str(dev))).embed)
try:
    import STAGATE_pyG as ST
    ada = ad.AnnData(X=ln.astype(np.float32)); ada.obsm["spatial"] = coords
    ST.Cal_Spatial_Net(ada, k_cutoff=6, model="KNN", verbose=False); torch.manual_seed(SEED)
    o = ST.train_STAGATE(ada, n_epochs=800, random_seed=SEED, device=dev, verbose=False)
    add("STAGATE", np.asarray(o.obsm["STAGATE"]))
except Exception as e:
    print("STAGATE fail", str(e)[:60])
try:
    import SEDR
    ada = ad.AnnData(X=ln.astype(np.float32)); ada.obsm["spatial"] = coords
    ada.obsm["X_pca"] = PCA(min(200, Z.shape[1]), random_state=0).fit_transform(Z).astype(np.float32)
    gd = SEDR.graph_construction(ada, 6); torch.manual_seed(SEED)
    net = SEDR.Sedr(ada.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
    feat, _, _, _ = net.process(); add("SEDR", np.asarray(feat))
except Exception as e:
    print("SEDR fail", str(e)[:60])
nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
add("BANKSY-style", PCA(20, random_state=SEED).fit_transform(
    np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1)))
add("floor:nonspatial", PCA(30, random_state=SEED).fit_transform(Z))
add("floor:smoothed", PCA(30, random_state=SEED).fit_transform(nbr))

# ---- compare to DLPFC findings ----
print("\n=== seqFISH: backend confound (GMM-KMeans ARI) ===")
for m in results:
    d = results[m]["gmm"]["ARI"] - results[m]["kmeans"]["ARI"]
    print(f"  {m:>16}: ΔARI = {round(d, 4)}")

print("\n=== seqFISH ARI ranking (GMM) — does Tessera flip? ===")
rows = [{"config": m, **results[m]["gmm"]} for m in results]
rows.sort(key=lambda r: -r["ARI"])
print(format_table(rows, ["config"] + KEYS))
tr = [r["config"] for r in rows].index("Tessera") + 1
print(f"\nTessera ARI rank (GMM): {tr}/{len(rows)}  [DLPFC was 5/7]")
print("best ARI:", rows[0]["config"], rows[0]["ARI"])
json.dump(results, open("experiments/seqfish_bench.json", "w"), indent=2)
print("wrote experiments/seqfish_bench.json")
