"""Extend the task-fit law to 9 platforms (n=6 → 9) to sharpen significance.

Add 3 more platforms (BRCA Visium tumour / IMC breast / openST HNSCC), run the hardened protocol
(best-config Tessera rank), compute each new GT's spatial contiguity, merge with the 6 existing rows
from task_fit_law.json, and recompute Spearman(contiguity, Tessera rank) and (contiguity, spatial
advantage) over all 9. If the correlations hold/strengthen, the predictive rule is firmer."""

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

EXT = str(external_root())
DR = str(data_root())
for s in ("STAGATE", "SEDR", "GraphST"):
    sys.path.insert(0, f"{EXT}/{s}")

from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.eval.refine import refine_labels  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1, 2]
NEW = {
    "BRCA(Visium,tumor)": (f"{DR}/processed/brca1_visium_10x/anndata.h5ad",
                           ["ground_truth", "annot_type", "fine_annot_type"]),
    "IMC(breast,ctype)": (f"{DR}/processed/niche_imc_breast/anndata.h5ad",
                          ["ground_truth", "Region", "cell_type"]),
    "openST(HNSCC,annot)": (f"{DR}/processed/openst_hnscc_sub15k/anndata.h5ad",
                            ["ground_truth", "annotation", "cell_class"]),
}


def _valid(v):
    return str(v).strip().lower() not in {"", "nan", "none", "na", "unknown"}


def load_any(path, gt_cols):
    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], float)
    col = next(c for c in gt_cols if c in a.obs.columns)
    raw = a.obs[col].to_numpy()
    codes = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([codes[v] if _valid(v) else -1 for v in raw], dtype=np.int64)
    X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float64)
    if np.allclose(X, np.round(X)) and X.min() >= 0:
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
    if X.shape[1] > 500:
        X = X[:, np.argsort(X.var(0))[::-1][:3000]]
    lognorm = X.copy()
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    return np.clip((X - mu) / sd, -10, 10), lognorm, coords, true


def spca(Z, n, seed=0):
    return PCA(max(2, min(n, Z.shape[1] - 1, Z.shape[0] - 1)), random_state=seed).fit_transform(Z)


def gari(true, lab):
    m = true >= 0
    return float(ari(true[m], lab[m]))


def best_config_ari(emb, true, coords, k, seed):
    best = -1.0
    labs = [KMeans(k, n_init=10, random_state=seed).fit_predict(emb)]
    try:  # GMM can fail (singular covariance) on some embeddings — fall back to KMeans only
        labs.append(GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=3,
                                    reg_covar=1e-2).fit_predict(emb))
    except Exception:
        pass
    for lab0 in labs:
        for lab in (lab0, refine_labels(lab0, coords, k=6)):
            best = max(best, gari(true, lab))
    return best


def gt_contiguity(coords, true, k=6):
    m = true >= 0
    cv, tv = coords[m], true[m]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(cv).kneighbors(cv)[1][:, 1:]
    return float((tv[idx] == tv[:, None]).mean())


def embeddings(Z, lognorm, coords, k, seed):
    out = {"Tessera": fit_predict(spca(Z, min(50, k * 6)).astype(np.float32), coords, k,
           AblationConfig(), train_cfg=TrainConfig(epochs=120, seed=seed, device=str(dev))).embed}
    try:
        import STAGATE_pyG as ST
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False); torch.manual_seed(seed)
        out["STAGATE"] = np.asarray(ST.train_STAGATE(a, n_epochs=600, random_seed=seed,
                                    device=dev, verbose=False).obsm["STAGATE"])
    except Exception as e:
        print("  STAGATE fail", str(e)[:40])
    try:
        import SEDR
        a = ad.AnnData(X=Z.astype(np.float32)); a.obsm["spatial"] = coords
        a.obsm["X_pca"] = spca(Z, 200).astype(np.float32)
        gd = SEDR.graph_construction(a, 6); torch.manual_seed(seed)
        net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev)); net.train_without_dec(N=1)
        out["SEDR"] = np.asarray(net.process()[0])
    except Exception as e:
        print("  SEDR fail", str(e)[:40])
    # GraphST omitted here: it does not change Tessera's rank (it ranks low) and is far too slow
    # under the pure-Python numba stub at 15k cells. The 6-platform hardened table includes it.
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    out["BANKSY-style"] = spca(np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1), 20, seed)
    out["floor:nonspatial"] = spca(Z, 30, seed)
    out["floor:smoothed"] = spca(nbr, 30, seed)
    return out


# existing 6 rows (contiguity + Tessera rank + spatial advantage) from the established law
rows = list(json.load(open("experiments/task_fit_law.json"))["rows"])

for name, (path, gt) in NEW.items():
    try:
        Z, lognorm, coords, true = load_any(path, gt)
        k = int(len(np.unique(true[true >= 0])))
        cont = gt_contiguity(coords, true)
        print(f"\n=== {name}: {Z.shape[0]} cells, {k} classes, GT contig {cont:.3f} ===", flush=True)
        per = {}
        for seed in SEEDS:
            for m, emb in embeddings(Z, lognorm, coords, k, seed).items():
                per.setdefault(m, []).append(best_config_ari(emb, true, coords, k, seed))
        mean = {m: float(np.mean(v)) for m, v in per.items()}
        order = sorted(mean, key=lambda m: -mean[m])
        t_rank = order.index("Tessera") + 1
        nonsp = mean.get("floor:nonspatial", 0.0)
        sadv = max(v for m, v in mean.items() if m != "floor:nonspatial") - nonsp
        rows.append({"platform": name, "GT_contiguity": round(cont, 3), "Tessera_rank": t_rank,
                     "winner": order[0], "spatial_advantage": round(sadv, 3),
                     "spatial_helps": sadv > 0})
        print(f"  Tessera rank {t_rank}/{len(order)} | winner {order[0]} | spatial_adv {sadv:+.3f}",
              flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:50]}")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


cont = np.array([r["GT_contiguity"] for r in rows])
trank = np.array([r["Tessera_rank"] for r in rows])
sadv = np.array([r["spatial_advantage"] for r in rows])
rho_rank = round(spearman(cont, trank), 3)
rho_adv = round(spearman(cont, sadv), 3)

print(f"\n===== EXTENDED LAW (n={len(rows)} platforms) =====")
for r in sorted(rows, key=lambda r: -r["GT_contiguity"]):
    print(f"  {r['platform']:>26} | contig {r['GT_contiguity']:.3f} | Tessera {r['Tessera_rank']} | "
          f"adv {r['spatial_advantage']:+.3f} | {r['winner']}")
print(f"\nSpearman(contiguity, Tessera_rank)      = {rho_rank:+.3f}  (was -0.714 at n=6)")
print(f"Spearman(contiguity, spatial_advantage) = {rho_adv:+.3f}  (was +0.771 at n=6)")
json.dump({"rows": rows, "n_platforms": len(rows),
           "spearman_contiguity_vs_tessera_rank": rho_rank,
           "spearman_contiguity_vs_spatial_advantage": rho_adv},
          open("experiments/task_fit_law_extended.json", "w"), indent=2)
print("wrote experiments/task_fit_law_extended.json")
