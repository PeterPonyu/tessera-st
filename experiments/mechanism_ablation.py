"""MECHANISM: why does Tessera win the true spatial-domain tasks? Run Tessera's curated component
ablation (full + 4 leave-one-out + backbone) on the two platforms it wins — MERFISH (domain) and
osmFISH (Region) — at best-config, 2 seeds. The component whose removal drops ARI most is the
mechanism. Hypothesis: the boundary-contrastive term is load-bearing (it was on DLPFC); confirming it
on the platforms Tessera actually wins ties the win to its boundary-aware design."""

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
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

DR = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/data"
EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"

from tessera_st.config import TrainConfig, curated_ablation_grid  # noqa: E402
from tessera_st.eval.refine import refine_labels  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1, 2]
PLATFORMS = {
    "MERFISH(domain)": (f"{DR}/baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad",
                        ["domain"]),
    "osmFISH(Region)": (f"{EXT}/../baselines/CellNiche-original/data/osmFISH_SScortex.h5ad",
                        ["Region"]),
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
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    return np.clip((X - mu) / sd, -10, 10), coords, true


def spca(Z, n, seed=0):
    return PCA(max(2, min(n, Z.shape[1] - 1, Z.shape[0] - 1)), random_state=seed).fit_transform(Z)


def best_config_ari(emb, true, coords, k, seed):
    best = -1.0
    m = true >= 0
    for lab0 in (KMeans(k, n_init=10, random_state=seed).fit_predict(emb),
                 GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=3,
                                 reg_covar=1e-4).fit_predict(emb)):
        for lab in (lab0, refine_labels(lab0, coords, k=6)):
            best = max(best, float(ari(true[m], lab[m])))
    return best


out = {}
for name, (path, gt) in PLATFORMS.items():
    Z, coords, true = load_any(path, gt)
    k = int(len(np.unique(true[true >= 0])))
    expr = spca(Z, min(50, k * 6)).astype(np.float32)
    print(f"\n=== {name}: {Z.shape[0]} cells, {k} classes ===", flush=True)
    rec = {}
    for cfg in curated_ablation_grid():
        aris = []
        for seed in SEEDS:
            res = fit_predict(expr, coords, k, cfg,
                              train_cfg=TrainConfig(epochs=120, seed=seed, device=str(dev)))
            aris.append(best_config_ari(res.embed, true, coords, k, seed))
        rec[cfg.name] = {"mean": round(float(np.mean(aris)), 4), "std": round(float(np.std(aris)), 4)}
        print(f"  {cfg.name:>26}: {rec[cfg.name]['mean']}±{rec[cfg.name]['std']}", flush=True)
    full = rec["full"]["mean"]
    # component contribution = full - (full with that component removed); larger drop = more load-bearing
    drops = {c.replace("no_", ""): round(full - rec[c]["mean"], 4)
             for c in rec if c.startswith("no_")}
    drops["all-components(vs backbone)"] = round(full - rec["backbone"]["mean"], 4)
    out[name] = {"per_config": rec, "component_drop": drops}
    print(f"  component contribution (full − leave-one-out): {drops}")

print("\n===== MECHANISM SUMMARY =====")
for name, r in out.items():
    d = r["component_drop"]
    key = max((c for c in d if c != "all-components(vs backbone)"), key=lambda c: d[c])
    print(f"  {name}: most load-bearing component = '{key}' (Δ {d[key]}) | all drops: {d}")
json.dump(out, open("experiments/mechanism_ablation.json", "w"), indent=2)
print("\nwrote experiments/mechanism_ablation.json")
