"""THE comparison: Tessera vs 4 real SOTA (STAGATE/SEDR/GraphST/BANKSY) + 2 floors, full 11-metric
panel, fair GMM(reg_covar) backend, 3 seeds, on DLPFC 151673. Reports Tessera's score and a
per-metric Tessera-vs-each-SOTA analysis. numba-stub lets SEDR/GraphST run without changing the env.
Unified backend =横向可比; each SOTA's own recommended backend may score higher (noted)."""

import sys

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
for sub in ("STAGATE", "SEDR", "GraphST"):
    sys.path.insert(0, f"{EXT}/{sub}")
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
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU", "ECE",
        "marker_purity"]


def gmm(emb, seed):
    return GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=5,
                           reg_covar=1e-4).fit_predict(emb)


def score(name, embeds, kind):
    rows = []
    for seed, emb in embeds:
        lab = gmm(emb, seed)
        rows.append(_metric_row("x", coords, true, lab, emb, _hard_confidence(emb, lab),
                                kind=kind, layer_scores=ls))
    s = {"config": name, "kind": kind}
    for key in KEYS:
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
    return s


summaries = []


def add(name, fn, kind="sota"):
    try:
        summaries.append(score(name, fn(), kind))
        print(f"{name}: ARI={summaries[-1].get('ARI')}")
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"{name} FAILED: {str(e)[:90]}")


# Tessera
def tessera():
    out = []
    expr = PCA(min(50, k * 8), random_state=0).fit_transform(Z).astype(np.float32)
    for s in [1, 2, 3]:
        r = fit_predict(expr, coords, k, AblationConfig(),
                        train_cfg=TrainConfig(epochs=120, seed=s, device=str(dev)))
        out.append((s, r.embed))
    return out


add("Tessera.full", tessera, "tessera")


def stagate():
    import STAGATE_pyG as ST
    a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
    out = []
    for s in [1, 2, 3]:
        ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False)
        torch.manual_seed(s)
        o = ST.train_STAGATE(a, n_epochs=1000, random_seed=s, device=dev, verbose=False)
        out.append((s, np.asarray(o.obsm["STAGATE"])))
    return out


add("SOTA:STAGATE", stagate)


def sedr():
    import SEDR
    a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
    a.obsm["X_pca"] = PCA(200, random_state=0).fit_transform(Z).astype(np.float32)
    gd = SEDR.graph_construction(a, 6)
    out = []
    for s in [1, 2, 3]:
        torch.manual_seed(s)
        net = SEDR.Sedr(a.obsm["X_pca"], gd, device=str(dev))
        net.train_without_dec(N=1)
        feat, _, _, _ = net.process()
        out.append((s, np.asarray(feat)))
    return out


add("SOTA:SEDR", sedr)


def graphst():
    from GraphST import GraphST as G
    out = []
    for s in [1, 2, 3]:
        a = ad.AnnData(X=ln[:, hvg].astype(np.float32)); a.obsm["spatial"] = coords
        m = G.GraphST(a, device=dev, epochs=600, random_seed=s)
        o = m.train()
        out.append((s, np.asarray(o.obsm["emb"])))
    return out


add("SOTA:GraphST", graphst)


def banksy():
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    aug = np.concatenate([np.sqrt(0.2) * Z, np.sqrt(0.8) * nbr], 1)
    return [(s, PCA(20, random_state=s).fit_transform(aug)) for s in [1, 2, 3]]


add("SOTA:BANKSY-style", banksy)


def floor_nonspatial():
    return [(s, PCA(30, random_state=s).fit_transform(Z)) for s in [1, 2, 3]]


def floor_smoothed():
    nbr = Z[NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)[1][:, 1:]].mean(1)
    return [(s, PCA(30, random_state=s).fit_transform(nbr)) for s in [1, 2, 3]]


add("floor:nonspatial", floor_nonspatial, "floor")
add("floor:smoothed", floor_smoothed, "floor")

print("\n" + format_table(summaries, ["config"] + KEYS))

# Tessera vs each method, per metric
tess = next(s for s in summaries if s["config"] == "Tessera.full")
from tessera_st.ablation import HIGHER_IS_BETTER  # noqa: E402

print("\nTessera vs each (✓ = Tessera better):")
for other in summaries:
    if other["config"] == "Tessera.full":
        continue
    wins = []
    for m in KEYS:
        a, b = tess.get(m), other.get(m)
        if a is None or b is None:
            continue
        wins.append(f"{m}{'✓' if (a - b) * HIGHER_IS_BETTER[m] > 0 else '✗'}")
    nw = sum("✓" in w for w in wins)
    print(f"  vs {other['config']:>18}: Tessera wins {nw}/{len(wins)} | " + " ".join(wins))

import json  # noqa: E402
json.dump({"summaries": summaries}, open("experiments/full_panel_v2.json", "w"), indent=2)
print("\nwrote experiments/full_panel_v2.json")
