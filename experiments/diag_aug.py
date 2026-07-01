"""Resilient improvement #1: fuse BANKSY's winning mechanism (neighbour feature augmentation) into
Tessera's input. Tessera's weakest axis is spatial coherence (CHAOS/PAS) and floor:smoothed beats it
on ARI — both because Tessera doesn't explicitly use neighbour means. Hypothesis: augmenting the
input with neighbour-mean features (BANKSY-style) lifts ARI + CHAOS while the edge gate keeps the
boundary_F1 lead. Compares Tessera-orig vs Tessera-aug(lam) under GMM, full metric set, 3 seeds."""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from tessera_st.ablation import _hard_confidence, _metric_row, format_table
from tessera_st.config import AblationConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS
from tessera_st.train import fit_predict

P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")
slide = load_h5ad(P, label_key="ground_truth", marker_dict=DLPFC_LAYER_MARKERS)
ls = slide.layer_marker_scores
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
dev = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "boundary_F1", "small_IoU", "ECE",
        "marker_purity"]

# raw scaled-HVG features behind Tessera's PCA input (reconstruct from slide.expr is lossy, so
# rebuild from the loader's pipeline): use slide.expr (already PCA50). For augmentation we need the
# pre-PCA space, so approximate neighbour mean in the PCA space itself (linear, so PCA(neighbour
# mean) == neighbour mean of PCA — valid).
base = slide.expr  # (n, 50) PCA
idx = NearestNeighbors(n_neighbors=7).fit(slide.coords).kneighbors(slide.coords)[1][:, 1:]
nbr = base[idx].mean(1)


def gmm(emb, seed):
    return GaussianMixture(n, covariance_type="tied", random_state=seed, n_init=5,
                           reg_covar=1e-4).fit_predict(emb)


def run(expr, name):
    rows = []
    for seed in [1, 2, 3]:
        r = fit_predict(expr.astype(np.float32), slide.coords, n, AblationConfig(),
                        train_cfg=TrainConfig(epochs=120, seed=seed, device=dev))
        lab = gmm(r.embed, seed)
        rows.append(_metric_row("x", slide.coords, slide.labels, lab, r.embed,
                                _hard_confidence(r.embed, lab), kind="t", layer_scores=ls))
    s = {"config": name}
    for key in KEYS:
        v = [x[key] for x in rows if x.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
    return s


summaries = [run(base, "Tessera-orig")]
for lam in [0.3, 0.5, 0.7]:
    aug = np.concatenate([np.sqrt(1 - lam) * base, np.sqrt(lam) * nbr], axis=1)
    summaries.append(run(aug, f"Tessera-aug(lam={lam})"))
    print(f"lam={lam}: ARI={summaries[-1]['ARI']} CHAOS={summaries[-1]['CHAOS']} "
          f"bF1={summaries[-1]['boundary_F1']}")

print("\nref: STAGATE ARI0.580 CHAOS1.005 bF10.406 | SEDR ARI0.570 | floor:smoothed ARI0.532 "
      "bF10.432 | Tessera-orig bF1 0.454\n")
print(format_table(summaries, ["config"] + KEYS))
