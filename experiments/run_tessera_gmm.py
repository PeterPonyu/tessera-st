"""Symmetric fairness: evaluate Tessera under BOTH clustering backends (KMeans and GMM).

STAGATE jumped 0.246->0.577 just by switching KMeans->GMM. Before concluding anything about
Tessera-vs-SOTA we must give Tessera the same backend choice. Each config+seed is trained ONCE;
both backends score the same embedding.
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from tessera_st.ablation import _hard_confidence, _metric_row, format_table
from tessera_st.config import AblationConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval.baselines import kmeans_labels
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS

P = (
    "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
    "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad"
)


def gmm_labels(emb, k, seed):
    return GaussianMixture(n_components=k, covariance_type="tied", random_state=seed,
                           n_init=5).fit_predict(emb)


slide = load_h5ad(P, label_key="ground_truth", marker_dict=DLPFC_LAYER_MARKERS)
ls = slide.layer_marker_scores
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
configs = {"full": AblationConfig(), "no_edge_gating": AblationConfig(edge_gating=False)}
keys = ["ARI", "NMI", "CHAOS", "PAS", "boundary_F1", "small_IoU", "ECE", "marker_purity"]

summaries = []
for cname, cfg in configs.items():
    rows = {"kmeans": [], "gmm": []}
    for seed in [1, 2, 3]:
        res = fit = None
        from tessera_st.train import fit_predict
        res = fit_predict(slide.expr, slide.coords, n, cfg,
                          train_cfg=TrainConfig(epochs=120, seed=seed, device="cuda"))
        for backend, fn in [("kmeans", kmeans_labels), ("gmm", gmm_labels)]:
            lab = fn(res.embed, n, seed)
            conf = _hard_confidence(res.embed, lab)
            rows[backend].append(
                _metric_row(f"{cname}+{backend}.s{seed}", slide.coords, slide.labels, lab,
                            res.embed, conf, kind="tessera", layer_scores=ls))
    for backend in ("kmeans", "gmm"):
        s = {"config": f"Tessera.{cname}+{backend}", "kind": "tessera"}
        for k in keys:
            vals = [r[k] for r in rows[backend] if r.get(k) is not None]
            if vals:
                s[k] = round(float(np.mean(vals)), 4)
                s[k + "_std"] = round(float(np.std(vals)), 4)
        summaries.append(s)
    print(f"{cname}: kmeans ARI={summaries[-2]['ARI']} | gmm ARI={summaries[-1]['ARI']}")

print()
print(format_table(summaries))
import json  # noqa: E402

json.dump({"summaries": summaries}, open("experiments/tessera_gmm_151673.json", "w"), indent=2)
print("\nwrote experiments/tessera_gmm_151673.json")
