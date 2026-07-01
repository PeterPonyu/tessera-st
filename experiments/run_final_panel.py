"""Final comprehensive panel: Tessera vs runnable SOTA (STAGATE, BANKSY-style), full metric set
incl. internal ASW/DBI/CAL, fair GMM backend, 3 seeds. Merges the SOTA panel with a fresh
full-metric Tessera run so every method is compared on every axis."""

import json

import numpy as np
from sklearn.mixture import GaussianMixture

from tessera_st.ablation import _hard_confidence, _metric_row, format_table
from tessera_st.config import AblationConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS
from tessera_st.train import fit_predict

P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU", "ECE",
        "marker_purity"]

slide = load_h5ad(P, label_key="ground_truth", marker_dict=DLPFC_LAYER_MARKERS)
ls = slide.layer_marker_scores
n = int(len(np.unique(slide.labels[slide.labels >= 0])))


def gmm(emb, seed):
    return GaussianMixture(n, covariance_type="tied", random_state=seed, n_init=5).fit_predict(emb)


def aggregate(rows, name, kind):
    s = {"config": name, "kind": kind}
    for key in KEYS:
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
            s[key + "_std"] = round(float(np.std(v)), 4)
    return s


rows = []
for seed in [1, 2, 3]:
    res = fit_predict(slide.expr, slide.coords, n, AblationConfig(),
                      train_cfg=TrainConfig(epochs=120, seed=seed, device="cuda"))
    lab = gmm(res.embed, seed)
    rows.append(_metric_row(f"Tessera.full+gmm.s{seed}", slide.coords, slide.labels, lab,
                            res.embed, _hard_confidence(res.embed, lab), kind="tessera",
                            layer_scores=ls))
tess = aggregate(rows, "Tessera.full", "tessera")

sota = json.load(open("experiments/sota_panel_151673.json"))["summaries"]
allrows = sota + [tess]

cols = ["config", "ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU",
        "ECE", "marker_purity"]
print(format_table(allrows, cols))

# per-method win count (best on each metric)
from tessera_st.ablation import HIGHER_IS_BETTER  # noqa: E402

print("\nper-metric winner:")
for m in ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU", "ECE",
          "marker_purity"]:
    d = HIGHER_IS_BETTER.get(m, 1)
    best = max((r for r in allrows if r.get(m) is not None), key=lambda r: d * r[m], default=None)
    if best:
        print(f"  {m:>13}: {best['config']} ({best[m]})")
json.dump({"rows": allrows}, open("experiments/final_panel_151673.json", "w"), indent=2)
print("\nwrote experiments/final_panel_151673.json")
