"""Under-fitting diagnostic: does ARI rise / variance fall as we train longer?

Tessera was rough-tested at only 120 epochs; graph-autoencoder domain methods commonly train
~1000. If 0.42±0.04 is an under-fitting artefact, ARI should climb and the seed std should
shrink with more epochs. Runs the FULL config only (cheap), 3 seeds per epoch budget.
"""

import numpy as np

from tessera_st.config import AblationConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval import ari, nmi
from tessera_st.train import fit_predict

P = (
    "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
    "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad"
)

slide = load_h5ad(P, label_key="ground_truth")
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
full = AblationConfig()  # all components on
print(f"DLPFC 151673: {slide.expr.shape}, {n} layers; config=full\n")

for ep in [120, 400, 800]:
    aris, nmis = [], []
    for sd in [1, 2, 3]:
        cfg = TrainConfig(epochs=ep, seed=sd, device="cuda")
        r = fit_predict(slide.expr, slide.coords, n, full, train_cfg=cfg)
        aris.append(ari(slide.labels, r.labels))
        nmis.append(nmi(slide.labels, r.labels))
    a, m = np.array(aris), np.array(nmis)
    print(f"epochs={ep:>4}: ARI {a.mean():.4f} ± {a.std():.4f}  NMI {m.mean():.4f} ± {m.std():.4f}"
          f"   seeds={[round(x, 3) for x in aris]}")
