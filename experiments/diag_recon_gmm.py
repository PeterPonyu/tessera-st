"""Ceiling probe under the WINNING recipe (GMM backend + reconstruction, STAGATE-style).

DEC hurt GMM; warmup (recon) + GMM already ~= STAGATE. So test which Tessera config, trained
longer and read out by GMM, climbs highest — and whether the boundary machinery (edge gating /
contrastive) helps or hurts once GMM is the backend. Single seed, ARI(gmm) vs epoch.
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from tessera_st.config import AblationConfig, ModelConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval import ari
from tessera_st.losses import total_loss
from tessera_st.model.graph import build_knn_edges
from tessera_st.model.tessera import TesseraNet
from _roots import data_root, external_root, spatial_omics_root

P = str(spatial_omics_root() / "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")

slide = load_h5ad(P, label_key="ground_truth")
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
dev = "cuda" if torch.cuda.is_available() else "cpu"

ei, ed = build_knn_edges(slide.coords, k=6)
x = torch.tensor(slide.expr, dtype=torch.float32, device=dev)
edge_index = torch.tensor(ei, dtype=torch.long, device=dev)
edge_dist = torch.tensor((ed - ed.mean()) / (ed.std() + 1e-8), dtype=torch.float32, device=dev)


def gmm_ari(z):
    lab = GaussianMixture(n, covariance_type="tied", random_state=1, n_init=5).fit_predict(z)
    return ari(slide.labels, lab)


CONFIGS = {
    "full(recon+contrast+gate+ms)": AblationConfig(),
    "recon+gate+ms(no_contrast)": AblationConfig(boundary_contrastive=False),
    "pure_recon(backbone)": AblationConfig(False, False, False, False),
}
cfg = TrainConfig(device=dev)
print("DLPFC 151673 — ARI(gmm) vs epoch. STAGATE+gmm ref = 0.577\n")
for name, abl in CONFIGS.items():
    torch.manual_seed(1)
    gen = torch.Generator(device=dev).manual_seed(1)
    model = TesseraNet(ModelConfig(in_dim=x.shape[1]), n, abl).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    traj = []
    for epoch in range(401):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index, edge_dist)
        loss, _ = total_loss(out, x, edge_index, cfg, abl, generator=gen)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                z = model(x, edge_index, edge_dist).z.cpu().numpy()
            traj.append((epoch, round(gmm_ari(z), 4)))
    print(f"{name:>30}: " + "  ".join(f"e{e}={a}" for e, a in traj))
