"""Improvement probe: anneal the contrastive weight so training stops degrading.

Diagnosis: full+GMM peaks ~0.566 at e100 then decays as the contrastive term over-spreads the
embedding. Fix idea: keep contrastive early (builds boundary structure -> the peak), then anneal it
to ~0 (approach pure reconstruction -> GMM-friendly, stable). If ARI(gmm) holds near the peak
instead of decaying, anneal is the integration-worthy change. Compared against constant weight.
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from tessera_st.config import AblationConfig, ModelConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval import ari
from tessera_st.losses import boundary_contrastive_loss, reconstruction_loss
from tessera_st.model.graph import build_knn_edges
from tessera_st.model.tessera import TesseraNet

P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")

slide = load_h5ad(P, label_key="ground_truth")
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
dev = "cuda" if torch.cuda.is_available() else "cpu"
ei, ed = build_knn_edges(slide.coords, k=6)
x = torch.tensor(slide.expr, dtype=torch.float32, device=dev)
edge_index = torch.tensor(ei, dtype=torch.long, device=dev)
edge_dist = torch.tensor((ed - ed.mean()) / (ed.std() + 1e-8), dtype=torch.float32, device=dev)


def gmm_ari(z):
    return ari(slide.labels, GaussianMixture(n, covariance_type="tied", random_state=1,
                                             n_init=5).fit_predict(z))


def run(anneal):
    torch.manual_seed(1)
    gen = torch.Generator(device=dev).manual_seed(1)
    cfg = TrainConfig(device=dev)
    model = TesseraNet(ModelConfig(in_dim=x.shape[1]), n, AblationConfig()).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    base, anneal_to = cfg.contrastive_weight, 250
    traj = []
    for epoch in range(401):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index, edge_dist)
        w = base * max(0.0, 1 - epoch / anneal_to) if anneal else base
        loss = reconstruction_loss(out.recon, x)
        if w > 0:
            loss = loss + w * boundary_contrastive_loss(
                out.z, edge_index, out.gates[-1], cfg.contrastive_temp, cfg.n_negatives, gen)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                z = model(x, edge_index, edge_dist).z.cpu().numpy()
            traj.append((epoch, round(gmm_ari(z), 4)))
    return traj


print("DLPFC 151673 — ARI(gmm) vs epoch. STAGATE+gmm ref = 0.577\n")
print("  constant w: " + "  ".join(f"e{e}={a}" for e, a in run(False)))
print("  annealed w: " + "  ".join(f"e{e}={a}" for e, a in run(True)))
