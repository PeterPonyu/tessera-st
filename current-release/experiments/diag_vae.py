"""Resilient improvement #2: fuse SEDR's winning mechanism (variational latent / VGAE) into Tessera.

SEDR leads NMI/small_IoU/ECE/marker via a variational autoencoder. Variational latent + KL acts at
the DISTRIBUTION level (global smoothness, calibration), whereas Tessera's edge gate acts at the
message-passing level (boundary sharpness) — so unlike BANKSY augmentation they may NOT conflict.
Test: Tessera encoder + variational head (mu/logvar, reparam, KL) + boundary-contrastive, vs
Tessera-orig. Does ARI/ECE rise while boundary_F1 holds? Full metrics, GMM, 3 seeds, 151673."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from tessera_st.ablation import _hard_confidence, _metric_row, format_table
from tessera_st.config import ModelConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS
from tessera_st.losses import boundary_contrastive_loss
from tessera_st.model.encoder import GatedEncoder
from tessera_st.model.graph import build_knn_edges
from _roots import data_root, external_root, spatial_omics_root

P = str(spatial_omics_root() / "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")
slide = load_h5ad(P, label_key="ground_truth", marker_dict=DLPFC_LAYER_MARKERS)
ls = slide.layer_marker_scores
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
dev = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "boundary_F1", "small_IoU", "ECE",
        "marker_purity"]

ei, ed = build_knn_edges(slide.coords, k=6)
x = torch.tensor(slide.expr, dtype=torch.float32, device=dev)
edge_index = torch.tensor(ei, dtype=torch.long, device=dev)
edge_dist = torch.tensor((ed - ed.mean()) / (ed.std() + 1e-8), dtype=torch.float32, device=dev)
mcfg = ModelConfig(in_dim=x.shape[1])
tcfg = TrainConfig(device=dev)


class VarTessera(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = GatedEncoder(cfg, gating=True, multi_scale=True)
        self.to_mu = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.to_lv = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dec = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.hidden_dim), nn.ReLU(),
                                 nn.Linear(cfg.hidden_dim, cfg.in_dim))

    def forward(self, x, ei, ed):
        z0, boundary, gates = self.enc(x, ei, ed)
        mu, lv = self.to_mu(z0), self.to_lv(z0).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return z, self.dec(z), gates, mu, lv


def gmm(emb, seed):
    return GaussianMixture(n, covariance_type="tied", random_state=seed, n_init=5,
                           reg_covar=1e-4).fit_predict(emb)


def run_vae(kl_w):
    rows = []
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        gen = torch.Generator(device=dev).manual_seed(seed)
        model = VarTessera(mcfg).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        for _ in range(120):
            model.train()
            opt.zero_grad()
            z, recon, gates, mu, lv = model(x, edge_index, edge_dist)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = (F.mse_loss(recon, x) + kl_w * kl
                    + tcfg.contrastive_weight * boundary_contrastive_loss(
                        z, edge_index, gates[-1], tcfg.contrastive_temp, tcfg.n_negatives, gen))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index, edge_dist)[0].cpu().numpy()
        lab = gmm(z, seed)
        rows.append(_metric_row("x", slide.coords, slide.labels, lab, z,
                                _hard_confidence(z, lab), kind="t", layer_scores=ls))
    s = {"config": f"Tessera-VAE(kl={kl_w})"}
    for key in KEYS:
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
    return s


summaries = []
for kl_w in [0.001, 0.01, 0.05]:
    summaries.append(run_vae(kl_w))
    print(f"kl={kl_w}: ARI={summaries[-1]['ARI']} bF1={summaries[-1]['boundary_F1']} "
          f"ECE={summaries[-1]['ECE']} CHAOS={summaries[-1]['CHAOS']}")

print("\nref: Tessera-orig ARI0.528 bF1 0.453 ECE0.205 | STAGATE ARI0.580 | SEDR ARI0.570 ECE0.189\n")
print(format_table(summaries, ["config"] + KEYS))
