"""Resilient improvement #4: DECOUPLE boundary and global into separate subspaces.

The 3 mechanism-transfers all traded off because contrastive (boundary) and recon (global) fought in
ONE embedding. Fix: split the projection into z_global (optimised by reconstruction only) and
z_boundary (optimised by boundary-contrastive only); cluster on concat[z_global, z_boundary]. They
no longer compete, so we may keep STAGATE-like global ARI AND Tessera's boundary_F1. vs orig + ablate
the split. Full metrics, GMM, 3 seeds, 151673."""

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

P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")
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


class Decoupled(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = GatedEncoder(cfg, gating=True, multi_scale=True)  # gives z0 (embed_dim), gates
        d = cfg.embed_dim
        self.to_g = nn.Linear(d, d)   # global subspace
        self.to_b = nn.Linear(d, d)   # boundary subspace
        self.dec = nn.Sequential(nn.Linear(d, cfg.hidden_dim), nn.ReLU(),
                                 nn.Linear(cfg.hidden_dim, cfg.in_dim))  # from z_global only

    def forward(self, x, ei, ed):
        z0, boundary, gates = self.enc(x, ei, ed)
        zg, zb = self.to_g(z0), self.to_b(z0)
        return zg, zb, self.dec(zg), gates


def gmm(emb, seed):
    return GaussianMixture(n, covariance_type="tied", random_state=seed, n_init=5,
                           reg_covar=1e-4).fit_predict(emb)


def run(use_concat, label):
    rows = []
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        gen = torch.Generator(device=dev).manual_seed(seed)
        model = Decoupled(mcfg).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        for _ in range(120):
            model.train()
            opt.zero_grad()
            zg, zb, recon, gates = model(x, edge_index, edge_dist)
            # global subspace <- reconstruction; boundary subspace <- contrastive (decoupled)
            loss = F.mse_loss(recon, x) + tcfg.contrastive_weight * boundary_contrastive_loss(
                zb, edge_index, gates[-1], tcfg.contrastive_temp, tcfg.n_negatives, gen)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            zg, zb, _, _ = model(x, edge_index, edge_dist)
            emb = (torch.cat([zg, zb], dim=1) if use_concat else zg).cpu().numpy()
        lab = gmm(emb, seed)
        rows.append(_metric_row("x", slide.coords, slide.labels, lab, emb,
                                _hard_confidence(emb, lab), kind="t", layer_scores=ls))
    s = {"config": label}
    for key in KEYS:
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
    return s


summaries = []
for use_concat, label in [(False, "global-only(z_g)"), (True, "decoupled concat[z_g,z_b]")]:
    summaries.append(run(use_concat, label))
    print(f"{label}: ARI={summaries[-1]['ARI']} bF1={summaries[-1]['boundary_F1']} "
          f"NMI={summaries[-1]['NMI']} CHAOS={summaries[-1]['CHAOS']}")

print("\nref: Tessera-orig ARI0.528 bF1 0.453 | STAGATE ARI0.580 bF10.406 | SEDR ARI0.570\n")
print(format_table(summaries, ["config"] + KEYS))
