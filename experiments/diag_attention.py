"""Architecture probe: GAT-style attention + Tessera's boundary gate (the un-tried fix).

Diagnosis converged: Tessera's mean-aggregation reconstruction backbone is weaker than STAGATE's
graph-ATTENTION, so loss-schedule tweaks (DEC, anneal) can't pass STAGATE. Here we fuse learned
attention (alpha_ij, GAT-style) WITH the boundary gate (g_ij): message weight = alpha_ij * g_ij.
If GMM ARI clears STAGATE's 0.577 and holds, this architecture is worth integrating into src.
Clean-room: GAT is a general idea (pyG primitives), no STAGATE code copied.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as gu
from sklearn.mixture import GaussianMixture

from tessera_st.config import ModelConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval import ari
from tessera_st.losses import boundary_contrastive_loss, reconstruction_loss
from tessera_st.model.gating import EdgeGate
from tessera_st.model.graph import build_knn_edges

P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")


class AttnGateLayer(nn.Module):
    def __init__(self, dim, gate_hidden, gating, dropout):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.a_src = nn.Linear(dim, 1, bias=False)
        self.a_dst = nn.Linear(dim, 1, bias=False)
        self.gate = EdgeGate(dim, hidden=gate_hidden, enabled=gating)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_dist):
        src, dst = edge_index[0], edge_index[1]
        Wh = self.W(h)
        e = F.leaky_relu(self.a_src(Wh).squeeze(-1)[src] + self.a_dst(Wh).squeeze(-1)[dst], 0.2)
        alpha = gu.softmax(e, dst)                      # attention over incoming edges
        g = self.gate(h, edge_index, edge_dist)         # boundary gate
        w = (alpha * g).unsqueeze(-1)                   # fused weight
        agg = gu.scatter(w * Wh[src], dst, dim=0, dim_size=h.shape[0], reduce="sum")
        return self.norm(h + self.drop(F.relu(agg))), g


class AttnNet(nn.Module):
    def __init__(self, cfg, gating=True):
        super().__init__()
        self.input = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            AttnGateLayer(cfg.hidden_dim, cfg.gate_hidden, gating, cfg.dropout)
            for _ in range(cfg.n_layers))
        self.project = nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        self.decoder = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.hidden_dim), nn.ReLU(),
                                     nn.Linear(cfg.hidden_dim, cfg.in_dim))

    def forward(self, x, ei, ed):
        h = F.relu(self.input(x))
        g = None
        for layer in self.layers:
            h, g = layer(h, ei, ed)
        z = self.project(h)
        return z, self.decoder(z), g


slide = load_h5ad(P, label_key="ground_truth")
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
dev = "cuda" if torch.cuda.is_available() else "cpu"
ei, ed = build_knn_edges(slide.coords, k=6)
x = torch.tensor(slide.expr, dtype=torch.float32, device=dev)
edge_index = torch.tensor(ei, dtype=torch.long, device=dev)
edge_dist = torch.tensor((ed - ed.mean()) / (ed.std() + 1e-8), dtype=torch.float32, device=dev)
mcfg = ModelConfig(in_dim=x.shape[1])
tcfg = TrainConfig(device=dev)


def gmm_ari(z):
    return ari(slide.labels, GaussianMixture(n, covariance_type="tied", random_state=1,
                                             n_init=5).fit_predict(z))


print("DLPFC 151673 — attention+gate encoder, ARI(gmm) vs epoch. STAGATE+gmm = 0.577\n")
for seed in [1, 2, 3]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model = AttnNet(mcfg, gating=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    traj = []
    for epoch in range(501):
        model.train()
        opt.zero_grad()
        z, recon, g = model(x, edge_index, edge_dist)
        loss = reconstruction_loss(recon, x) + tcfg.contrastive_weight * boundary_contrastive_loss(
            z, edge_index, g, tcfg.contrastive_temp, tcfg.n_negatives, gen)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                zz = model(x, edge_index, edge_dist)[0].cpu().numpy()
            traj.append((epoch, round(gmm_ari(zz), 4)))
    print(f"seed {seed}: " + "  ".join(f"e{e}={a}" for e, a in traj))
