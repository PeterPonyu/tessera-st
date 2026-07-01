"""Resilient improvement #3 (re-examined): GAT attention is SELECTIVE aggregation, not smoothing —
so unlike BANKSY-aug / VAE it may raise ARI WITHOUT blurring boundaries. Earlier attention probe
only logged ARI; here we score the FULL panel (esp. boundary_F1) to see if attention is the rare
non-trade-off improvement: global structure up AND boundary sharpness held. vs Tessera-orig."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as gu
from sklearn.mixture import GaussianMixture

from tessera_st.ablation import _hard_confidence, _metric_row, format_table
from tessera_st.config import ModelConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval.markers import DLPFC_LAYER_MARKERS
from tessera_st.losses import boundary_contrastive_loss
from tessera_st.model.gating import EdgeGate
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


class AttnGateLayer(nn.Module):
    def __init__(self, dim, gate_hidden, dropout):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.a_src = nn.Linear(dim, 1, bias=False)
        self.a_dst = nn.Linear(dim, 1, bias=False)
        self.gate = EdgeGate(dim, hidden=gate_hidden, enabled=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, ei, ed):
        src, dst = ei[0], ei[1]
        Wh = self.W(h)
        e = F.leaky_relu(self.a_src(Wh).squeeze(-1)[src] + self.a_dst(Wh).squeeze(-1)[dst], 0.2)
        alpha = gu.softmax(e, dst)
        g = self.gate(h, ei, ed)
        w = (alpha * g).unsqueeze(-1)
        agg = gu.scatter(w * Wh[src], dst, dim=0, dim_size=h.shape[0], reduce="sum")
        return self.norm(h + self.drop(F.relu(agg))), g


class AttnNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            AttnGateLayer(cfg.hidden_dim, cfg.gate_hidden, cfg.dropout)
            for _ in range(cfg.n_layers))
        self.project = nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        self.dec = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.hidden_dim), nn.ReLU(),
                                 nn.Linear(cfg.hidden_dim, cfg.in_dim))

    def forward(self, x, ei, ed):
        h = F.relu(self.input(x))
        g = None
        for layer in self.layers:
            h, g = layer(h, ei, ed)
        z = self.project(h)
        return z, self.dec(z), g


def gmm(emb, seed):
    return GaussianMixture(n, covariance_type="tied", random_state=seed, n_init=5,
                           reg_covar=1e-4).fit_predict(emb)


def run(epochs):
    rows = []
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        gen = torch.Generator(device=dev).manual_seed(seed)
        model = AttnNet(mcfg).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            z, recon, g = model(x, edge_index, edge_dist)
            loss = F.mse_loss(recon, x) + tcfg.contrastive_weight * boundary_contrastive_loss(
                z, edge_index, g, tcfg.contrastive_temp, tcfg.n_negatives, gen)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index, edge_dist)[0].cpu().numpy()
        lab = gmm(z, seed)
        rows.append(_metric_row("x", slide.coords, slide.labels, lab, z,
                                _hard_confidence(z, lab), kind="t", layer_scores=ls))
    s = {"config": f"Tessera-Attn(ep={epochs})"}
    for key in KEYS:
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            s[key] = round(float(np.mean(v)), 4)
    return s


summaries = []
for ep in [120, 250]:
    summaries.append(run(ep))
    print(f"ep={ep}: ARI={summaries[-1]['ARI']} bF1={summaries[-1]['boundary_F1']} "
          f"NMI={summaries[-1]['NMI']} ECE={summaries[-1]['ECE']}")

print("\nref: Tessera-orig ARI0.528 bF1 0.453 | STAGATE ARI0.580 bF10.406 | SEDR ARI0.570 bF10.412\n")
print(format_table(summaries, ["config"] + KEYS))
