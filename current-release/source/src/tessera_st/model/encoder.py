"""Edge-gated multi-scale message passing (Components 1 & 2)."""

from __future__ import annotations

import torch
from torch import nn

from tessera_st.model.gating import EdgeGate


def scatter_mean_gated(
    messages: torch.Tensor,
    dst: torch.Tensor,
    gate: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Gate-weighted mean aggregation of neighbour messages into destination nodes.

    out_i = sum_j g_ij * msg_j  /  (sum_j g_ij + eps)
    With gate==1 everywhere this is plain neighbourhood mean (the over-smoothing baseline).
    """
    dim = messages.shape[1]
    weighted = messages * gate.unsqueeze(-1)
    out = torch.zeros(n_nodes, dim, device=messages.device, dtype=messages.dtype)
    out.index_add_(0, dst, weighted)
    norm = torch.zeros(n_nodes, device=messages.device, dtype=messages.dtype)
    norm.index_add_(0, dst, gate)
    return out / (norm.unsqueeze(-1) + 1e-8)


class GatedLayer(nn.Module):
    def __init__(self, dim: int, gate_hidden: int, gating: bool, dropout: float):
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = EdgeGate(dim, hidden=gate_hidden, enabled=gating)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_dist):
        src, dst = edge_index[0], edge_index[1]
        g = self.gate(h, edge_index, edge_dist)
        msg = self.transform(h)[src]
        agg = scatter_mean_gated(msg, dst, g, h.shape[0])
        h = self.norm(h + self.drop(torch.relu(agg)))  # residual: extra anti-over-smoothing
        return h, g


class MultiScaleFusion(nn.Module):
    """Component 2 — attention over per-layer embeddings (the 'scales')."""

    def __init__(self, dim: int, n_scales: int, enabled: bool):
        super().__init__()
        self.enabled = enabled
        self.score = nn.Linear(dim, 1)
        self.n_scales = n_scales

    def forward(self, scales: list[torch.Tensor]) -> torch.Tensor:
        if not self.enabled:
            return scales[-1]  # last layer only
        stacked = torch.stack(scales, dim=1)  # (n, S, dim)
        attn = torch.softmax(self.score(stacked).squeeze(-1), dim=1)  # (n, S)
        return (stacked * attn.unsqueeze(-1)).sum(dim=1)


class GatedEncoder(nn.Module):
    def __init__(self, cfg, gating: bool, multi_scale: bool):
        super().__init__()
        self.input = nn.Linear(cfg.in_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            GatedLayer(cfg.hidden_dim, cfg.gate_hidden, gating, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.fusion = MultiScaleFusion(cfg.hidden_dim, cfg.n_layers, multi_scale)
        self.project = nn.Linear(cfg.hidden_dim, cfg.embed_dim)

    def forward(self, x, edge_index, edge_dist):
        h = torch.relu(self.input(x))
        scales, gates = [], []
        for layer in self.layers:
            h, g = layer(h, edge_index, edge_dist)
            scales.append(h)
            gates.append(g)
        fused = self.fusion(scales)
        z = self.project(fused)
        # boundary score per node: 1 - mean incident gate (last layer)
        last_gate = gates[-1]
        node_gate_sum = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        node_gate_cnt = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        node_gate_sum.index_add_(0, edge_index[1], last_gate)
        node_gate_cnt.index_add_(0, edge_index[1], torch.ones_like(last_gate))
        boundary = 1.0 - node_gate_sum / (node_gate_cnt + 1e-8)
        return z, boundary, gates
