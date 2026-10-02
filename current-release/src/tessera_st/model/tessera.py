"""TesseraNet — assembles the four components per AblationConfig."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tessera_st.config import AblationConfig, ModelConfig
from tessera_st.model.encoder import GatedEncoder
from tessera_st.model.heads import ClusterHead, FeatureDecoder


@dataclass
class TesseraOutput:
    z: torch.Tensor  # (n, embed_dim) node embedding
    recon: torch.Tensor  # (n, in_dim) reconstructed features
    boundary: torch.Tensor  # (n,) boundary score in [0,1]
    q: torch.Tensor  # (n, n_clusters) soft assignment
    gates: list[torch.Tensor]  # per-layer edge gates


class TesseraNet(nn.Module):
    def __init__(self, model_cfg: ModelConfig, n_clusters: int, ablation: AblationConfig):
        super().__init__()
        self.ablation = ablation
        self.encoder = GatedEncoder(model_cfg, gating=ablation.edge_gating,
                                    multi_scale=ablation.multi_scale)
        self.decoder = FeatureDecoder(model_cfg.embed_dim, model_cfg.in_dim,
                                      hidden=model_cfg.hidden_dim)
        self.cluster = ClusterHead(model_cfg.embed_dim, n_clusters)

    def forward(self, x, edge_index, edge_dist) -> TesseraOutput:
        z, boundary, gates = self.encoder(x, edge_index, edge_dist)
        recon = self.decoder(z)
        q = self.cluster(z, calibrated=self.ablation.calibrated_uncertainty)
        return TesseraOutput(z=z, recon=recon, boundary=boundary, q=q, gates=gates)
