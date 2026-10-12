"""Feature decoder and legacy untrained soft-assignment head (not confidence)."""

from __future__ import annotations

import torch
from torch import nn


class FeatureDecoder(nn.Module):
    """Reconstruct input features from the embedding (graph-autoencoder objective)."""

    def __init__(self, embed_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return self.net(z)


class ClusterHead(nn.Module):
    """Distance-softmax head retained for archived model compatibility.

    Its parameters have no term in the current training objective. There is no
    held-out temperature or conformal calibration in fit_predict, and these
    centroids are not the KMeans centroids that produce the reported labels.
    """

    def __init__(self, embed_dim: int, n_clusters: int):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)
        self.log_temp = nn.Parameter(torch.zeros(()))  # temperature = exp(log_temp)

    def forward(self, z, calibrated: bool = True):
        d2 = torch.cdist(z, self.centroids) ** 2
        temp = torch.exp(self.log_temp) if calibrated else torch.ones((), device=z.device)
        logits = -d2 / (temp + 1e-6)
        q = torch.softmax(logits, dim=-1)
        return q

    def set_temperature(self, value: float) -> None:
        with torch.no_grad():
            self.log_temp.fill_(float(torch.log(torch.tensor(max(value, 1e-3)))))
