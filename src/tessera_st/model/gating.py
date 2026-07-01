"""Component 1 — the anti-over-smoothing core: a learned per-edge gate.

For every edge (i, j) the gate ``g_ij in [0, 1]`` decides how much of neighbour j's message
is allowed to flow into i. A gate near 0 means "this edge crosses a boundary — stop smoothing".
When the component is ablated the gate is pinned to 1, recovering plain mean aggregation
(the classic over-smoothing graph encoder).
"""

from __future__ import annotations

import torch
from torch import nn


class EdgeGate(nn.Module):
    """g_ij = sigmoid( MLP([ |h_i - h_j| , dist_ij ]) )."""

    def __init__(self, dim: int, hidden: int = 16, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-edge gate values, shape (E,)."""
        if not self.enabled:
            return torch.ones(edge_index.shape[1], device=h.device, dtype=h.dtype)
        src, dst = edge_index[0], edge_index[1]
        feat = torch.cat([(h[dst] - h[src]).abs(), edge_dist.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.net(feat)).squeeze(-1)
