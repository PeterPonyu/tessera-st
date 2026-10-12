"""Training objectives.

- reconstruction: graph-autoencoder feature reconstruction (always on).
- boundary_contrastive (Component 3): InfoNCE that pulls high-gate (intra-domain) neighbours
  together and pushes random distant nodes apart, sharpening the embedding around seams.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, x)


def boundary_contrastive_loss(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    gates: torch.Tensor,
    temp: float,
    n_negatives: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """InfoNCE with gate-weighted positives.

    For each edge (i, j) the positive pull is weighted by the gate g_ij: edges the model
    believes are intra-domain (high gate) contribute the most attraction, so the contrastive
    geometry co-adapts with the boundary belief instead of fighting it.
    """
    src, dst = edge_index[0], edge_index[1]
    zi = F.normalize(z[src], dim=-1)
    zj = F.normalize(z[dst], dim=-1)
    pos = (zi * zj).sum(-1) / temp  # (E,)

    n = z.shape[0]
    neg_idx = torch.randint(0, n, (src.shape[0], n_negatives), device=z.device,
                            generator=generator)
    zneg = F.normalize(z[neg_idx], dim=-1)  # (E, K, d)
    neg = torch.bmm(zneg, zi.unsqueeze(-1)).squeeze(-1) / temp  # (E, K)

    logits = torch.cat([pos.unsqueeze(-1), neg], dim=-1)  # (E, 1+K)
    target = torch.zeros(src.shape[0], dtype=torch.long, device=z.device)
    per_edge = F.cross_entropy(logits, target, reduction="none")
    # weight each edge's contrastive loss by its gate (intra-domain edges matter most)
    w = gates.detach()
    return (per_edge * w).sum() / (w.sum() + 1e-8)


def total_loss(out, x, edge_index, cfg, ablation, generator=None):
    """Compose the objective according to the ablation config. Returns (loss, parts)."""
    rec = reconstruction_loss(out.recon, x)
    parts = {"recon": float(rec.detach())}
    loss = cfg.recon_weight * rec
    if ablation.boundary_contrastive:
        con = boundary_contrastive_loss(
            out.z, edge_index, out.gates[-1], cfg.contrastive_temp,
            cfg.n_negatives, generator=generator,
        )
        loss = loss + cfg.contrastive_weight * con
        parts["contrastive"] = float(con.detach())
    return loss, parts
