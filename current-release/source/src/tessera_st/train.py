"""Training + inference loop. Pure-torch; no data-origin branching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera_st.config import AblationConfig, ModelConfig, TrainConfig
from tessera_st.losses import total_loss


@dataclass
class FitResult:
    labels: np.ndarray
    embed: np.ndarray
    boundary: np.ndarray
    confidence: np.ndarray | None
    history: list[dict]


def _resolve_device(name: str):
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def fit_predict(
    expr: np.ndarray,
    coords: np.ndarray,
    n_clusters: int,
    ablation: AblationConfig,
    train_cfg: TrainConfig | None = None,
    model_cfg: ModelConfig | None = None,
    knn_k: int = 6,
    refine: bool = False,
    field_ids=None,
) -> FitResult:
    """Train TesseraNet on one slide and return domain labels + boundary + confidence.

    refine=True applies the standard spatial label-refinement post-step (majority vote over the
    spatial kNN) — recommended; it lifts ARI/spatial-coherence and benefits Tessera most.
    """
    import torch

    from tessera_st.eval.baselines import kmeans_labels
    from tessera_st.model.graph import build_knn_edges
    from tessera_st.model.tessera import TesseraNet

    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or ModelConfig(in_dim=expr.shape[1])
    device = _resolve_device(train_cfg.device)
    torch.manual_seed(train_cfg.seed)
    gen = torch.Generator(device=device).manual_seed(train_cfg.seed)

    if len(expr) != len(coords):
        raise ValueError("expression and coordinate rows must match")
    edge_index_np, edge_dist_np = build_knn_edges(coords, k=knn_k, field_ids=field_ids)
    if edge_index_np.shape[1] == 0:
        raise ValueError("no within-field spatial edges; model training is undefined")
    x = torch.tensor(expr, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_dist = torch.tensor(_zscore(edge_dist_np), dtype=torch.float32, device=device)

    model = TesseraNet(model_cfg, n_clusters, ablation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)

    history: list[dict] = []
    model.train()
    for epoch in range(train_cfg.epochs):
        opt.zero_grad()
        out = model(x, edge_index, edge_dist)
        loss, parts = total_loss(out, x, edge_index, train_cfg, ablation, generator=gen)
        loss.backward()
        opt.step()
        if epoch % max(1, train_cfg.epochs // 10) == 0:
            history.append({"epoch": epoch, "loss": float(loss.detach()), **parts})

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_dist)
        embed = out.z.cpu().numpy()
        boundary = out.boundary.cpu().numpy()

    labels = kmeans_labels(embed, n_clusters, seed=train_cfg.seed)
    if refine:
        from tessera_st.eval.refine import refine_labels
        labels = refine_labels(labels, coords, k=knn_k, field_ids=field_ids)
    # The legacy soft head has no loss term or held-out calibration, and its
    # centroids are not the KMeans centroids used for labels. Do not publish
    # its random-head maximum as confidence in those predictions.
    confidence = None
    return FitResult(labels=labels, embed=embed, boundary=boundary,
                     confidence=confidence, history=history)


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-8)
