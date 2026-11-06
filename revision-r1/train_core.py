"""Training core equivalent to the frozen ``run_experiments.train_net``.

The frozen function fixes the training budget at 120 epochs and the graph at six
query neighbours. This re-implementation exposes those settings, the contrastive
weight and temperature, the size of the untrained cluster head and evaluation
checkpoints. At the frozen settings it reproduces ``train_net`` exactly on CPU
with one thread (see test_r1.py).
"""
from __future__ import annotations

import time

import numpy as np
import torch

import common_r1  # noqa: F401  (sets the frozen import paths)
import run_experiments as rx
from tessera_st.config import AblationConfig, ModelConfig, TrainConfig
from tessera_st.losses import total_loss
from tessera_st.model.graph import build_knn_edges
from tessera_st.model.tessera import TesseraNet

ARMS = ['Tessera', 'NoGate', 'NoContrast', 'SelfOnly', 'PermutedXY-NoContrast']


def ablation(arm: str) -> AblationConfig:
    if arm == 'PermutedXY-NoContrast':
        return AblationConfig(edge_gating=True, multi_scale=True, boundary_contrastive=False,
                              calibrated_uncertainty=False)
    return rx.configuration(arm)


def build_graph(xy, fields, arm: str, seed: int, graph_k: int = 6):
    """Frozen graph rule with the neighbour count exposed (k=6 reproduces rx.graph)."""
    if arm == 'SelfOnly':
        row = np.arange(len(xy), dtype=np.int64)
        return np.stack([row, row]), np.zeros(len(row), np.float32)
    xy = np.asarray(xy, float).copy()
    if arm in ('PermutedXY', 'PermutedXY-NoContrast'):
        rng = np.random.RandomState(1000 + seed)
        for f in np.unique(fields):
            ix = np.flatnonzero(fields == f)
            xy[ix] = xy[rng.permutation(ix)]
    edge, dist = build_knn_edges(xy, k=graph_k, field_ids=fields)
    if (fields[edge[0]] != fields[edge[1]]).any() or (edge[0] == edge[1]).any():
        raise ValueError('cross-field or self graph edge')
    return edge, ((dist - dist.mean()) / (dist.std() + 1e-8)).astype(np.float32)


def train_core(x, xy, fields, head_k: int, arm: str, seed: int, device: str, *, epochs: int = 120,
               graph_k: int = 6, contrastive_weight: float = 0.5, contrastive_temp: float = 0.2,
               checkpoints=None, on_checkpoint=None) -> dict:
    """Train one encoder; return embeddings, boundary scores and last gates at each checkpoint.

    ``on_checkpoint(epoch, model)`` is called in evaluation mode at each checkpoint, for
    example to embed a held-out patient with the current parameters.
    """
    checkpoints = sorted(set(checkpoints or [epochs]) | {epochs})
    if checkpoints[-1] > epochs:
        raise ValueError('checkpoint beyond training budget')
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device=device).manual_seed(seed)
    tc = TrainConfig(epochs=epochs, seed=seed, device=device, contrastive_weight=contrastive_weight,
                     contrastive_temp=contrastive_temp)
    ab = ablation(arm)
    model = TesseraNet(ModelConfig(in_dim=x.shape[1]), head_k, ab).to(device)
    edge, dist = build_graph(xy, fields, arm, seed, graph_k)
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    te = torch.as_tensor(edge, dtype=torch.long, device=device)
    td = torch.as_tensor(dist, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    history, saved, extra = [], {}, {}
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(tx, te, td)
        loss, parts = total_loss(out, tx, te, tc, ab, generator=gen)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        opt.step()
        if epoch % 12 == 0 or epoch == epochs - 1:
            history.append(dict(epoch=epoch, loss=float(loss.detach()), **parts))
        done = epoch + 1
        if done in checkpoints:
            model.eval()
            with torch.no_grad():
                out = model(tx, te, td)
                saved[done] = dict(z=out.z.cpu().numpy(), boundary=out.boundary.cpu().numpy(),
                                   gate=out.gates[-1].cpu().numpy(),
                                   seconds=time.perf_counter() - started)
                if on_checkpoint is not None:
                    extra[done] = on_checkpoint(done, model)
            model.train()
    model.eval()
    memory = int(torch.cuda.max_memory_allocated()) if device == 'cuda' else None
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(checkpoints=saved, extra=extra, edge=edge, dist=dist, history=history,
                seconds=time.perf_counter() - started, memory=memory, params=params, model=model)


def same_graph_mean(x, xy, fields, graph_k: int = 6):
    """Unweighted mean of incoming neighbours on the symmetrized multiset graph."""
    edge, _ = build_graph(xy, fields, 'Tessera', 1, graph_k)
    return rx.neighbor_embedding(x, edge).astype(np.float32), edge
