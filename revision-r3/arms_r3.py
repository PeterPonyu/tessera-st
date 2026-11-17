"""Round-3 mechanism arms (WP3) that tie the Tessera-ST result to graph-learning theory.

- ContrastNoMP: spatial positives without message passing (the Graph-MLP design).
  The released network runs on self edges only, as in the self-only arm, so no
  message is ever passed; the released contrastive loss is applied to the
  six-neighbour multiset of Eq. 4 with uniform weights, because no gates exist for
  edges the network never sees. Reconstruction, budget and every setting are
  released values.
- propagate: parameter-free propagation P^L x on an encoder's own graph, with P the
  row-normalized mean over incoming edges (the same-graph mean applied L times).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_r3  # noqa: E402,F401  (frozen import paths)
import train_core as tc  # noqa: E402
from tessera_st.config import ModelConfig, TrainConfig  # noqa: E402
from tessera_st.losses import boundary_contrastive_loss, reconstruction_loss  # noqa: E402
from tessera_st.model.tessera import TesseraNet  # noqa: E402

rx = tc.rx


def train_contrast_no_mp(x, xy, fields, head_k: int, seed: int, device: str = 'cpu', *, epochs: int = 120,
                         graph_k: int = 6, contrastive_weight: float = 0.5, contrastive_temp: float = 0.2) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    tc_cfg = TrainConfig(epochs=epochs, seed=seed, device=device, contrastive_weight=contrastive_weight,
                         contrastive_temp=contrastive_temp)
    model = TesseraNet(ModelConfig(in_dim=x.shape[1]), head_k, rx.configuration('SelfOnly')).to(device)
    self_edge, self_dist = tc.build_graph(xy, fields, 'SelfOnly', seed, graph_k)
    spatial_edge, _ = tc.build_graph(xy, fields, 'Tessera', seed, graph_k)
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    te_self = torch.as_tensor(self_edge, dtype=torch.long, device=device)
    td_self = torch.as_tensor(self_dist, dtype=torch.float32, device=device)
    te_sp = torch.as_tensor(spatial_edge, dtype=torch.long, device=device)
    weights = torch.ones(te_sp.shape[1], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=tc_cfg.lr, weight_decay=tc_cfg.weight_decay)
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(tx, te_self, td_self)
        loss = tc_cfg.recon_weight * reconstruction_loss(out.recon, tx) + contrastive_weight * \
            boundary_contrastive_loss(out.z, te_sp, weights, contrastive_temp, tc_cfg.n_negatives, generator=gen)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        z = model(tx, te_self, td_self).z.cpu().numpy()
    return dict(z=z, forward_edges=self_edge, contrastive_edges=spatial_edge, seconds=time.perf_counter() - started)


def propagate(x, edge, L: int) -> np.ndarray:
    """P^L x on the graph ``edge`` (2, E): each step averages incoming neighbours, no self term."""
    z = np.asarray(x, np.float64)
    for _ in range(L):
        z = rx.neighbor_embedding(z, edge)
    return z.astype(np.float32)
