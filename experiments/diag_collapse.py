"""Root-cause probe: is the training degradation representation collapse?

Trains the full model on DLPFC 151673 and, every 50 epochs, logs embedding-collapse signals
alongside ARI. If ARI falls *as* effective-rank / embedding-variance / mean-pairwise-distance
fall, the encoder is collapsing — and the fix is an anti-collapse term, not more epochs.
"""

import numpy as np
import torch

from tessera_st.config import AblationConfig, ModelConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval import ari
from tessera_st.eval.baselines import kmeans_labels
from tessera_st.losses import total_loss
from tessera_st.model.graph import build_knn_edges
from tessera_st.model.tessera import TesseraNet

P = (
    "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
    "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad"
)


def effective_rank(z: np.ndarray) -> float:
    zc = z - z.mean(axis=0)
    s = np.linalg.svd(zc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def mean_pairwise(z: np.ndarray, k: int = 500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), size=min(k, len(z)), replace=False)
    sub = z[idx]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    return float(d[np.triu_indices(len(sub), 1)].mean())


slide = load_h5ad(P, label_key="ground_truth")
n_clusters = int(len(np.unique(slide.labels[slide.labels >= 0])))
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)
gen = torch.Generator(device=device).manual_seed(1)

ei_np, ed_np = build_knn_edges(slide.coords, k=6)
x = torch.tensor(slide.expr, dtype=torch.float32, device=device)
edge_index = torch.tensor(ei_np, dtype=torch.long, device=device)
edge_dist = torch.tensor((ed_np - ed_np.mean()) / (ed_np.std() + 1e-8),
                         dtype=torch.float32, device=device)

cfg = TrainConfig(device=device)
abl = AblationConfig()
model = TesseraNet(ModelConfig(in_dim=x.shape[1]), n_clusters, abl).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

print(f"DLPFC 151673 {slide.expr.shape}, {n_clusters} layers, device={device}, embed_dim=32\n")
print(f"{'epoch':>6} | {'ARI':>7} | {'eff_rank':>8} | {'emb_var':>8} | {'pairwise':>8} | {'loss':>8}")
print("-" * 60)

for epoch in range(801):
    model.train()
    opt.zero_grad()
    out = model(x, edge_index, edge_dist)
    loss, _ = total_loss(out, x, edge_index, cfg, abl, generator=gen)
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index, edge_dist).z.cpu().numpy()
        lab = kmeans_labels(z, n_clusters, seed=1)
        print(f"{epoch:>6} | {ari(slide.labels, lab):>7.4f} | {effective_rank(z):>8.3f} | "
              f"{z.var(axis=0).mean():>8.4f} | {mean_pairwise(z):>8.4f} | {float(loss):>8.4f}")
