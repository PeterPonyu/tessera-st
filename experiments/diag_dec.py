"""Mechanism fix probe: does a SEDR-style DEC self-training objective cure Tessera's degradation?

Borrowed mechanism (clean-room, idea only): join the clustering objective to training via a DEC
KL(P||Q) term on the existing student-t cluster head, with KMeans-initialised centroids after a
reconstruction warm-up. Hypothesis: ARI rises monotonically and stops degrading, and the embedding
becomes GMM/KMeans-friendly. Logs ARI under BOTH backends vs epoch.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from tessera_st.config import AblationConfig, ModelConfig, TrainConfig
from tessera_st.data.dlpfc import load_h5ad
from tessera_st.eval import ari
from tessera_st.losses import total_loss
from tessera_st.model.graph import build_knn_edges
from tessera_st.model.tessera import TesseraNet

P = ("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/"
     "data/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad")


def soft_assign(z, centroids, alpha=1.0):
    """Numerically-stable student-t kernel (power-law tail, no softmax underflow)."""
    d2 = torch.cdist(z, centroids) ** 2
    q = (1.0 + d2 / alpha).pow(-(alpha + 1) / 2)
    return q / (q.sum(1, keepdim=True) + 1e-12)


def target_distribution(q):
    w = q ** 2 / (q.sum(0) + 1e-12)
    return (w.t() / (w.sum(1) + 1e-12)).t()


slide = load_h5ad(P, label_key="ground_truth")
n = int(len(np.unique(slide.labels[slide.labels >= 0])))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)
gen = torch.Generator(device=dev).manual_seed(1)

ei, ed = build_knn_edges(slide.coords, k=6)
x = torch.tensor(slide.expr, dtype=torch.float32, device=dev)
edge_index = torch.tensor(ei, dtype=torch.long, device=dev)
edge_dist = torch.tensor((ed - ed.mean()) / (ed.std() + 1e-8), dtype=torch.float32, device=dev)

cfg = TrainConfig(device=dev)
abl = AblationConfig()                                   # warmup: recon + boundary-contrastive
abl_recon = AblationConfig(boundary_contrastive=False)   # DEC phase: recon only (no over-spread)
model = TesseraNet(ModelConfig(in_dim=x.shape[1]), n, abl).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

WARMUP, TOTAL, DEC_W = 100, 600, 0.5
print(f"DLPFC 151673, full+DEC (warmup={WARMUP}, dec_w={DEC_W})")
print(f"{'epoch':>6} | {'ARI(km)':>8} | {'ARI(gmm)':>8} | {'phase':>6}")
print("-" * 42)

for epoch in range(TOTAL + 1):
    model.train()
    opt.zero_grad()
    out = model(x, edge_index, edge_dist)
    phase_abl = abl if epoch <= WARMUP else abl_recon  # drop contrastive once DEC takes over
    loss, _ = total_loss(out, x, edge_index, cfg, phase_abl, generator=gen)
    if epoch == WARMUP:  # init centroids from embedding (DEC requirement)
        with torch.no_grad():
            z0 = out.z.detach().cpu().numpy()
        km = KMeans(n, n_init=10, random_state=1).fit(z0)
        model.cluster.centroids.data = torch.tensor(
            km.cluster_centers_, dtype=torch.float32, device=dev)
    if epoch > WARMUP:  # DEC self-training phase: recon + KL(P||Q), stable student-t q
        q = soft_assign(out.z, model.cluster.centroids)
        p = target_distribution(q).detach()
        loss = loss + DEC_W * F.kl_div((q + 1e-12).log(), p, reduction="batchmean")
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index, edge_dist).z.cpu().numpy()
        a_km = ari(slide.labels, KMeans(n, n_init=10, random_state=1).fit_predict(z))
        a_gmm = ari(slide.labels, GaussianMixture(n, covariance_type="tied",
                    random_state=1, n_init=5).fit_predict(z))
        print(f"{epoch:>6} | {a_km:>8.4f} | {a_gmm:>8.4f} | "
              f"{'warmup' if epoch <= WARMUP else 'DEC':>6}")
