"""Evaluation metrics.

Standard three (ARI/NMI/ASW) keep us on the established leaderboard; boundary-F1 and ECE
are the additive, boundary-aware metrics that motivate Tessera. All operate on plain arrays.
"""

from __future__ import annotations

import numpy as np


def _valid(labels: np.ndarray) -> np.ndarray:
    return labels >= 0


def ari(true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score

    m = _valid(true)
    return float(adjusted_rand_score(true[m], pred[m]))


def nmi(true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    m = _valid(true)
    return float(normalized_mutual_info_score(true[m], pred[m]))


def asw(embed: np.ndarray, pred: np.ndarray) -> float:
    """Average silhouette width of predicted domains in embedding space."""
    from sklearn.metrics import silhouette_score

    if len(np.unique(pred)) < 2 or len(pred) < 3:
        return float("nan")
    return float(silhouette_score(embed, pred))


def boundary_f1(
    coords: np.ndarray, true: np.ndarray, pred: np.ndarray, k: int = 6
) -> float:
    """F1 between predicted boundary spots and ground-truth boundary spots.

    A spot is a boundary spot if any kNN neighbour carries a different domain label. This
    rewards methods that keep seams sharp instead of smoothing them away.
    """
    from sklearn.neighbors import NearestNeighbors

    m = _valid(true)
    coords, true, pred = coords[m], true[m], pred[m]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    _, idx = nn.kneighbors(coords)
    neigh = idx[:, 1:]
    true_b = (true[neigh] != true[:, None]).any(axis=1)
    pred_b = (pred[neigh] != pred[:, None]).any(axis=1)
    tp = np.sum(true_b & pred_b)
    fp = np.sum(~true_b & pred_b)
    fn = np.sum(true_b & ~pred_b)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over confidence vs. empirical accuracy. Lower is better-calibrated."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# --- spatial-coherence dimension (orthogonal to ARI/NMI label-agreement) -------------------

def chaos_score(coords: np.ndarray, pred: np.ndarray) -> float:
    """Spatial Chaos Score (CHAOS). Lower = spatially more coherent domains.

    For each predicted domain, the mean nearest-neighbour *spatial* distance among its own
    spots; averaged over domains and normalised by the global mean nearest-neighbour distance
    so it is scale-free. A method can score high ARI yet high CHAOS if its domains are spatially
    shattered — exactly the failure ARI/NMI cannot see.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(coords) < 3:
        return float("nan")
    g = NearestNeighbors(n_neighbors=2).fit(coords)
    global_d = g.kneighbors(coords)[0][:, 1].mean()
    vals = []
    for c in np.unique(pred):
        cc = coords[pred == c]
        if len(cc) < 2:
            continue
        d = NearestNeighbors(n_neighbors=2).fit(cc).kneighbors(cc)[0][:, 1]
        vals.append(d.mean())
    if not vals:
        return float("nan")
    return float(np.mean(vals) / (global_d + 1e-8))


def percentage_abnormal_spots(coords: np.ndarray, pred: np.ndarray, k: int = 6) -> float:
    """PAS — fraction of spots whose predicted label disagrees with most spatial neighbours.

    Lower = less spatial fragmentation. Orthogonal to CHAOS (PAS is local/boundary-ish,
    CHAOS is global/compactness) and to ARI (label agreement).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    neigh = nn.kneighbors(coords)[1][:, 1:]
    same = (pred[neigh] == pred[:, None]).sum(axis=1)
    return float((same < neigh.shape[1] / 2.0).mean())


# --- geometric-separation dimension --------------------------------------------------------

def davies_bouldin(embed: np.ndarray, pred: np.ndarray) -> float:
    """Davies-Bouldin index in embedding space. Lower = better-separated clusters.

    A geometric view (intra/inter cluster scatter) distinct from ASW's silhouette formulation.
    """
    from sklearn.metrics import davies_bouldin_score

    if len(np.unique(pred)) < 2:
        return float("nan")
    return float(davies_bouldin_score(embed, pred))


def calinski_harabasz(embed: np.ndarray, pred: np.ndarray) -> float:
    """Calinski-Harabasz (variance-ratio) index. Higher = better-separated clusters.

    A third internal/geometric view (between- over within-cluster dispersion). Internal metrics
    are semi-circular (computed in each method's own embedding) — reported as one more angle,
    never as a sole judge.
    """
    from sklearn.metrics import calinski_harabasz_score

    if len(np.unique(pred)) < 2:
        return float("nan")
    return float(calinski_harabasz_score(embed, pred))


# --- small-domain recovery dimension (the over-smoothing casualty) -------------------------

def small_domain_iou(true: np.ndarray, pred: np.ndarray) -> float:
    """IoU between the *smallest* GT domain and its dominant predicted cluster. Higher = better.

    The smallest GT domain is the first casualty of uniform smoothing — either dissolved into a
    big neighbour or shattered. IoU punishes both: if the small domain is absorbed, its dominant
    predicted cluster is huge (low IoU); if it is split, the overlap is small (low IoU). ARI
    barely moves when a tiny domain is lost, so this is a genuinely different axis.
    """
    m = _valid(true)
    t, p = true[m], pred[m]
    if len(t) == 0:
        return float("nan")
    sizes = {int(c): int((t == c).sum()) for c in np.unique(t)}
    small = min(sizes, key=lambda c: sizes[c])
    small_mask = t == small
    in_small = p[small_mask]
    if len(in_small) == 0:
        return float("nan")
    dominant = np.bincount(in_small).argmax()
    pred_mask = p == dominant
    inter = float((small_mask & pred_mask).sum())
    union = float((small_mask | pred_mask).sum())
    return float(inter / (union + 1e-12))
