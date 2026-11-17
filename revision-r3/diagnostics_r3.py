"""Label diagnostics of reference annotations (round-3 revision, WP1).

Every quantity is computed within coordinate frames: no pair of observations from
two frames is ever compared. Definitions follow PROTOCOL-R3.md, section 2, and are
reused unchanged on cohorts analyzed after the protocol freeze.

- homophily_profile: edge homophily, frame-stratified adjusted homophily
  (Platonov et al.) and frame-conditional label informativeness at each scale k.
- correlation_length: rank (and distance) at which the chance-corrected excess of
  same-label neighbours falls to 1/e of its nearest-neighbour value.
- predictability: spatially blocked cross-validated agreement between a label and
  the observation's own expression, and the gain from adding a neighbour mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

KS = (6, 10, 15, 30, 50, 100, 200)
MIN_COVERAGE = 0.9          # a scale is reported only if frames large enough for it hold >= 90% of labelled nodes
SHELLS = ((1, 1), (2, 3), (4, 6), (7, 10), (11, 15), (16, 30), (31, 50), (51, 100), (101, 200))
NO_COHERENCE = 0.05         # nearest-shell excess below this means no spatial coherence (length 0)
BLOCK_D1 = 10.0             # side of a cross-validation block in median nearest-neighbour distances
N_FOLDS = 5


def frame_knn(coords, frames, kmax):
    """Self-excluding nearest neighbours within each frame, never padded.

    Returns ``idx`` (n, kmax) with -1 where a frame has fewer than kmax + 1
    observations, ``dist`` with NaN there, and the frame size of every observation.
    The query is removed by index, so distinct observations at identical
    coordinates stay neighbours.
    """
    coords = np.asarray(coords, float)
    frames = np.asarray(frames)
    n = len(coords)
    idx = np.full((n, kmax), -1, np.int64)
    dist = np.full((n, kmax), np.nan)
    size = np.zeros(n, np.int64)
    for f in np.unique(frames):
        ix = np.flatnonzero(frames == f)
        size[ix] = len(ix)
        kk = min(kmax, len(ix) - 1)
        if kk < 1:
            continue
        d, j = NearestNeighbors(n_neighbors=kk + 1).fit(coords[ix]).kneighbors(coords[ix])
        keep = j != np.arange(len(ix))[:, None]
        keep[~(~keep).any(1), -1] = False          # query not returned: drop the farthest instead
        idx[ix, :kk] = ix[j[keep].reshape(len(ix), kk)]
        dist[ix, :kk] = d[keep].reshape(len(ix), kk)
    return idx, dist, size


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _edges(idx, size, y, k):
    """Directed edges from each labelled node to its first k labelled-or-not neighbours."""
    lab = y >= 0
    node = lab & (size - 1 >= k)
    src = np.repeat(np.flatnonzero(node), k)
    dst = idx[node, :k].ravel()
    keep = lab[dst]
    return src[keep], dst[keep], node.sum() / max(lab.sum(), 1)


def homophily_profile(coords, frames, y, ks=KS) -> pd.DataFrame:
    """Edge homophily, frame-stratified adjusted homophily and label informativeness per scale.

    The chance term and label informativeness are computed on the symmetrized
    multiset of labelled edges within each frame and combined with edge weights,
    so differences in class composition between frames do not count as homophily.
    """
    y = np.asarray(y, np.int64)
    frames = np.asarray(frames)
    classes = int(y.max()) + 1
    idx, _, size = frame_knn(coords, frames, max(ks))
    rows = []
    for k in ks:
        src, dst, coverage = _edges(idx, size, y, k)
        row = dict(k=k, coverage=float(coverage), n_edges=int(len(src)))
        if coverage < MIN_COVERAGE or not len(src):
            rows.append(row)
            continue
        same = y[src] == y[dst]
        chance = info = ent = 0.0
        for f in np.unique(frames[src]):
            m = frames[src] == f
            a = np.concatenate([y[src[m]], y[dst[m]]])
            b = np.concatenate([y[dst[m]], y[src[m]]])
            w = m.sum() / len(src)
            p = np.bincount(a, minlength=classes) / len(a)
            joint = np.zeros((classes, classes))
            np.add.at(joint, (a, b), 1.0)
            joint /= joint.sum()
            chance += w * float((p ** 2).sum())
            info += w * (2 * _entropy(p) - _entropy(joint.ravel()))
            ent += w * _entropy(p)
        h_edge = float(same.mean())
        row.update(h_edge=h_edge, chance=chance,
                   h_adj=(h_edge - chance) / (1 - chance) if chance < 1 else np.nan,
                   li=info / ent if ent > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_length(coords, frames, y, scale_um: float | None = None) -> dict:
    """Rank and distance at which the same-label excess falls to 1/e of its first-shell value.

    Shells are neighbour ranks (SHELLS). In each shell the excess is
    (same - chance) / (1 - chance), with chance the frame-weighted sum of squared
    class frequencies of labelled observations. The crossing is interpolated
    linearly in log rank between geometric shell midpoints. Lengths are censored at
    the last available shell and set to zero when the first-shell excess is below
    NO_COHERENCE. Distances are given in median nearest-neighbour distances (d1)
    and in micrometres when ``scale_um`` (micrometres per coordinate unit) is known.
    """
    y = np.asarray(y, np.int64)
    frames = np.asarray(frames)
    kmax = SHELLS[-1][1]
    idx, dist, size = frame_knn(coords, frames, kmax)
    lab = y >= 0
    d1 = float(np.nanmedian(dist[lab, 0]))
    classes = int(y.max()) + 1
    chance_f = {}
    for f in np.unique(frames[lab]):
        p = np.bincount(y[lab & (frames == f)], minlength=classes) / (lab & (frames == f)).sum()
        chance_f[f] = float((p ** 2).sum())
    mids, excess, rdist = [], [], []
    for lo, hi in SHELLS:
        node = lab & (size - 1 >= hi)
        if node.sum() < MIN_COVERAGE * lab.sum():
            break
        cols = np.arange(lo - 1, hi)
        src = np.repeat(np.flatnonzero(node), len(cols))
        dst = idx[node][:, cols].ravel()
        dd = dist[node][:, cols].ravel()
        keep = lab[dst]
        src, dst, dd = src[keep], dst[keep], dd[keep]
        chance = np.mean([chance_f[f] for f in frames[src]])
        same = float((y[src] == y[dst]).mean())
        mids.append(np.sqrt(lo * hi))
        excess.append((same - chance) / (1 - chance) if chance < 1 else np.nan)
        rdist.append(float(np.median(dd)) / d1)
    out = dict(d1=d1, shells=len(mids), first_excess=float(excess[0]) if excess else np.nan,
               excess=[float(e) for e in excess])
    if not excess or not np.isfinite(excess[0]) or excess[0] < NO_COHERENCE:
        out.update(ell_k=0.0, ell_d1=0.0, censored=False)
    else:
        target = excess[0] / np.e
        below = [i for i, e in enumerate(excess) if e <= target]
        if not below:
            out.update(ell_k=float(SHELLS[len(mids) - 1][1]), ell_d1=float(rdist[-1]), censored=True)
        else:
            i = below[0]
            t = (excess[i - 1] - target) / (excess[i - 1] - excess[i])
            logk = np.log(mids[i - 1]) + t * (np.log(mids[i]) - np.log(mids[i - 1]))
            out.update(ell_k=float(np.exp(logk)), ell_d1=float(rdist[i - 1] + t * (rdist[i] - rdist[i - 1])),
                       censored=False)
    out['ell_um'] = out['ell_d1'] * d1 * scale_um if scale_um else None
    return out


def spatial_folds(coords, frames, d1, n_folds=N_FOLDS, block_d1=BLOCK_D1, seed=0):
    """Fold of every observation from square blocks of side block_d1 * d1, shuffled round-robin.

    Returns None when fewer than ``n_folds`` non-empty blocks exist and fewer than
    three frames are available; with three or more frames but too few blocks,
    whole frames become folds.
    """
    coords = np.asarray(coords, float)
    frames = np.asarray(frames)
    side = block_d1 * d1
    keys = np.empty(len(coords), dtype=object)
    for f in np.unique(frames):
        m = frames == f
        cell = np.floor((coords[m] - coords[m].min(0)) / side).astype(np.int64)
        keys[m] = [f'{f}|{a}|{b}' for a, b in cell[:, :2]]
    blocks = np.unique(keys)
    if len(blocks) >= n_folds:
        order = np.random.default_rng(seed).permutation(len(blocks))
        fold_of = {blocks[j]: r % n_folds for r, j in enumerate(order)}
        return np.array([fold_of[k] for k in keys])
    uniq = np.unique(frames)
    if len(uniq) >= 3:
        return np.searchsorted(uniq, frames)
    return None


def _cv_scores(features, y, folds, estimator=None):
    lab = y >= 0
    pred = np.full(len(y), -1, np.int64)
    for fold in np.unique(folds[lab]):
        test = lab & (folds == fold)
        train = lab & (folds != fold)
        if len(np.unique(y[train])) < 2 or not test.any():
            continue
        model = estimator() if estimator else LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
        scaler = StandardScaler().fit(features[train])
        model.fit(scaler.transform(features[train]), y[train])
        pred[test] = model.predict(scaler.transform(features[test]))
    ok = lab & (pred >= 0)
    if ok.sum() < 10:
        return np.nan, np.nan
    return (float(balanced_accuracy_score(y[ok], pred[ok])), float(cohen_kappa_score(y[ok], pred[ok])))


def predictability(x, coords, frames, y, ks=KS, estimator=None, seed=0) -> dict:
    """Blocked cross-validated agreement of a label with own expression, and with a neighbour mean added.

    ``kappa_self`` uses the observation's own features; ``kappa_k`` adds the mean of
    its k nearest neighbours in the frame (fewer when the frame is smaller). The
    random-fold value is reported alongside as the leakage-prone upper bound.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, np.int64)
    idx, dist, _ = frame_knn(coords, frames, max(ks))
    d1 = float(np.nanmedian(dist[y >= 0, 0]))
    folds = spatial_folds(coords, frames, d1, seed=seed)
    lab = y >= 0
    present = np.unique(y[lab])
    out = dict(chance_ba=1.0 / len(present), n_blocks_ok=folds is not None)
    random_folds = np.random.default_rng(seed).integers(0, N_FOLDS, len(y))
    out['ba_self_random'], out['kappa_self_random'] = _cv_scores(x, y, random_folds, estimator)
    if folds is None:
        out.update(ba_self=np.nan, kappa_self=np.nan, info_gain_max=np.nan, info_gain_k=np.nan)
        return out
    out['ba_self'], out['kappa_self'] = _cv_scores(x, y, folds, estimator)
    gains = {}
    for k in ks:
        rows = idx[:, :k]
        valid = rows >= 0
        mean = np.where(valid.any(1, keepdims=True),
                        (x[np.maximum(rows, 0)] * valid[..., None]).sum(1) / np.maximum(valid.sum(1, keepdims=True), 1),
                        x)
        _, kappa = _cv_scores(np.hstack([x, mean]), y, folds, estimator)
        gains[k] = kappa - out['kappa_self']
    best = max(gains, key=lambda k: -np.inf if not np.isfinite(gains[k]) else gains[k])
    out.update(info_gain_max=float(gains[best]), info_gain_k=int(best),
               info_gain={int(k): float(v) for k, v in gains.items()})
    return out
