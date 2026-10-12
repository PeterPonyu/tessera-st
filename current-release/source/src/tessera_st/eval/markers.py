"""A LESS-circular (not non-circular) evaluation view: predefined literature layer markers.

`marker_purity` lowers — but does NOT remove — circularity. It is not computed from re-clustering
this dataset, which is its only advantage. But it is far from external truth: the marker list is a
prior from *other* datasets/conditions, and marker specificity drifts across platform, individual,
disease and batch — "every dataset's conditions differ, so its markers differ too". So treat it as
one more conditioned, partial view, NOT a tie-breaking ground truth. It is used alongside the rest
to triangulate; it must never become the new single metric we enthrone.

DLPFC cortical-layer markers (canonical, used by spatialLIBD / Maynard et al.). Genes are HGNC
symbols; any missing from a dataset are skipped. On a different tissue/platform this list is wrong
and must be swapped — that data-dependence is exactly why it is not external truth.
"""

from __future__ import annotations

import numpy as np

DLPFC_LAYER_MARKERS: dict[str, list[str]] = {
    "L1": ["RELN", "NDNF"],
    "L2_3": ["CUX2", "CALB1", "HPCAL1"],
    "L4": ["RORB"],
    "L5": ["PCP4", "BCL11B", "FEZF2"],
    "L6": ["TLE4", "FOXP2", "NTNG2"],
    "WM": ["MOBP", "MBP", "PLP1"],
}


def compute_layer_scores(
    marker_expr: np.ndarray,
    marker_names: list[str],
    marker_dict: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Per-spot z-scored mean expression for each layer's marker set.

    marker_expr: (n_spots, n_present_markers) log-normalised expression.
    Returns {layer: (n_spots,) score}; layers with no present markers are dropped.
    """
    idx_of = {g: i for i, g in enumerate(marker_names)}
    out: dict[str, np.ndarray] = {}
    for layer, genes in marker_dict.items():
        cols = [idx_of[g] for g in genes if g in idx_of]
        if not cols:
            continue
        sub = marker_expr[:, cols]
        z = (sub - sub.mean(axis=0)) / (sub.std(axis=0) + 1e-8)
        out[layer] = z.mean(axis=1)
    return out


def marker_purity(labels: np.ndarray, layer_scores: dict[str, np.ndarray]) -> float | None:
    """How specifically each predicted domain is marked by one layer's markers. Higher = better.

    For each domain, take its mean score per layer; specificity = (top layer - mean over layers).
    Domain-size-weighted average. It does not use the manual-layer GT, so it LOWERS circularity vs
    ARI/ASW/CHAOS — but it is conditioned on the chosen marker prior, which may not fit this data, so
    it is not a ground truth. A degenerate one-big-domain solution scores ~0 (mean profile ≈ global
    zero), so it is not gameable the way CHAOS is — one modest robustness it does have.
    """
    if not layer_scores:
        return None
    layers = list(layer_scores.keys())
    S = np.stack([layer_scores[l] for l in layers], axis=1)  # (n, L)
    specs, weights = [], []
    for c in np.unique(labels):
        m = labels == c
        prof = S[m].mean(axis=0)
        specs.append(float(prof.max() - prof.mean()))
        weights.append(int(m.sum()))
    return float(np.average(specs, weights=weights))
