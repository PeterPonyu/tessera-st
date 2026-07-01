#!/usr/bin/env python3
"""
Search-corrected (max-statistic / Westfall-Young) permutation p-value for the
label-free coh_gain proxy.

We selected coh_gain out of 6 candidate label-free vectors by its Spearman rho
against `spatial_advantage`. Reporting the naive per-candidate p-value ignores
that we searched over 6 candidates. This script builds the correct null by, for
each permutation of the `spatial_advantage` labels, recomputing the MAX |rho|
over all 6 candidates -- the Westfall-Young max-T procedure that controls the
family-wise error rate across the search.

Inputs : experiments/gtfree_proxy_search.json  (on-disk; no re-run of KMeans)
Outputs: experiments/proxy_search_corrected.json
Deterministic: fixed seed, fixed B. No new data.
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "gtfree_proxy_search.json")
OUT_PATH = os.path.join(HERE, "proxy_search_corrected.json")

SEED = 1234
B = 10000
CANDIDATES = [
    "raw_coh",
    "smooth_coh",
    "coh_gain",
    "partition_shift",
    "sil_gain",
    "smoothing_score",
]
TARGET = "spatial_advantage"
SELECTED = "coh_gain"


def spearman_rho(x, y):
    """Spearman rho via Pearson on ranks (average ties)."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    return np.corrcoef(rx, ry)[0, 1]


def main():
    with open(IN_PATH) as fh:
        data = json.load(fh)
    rows = data["rows"]
    n = len(rows)

    # Build candidate matrix X (n x 6) and target y (n,)
    X = np.array([[r[c] for c in CANDIDATES] for r in rows], dtype=float)
    y = np.array([r[TARGET] for r in rows], dtype=float)

    # Observed per-candidate rho and the selected/observed max|rho|
    obs_rho = {c: spearman_rho(X[:, j], y) for j, c in enumerate(CANDIDATES)}
    obs_maxabs = max(abs(v) for v in obs_rho.values())
    selected_rho = obs_rho[SELECTED]
    selected_is_argmax = (
        SELECTED == max(obs_rho, key=lambda c: abs(obs_rho[c]))
    )

    # Uncorrected per-candidate two-sided permutation p for the selected proxy.
    # (recompute against a simple B-permutation null of the selected candidate)
    rng = np.random.default_rng(SEED)

    # Pre-rank columns once (Spearman only depends on ranks; permuting labels
    # permutes ranks of y identically to permuting y).
    Xr = np.apply_along_axis(stats.rankdata, 0, X)  # n x 6 ranks
    yr = stats.rankdata(y)

    # Center ranks for fast Pearson-on-ranks correlation
    Xr_c = Xr - Xr.mean(axis=0, keepdims=True)
    Xr_norm = Xr_c / np.linalg.norm(Xr_c, axis=0, keepdims=True)

    null_maxabs = np.empty(B, dtype=float)
    null_selected_abs = np.empty(B, dtype=float)
    sel_idx = CANDIDATES.index(SELECTED)

    for b in range(B):
        perm = rng.permutation(n)
        yp = yr[perm]
        yp_c = yp - yp.mean()
        yp_norm = yp_c / np.linalg.norm(yp_c)
        # rho for all 6 candidates at once
        rhos = Xr_norm.T @ yp_norm  # (6,)
        aabs = np.abs(rhos)
        null_maxabs[b] = aabs.max()
        null_selected_abs[b] = aabs[sel_idx]

    # Search-corrected (family-wise, max-T) p-value
    p_corrected = float((np.sum(null_maxabs >= abs(obs_maxabs) - 1e-12) + 0) / B)
    # +1 smoothing variant (conservative, avoids p=0)
    p_corrected_plus1 = float((np.sum(null_maxabs >= abs(obs_maxabs) - 1e-12) + 1) / (B + 1))

    # Uncorrected two-sided permutation p for the selected candidate alone
    p_uncorrected = float((np.sum(null_selected_abs >= abs(selected_rho) - 1e-12) + 0) / B)
    p_uncorrected_plus1 = float((np.sum(null_selected_abs >= abs(selected_rho) - 1e-12) + 1) / (B + 1))

    quant_levels = [0.5, 0.9, 0.95, 0.975, 0.99]
    null_quantiles = {
        f"q{int(q*1000)/10}": float(np.quantile(null_maxabs, q))
        for q in quant_levels
    }

    out = {
        "description": (
            "Westfall-Young max-statistic permutation test. Null = max over 6 "
            "label-free candidates of |Spearman rho| vs permuted spatial_advantage. "
            "Search-corrected p controls family-wise error over the 6-candidate search."
        ),
        "n_platforms": n,
        "candidates": CANDIDATES,
        "target": TARGET,
        "selected_candidate": SELECTED,
        "selected_is_argmax_over_search": bool(selected_is_argmax),
        "seed": SEED,
        "B": B,
        "observed_rho_per_candidate": {c: float(v) for c, v in obs_rho.items()},
        "observed_selected_rho": float(selected_rho),
        "observed_max_abs_rho": float(obs_maxabs),
        "p_uncorrected_selected": p_uncorrected,
        "p_uncorrected_selected_plus1": p_uncorrected_plus1,
        "p_search_corrected": p_corrected,
        "p_search_corrected_plus1": p_corrected_plus1,
        "null_max_abs_rho_quantiles": null_quantiles,
        "sanity_coh_gain_rho_matches_0p551": bool(abs(selected_rho - 0.551) < 0.01),
    }

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
