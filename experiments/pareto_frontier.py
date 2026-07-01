"""G002 TradeoffFrontier: quantify the boundary-sharpness vs label-agreement Pareto frontier.

Reads rigorous_bench.json (cross-section GMM results). Plots (ASCII) ARI vs boundary_F1, computes
the non-dominated frontier, and labels where each method sits. The mechanism story (contrastive ->
boundary end; reconstruction/smoothing -> global end) is corroborated by the 4 failed transfer
experiments recorded in CLAIM_LEDGER (BANKSY-aug, SEDR-VAE, GAT-attn, decoupled — all traded ARI
against boundary_F1). Emits a figure spec + frontier JSON for the manuscript.
"""

import json

import numpy as np

d = json.load(open("experiments/rigorous_bench.json"))
res = d["results"]

# cross-section mean (GMM backend) for ARI and boundary_F1 per method
pts = {}
for m in res:
    aris, bf1 = [], []
    for s in res[m]:
        g = res[m][s].get("gmm", {})
        if "ARI" in g:
            aris.append(g["ARI"])
        if "boundary_F1" in g:
            bf1.append(g["boundary_F1"])
    if aris and bf1:
        pts[m] = {"ARI": float(np.mean(aris)), "bF1": float(np.mean(bf1))}


def dominates(b, a):  # b dominates a if >= on both and > on one
    return (b["ARI"] >= a["ARI"] and b["bF1"] >= a["bF1"]
            and (b["ARI"] > a["ARI"] or b["bF1"] > a["bF1"]))


frontier = [m for m in pts if not any(dominates(pts[o], pts[m]) for o in pts if o != m)]

print("=== Pareto frontier: label-agreement (ARI↑) vs boundary-sharpness (boundary_F1↑), GMM ===\n")
order = sorted(pts, key=lambda m: pts[m]["ARI"], reverse=True)
print(f"{'method':>18} | {'ARI':>6} | {'bF1':>6} | frontier?")
print("-" * 48)
for m in order:
    tag = "  ◀ PARETO" if m in frontier else ""
    print(f"{m:>18} | {pts[m]['ARI']:.3f} | {pts[m]['bF1']:.3f} |{tag}")

# ASCII scatter
print("\nASCII map (x=ARI, y=boundary_F1):")
xs = [pts[m]["ARI"] for m in pts]; ys = [pts[m]["bF1"] for m in pts]
x0, x1 = min(xs) - 0.01, max(xs) + 0.01
y0, y1 = min(ys) - 0.01, max(ys) + 0.01
W, H = 50, 14
grid = [[" "] * W for _ in range(H)]
for m in pts:
    cx = int((pts[m]["ARI"] - x0) / (x1 - x0) * (W - 1))
    cy = int((pts[m]["bF1"] - y0) / (y1 - y0) * (H - 1))
    grid[H - 1 - cy][cx] = m[0]  # first letter
for row in grid:
    print("  |" + "".join(row) + "|")
print(f"  ARI {x0:.2f} -> {x1:.2f}   bF1 {y0:.2f}(bottom) -> {y1:.2f}(top)")
print("  legend:", {m[0]: m for m in pts})

analysis = {
    "points": pts,
    "frontier": frontier,
    "boundary_end": max(pts, key=lambda m: pts[m]["bF1"]),
    "global_end": max(pts, key=lambda m: pts[m]["ARI"]),
    "mechanism": ("contrastive loss -> boundary end (high bF1, mid ARI); reconstruction/neighbour-"
                  "smoothing -> global end (high ARI, lower bF1). 4 transfer experiments "
                  "(BANKSY-aug, SEDR-VAE, GAT-attn, decoupled) all moved ALONG the frontier "
                  "(traded one for the other), none pushed it outward — evidence the trade-off is "
                  "structural for this method family."),
}
json.dump(analysis, open("experiments/pareto_frontier.json", "w"), indent=2)
print(f"\nboundary end: {analysis['boundary_end']} | global end: {analysis['global_end']}")
print("frontier members:", frontier)
print("\nwrote experiments/pareto_frontier.json")
