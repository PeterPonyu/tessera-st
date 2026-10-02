"""n=12 expansion: Zhuang-ABCA-1 section 069 under the published-panel protocol.

Same functions as expand_data.py (best-of KMeans/GMM-tied x refine off/on,
3 seeds, 16k subsample, CORE methods). Writes only to the round-2 ledger.
Does not overwrite expand_data.json.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from protocol_guard import (output_path, require_cached_input, require_candidate_rows,
    preflight_coordinate_frames)
OUTDIR = Path(output_path("n12"))
OUTDIR.mkdir(parents=True, exist_ok=True)
PARTIAL = OUTDIR / "n12_partial.json"

# Load expand_data helpers without running its n=14 loop.
_src = (HERE / "expand_data.py").read_text()
_head = _src.split("# Resumable:")[0]
exec(compile(_head, str(HERE / "expand_data.py"), "exec"), globals())  # noqa: S102

PATH = (
    Path(DR) / "processed/niche_abc_zhuang1_brain_slice/anndata.h5ad"
)
NAME = "Zhuang-ABCA-1.069(MERFISH,domain)"
GT_COLS = ["domain"]


def spearman_ci(x, y, rng, b=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, p = spearmanr(x, y)
    n = len(x)
    boots = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        boots.append(spearmanr(x[i], y[i]).correlation)
    boots = np.asarray(boots)
    return {"rho": float(rho), "p": float(p), "n": n, "n_boot": int(len(boots)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "frac_positive": float((boots > 0).mean())}


def run_one():
    if PARTIAL.exists():
        cached = json.loads(PARTIAL.read_text())
        require_cached_input(cached, [PATH, __file__, HERE / "expand_data.py"])
        print("cached", cached["platform"], flush=True)
        return cached
    Z, lognorm, counts, coords, true, subbed = load_any(str(PATH), GT_COLS)
    k = int(len(np.unique(true[true >= 0])))
    cont = gt_contiguity(coords, true)
    print(f"=== {NAME}: {Z.shape[0]} cells (subsampled={subbed}), k={k}, "
          f"contig={cont:.3f} device={dev} ===", flush=True)
    per = {}
    for seed in SEEDS:
        print(f" seed {seed} embeddings...", flush=True)
        for m, emb in embeddings(Z, lognorm, counts, coords, k, seed).items():
            record_seed(per, m, seed, best_config_ari(emb, true, coords, k, seed, method=m))
            print(f"  {m} ARI={per[m][seed]:.3f}", flush=True)
        try:
            sl = spatial_leiden_labels(Z, coords, k, seed)
            record_seed(per, "SpatialLeiden", seed,
                max(gari(true, sl), gari(true, refine_labels(sl, coords, k=6))))
            print(f"  SpatialLeiden ARI={per['SpatialLeiden'][seed]:.3f}", flush=True)
        except Exception as e:
            print("  SpatialLeiden fail", str(e)[:80], flush=True)
    mean, std, missing_seeds = complete_seed_summary(per, SEEDS)
    core_mean = {m: mean[m] for m in CORE if m in mean}
    order = sorted(core_mean, key=lambda m: -core_mean[m])
    nonsp = core_mean["floor:nonspatial"]
    sadv = max(v for m, v in core_mean.items() if m != "floor:nonspatial") - nonsp
    row = {
        "platform": NAME,
        "protocol_id": PROTOCOL_ID,
        "input_fingerprint": fingerprint([PATH, __file__, HERE / "expand_data.py"]),
        "missing_seeds": missing_seeds,
        "path": str(PATH),
        "gt_column": "domain",
        "gt_provenance": (
            "Allen-CCF-2020 anatomical division (AllenCCF-Ontology-2017); "
            "per-cell CCF registration, non-clustering-derived"
        ),
        "n_cells_used": int(Z.shape[0]),
        "n_cells_source": 22072,
        "subsampled": bool(subbed),
        "k": k,
        "GT_contiguity": round(float(cont), 3),
        "GT_contiguity_raw": float(cont),
        "winner": order[0],
        "spatial_advantage": round(float(sadv), 3),
        "spatial_advantage_raw": float(sadv),
        "means": {m: round(mean[m], 4) for m in mean},
        "seeds": list(SEEDS), "per_seed": per,
        "seeds": SEEDS,
        "protocol": "repaired candidate; expression-only control never refined; no mixing with historical panel",
    }
    PARTIAL.write_text(json.dumps(row, indent=2))
    print(f"winner {order[0]} adv {sadv:+.3f}", flush=True)
    return row


def main():
    t0 = datetime.now(timezone.utc).isoformat()
    pub = json.loads(Path(output_path("expanded_bench.json")).read_text())["rows"]
    require_candidate_rows(pub)
    preflight_coordinate_frames({NAME: (str(PATH), GT_COLS)})
    row = run_one()
    rng = np.random.default_rng(20260918)
    n11_c = [r["GT_contiguity"] for r in pub]
    n11_a = [r["spatial_advantage"] for r in pub]
    n12_c = n11_c + [row["GT_contiguity"]]
    n12_a = n11_a + [row["spatial_advantage"]]
    out = {
        "role": "n=12 Zhuang-ABCA-1.069 under published-panel protocol",
        "started_utc": t0,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(dev),
        "new_row": row,
        "n11": spearman_ci(n11_c, n11_a, rng),
        "n12": spearman_ci(n12_c, n12_a, rng),
        "points_n12": [
            {"platform": r["platform"], "contiguity": r["GT_contiguity"],
             "advantage": r["spatial_advantage"]}
            for r in pub
        ] + [{"platform": row["platform"], "contiguity": row["GT_contiguity"],
              "advantage": row["spatial_advantage"]}],
    }
    path = OUTDIR / "n12_zhuang.json"
    path.write_text(json.dumps(out, indent=2))
    print("WROTE", path)
    print("n11", out["n11"])
    print("n12", out["n12"])


if __name__ == "__main__":
    main()
