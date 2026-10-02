"""Fill the two n=11 SpaceFlow holes on protein panels (MIBI-TOF, CODEX).

Reason recorded in verify_manuscript.py: SpaceFlow failed on 2 protein panels,
so ranks are n=9. This script reruns ONLY SpaceFlow under the published
expanded_bench protocol (3 seeds, 16k subsample, best-of KMeans/GMM-tied).
It writes a ledger. It does not invent ARI if the run fails.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "manuscript" / "experiments"))

from _roots import data_root, external_root  # noqa: E402

DR = data_root()
EXT = external_root()
for name in ("SpaceFlow",):
    sys.path.insert(0, str(EXT / name))

import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

SEEDS = [1, 2, 3]
SUBSAMPLE = 16000
OUT = Path(__file__).resolve().parent / "n11_spaceflow_protein_panels.json"

PANELS = {
    "MIBI-TOF(protein,Cluster)": (
        DR / "raw/squidpy/mibitof.h5ad",
        ["Cluster"],
    ),
    "CODEX(spleen,niche)": (
        DR / "processed/codex_spleen_goltsev2018/anndata.h5ad",
        ["niche", "cell_type"],
    ),
}


def _valid(v) -> bool:
    return str(v).strip().lower() not in {"", "nan", "none", "na", "unknown"}


def load_panel(path: Path, gt_cols):
    import anndata as ad

    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], float)
    col = next(c for c in gt_cols if c in a.obs.columns)
    raw = a.obs[col].to_numpy()
    X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float64)
    n = X.shape[0]
    subsampled = n > SUBSAMPLE
    if subsampled:
        keep = np.sort(np.random.RandomState(0).choice(n, SUBSAMPLE, replace=False))
        X, coords, raw = X[keep], coords[keep], raw[keep]
    codes = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([codes[v] if _valid(v) else -1 for v in raw], dtype=np.int64)
    return X.copy(), coords, true, subsampled, n, int(true.max() + 1), col


def best_ari(emb, true, k, seed) -> float:
    labs = [KMeans(k, n_init=10, random_state=seed).fit_predict(emb)]
    try:
        labs.append(
            GaussianMixture(
                k, covariance_type="tied", random_state=seed, n_init=3, reg_covar=1e-2
            ).fit_predict(emb)
        )
    except Exception:
        pass
    m = true >= 0
    return float(max(ari(true[m], lab[m]) for lab in labs))


def run_spaceflow(counts, coords, seed):
    from SpaceFlow.SpaceFlow import SpaceFlow
    import anndata as ad

    a = ad.AnnData(X=counts.astype(np.float32))
    a.obsm["spatial"] = coords
    sf = SpaceFlow(adata=a)
    sf.preprocessing_data(n_top_genes=min(3000, counts.shape[1]))
    emb = np.asarray(
        sf.train(
            embedding_save_filepath=str(OUT.parent / f"_sf_emb_{seed}.tsv"),
            epochs=400,
            random_seed=seed,
            z_dim=50,
            min_stop=80,
            max_patience=30,
        )
    )
    return emb


def main() -> int:
    payload = {
        "role": "n=11 SpaceFlow fill on the two protein panels that were NA",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "expanded_bench.py SpaceFlow + best-of KMeans/GMM-tied, 3 seeds, 16k subsample",
        "headline_remains_n11": True,
        "do_not_invent": True,
        "panels": {},
        "blocked": [],
    }
    print("data_root", DR)
    print("external SpaceFlow", EXT / "SpaceFlow")
    for name, (path, cols) in PANELS.items():
        rec = {
            "path": str(path),
            "exists": path.is_file(),
            "gt_cols": cols,
            "status": "pending",
        }
        payload["panels"][name] = rec
        if not path.is_file():
            rec["status"] = "blocked_missing_h5ad"
            payload["blocked"].append({"platform": name, "reason": "missing_h5ad", "path": str(path)})
            print("MISSING", name, path)
            continue
        try:
            counts, coords, true, subsampled, n_src, k, col = load_panel(path, cols)
        except Exception as exc:
            rec["status"] = "blocked_load"
            rec["error"] = str(exc)
            payload["blocked"].append({"platform": name, "reason": "load", "error": str(exc)})
            print("LOAD FAIL", name, exc)
            continue
        rec.update(
            {
                "gt_column": col,
                "n_cells_source": n_src,
                "n_cells_used": int(counts.shape[0]),
                "n_genes": int(counts.shape[1]),
                "k": k,
                "subsampled": subsampled,
            }
        )
        seed_ari = []
        seed_err = []
        for seed in SEEDS:
            try:
                emb = run_spaceflow(counts, coords, seed)
                val = best_ari(emb, true, k, seed)
                seed_ari.append(val)
                print(f"  {name} seed={seed} ARI={val:.4f}")
            except Exception as exc:
                seed_err.append({"seed": seed, "error": str(exc), "trace": traceback.format_exc()[-800:]})
                print(f"  {name} seed={seed} FAIL {exc}")
        if seed_ari:
            rec["per_seed_ari"] = seed_ari
            rec["mean_ari"] = float(np.mean(seed_ari))
            rec["n_seeds_ok"] = len(seed_ari)
            rec["status"] = "ok" if len(seed_ari) == 3 else "partial"
        else:
            rec["status"] = "blocked_runtime"
            rec["seed_errors"] = seed_err
            payload["blocked"].append({"platform": name, "reason": "spaceflow_runtime", "errors": seed_err})
        if seed_err and seed_ari:
            rec["seed_errors"] = seed_err
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    payload["n_ok"] = sum(1 for r in payload["panels"].values() if r.get("status") in {"ok", "partial"})
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT)
    return 0 if payload["n_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
