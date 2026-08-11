"""BAYESSPACE BASELINE-BREADTH EXTENSION --- STAGED, NOT YET EXECUTED.

WHY THIS EXISTS
    Peer benchmarks of spatial-domain detection carry 13-26 methods and always include at least one
    Bayesian / hidden-Markov-random-field (HMRF) method (BayesSpace, BASS, SpatialPCA). This paper's
    panel currently has 6-8 methods and ZERO Bayesian/HMRF family member. This script closes exactly
    that gap: it adds BayesSpace (Zhao et al., Nat. Biotechnol. 2021) --- the canonical Bayesian
    t-error + Potts/MRF spatial clusterer --- as a NEW method family in the existing 11-platform harness,
    then re-derives the headline (distinct winners; Spearman(GT contiguity, spatial-prior advantage))
    with the HMRF family included, to test whether "no universal SOTA" and the +0.72 law survive a
    broader, peer-normed baseline set.

    This is genuinely NEW compute (a BayesSpace MCMC run per platform). Per this repo's CLAUDE.md
    (no-new-heavy-recompute-by-default) it is therefore STAGED, not run: the file is committed so the
    user can execute it deliberately after installing BayesSpace. It is ADDITIVE --- it does not touch
    any number the current manuscript claims, and verify_manuscript.py does not depend on its output.

WHAT IT DOES (when the user runs it)
    1. Reuses the EXACT already-computed per-method ARIs from experiments/expanded_bench.json (the
       published n=11, 3-seed panel) --- so the deep methods are NOT re-run; the only new compute is
       BayesSpace. This is the cheap, correct way to slot one new baseline into a fixed harness.
    2. For each of the 11 platforms: loads the same subsampled spots (seed-0 subsample, identical to
       expanded_bench so BayesSpace is scored on the same cells), runs BayesSpace on PCA(log-norm, d=15)
       over the same spatial neighbourhood, and takes its best-config ARI over {raw, refine} (the same
       best-config rule every other method in the panel gets).
    3. Adds "BayesSpace" into each platform's method set, recomputes the per-platform winner, the number
       of DISTINCT winners across the panel, and the spatial-prior advantage (max spatial method minus
       the non-spatial floor --- BayesSpace is a spatial method, so it can only raise the max), then the
       Spearman(GT contiguity, spatial-prior advantage) with BayesSpace included.
    4. Writes experiments/bayesspace_bench.json and prints whether: (a) BayesSpace becomes a near-universal
       winner (>2 wins would threaten no-universal-SOTA --- same guard the SpaGCN panel uses), and
       (b) the +0.72 law holds with the HMRF family in the panel.

DEPENDENCIES (the reason this is staged, not run in-place)
    * R + Bioconductor:  run  Rscript experiments/install_bayesspace.R  (NOT a bare
      BiocManager::install(c("BayesSpace","SingleCellExperiment")): on this toolchain --- system R
      4.3.3 + current RcppArmadillo >=15 (needs C++14+) vs BayesSpace 1.12.0's CXX_STD=CXX11 --- the
      naive install fails the Armadillo C++14 compiler check; the helper adds a scoped
      CXX11STD=-std=gnu++17 override so the pinned C++11 build compiles as C++17. Verified working:
      BayesSpace 1.12.0 + SingleCellExperiment 1.24.0 load and run under R 4.3.3.)
      BayesSpace is R-only. This repo already shells out to R for one thing --- mclust, via
      experiments/_mclust.R called with `Rscript` + a CSV round-trip (rpy2 does not build on this
      Python 3.13 env). BayesSpace reuses that exact pattern through the sibling worker
      experiments/_bayesspace.R. mclust is only an R *backend* here; BayesSpace is the first R-based
      *method*, so its worker returns cluster labels directly (it slots in like SpatialLeiden, which
      also yields labels rather than an embedding).
    * Python side needs only what the rest of experiments/ already uses (numpy, anndata, sklearn,
      scipy) --- no new Python packages.

    NEIGHBOUR-STRUCTURE CAVEAT (#1 thing to validate on first run). BayesSpace's MRF prior is defined
    on a Visium/ST *lattice*: it derives neighbours from integer (row,col) array indices. Genuine Visium
    platforms here (DLPFC, BRCA) have a real lattice and pass platform="Visium". The imaging / bead
    platforms (MERFISH, osmFISH, seqFISH, STARmap, MIBI-TOF, IMC, CODEX, Slide-seqV2, openST) have NO
    native lattice, so _bayesspace.R snaps their float coordinates onto an integer grid and uses
    platform="ST" (square lattice). That snap is an APPROXIMATION of BayesSpace's intended input and is
    the main reason to sanity-check BayesSpace's per-platform output before trusting it on non-Visium
    data --- a reviewer may (fairly) prefer BayesSpace restricted to the true-lattice platforms only.
    Set BAYESSPACE_LATTICE_ONLY=1 to run it only where a native lattice exists (DLPFC, BRCA) and skip
    the rest rather than rely on the snap.

RUNTIME ESTIMATE (inferred --- no BayesSpace run exists on disk to calibrate against)
    BayesSpace spatialCluster is MCMC; cost ~ nrep * n_spots. At the BayesSpace default nrep=50000:
      * small/lattice platforms (~1.2k-5.5k spots: STARmap, MIBI-TOF, DLPFC, BRCA, IMC, osmFISH,
        MERFISH) --- roughly 2-12 min each;
      * the 16k-subsampled platforms (seqFISH, openST, Slide-seqV2, CODEX) --- the long pole, roughly
        20-60 min each.
    Whole 11-platform panel at 1 seed: ~2-5 h wall (single-threaded R), comparable per-platform to the
    real-mclust R-subprocess overhead the native_baselines.py pass already tolerated (20 mclust-R calls).
    Lower BAYESSPACE_NREP (e.g. 10000) for a first ~15-40 min smoke sweep before the full run; drop to
    BAYESSPACE_LATTICE_ONLY=1 for a ~5-15 min DLPFC+BRCA-only check. These are order-of-magnitude
    estimates from BayesSpace's published scaling, NOT measured here.

DO NOT add BayesSpace numbers to the manuscript from a single unvalidated run: fold results in only
after the neighbour-structure caveat is checked and (per CLAIM_LEDGER.md's LOCKED->graduated gate) the
panel-count / winner claims are re-verified.
"""

import os
import sys
from _roots import data_root, external_root, spatial_omics_root

sys.path.insert(0, "experiments")

import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from tessera_st.eval.refine import refine_labels  # noqa: E402

# Same real-data roots + the same 11 platforms/GT columns as native_baselines.py (the published panel).
DR = str(data_root())
EXT = str(external_root())
SUBSAMPLE = 16000
NREP = int(os.environ.get("BAYESSPACE_NREP", "50000"))
LATTICE_ONLY = os.environ.get("BAYESSPACE_LATTICE_ONLY", "0") == "1"
_BSPACE_R = os.path.join("experiments", "_bayesspace.R")

PLATFORMS = {
    "DLPFC(Visium,layer)": (f"{DR}/raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad",
                            ["layer_label", "Region", "ground_truth"]),
    "seqFISH(embryo,ctype)": (f"{DR}/raw/squidpy/seqfish.h5ad", ["celltype_mapped_refined"]),
    "MERFISH(hypo,domain)": (f"{DR}/baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad",
                             ["domain"]),
    "STARmap(cortex,region)": (f"{DR}/processed/starmap_mouse_vcortex_wang2018/anndata.h5ad",
                               ["layer_label", "Region", "ground_truth"]),
    "osmFISH(cortex,Region)": (f"{EXT}/../baselines/CellNiche-original/data/osmFISH_SScortex.h5ad",
                               ["Region"]),
    "MIBI-TOF(protein,Cluster)": (f"{DR}/raw/squidpy/mibitof.h5ad", ["Cluster"]),
    "BRCA(Visium,tumor)": (f"{DR}/processed/brca1_visium_10x/anndata.h5ad",
                           ["ground_truth", "annot_type", "fine_annot_type"]),
    "IMC(breast,ctype)": (f"{DR}/processed/niche_imc_breast/anndata.h5ad",
                          ["ground_truth", "Region", "cell_type"]),
    "openST(HNSCC,annot)": (f"{DR}/processed/openst_hnscc_sub15k/anndata.h5ad",
                            ["ground_truth", "annotation", "cell_class"]),
    "SlideseqV2(hippo,Region)": (f"{DR}/processed/slideseqv2_hippo/anndata.h5ad",
                                 ["Region", "ground_truth"]),
    "CODEX(spleen,niche)": (f"{DR}/processed/codex_spleen_goltsev2018/anndata.h5ad",
                            ["niche", "cell_type"]),
}
# non-spatial floor name in expanded_bench.json (the advantage denominator)
NONSPATIAL = "floor:nonspatial"


def _valid(v):
    return str(v).strip().lower() not in {"", "nan", "none", "na", "unknown"}


def load_any(path, gt_cols):
    """Identical load + seed-0 subsample as native_baselines.py, so BayesSpace is scored on exactly the
    same spots as the published panel. Returns log-normalised expression, coords, and integer GT."""
    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], float)
    col = next(c for c in gt_cols if c in a.obs.columns)
    raw = a.obs[col].to_numpy()
    X = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float64)
    n = X.shape[0]
    if n > SUBSAMPLE:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(n, SUBSAMPLE, replace=False))
        X, coords, raw = X[keep], coords[keep], raw[keep]
    codes = {v: i for i, v in enumerate(sorted({x for x in raw if _valid(x)}, key=str))}
    true = np.array([codes[v] if _valid(v) else -1 for v in raw], dtype=np.int64)
    if np.allclose(X, np.round(X)) and X.min() >= 0:
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
    if X.shape[1] > 500:
        X = X[:, np.argsort(X.var(0))[::-1][:3000]]
    return X, coords, true


def gari(true, lab):
    m = true >= 0
    return float(ari(true[m], lab[m]))


def gt_contiguity(coords, true, k=6):
    m = true >= 0
    cv, tv = coords[m], true[m]
    idx = NearestNeighbors(n_neighbors=k + 1).fit(cv).kneighbors(cv)[1][:, 1:]
    return float((tv[idx] == tv[:, None]).mean())


def bayesspace_labels(lognorm, coords, k, platform_arg, seed):
    """Run BayesSpace via Rscript + CSV round-trip (the _mclust.R pattern). BayesSpace natively clusters
    PCA(log-norm); we hand it PCA capped to d=15 (BayesSpace's default working dimensionality). Returns
    integer labels, or None on failure (logged, dropped for that platform --- fail-soft)."""
    d = min(15, lognorm.shape[1] - 1, lognorm.shape[0] - 1)
    emb = PCA(d, random_state=seed).fit_transform(lognorm)
    tmp = tempfile.mkdtemp(prefix="bayesspace_")
    fe, fc, fo = (os.path.join(tmp, "emb.csv"), os.path.join(tmp, "coords.csv"),
                  os.path.join(tmp, "lab.txt"))
    try:
        np.savetxt(fe, emb, delimiter=",",
                   header=",".join(f"V{i}" for i in range(emb.shape[1])), comments="")
        np.savetxt(fc, coords, delimiter=",", header="x,y", comments="")
        r = subprocess.run(["Rscript", _BSPACE_R, fe, fc, str(k), fo, str(seed),
                            platform_arg, str(NREP)],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode != 0 or not os.path.exists(fo):
            print("  BayesSpace fail:", (r.stderr or "")[-200:])
            return None
        return np.loadtxt(fo).astype(int)
    except Exception as e:
        print("  BayesSpace exception:", str(e)[:120])
        return None
    finally:
        for f in (fe, fc, fo):
            try:
                os.remove(f)
            except OSError:
                pass


def best_config_ari(labels, true, coords, k):
    """Best ARI over {raw labels, spatially-refined labels} --- the same best-config rule the panel
    gives every method (BayesSpace already MRF-smooths, so refine is usually a no-op/slightly worse,
    but we apply the identical rule for fairness)."""
    return max(gari(true, labels), gari(true, refine_labels(labels, coords, k=6)))


def main():
    if not os.path.exists("experiments/expanded_bench.json"):
        sys.exit("expanded_bench.json missing --- run the base panel first; BayesSpace slots into it.")
    base = {r["platform"]: r for r in json.load(open("experiments/expanded_bench.json"))["rows"]}

    partial = "experiments/bayesspace_bench_partial.json"
    rows = json.load(open(partial))["rows"] if os.path.exists(partial) else []
    done = {r["platform"] for r in rows}

    for name, (path, gt) in PLATFORMS.items():
        if name in done:
            print(f"=== {name}: cached, skipping ===", flush=True); continue
        platform_arg = "Visium" if "Visium" in name else "ST"
        if LATTICE_ONLY and platform_arg != "Visium":
            print(f"=== {name}: skipped (BAYESSPACE_LATTICE_ONLY, no native lattice) ===", flush=True)
            continue
        if name not in base:
            print(f"=== {name}: not in expanded_bench.json, skipping ==="); continue
        try:
            lognorm, coords, true = load_any(path, gt)
            k = int(len(np.unique(true[true >= 0])))
            cont = base[name]["GT_contiguity"]
            print(f"\n=== {name}: {lognorm.shape[0]} cells, {k} domains, contig {cont:.3f}, "
                  f"platform={platform_arg}, nrep={NREP} ===", flush=True)
            lab = bayesspace_labels(lognorm, coords, k, platform_arg, seed=1)
            if lab is None:
                continue
            bs = round(best_config_ari(lab, true, coords, k), 4)
            means = dict(base[name]["means"]); means["BayesSpace"] = bs
            winner = max(means, key=lambda m: means[m])
            nonsp = means.get(NONSPATIAL, 0.0)
            sadv = round(max(v for m, v in means.items() if m != NONSPATIAL) - nonsp, 3)
            rows.append({"platform": name, "GT_contiguity": cont, "k": k,
                         "BayesSpace_ari": bs, "winner_with_bayesspace": winner,
                         "spatial_advantage_with_bayesspace": sadv,
                         "spatial_advantage_published": base[name]["spatial_advantage"],
                         "means_with_bayesspace": {m: round(v, 4) for m, v in means.items()}})
            json.dump({"rows": rows}, open(partial, "w"), indent=2)
            print(f"  BayesSpace ARI {bs} | winner now {winner} | adv {sadv:+.3f} "
                  f"(published {base[name]['spatial_advantage']:+.3f})", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"{name} FAILED {str(e)[:80]}")

    if not rows:
        sys.exit("no BayesSpace results produced.")
    cont = np.array([r["GT_contiguity"] for r in rows])
    sadv = np.array([r["spatial_advantage_with_bayesspace"] for r in rows])
    rho = spearmanr(cont, sadv)
    winners = sorted({r["winner_with_bayesspace"] for r in rows})
    bs_wins = sum(r["winner_with_bayesspace"] == "BayesSpace" for r in rows)

    out = {
        "rows": rows, "n_platforms": len(rows), "new_method": "BayesSpace",
        "method_family": "Bayesian t-error + Potts/MRF (HMRF family)",
        "nrep": NREP, "lattice_only": LATTICE_ONLY,
        "distinct_winners_with_bayesspace": winners,
        "n_distinct_winners_with_bayesspace": len(winners),
        "n_bayesspace_wins": bs_wins,
        "spearman_contiguity_vs_spatial_advantage_with_bayesspace": round(float(rho.correlation), 3),
        "p_with_bayesspace": round(float(rho.pvalue), 4),
        "published_spearman_advantage": 0.724, "published_p": 0.0117,
    }
    json.dump(out, open("experiments/bayesspace_bench.json", "w"), indent=2)
    print("\n===== BAYESSPACE (HMRF family) ADDED TO THE PANEL =====")
    print(f"distinct winners now: {winners} ({len(winners)})")
    print(f"BayesSpace wins {bs_wins} platform(s) "
          f"[>2 would threaten no-universal-SOTA --- same guard as the SpaGCN panel]")
    print(f"Spearman(contiguity, spatial_advantage) with BayesSpace = {rho.correlation:+.3f} "
          f"(p={rho.pvalue:.4f})  [published without it: +0.724, p=0.0117]")
    print("wrote experiments/bayesspace_bench.json")


if __name__ == "__main__":
    main()
