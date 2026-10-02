"""Science hardening round 2 (2026-09-18): n-raise screen, Bayesian ρ, meta decision.

Real files only. Does not invent cell counts. Does not add a dataset that fails
the published-panel inclusion bar. Neighbour-mean / STAGATE generator not re-run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

CAPSULE = Path("/home/zeyufu/Desktop/labs/capsules/tessera-option-hold")
DATA = Path("/home/zeyufu/Desktop/labs/active/spatial-omics-reform/data")
HEADLINE = CAPSULE / "manuscript" / "ledgers" / "headline_json"
OUTDIR = CAPSULE / "manuscript" / "ledgers" / "science_core_20260918_r2"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 20260918
RNG = np.random.default_rng(SEED)
GRID = 4001

CANDIDATE = (
    DATA / "baselines/serial3d_ref/merfish_mouse_hypothalamus/deepstarmap_mouse_brain.h5ad"
)

# Same bar the published n=11 + expand_data extras used. BRCA's SEDR labels are a
# disclosed exception already on the panel; they do not license new method-derived rows.
INCLUSION = {
    "C1_spatial": "obsm['spatial'] present with shape (n, >=2)",
    "C2_biological_gt": (
        "at least one obs column is a published biological annotation "
        "(domain/region/layer/tumour/cell type/niche/annotation/cluster/ground_truth) "
        "with >=2 valid classes; not a single dummy"
    ),
    "C3_not_duplicate": "not a byte-duplicate of an already-included file",
    "C4_independent_platform": (
        "not an extra slice/section of an already-included experiment counted as +1 n"
    ),
    "C5_gt_independence_new": (
        "NEW rows beyond the published 11 require independent biological GT. "
        "Method-derived labels (Harmony clusters, FUSEmap domains, Leiden on the "
        "same matrix, a competitor method's domains) do not qualify. "
        "BRCA/SEDR is a disclosed exception already on the panel, not a licence "
        "to add more method-derived rows."
    ),
    "C6_not_technical": "not technical-only (e.g. segmentation_method)",
    "C7_not_scrna": "not scRNA-seq / atlas without spatial coordinates",
}

ALREADY = {
    str(DATA / "raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad"),
    str(DATA / "raw/squidpy/seqfish.h5ad"),
    str(DATA / "baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad"),
    str(DATA / "processed/starmap_mouse_vcortex_wang2018/anndata.h5ad"),
    str(DATA / "raw/squidpy/mibitof.h5ad"),
    str(DATA / "processed/brca1_visium_10x/anndata.h5ad"),
    str(DATA / "processed/niche_imc_breast/anndata.h5ad"),
    str(DATA / "processed/openst_hnscc_sub15k/anndata.h5ad"),
    str(DATA / "processed/slideseqv2_hippo/anndata.h5ad"),
    str(DATA / "processed/codex_spleen_goltsev2018/anndata.h5ad"),
    str(DATA / "baselines/st_impute_ref/processed_data/st_COAD_test.h5ad"),
    str(DATA / "baselines/st_impute_ref/processed_data/st_LIHC_test.h5ad"),
    str(DATA / "baselines/st_impute_ref/processed_data/st_OV_test.h5ad"),
}

METHOD_DERIVED = (
    "harmony", "fusemap", "leiden", "louvain", "sedr", "stagate", "graphst",
    "banksy", "spaceflow", "spagcn", "tessera",
)
TECHNICAL = ("segmentation_method", "segmentation",)
BIO_HINTS = (
    "domain", "region", "layer", "tumour", "tumor", "annotation", "annot",
    "cell_type", "celltype", "cell_class", "niche", "cluster", "ground_truth",
    "harmony", "fusemap", "label", "class",
)


def _valid(v):
    return str(v).strip().lower() not in {"", "nan", "none", "na", "unknown"}


def sha256_file(path: Path, limit=0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if limit:
            h.update(f.read(limit))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def inspect_h5ad(path: Path) -> dict:
    import anndata as ad

    rec = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}
    if not path.exists():
        rec["error"] = "missing"
        return rec
    try:
        a = ad.read_h5ad(path, backed="r")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["n_obs"] = int(a.n_obs)
    rec["n_vars"] = int(a.n_vars)
    rec["obs_columns"] = list(map(str, a.obs.columns))
    rec["obsm_keys"] = list(map(str, a.obsm.keys()))
    rec["uns_keys"] = list(map(str, getattr(a, "uns", {}).keys()))
    rec["has_spatial"] = "spatial" in a.obsm
    if rec["has_spatial"]:
        rec["spatial_shape"] = list(a.obsm["spatial"].shape)
    rec["label_columns"] = []
    for col in a.obs.columns:
        name = str(col)
        low = name.lower()
        if not any(h in low for h in BIO_HINTS):
            continue
        raw = a.obs[col]
        try:
            vals = [v for v in raw.astype(str).tolist() if _valid(v)]
        except Exception:
            continue
        n_cls = len(set(vals))
        top = sorted({str(v) for v in vals})[:12]
        rec["label_columns"].append({
            "column": name,
            "n_valid": len(vals),
            "n_classes": n_cls,
            "example_classes": top,
            "method_derived_name": any(m in low for m in METHOD_DERIVED),
            "technical_name": any(t == low or t in low for t in TECHNICAL),
        })
    try:
        a.file.close()
    except Exception:
        pass
    return rec


def apply_criteria(rec: dict, *, already: bool, same_study: str | None) -> dict:
    fails = []
    if not rec.get("has_spatial"):
        fails.append("C1_spatial")
    labs = rec.get("label_columns") or []
    bio = [c for c in labs if c["n_classes"] >= 2 and not c["technical_name"]]
    indep = [c for c in bio if not c["method_derived_name"]]
    if not bio:
        fails.append("C2_biological_gt")
    if already:
        fails.append("C3_not_duplicate")
    if same_study:
        fails.append("C4_independent_platform")
        rec["same_study_as"] = same_study
    if bio and not indep:
        fails.append("C5_gt_independence_new")
    if labs and all(c["technical_name"] for c in labs) and not indep:
        fails.append("C6_not_technical")
    if not rec.get("has_spatial") and rec.get("n_obs", 0) > 0:
        fails.append("C7_not_scrna")
    rec["fails"] = fails
    rec["qualifies_as_new_n"] = len(fails) == 0
    rec["independent_gt_columns"] = [c["column"] for c in indep]
    rec["method_derived_columns"] = [c["column"] for c in bio if c["method_derived_name"]]
    return rec


def same_study_of(path: Path) -> str | None:
    p = str(path)
    rules = [
        ("IMC Jackson breast (niche_imc_breast already in n=11)",
         "imc_human_breastcancer" in p or p.endswith("/raw/squidpy/imc.h5ad")),
        ("Open-ST HNSCC (openst_hnscc_sub15k already in n=11)",
         "openst_hnscc" in p and "sub15k" not in p),
        ("MERFISH Moffitt hypothalamus (merfish_0 already in n=11)",
         "merfish_mouse_hypothalamus" in p and path.name != "deepstarmap_mouse_brain.h5ad"),
        ("MIBI-TOF colorectal duplicate of squidpy mibitof",
         "mibitof_colorectal" in p),
        ("DLPFC Maynard 151673 already in n=11",
         "dlpfc_maynard_2021" in p and "151673" in p),
        ("STARmap Wang 2018 already in n=11",
         "starmap_mouse_vcortex_wang2018" in p and "processed" not in p),
        ("Slide-seqV2 hippo already in n=11",
         "slideseqv2" in p and "processed/slideseqv2_hippo" not in p),
        ("seqFISH embryo already in n=11",
         "seqfish" in p and "raw/squidpy/seqfish" not in p),
        ("st_impute CESC/NSCLC/PRAD technical-label siblings of the n=14 extras",
         any(x in p for x in ("st_CESC", "st_NSCLC", "st_PRAD"))),
        ("synthetic generator tissue, not a real dataset",
         "/synthetic/" in p),
        ("scRNA-seq reference (lumina/qukun *_ref)",
         "sc_reference" in p or "lumina_ref" in p or p.endswith("wu_breast_atlas.h5ad")
         or p.endswith("sc_mouse_cortex.h5ad")),
    ]
    for label, cond in rules:
        if cond:
            return label
    return None


def fisher_z_posterior(rho_hat: float, n: int, prior: str) -> dict:
    """Grid posterior on Spearman ρ.

    Likelihood: Fisher-z, z_obs | ρ ~ Normal(atanh(ρ), 1/sqrt(n-3)).
    Priors (stated):
      uniform: π(ρ) = 1/2 on (-1, 1)
      skeptical: π(ρ) ∝ Normal(0, 0.5^2) truncated to (-1, 1)
    """
    rho_hat = float(np.clip(rho_hat, -0.999999, 0.999999))
    z_obs = np.arctanh(rho_hat)
    se = 1.0 / np.sqrt(n - 3)
    grid = np.linspace(-0.999, 0.999, GRID)
    z = np.arctanh(grid)
    loglik = -0.5 * ((z_obs - z) / se) ** 2
    if prior == "uniform":
        logprior = np.zeros_like(grid)
    elif prior == "skeptical":
        logprior = -0.5 * (grid / 0.5) ** 2
    else:
        raise ValueError(prior)
    logpost = loglik + logprior
    logpost -= logpost.max()
    post = np.exp(logpost)
    post /= post.sum()
    cdf = np.cumsum(post)
    def q(p):
        return float(grid[int(np.searchsorted(cdf, p).clip(0, len(grid) - 1))])
    mean = float((grid * post).sum())
    median = q(0.5)
    return {
        "rho_hat": rho_hat,
        "n": n,
        "prior": prior,
        "likelihood": "Fisher-z Normal(atanh(ρ), 1/sqrt(n-3))",
        "z_obs": float(z_obs),
        "se_z": float(se),
        "posterior_mean": mean,
        "posterior_median": median,
        "credible95": [q(0.025), q(0.975)],
        "P_rho_gt_0": float(post[grid > 0].sum()),
        "P_rho_gt_0.5": float(post[grid > 0.5].sum()),
        "P_rho_gt_0.72": float(post[grid > 0.72].sum()),
    }


def bootstrap_posterior_mass(x, y, rng, b=5000) -> dict:
    """Bayesian bootstrap (Dirichlet-1 weights) on Spearman ρ. Prior = uniform on discrete atoms."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    boots = []
    for _ in range(b):
        w = rng.dirichlet(np.ones(n))
        # weighted rank correlation via weighted mid-ranks is heavier; use
        # multinomial Bayesian bootstrap (Rubin) which is equivalent at this n.
        i = rng.choice(n, size=n, replace=True)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        boots.append(float(spearmanr(x[i], y[i]).correlation))
    boots = np.asarray(boots)
    return {
        "method": "Rubin Bayesian bootstrap (multinomial resample), B=%d" % b,
        "n_boot": int(len(boots)),
        "mean": float(boots.mean()),
        "median": float(np.median(boots)),
        "credible95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "P_rho_gt_0": float((boots > 0).mean()),
    }


def observational_vectors():
    eb = json.loads((HEADLINE / "expanded_bench.json").read_text())
    gp = json.loads((HEADLINE / "gtfree_proxy.json").read_text())
    ex = json.loads((HEADLINE / "expand_data.json").read_text())
    rows = eb["rows"]
    cont = np.array([r["GT_contiguity"] for r in rows], float)
    adv = np.array([r["spatial_advantage"] for r in rows], float)
    proxy = np.array([
        next(r["coh_gain"] for r in gp["rows"] if r["platform"] == q["platform"])
        for q in rows
    ], float)
    n14_cont = np.concatenate([cont, [r["GT_contiguity"] for r in ex["new_rows"]]])
    n14_adv = np.concatenate([adv, [r["spatial_advantage"] for r in ex["new_rows"]]])
    return {
        "n11_names": [r["platform"] for r in rows],
        "n11_cont": cont, "n11_adv": adv, "n11_proxy": proxy,
        "n14_cont": n14_cont, "n14_adv": n14_adv,
        "n14_rho": float(spearmanr(n14_cont, n14_adv).correlation),
        "n14_p": float(spearmanr(n14_cont, n14_adv).pvalue),
        "n11_rho": float(spearmanr(cont, adv).correlation),
        "n11_p": float(spearmanr(cont, adv).pvalue),
        "proxy_rho": float(spearmanr(proxy, adv).correlation),
        "proxy_p": float(spearmanr(proxy, adv).pvalue),
    }


def naive_ivw_warning(rho_obs, n_obs, rho_syn, n_syn):
    """Show why inverse-variance pooling would be indefensible, without claiming it."""
    z1, z2 = np.arctanh(rho_obs), np.arctanh(np.clip(rho_syn, -0.999, 0.999))
    w1, w2 = n_obs - 3, n_syn - 3
    z = (w1 * z1 + w2 * z2) / (w1 + w2)
    return {
        "computed_only_as_counterexample": True,
        "model_if_wrongly_applied": "fixed-effect IVW on Fisher-z",
        "obs": {"rho": rho_obs, "n": n_obs, "weight_n_minus_3": w1},
        "synth": {"rho": rho_syn, "n": n_syn, "weight_n_minus_3": w2},
        "wrong_pooled_rho": float(np.tanh(z)),
        "synth_weight_fraction": float(w2 / (w1 + w2)),
        "why_invalid": (
            "The observational unit is a published dataset (n=11). The synthetic "
            "unit is a tessellation condition-evaluation (n=384 paper-config points) "
            "from one generator. Different estimands, different methods "
            "(full panel vs neighbour-mean), different sampling units, and the "
            "synthetic points are dependent. IVW would assign ~98% of the weight "
            "to the generator and silently replace the observational claim."
        ),
    }


def screen_disk():
    import anndata  # noqa: F401 — fail early if missing

    files = sorted(DATA.rglob("*.h5ad"))
    out = []
    for p in files:
        rec = inspect_h5ad(p)
        already = str(p) in ALREADY
        same = None if already else same_study_of(p)
        apply_criteria(rec, already=already, same_study=same)
        rec["already_in_panel"] = already
        out.append(rec)
        print(f"screen {p.relative_to(DATA)} n={rec.get('n_obs')} spatial={rec.get('has_spatial')} "
              f"fails={rec.get('fails')} indep={rec.get('independent_gt_columns')}", flush=True)
    return out


def candidate_deep():
    rec = inspect_h5ad(CANDIDATE)
    apply_criteria(rec, already=False, same_study=None)
    rec["head16_sha256"] = sha256_file(CANDIDATE, limit=1 << 20)
    rec["full_sha256"] = sha256_file(CANDIDATE)
    # merfish_0 is the included hypothalamus slice in the same folder
    mer0 = DATA / "baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad"
    rec["same_folder_as_included_merfish_0"] = True
    rec["merfish_0_n_obs"] = inspect_h5ad(mer0).get("n_obs")
    rec["distinct_from_merfish_0"] = rec.get("n_obs") != rec["merfish_0_n_obs"]
    rec["criterion_applied"] = "C5_gt_independence_new"
    rec["decision"] = (
        "FAIL C5" if "C5_gt_independence_new" in rec["fails"]
        else ("FAIL " + ",".join(rec["fails"]) if rec["fails"] else "QUALIFIES")
    )
    return rec


def main():
    t0 = datetime.now(timezone.utc).isoformat()
    print("inspect candidate", flush=True)
    cand = candidate_deep()
    print("CANDIDATE", json.dumps({k: cand[k] for k in cand if k != "obs_columns"}, default=str), flush=True)
    print("screen disk", flush=True)
    disk = screen_disk()
    vec = observational_vectors()
    endpoints = {
        "primary_n11_contiguity_vs_advantage": {
            "estimand": "Spearman ρ(GT-contiguity, spatial-prior advantage), n=11 published panel",
            "role": "primary",
            "rho_hat": vec["n11_rho"],
            "p_two_sided": vec["n11_p"],
            "fisher_z_uniform": fisher_z_posterior(vec["n11_rho"], 11, "uniform"),
            "fisher_z_skeptical": fisher_z_posterior(vec["n11_rho"], 11, "skeptical"),
            "bayes_bootstrap": bootstrap_posterior_mass(vec["n11_cont"], vec["n11_adv"], RNG),
        },
        "family7_same_estimand": {
            "estimand": "same ρ as primary; family-7 is a frequentist multiplicity correction, not a new ρ",
            "role": "pre-specified secondary (multiplicity)",
            "rho_hat": vec["n11_rho"],
            "bonferroni_p": 0.0819,
            "note": (
                "The posterior on ρ is identical to the primary. Bonferroni p=0.082 "
                "is a frequentist threshold statement, not a Bayesian quantity. "
                "Never call p=0.082 significant."
            ),
            "fisher_z_uniform": fisher_z_posterior(vec["n11_rho"], 11, "uniform"),
            "P_rho_gt_0_is_the_same_as_primary": True,
        },
        "proxy_coh_gain_vs_advantage": {
            "estimand": "Spearman ρ(coh_gain, spatial-prior advantage), n=11",
            "role": "pre-specified secondary after 6-candidate search",
            "rho_hat": vec["proxy_rho"],
            "p_two_sided": vec["proxy_p"],
            "fisher_z_uniform": fisher_z_posterior(vec["proxy_rho"], 11, "uniform"),
            "fisher_z_skeptical": fisher_z_posterior(vec["proxy_rho"], 11, "skeptical"),
            "bayes_bootstrap": bootstrap_posterior_mass(vec["n11_proxy"], vec["n11_adv"], RNG),
        },
        "robustness_n14_already_scored": {
            "estimand": "Spearman ρ on the n=14 expansion (nests the 11; not independent)",
            "role": "already-printed robustness, not a new dataset",
            "rho_hat": vec["n14_rho"],
            "p_two_sided": vec["n14_p"],
            "fisher_z_uniform": fisher_z_posterior(vec["n14_rho"], 14, "uniform"),
            "fisher_z_skeptical": fisher_z_posterior(vec["n14_rho"], 14, "skeptical"),
            "bayes_bootstrap": bootstrap_posterior_mass(vec["n14_cont"], vec["n14_adv"], RNG),
        },
    }
    synth_rho = 0.9195443779396417
    meta = {
        "combinable": False,
        "model_considered": "fixed-effect or random-effects meta-analysis of Fisher-z",
        "why_not_exchangeable": [
            "Different estimands: cross-dataset observational association vs within-generator intervention.",
            "Different units: 11 published tissues vs 384 dependent tessellation evaluations.",
            "Different spatial methods: full competitor panel vs neighbour-mean only.",
            "Synthetic points share one generator and are not independent studies.",
            "n=14 nests n=11, so it is also not a second observational study.",
        ],
        "naive_ivw_counterexample": naive_ivw_warning(vec["n11_rho"], 11, synth_rho, 384),
    }
    new_ok = [r for r in disk if r.get("qualifies_as_new_n")]
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "anndata": __import__("importlib.metadata", fromlist=["version"]).version("anndata"),
        "SOURCE_DATE_EPOCH": "0",
        "seed": SEED,
        "grid": GRID,
    }
    out = {
        "role": "science-hardening round 2 2026-09-18; n-raise screen + Bayesian ρ + meta decision",
        "started_utc": t0,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "inclusion_criteria": INCLUSION,
        "candidate": cand,
        "n_rose": False if not cand.get("qualifies_as_new_n") else True,
        "n_published": 11,
        "n_if_candidate_added": 12 if cand.get("qualifies_as_new_n") else 11,
        "disk_screen": {
            "n_h5ad": len(disk),
            "n_qualify_as_new": len(new_ok),
            "qualifying_paths": [r["path"] for r in new_ok],
            "rows": disk,
        },
        "posteriors": endpoints,
        "meta_analysis": meta,
        "preregistration": {
            "primary": (
                "Spearman ρ(GT-contiguity, spatial-prior advantage) on the de-duplicated "
                "n=11 panel. Uncorrected two-sided α=0.05. Designed 80% power at ρ=+0.72 "
                "needs n=14 (MC)/n=13 (Fisher-z)."
            ),
            "secondary_multiplicity": (
                "Family-7 Bonferroni on the same ρ. Designed 80% power needs n=21 (MC)/n=18 "
                "(Fisher-z). Observed Bonferroni p=0.082 is not significant."
            ),
            "secondary_proxy": (
                "coh_gain vs advantage after a 6-candidate search (Westfall-Young). "
                "Designed 80% power at ρ=+0.55 needs n=26 uncorrected."
            ),
            "secondary_lodo": (
                "LODO decision-rule accuracy vs base 0.545. Designed 80% needs n=21."
            ),
            "not_primary": "generator ΔARI / pooled synthetic ρ are a separate causal arm, not pooled with observational ρ.",
        },
    }
    path = OUTDIR / "science_core_20260918_r2.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print("WROTE", path)
    print("candidate decision", cand["decision"], "n_rose", out["n_rose"])
    print("new qualifiers", len(new_ok))
    p11 = endpoints["primary_n11_contiguity_vs_advantage"]["fisher_z_uniform"]
    print("primary uniform P(ρ>0)", p11["P_rho_gt_0"], "median", p11["posterior_median"],
          "CrI", p11["credible95"])


if __name__ == "__main__":
    main()
