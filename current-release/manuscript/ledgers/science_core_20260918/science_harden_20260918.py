"""Science hardening 2026-09-18: scale the cheap neighbour-mean generator,
power the observational arm, full LOO, and pre-specified secondary endpoints.

Neighbour-mean only (no STAGATE). Same substrate as mechanism_synth.py.
Does not overwrite experiments/mechanism_synth.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
CAPSULE = Path("/home/zeyufu/Desktop/labs/capsules/tessera-option-hold")
WB_SRC = Path("/home/zeyufu/Desktop/labs/active/tessera-st/src")
VENDORED_SRC = HERE / "vendored_src"
for _p in (WB_SRC, VENDORED_SRC):
    if _p.exists():
        sys.path.insert(0, str(_p))
        break
from tessera_st.data.synthetic import make_tessellation  # noqa: E402

LEDGER = CAPSULE / "manuscript" / "ledgers"
HEADLINE = LEDGER / "headline_json"
OUTDIR = LEDGER / "science_core_20260918"
OUTDIR.mkdir(parents=True, exist_ok=True)

SCRAMBLE = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0]
PAPER_SEEDS = list(range(1, 49))  # original {1,2,3} plus 45 new
PAPER_CFG = dict(n_side=34, n_genes=40, n_domains=5, domain_signal=0.3,
                 iid_noise=1.0, small_domain=True)
EXTRA = [
    ("small_grid", dict(n_side=24, n_genes=40, n_domains=4, domain_signal=0.3,
                        iid_noise=1.0, small_domain=True), list(range(101, 117))),
    ("large_grid", dict(n_side=40, n_genes=40, n_domains=6, domain_signal=0.3,
                        iid_noise=1.0, small_domain=True), list(range(201, 217))),
    ("weak_signal", dict(n_side=34, n_genes=40, n_domains=5, domain_signal=0.2,
                         iid_noise=1.0, small_domain=True), list(range(301, 317))),
    ("strong_signal", dict(n_side=34, n_genes=40, n_domains=5, domain_signal=0.45,
                           iid_noise=1.0, small_domain=True), list(range(401, 417))),
    ("more_domains", dict(n_side=34, n_genes=40, n_domains=8, domain_signal=0.3,
                          iid_noise=1.0, small_domain=True), list(range(501, 517))),
]
B_BOOT = 5000
MC_POWER = 4000
RNG = np.random.default_rng(20260918)


def scramble_positions(coords, frac, rng):
    n = coords.shape[0]
    m = int(round(frac * n))
    if m < 2:
        return coords.copy()
    sel = rng.choice(n, m, replace=False)
    out = coords.copy()
    out[sel] = coords[rng.permutation(sel)]
    return out


def knn_idx(coords, k=6):
    return NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)[1][:, 1:]


def contiguity(labels, idx):
    return float((labels[idx] == labels[:, None]).mean())


def best_ari(emb, labels, k, seed):
    best = ari(labels, KMeans(k, n_init=10, random_state=seed).fit_predict(emb))
    try:
        g = GaussianMixture(k, covariance_type="tied", n_init=3, random_state=seed,
                            reg_covar=1e-4).fit_predict(emb)
        best = max(best, ari(labels, g))
    except Exception:
        pass
    return float(best)


def one_condition(cfg, scramble, seed):
    rng = np.random.default_rng(seed)
    sl = make_tessellation(seed=seed, **cfg)
    expr, labels = sl.expr.astype(np.float64), sl.labels
    coords = scramble_positions(sl.coords, scramble, rng)
    k = int(labels.max()) + 1
    mu, sd = expr.mean(0), expr.std(0)
    sd[sd == 0] = 1
    z = np.clip((expr - mu) / sd, -10, 10)
    idx = knn_idx(coords, 6)
    zpca = PCA(min(30, z.shape[1], z.shape[0] - 1), random_state=seed).fit_transform(z)
    nbr = PCA(min(30, z.shape[1], z.shape[0] - 1), random_state=seed).fit_transform(z[idx].mean(1))
    a_non = best_ari(zpca, labels, k, seed)
    a_sm = best_ari(nbr, labels, k, seed)
    return {"scramble": scramble, "seed": seed, "contiguity": contiguity(labels, idx),
            "nonspatial_ari": a_non, "smooth_ari": a_sm, "adv_smooth": a_sm - a_non}


def _job(args):
    name, cfg, scramble, seed = args
    rec = one_condition(cfg, scramble, seed)
    rec["config"] = name
    return rec


def run_grid(name, cfg, seeds, workers=8):
    jobs = [(name, cfg, s, seed) for seed in seeds for s in SCRAMBLE]
    rows = []
    if workers <= 1:
        it = map(_job, jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=workers)
        it = pool.map(_job, jobs, chunksize=2)
    for rec in it:
        rows.append(rec)
        print(f"{name} seed={rec['seed']} s={rec['scramble']:.2f} "
              f"contig={rec['contiguity']:.3f} adv={rec['adv_smooth']:+.3f}", flush=True)
    if workers > 1:
        pool.shutdown()
    return rows


def spearman_ci(x, y, rng, b=B_BOOT):
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


def mean_ci(v, rng, b=B_BOOT):
    v = np.asarray(v, float)
    boots = [float(np.mean(v[rng.integers(0, len(v), len(v))])) for _ in range(b)]
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n": int(len(v)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def per_seed_spearman(rows):
    by = {}
    for r in rows:
        by.setdefault(r["seed"], []).append(r)
    out = []
    for seed, rr in sorted(by.items()):
        c = [x["contiguity"] for x in rr]
        a = [x["adv_smooth"] for x in rr]
        rho, p = spearmanr(c, a)
        adv0 = next(x["adv_smooth"] for x in rr if x["scramble"] == 0.0)
        adv1 = next(x["adv_smooth"] for x in rr if x["scramble"] == 1.0)
        out.append({"seed": seed, "rho": float(rho), "p": float(p),
                    "delta_adv_s0_minus_s1": float(adv0 - adv1),
                    "sign_flip": bool(adv0 > 0 > adv1),
                    "nonsp_range": float(max(x["nonspatial_ari"] for x in rr)
                                         - min(x["nonspatial_ari"] for x in rr))})
    return out


def condition_means(rows):
    by = {}
    for r in rows:
        by.setdefault(r["scramble"], []).append(r)
    recs = []
    for s in SCRAMBLE:
        rr = by[s]
        recs.append({"scramble": s,
                     "contiguity": float(np.mean([x["contiguity"] for x in rr])),
                     "adv_smooth": float(np.mean([x["adv_smooth"] for x in rr])),
                     "nonspatial_ari": float(np.mean([x["nonspatial_ari"] for x in rr]))})
    return recs


def fisher_n_for_power(rho, alpha=0.05, power=0.80):
    z_r = np.arctanh(rho)
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    n = 3 + ((z_a + z_b) / z_r) ** 2
    return int(np.ceil(n))


def mc_spearman_power(rho, n, alpha, n_sim, rng):
    z = np.arctanh(rho)
    hits = 0
    for _ in range(n_sim):
        x = rng.normal(size=n)
        y = z * x + rng.normal(size=n)  # not exact Spearman; rank after
        # Gaussian copula with Pearson ≈ Fisher-z target, then Spearman
        y = rho * x + np.sqrt(max(1 - rho ** 2, 0)) * rng.normal(size=n)
        p = spearmanr(x, y).pvalue
        hits += int(p < alpha)
    return float(hits / n_sim)


def n_for_mc_power(rho, alpha, target=0.80, n_min=6, n_max=60):
    for n in range(n_min, n_max + 1):
        pw = mc_spearman_power(rho, n, alpha, MC_POWER, RNG)
        if pw >= target:
            return n, pw
    return n_max, mc_spearman_power(rho, n_max, alpha, MC_POWER, RNG)


def observational():
    eb = json.loads((HEADLINE / "expanded_bench.json").read_text())
    sr = json.loads((HEADLINE / "stat_rigor.json").read_text())
    ra = json.loads((HEADLINE / "robustness_audit.json").read_text())
    gp = json.loads((HEADLINE / "gtfree_proxy.json").read_text())
    psc = json.loads((HEADLINE / "proxy_search_corrected.json").read_text())
    ex = json.loads((HEADLINE / "expand_data.json").read_text())
    rows = eb["rows"]
    names = [r["platform"].split("(")[0] for r in rows]
    cont = np.array([r["GT_contiguity"] for r in rows], float)
    adv = np.array([r["spatial_advantage"] for r in rows], float)
    rho_all, p_all = spearmanr(cont, adv)
    loo = []
    for i, name in enumerate(names):
        m = np.ones(len(names), bool)
        m[i] = False
        rho, p = spearmanr(cont[m], adv[m])
        loo.append({"dropped": name, "platform": rows[i]["platform"],
                    "contiguity": float(cont[i]), "advantage": float(adv[i]),
                    "loo_rho": float(rho), "loo_p": float(p),
                    "delta_rho": float(rho - rho_all)})
    loo = sorted(loo, key=lambda d: d["delta_rho"])
    proxy_c = np.array([next(r["coh_gain"] for r in gp["rows"] if r["platform"] == q["platform"])
                        for q in rows], float)
    proxy = spearman_ci(proxy_c, adv, RNG)
    family7 = ra["multiplicity"]["family7"]
    n_unc, pw_unc = n_for_mc_power(0.724, 0.05)
    n_f7, pw_f7 = n_for_mc_power(0.724, 0.05 / 7)
    n_pr, pw_pr = n_for_mc_power(0.551, 0.05)
    # LODO binomial power at observed accuracy 0.818 vs base 0.545
    lodo_n = None
    for n in range(11, 81):
        k = int(np.ceil(0.818 * n))
        # power ≈ P(Binom(n, 0.818) yields p < 0.05 vs p0=0.545)
        # simulate
        ok = 0
        for _ in range(2000):
            kk = RNG.binomial(n, 0.818)
            if binomtest(kk, n, 0.545, alternative="greater").pvalue < 0.05:
                ok += 1
        if ok / 2000 >= 0.80:
            lodo_n = n
            break
    return {
        "n11": {"rho": float(rho_all), "p": float(p_all),
                "bootstrap_ci95": sr["bootstrap_gt_contiguity"]["ci95"],
                "frac_positive": sr["bootstrap_gt_contiguity"]["frac_positive"]},
        "loo": loo,
        "loo_range": [min(x["loo_rho"] for x in loo), max(x["loo_rho"] for x in loo)],
        "most_influential": loo[0]["dropped"],
        "family7": {
            "role": "pre-specified multiplicity family of 7 predictor-advantage tests",
            "raw_p": family7["contiguity_corrected"]["raw"],
            "bonferroni_p": family7["contiguity_corrected"]["bonferroni"],
            "bh_p": family7["contiguity_corrected"]["bh"],
            "effect_rho": 0.724,
            "effect_ci95": sr["bootstrap_gt_contiguity"]["ci95"],
            "significant_at_0.05": False,
            "note": "Bonferroni p=0.082 is not significant; do not call it a discovery",
        },
        "proxy": {
            "role": "pre-specified secondary: selected coh_gain after 6-candidate search",
            "rho": 0.551,
            "bootstrap_ci95": sr["bootstrap_gtfree_cohgain"]["ci95"],
            "frac_positive": sr["bootstrap_gtfree_cohgain"]["frac_positive"],
            "asymptotic_uncorrected_p": 0.0788,
            "permutation_uncorrected_p": psc["p_uncorrected_selected"],
            "westfall_young_search_corrected_p": psc["p_search_corrected"],
            "candidates": psc["candidates"],
            "significant_at_0.05": False,
            "this_run_bootstrap": proxy,
        },
        "lodo": ra["lopo_binomial"],
        "n14": {"rho": ex["spearman_expanded"], "p": ex["p_expanded"], "n_new": ex["n_new"]},
        "power": {
            "observed_rho": 0.724,
            "fisher_z_n80_uncorrected_alpha0.05": fisher_n_for_power(0.724, 0.05, 0.80),
            "fisher_z_n80_family7_alpha": fisher_n_for_power(0.724, 0.05 / 7, 0.80),
            "fisher_z_n80_proxy_uncorrected": fisher_n_for_power(0.551, 0.05, 0.80),
            "mc_n80_uncorrected": {"n": n_unc, "achieved_power": pw_unc, "n_sim": MC_POWER},
            "mc_n80_family7_bonferroni": {"n": n_f7, "achieved_power": pw_f7, "n_sim": MC_POWER},
            "mc_n80_proxy_uncorrected": {"n": n_pr, "achieved_power": pw_pr, "n_sim": MC_POWER},
            "lodo_binomial_n80_one_sided": lodo_n,
            "method": "Fisher-z analytic + Gaussian-copula Monte Carlo Spearman; LODO binomial sim",
        },
        "do_not_pool": {
            "observational_vs_synthetic": (
                "Different estimands (cross-platform association vs within-generator intervention). "
                "Do not inverse-variance meta-analyze +0.72 with +0.98."
            ),
            "family7_vs_proxy": (
                "Same n=11 platforms and the same advantage vector; p-values are dependent. "
                "Fisher/Stouffer combination is not valid without a joint null."
            ),
            "n11_vs_n14": (
                "n=14 nests the original 11; not an independent replication. "
                "The honest combined observational read is the n=14 Spearman already reported."
            ),
        },
    }


def summarize_synth(name, rows, rng):
    per = per_seed_spearman(rows)
    means = condition_means(rows)
    cond = spearman_ci([m["contiguity"] for m in means], [m["adv_smooth"] for m in means], rng)
    pooled = spearman_ci([r["contiguity"] for r in rows], [r["adv_smooth"] for r in rows], rng)
    rho_m = mean_ci([p["rho"] for p in per], rng)
    dlt = mean_ci([p["delta_adv_s0_minus_s1"] for p in per], rng)
    conv = []
    seeds = sorted({r["seed"] for r in rows})
    for n in [3, 6, 12, 24, 36, 48]:
        if n > len(seeds):
            continue
        use = set(seeds[:n])
        sub = [r for r in rows if r["seed"] in use]
        pm = per_seed_spearman(sub)
        conv.append({"n_seeds": n,
                     "mean_per_seed_rho": mean_ci([p["rho"] for p in pm], rng),
                     "mean_delta_adv": mean_ci([p["delta_adv_s0_minus_s1"] for p in pm], rng),
                     "sign_flip_rate": float(np.mean([p["sign_flip"] for p in pm])),
                     "condition_mean_spearman": float(spearmanr(
                         [m["contiguity"] for m in condition_means(sub)],
                         [m["adv_smooth"] for m in condition_means(sub)]).correlation)})
    return {
        "config": name, "n_seeds": len(seeds), "n_conditions": len(SCRAMBLE),
        "n_evaluations": len(rows),
        "condition_mean_spearman": cond,
        "pooled_point_spearman": pooled,
        "per_seed_rho": rho_m,
        "causal_delta_adv_s0_minus_s1": dlt,
        "sign_flip_rate": float(np.mean([p["sign_flip"] for p in per])),
        "mean_nonspatial_ari_range": float(np.mean([p["nonsp_range"] for p in per])),
        "condition_means": means,
        "convergence": conv,
        "per_seed": per,
    }


def main():
    t0 = datetime.now(timezone.utc).isoformat()
    paper_rows = run_grid("paper", PAPER_CFG, PAPER_SEEDS)
    extra = {}
    extra_rows = []
    for name, cfg, seeds in EXTRA:
        rr = run_grid(name, cfg, seeds)
        extra[name] = summarize_synth(name, rr, RNG)
        extra_rows.extend(rr)
    paper = summarize_synth("paper", paper_rows, RNG)
    all_rows = paper_rows + extra_rows
    all_sum = summarize_synth("all_configs", all_rows, RNG)
    (OUTDIR / "synth_rows.json").write_text(json.dumps(all_rows))
    (OUTDIR / "synth_summaries.json").write_text(json.dumps({
        "paper": paper, "extra": extra, "all_configs": all_sum}, indent=2))
    print("wrote intermediate synth JSON", flush=True)
    obs = observational()
    versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "sklearn": __import__("sklearn").__version__,
                "SOURCE_DATE_EPOCH": "0", "seed_master": 20260918,
                "B_bootstrap": B_BOOT, "MC_POWER": MC_POWER}
    out = {
        "role": "science-hardening 2026-09-18; neighbour-mean generator scale-up + observational diagnostics",
        "started_utc": t0,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "original_paper_synth": {"n_seeds": 3, "n_evaluations": 24,
                                 "spearman_condition_means": 0.976},
        "paper_config_scaled": paper,
        "extra_generator_configs": extra,
        "all_configs": all_sum,
        "counts": {
            "original_evaluations": 24,
            "new_paper_seed_evaluations": len(paper_rows) - 24,
            "extra_config_evaluations": len(extra_rows),
            "total_evaluations_this_run": len(all_rows),
            "n_paper_seeds": 48,
            "n_extra_tissues": sum(len(v[2]) for v in EXTRA),
        },
        "observational": obs,
        "real_data_ceiling": {
            "published_n": 11,
            "already_scored_extra": ["st_OV", "st_COAD", "st_LIHC"],
            "local_present_but_not_added": [
                "st_CESC/NSCLC/PRAD: technical segmentation_method only; region is a single dummy",
                "wu_breast_atlas / sc_mouse_cortex: scRNA-seq, no spatial coordinates",
                "Zhuang-ABCA-1: 4.1e6 cells, section label only, no spatial obsm",
                "bento merfish processed: n=15, no labels",
                "deepstarmap_mouse_brain: method-derived Harmony labels, not independent GT",
            ],
            "added_this_pass": 0,
            "ceiling": "n=11 published; n=14 already includes the only local independent ctype GTs",
        },
    }
    path = OUTDIR / "science_core_20260918.json"
    path.write_text(json.dumps(out, indent=2))
    print("\nWROTE", path)
    print("paper mean per-seed rho", paper["per_seed_rho"])
    print("paper delta adv", paper["causal_delta_adv_s0_minus_s1"])
    print("paper condition-mean spearman", paper["condition_mean_spearman"])
    print("power", obs["power"])
    print("LOO most influential", obs["most_influential"], "range", obs["loo_range"])


if __name__ == "__main__":
    main()
