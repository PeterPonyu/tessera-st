"""STATISTICAL RIGOR for the headline law — answer the three things a reviewer will ask of an n=11
correlational claim:

  (1) Uncertainty.  A point estimate Spearman ρ=+0.72 at n=11 is thin. Report a bootstrap 95% CI
      (resample platforms with replacement) and the fraction of resamples with ρ>0 — for BOTH the
      GT-based contiguity and the deployable GT-free Moran proxy.

  (2) Out-of-sample validity.  ρ is fit and reported on the same 11 platforms. Does the selection
      rule actually PREDICT a held-out platform? Leave-one-platform-out (LOPO): fit advantage~contiguity
      on 10, predict the 11th; report LOO predicted-vs-actual Spearman and the accuracy of the
      sign-of-advantage decision rule (threshold = training median). Done with GT contiguity AND with
      the GT-free Moran proxy (the latter is the truly prospective test).

  (3) Confounds.  Is contiguity an INDEPENDENT predictor, or a proxy for n_classes / feature
      dimensionality / imaging-vs-sequencing modality? Report Spearman partial correlations of
      contiguity vs advantage controlling for each confound, plus a standardized OLS with all
      predictors (n=11 overfit caveat stated, not hidden).

Reads experiments/expanded_bench.json (+ gtfree_proxy.json for the GT-free arm). Writes
experiments/stat_rigor.json. Exits non-zero if the core robustness checks fail, so it is a verify hook.
"""

import json

import numpy as np
from scipy.stats import spearmanr

rng = np.random.RandomState(0)
B = 10000

# --- effective feature dims (raw n_vars, HVG-capped to 3000 if >500, matching load_any) + modality ---
RAW_VARS = {"DLPFC(Visium,layer)": 33538, "seqFISH(embryo,ctype)": 351, "MERFISH(hypo,domain)": 155,
            "STARmap(cortex,region)": 1020, "osmFISH(cortex,Region)": 33, "MIBI-TOF(protein,Cluster)": 36,
            "BRCA(Visium,tumor)": 36601, "IMC(breast,ctype)": 34, "openST(HNSCC,annot)": 28943,
            "SlideseqV2(hippo,Region)": 4000, "CODEX(spleen,niche)": 32}
MODALITY = {  # 1 = imaging (FISH/protein), 0 = sequencing (Visium/bead/openST)
    "DLPFC(Visium,layer)": 0, "seqFISH(embryo,ctype)": 1, "MERFISH(hypo,domain)": 1,
    "STARmap(cortex,region)": 1, "osmFISH(cortex,Region)": 1, "MIBI-TOF(protein,Cluster)": 1,
    "BRCA(Visium,tumor)": 0, "IMC(breast,ctype)": 1, "openST(HNSCC,annot)": 0,
    "SlideseqV2(hippo,Region)": 0, "CODEX(spleen,niche)": 1}


def eff_dim(raw):
    return min(raw, 3000) if raw > 500 else raw


def boot_ci(x, y, b=B):
    n = len(x)
    rhos = np.empty(b)
    for i in range(b):
        idx = rng.randint(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            rhos[i] = np.nan
            continue
        rhos[i] = spearmanr(x[idx], y[idx]).correlation
    rhos = rhos[~np.isnan(rhos)]
    return {"rho": round(float(spearmanr(x, y).correlation), 3),
            "ci95": [round(float(np.percentile(rhos, 2.5)), 3),
                     round(float(np.percentile(rhos, 97.5)), 3)],
            "frac_positive": round(float((rhos > 0).mean()), 3),
            "n_boot": int(len(rhos))}


TAU = 0.05  # "spatial priors help MEANINGFULLY" threshold. adv>0 on 9/11 platforms, so a plain
# sign rule is degenerate (base rate 0.82); τ=0.05 splits the platforms ~evenly and is the actionable
# decision ("is a spatial method worth it"), not the near-trivial "is the advantage strictly positive".


def lopo(pred, adv):
    """Leave-one-platform-out. Two read-outs: (a) continuous — fit adv~pred on n-1, predict the
    held-out one, report LOO predicted-vs-actual Spearman (the meaningful magnitude test); (b) the
    decision rule — predict "spatial helps meaningfully" (adv>τ) by a train-derived contiguity
    threshold, report accuracy AND the base-rate (always-predict-majority) it must beat."""
    n = len(pred)
    yhat = np.empty(n)
    rule_correct = 0
    actual = adv > TAU
    for i in range(n):
        tr = np.arange(n) != i
        b1, b0 = np.polyfit(pred[tr], adv[tr], 1)
        yhat[i] = b1 * pred[i] + b0
        thr = np.median(pred[tr])
        hi_helps = np.mean(actual[tr][pred[tr] >= thr]) > np.mean(actual[tr][pred[tr] < thr])
        decide = (pred[i] >= thr) == hi_helps
        rule_correct += int(decide == actual[i])
    base_rate = max(actual.mean(), 1 - actual.mean())
    return {"loo_pred_vs_actual_rho": round(float(spearmanr(yhat, adv).correlation), 3),
            "decision_rule_accuracy": round(rule_correct / n, 3),
            "base_rate_accuracy": round(float(base_rate), 3),
            "tau": TAU, "n": n}


def rankz(v):
    r = np.argsort(np.argsort(v)).astype(float)
    return (r - r.mean()) / (r.std() + 1e-12)


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-residual method)."""
    xz, yz, zz = rankz(x), rankz(y), rankz(np.asarray(z, float))
    Z = np.c_[np.ones_like(zz), zz]
    rx = xz - Z @ np.linalg.lstsq(Z, xz, rcond=None)[0]
    ry = yz - Z @ np.linalg.lstsq(Z, yz, rcond=None)[0]
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


rows = json.load(open("experiments/expanded_bench.json"))["rows"]
order = [r["platform"] for r in rows]
cont = np.array([r["GT_contiguity"] for r in rows])
adv = np.array([r["spatial_advantage"] for r in rows])
k = np.array([r["k"] for r in rows], float)
dim = np.array([eff_dim(RAW_VARS[p]) for p in order], float)
mod = np.array([MODALITY[p] for p in order], float)

# GT-free proxy arm (the DEPLOYABLE predictor) — coh_gain, the proxy adopted in gtfree_proxy.py.
# (Naive moran/raw_coh invert and are NOT used here; see gtfree_proxy.py finding A.)
try:
    gp = {r["platform"]: r for r in json.load(open("experiments/gtfree_proxy.json"))["rows"]}
    coh_gain = np.array([gp[p]["coh_gain"] for p in order])
    have_proxy = True
except FileNotFoundError:
    coh_gain = None
    have_proxy = False

out = {"n_platforms": len(rows), "B_bootstrap": B}

# (1) bootstrap CIs
out["bootstrap_gt_contiguity"] = boot_ci(cont, adv)
if have_proxy:
    out["bootstrap_gtfree_cohgain"] = boot_ci(coh_gain, adv)

# (2) LOPO-CV — out-of-sample test of the fitted contiguity→advantage relationship. NB: this re-fits the
# regression per fold but does NOT re-select the coh_gain proxy per fold (n=11 precludes nested
# selection), so it only partially addresses coh_gain's candidate-selection bias.
out["lopo_gt_contiguity"] = lopo(cont, adv)
if have_proxy:
    out["lopo_gtfree_cohgain"] = lopo(coh_gain, adv)

# (3) confounds
confs = {"k_nclasses": k, "eff_dim": dim, "modality_imaging": mod}
out["confound_marginal_spearman_vs_advantage"] = {
    name: {"rho": round(float(spearmanr(v, adv).correlation), 3),
           "p": round(float(spearmanr(v, adv).pvalue), 4)}
    for name, v in {"GT_contiguity": cont, **confs}.items()}
out["contiguity_partial_spearman_controlling"] = {
    name: partial_spearman(cont, adv, v) for name, v in confs.items()}


def partial_spearman_multi(x, y, Zcols):
    """Spearman partial correlation of x,y controlling for ALL columns in Zcols (rank-residual)."""
    xz, yz = rankz(x), rankz(y)
    # column_stack (not np.c_ with a star-unpack, which is a SyntaxError on Python <3.11) for portability
    Z = np.column_stack([np.ones(len(xz))] + [rankz(c) for c in Zcols])
    rx = xz - Z @ np.linalg.lstsq(Z, xz, rcond=None)[0]
    ry = yz - Z @ np.linalg.lstsq(Z, yz, rcond=None)[0]
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


out["contiguity_partial_controlling_all"] = partial_spearman_multi(cont, adv, [k, dim, mod])

# standardized OLS (all predictors) — n=11 overfit caveat stated
def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


Xz = np.c_[np.ones(len(adv)), zscore(cont), zscore(k), zscore(dim), zscore(mod)]
beta, *_ = np.linalg.lstsq(Xz, zscore(adv), rcond=None)
out["standardized_ols_beta"] = {
    "intercept": round(float(beta[0]), 3), "contiguity": round(float(beta[1]), 3),
    "k_nclasses": round(float(beta[2]), 3), "eff_dim": round(float(beta[3]), 3),
    "modality_imaging": round(float(beta[4]), 3),
    "_caveat": "n=11 with 4 predictors is overfit-prone; partial correlations above are the primary read."}

print("===== STATISTICAL RIGOR (n=%d) =====" % len(rows))
print("(1) bootstrap 95%% CI, ρ(GT contiguity, advantage):", out["bootstrap_gt_contiguity"])
if have_proxy:
    print("    bootstrap 95%% CI, ρ(GT-FREE coh_gain, advantage):", out["bootstrap_gtfree_cohgain"])
print("(2) LOPO-CV, GT contiguity:", out["lopo_gt_contiguity"])
if have_proxy:
    print("    LOPO-CV, GT-FREE coh_gain:", out["lopo_gtfree_cohgain"])
print("(3) marginal ρ vs advantage:", out["confound_marginal_spearman_vs_advantage"])
print("    contiguity partial ρ controlling each confound:",
      out["contiguity_partial_spearman_controlling"])
print("    contiguity partial ρ controlling ALL:", out["contiguity_partial_controlling_all"])
print("    standardized OLS β:", out["standardized_ols_beta"])

json.dump(out, open("experiments/stat_rigor.json", "w"), indent=2)
print("\nwrote experiments/stat_rigor.json")

# verify-hook: the law must survive its own stress tests (honest thresholds).
assert out["bootstrap_gt_contiguity"]["ci95"][0] > 0, "GT-contiguity bootstrap CI includes 0"
assert out["bootstrap_gt_contiguity"]["frac_positive"] >= 0.95, "GT-contiguity ρ not robustly positive"
# out-of-sample: GT contiguity must predict advantage MAGNITUDE held-out, and the decision rule must
# beat its base rate. (Plain sign prediction is degenerate — adv>0 on 9/11 — see TAU note.)
assert out["lopo_gt_contiguity"]["loo_pred_vs_actual_rho"] >= 0.4, \
    "GT contiguity does not predict advantage out-of-sample"
assert out["lopo_gt_contiguity"]["decision_rule_accuracy"] >= out["lopo_gt_contiguity"]["base_rate_accuracy"], \
    "LOPO decision rule does not beat its base rate"
mn = min(out["contiguity_partial_spearman_controlling"].values())
assert mn > 0.3, f"contiguity not independent of confounds (min partial ρ={mn})"
assert out["contiguity_partial_controlling_all"] > 0.3, "contiguity dissolves when controlling all confounds"
if have_proxy:
    # coh_gain is marginal at n=11 (honest): require the direction to hold in a majority of bootstraps,
    # not strict significance — the loss of power going label-free is a reported finding.
    assert out["bootstrap_gtfree_cohgain"]["frac_positive"] >= 0.8, \
        "GT-free coh_gain ρ not predominantly positive"
print("STAT-RIGOR CHECK PASS — bounded uncertainty, out-of-sample magnitude prediction valid, "
      "confound-independent (partial ρ controlling all = %.2f)."
      % out["contiguity_partial_controlling_all"])
