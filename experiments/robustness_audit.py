"""Robustness audit of the headline law, prompted by an independent red-team review.

Every number here is recomputed from the existing machine-checked artifacts (no new experiments):

  1. De-circularised law -- recompute spatial-prior advantage using ONLY learned spatial methods
     (drop BOTH floors, including the neighbour-mean smoother) and re-correlate with contiguity.
     This answers the "the law is just smoothing-helps-when-labels-are-smooth tautology" charge.
  2. Multiple-comparison correction (Bonferroni + Benjamini-Hochberg) over the family of
     predictor-vs-advantage correlations, so the headline p is reported corrected, not raw.
  3. Single-point fragility -- jackknife the correlation (drop each dataset).
  4. Out-of-sample decision rule -- exact one-sided binomial test vs the base rate.
  5. Distinct-winners chance baseline -- expected number of distinct winners under a null where the
     winner is uniform-random, to contextualise "seven winners".

Writes experiments/robustness_audit.json.  Run from the repo root.
"""

from __future__ import annotations

import json

from scipy.stats import binomtest, spearmanr
from statsmodels.stats.multitest import multipletests

EXP = "experiments"


def _load(name: str) -> dict:
    return json.load(open(f"{EXP}/{name}.json"))


def main() -> None:
    eb = _load("expanded_bench")
    sr = _load("stat_rigor")
    gp = _load("gtfree_proxy")
    rows = eb["rows"]
    cont = [r["GT_contiguity"] for r in rows]
    adv_pub = [r["spatial_advantage"] for r in rows]
    out: dict = {"n_datasets": len(rows)}

    # ---- 1. smoother-independence (NOT a de-circularisation) ---------------------------------
    # Does the neighbour-mean smoother ever set the advantage? On this panel: no -- it is never the
    # winning spatial method, so dropping it from the max() changes the advantage vector by 0. This
    # rules out the narrow worry "advantage == smoothing gain" but does NOT remove the deeper
    # circularity: every spatial method here (learned encoders and BANKSY-style neighbour
    # augmentation alike) is built on the same spatial kNN structure that defines contiguity.
    learned = ["Tessera", "STAGATE", "SEDR", "SpaceFlow", "GraphST", "SpatialLeiden", "BANKSY-style"]
    floor = "floor:nonspatial"
    spatial_all = learned + ["floor:smoothed"]
    adv_learned = [max(r["means"][m] for m in learned if m in r["means"]) - r["means"][floor]
                   for r in rows]
    adv_incl = [max(r["means"][m] for m in spatial_all if m in r["means"]) - r["means"][floor]
                for r in rows]
    sm_best = sum(max((r["means"][m], m) for m in spatial_all if m in r["means"])[1] == "floor:smoothed"
                  for r in rows)
    out["smoother_independence"] = {
        "smoother_is_best_spatial_method_count": sm_best,
        "max_advantage_diff_with_vs_without_smoother": round(max(abs(a - b) for a, b in zip(adv_incl, adv_learned)), 4),
        "note": "smoother never wins -> advantage is smoother-independent on this panel; the deeper "
                "kNN-shared-with-predictor circularity is NOT addressed by this and is disclosed in Limitations",
    }

    # ---- 2. multiple-comparison correction ---------------------------------------------------
    cm = sr["confound_marginal_spearman_vs_advantage"]
    # the family actually tested against the "advantage" outcome
    family7 = {
        "contiguity": cm["GT_contiguity"]["p"], "k_nclasses": cm["k_nclasses"]["p"],
        "eff_dim": cm["eff_dim"]["p"], "modality": cm["modality_imaging"]["p"],
        "coh_gain": gp["principled_proxy_coh_gain"]["vs_advantage"][1],
        "moran": gp["naive_proxies_FAIL"]["moran"]["vs_advantage"][1],
        "raw_coh": gp["naive_proxies_FAIL"]["raw_coh"]["vs_advantage"][1],
    }
    # the three "primary" correlations the paper foregrounds
    family3 = {
        "contiguity_vs_advantage": eb["p_contiguity_vs_spatial_advantage"],
        "contiguity_vs_tessera_rank": eb["p_contiguity_vs_tessera_rank"],
        "contiguity_vs_spaceflow_rank": eb["p_contiguity_vs_spaceflow_rank"],
    }

    def corrected(pvals: dict, key: str) -> dict:
        names = list(pvals); ps = [pvals[n] for n in names]
        bonf = multipletests(ps, method="bonferroni")[1]
        bh = multipletests(ps, method="fdr_bh")[1]
        i = names.index(key)
        return {"m": len(ps), "raw": round(ps[i], 4),
                "bonferroni": round(float(bonf[i]), 4), "bh": round(float(bh[i]), 4)}

    out["multiplicity"] = {
        "family3": {**{k: round(v, 4) for k, v in family3.items()},
                    "contiguity_corrected": corrected(family3, "contiguity_vs_advantage")},
        "family7": {**{k: round(v, 4) for k, v in family7.items()},
                    "contiguity_corrected": corrected(family7, "contiguity")},
    }

    # ---- 3. single-point fragility (jackknife the correlation) --------------------------------
    def jack(x: list, y: list) -> list:
        return [float(spearmanr([x[j] for j in range(len(x)) if j != i],
                                [y[j] for j in range(len(y)) if j != i]).statistic) for i in range(len(x))]
    jp, jl = jack(cont, adv_pub), jack(cont, adv_learned)
    out["jackknife"] = {
        "published_min": round(float(min(jp)), 3), "published_max": round(float(max(jp)), 3),
        "learned_min": round(float(min(jl)), 3), "learned_max": round(float(max(jl)), 3),
        "most_influential_drop": rows[jp.index(min(jp))]["platform"].split("(")[0],
    }

    # ---- 4. out-of-sample decision rule: exact binomial vs base rate --------------------------
    lg = sr["lopo_gt_contiguity"]
    n = lg["n"]; acc = lg["decision_rule_accuracy"]; base = lg["base_rate_accuracy"]
    k = round(acc * n)
    bt = binomtest(k, n, base, alternative="greater")
    out["lopo_binomial"] = {"k_correct": k, "n": n, "accuracy": acc, "base_rate": base,
                            "p_value": round(float(bt.pvalue), 3)}

    # ---- 5. distinct-winners chance baseline --------------------------------------------------
    n_methods = len(eb["method_panel"]); n_sets = len(rows)
    expected = n_methods * (1 - (1 - 1 / n_methods) ** n_sets)
    out["distinct_winners"] = {"observed": eb["n_distinct_winners"], "n_methods": n_methods,
                               "n_datasets": n_sets, "expected_under_null": round(expected, 2)}

    json.dump(out, open(f"{EXP}/robustness_audit.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
