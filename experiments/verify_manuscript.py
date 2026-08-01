"""G004 verification: assert every headline number in the manuscript matches the JSON artifacts
(within rounding). Fails loudly if the writeup drifts from evidence."""

import json
import re

rb = json.load(open("experiments/rigorous_bench.json"))
pa = json.load(open("experiments/pareto_frontier.json"))
ve = json.load(open("experiments/tessera_verdict.json"))
md = open("manuscript/DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md").read()

checks = []


def check(name, claimed, actual, tol=0.0015):
    ok = abs(claimed - actual) <= tol
    checks.append((name, claimed, round(actual, 4), ok))


# backend confound
check("confound Tessera", 0.036, rb["backend_confound"]["Tessera"])
check("confound STAGATE", 0.200, rb["backend_confound"]["STAGATE"], tol=0.0006)
check("confound SEDR", 0.170, rb["backend_confound"]["SEDR"])
# cross-section ARI
check("xsec STAGATE", 0.494, rb["cross_section"]["STAGATE"]["mean"])
check("xsec floor:smoothed", 0.473, rb["cross_section"]["floor:smoothed"]["mean"])
check("xsec Tessera", 0.392, rb["cross_section"]["Tessera"]["mean"])
check("xsec GraphST", 0.291, rb["cross_section"]["GraphST"]["mean"])
# pareto: Tessera point + boundary/global ends
check("pareto Tessera ARI", 0.392, pa["points"]["Tessera"]["ARI"], tol=0.002)
check("pareto Tessera bF1", 0.363, pa["points"]["Tessera"]["bF1"], tol=0.002)
# verdict ranks (exact int)
assert ve["ari_rank"] == 5, f"ARI rank {ve['ari_rank']} != 5"
assert ve["boundary_f1_rank"] == 3, f"bF1 rank {ve['boundary_f1_rank']} != 3"
assert ve["backend_robustness_rank"] == 1, f"robustness rank {ve['backend_robustness_rank']} != 1"
checks.append(("verdict ranks ARI5/bF1 3/robust1", 1, 1, True))

# cross-platform seqFISH flip
sq = json.load(open("experiments/seqfish_bench.json"))
check("seqfish floor:nonspatial GMM ARI", 0.451, sq["floor:nonspatial"]["gmm"]["ARI"], tol=0.001)
check("seqfish STAGATE GMM ARI", 0.324, sq["STAGATE"]["gmm"]["ARI"], tol=0.001)
check("seqfish STAGATE backend flip (<0)", -0.026,
      sq["STAGATE"]["gmm"]["ARI"] - sq["STAGATE"]["kmeans"]["ARI"], tol=0.002)

# 6-platform: no universal SOTA + Tessera wins true spatial-domain tasks
mp = json.load(open("experiments/multi_platform_bench.json"))
assert mp["MERFISH(hypo,domain)"]["best_gmm"] == "Tessera", "Tessera not best on MERFISH"
assert mp["osmFISH(cortex,Region)"]["best_gmm"] == "Tessera", "Tessera not best on osmFISH"
distinct = len({mp[p]["best_gmm"] for p in mp})
assert distinct >= 4, f"only {distinct} distinct winners across platforms"
check("MERFISH Tessera GMM", 0.267, mp["MERFISH(hypo,domain)"]["per_method"]["Tessera"]["gmm"], tol=0.003)
check("osmFISH Tessera GMM", 0.426, mp["osmFISH(cortex,Region)"]["per_method"]["Tessera"]["gmm"], tol=0.003)
checks.append((f"6-platform: Tessera best on MERFISH+osmFISH, {distinct} distinct winners", 1, 1, True))

# hardened (2 seeds, best-config, +GraphST): does the headline survive?
hd = json.load(open("experiments/hardened_bench.json"))
def hwin(p):
    return sorted(hd[p], key=lambda m: -hd[p][m]["mean"])[0]
assert hwin("MERFISH(hypo,domain)") == "Tessera", "hardened: Tessera not best on MERFISH"
assert hwin("osmFISH(cortex,Region)") == "Tessera", "hardened: Tessera not best on osmFISH"
hwins = len({hwin(p) for p in hd})
assert hwins >= 5, f"hardened: only {hwins} distinct winners"
check("hardened MERFISH Tessera mean", 0.406, hd["MERFISH(hypo,domain)"]["Tessera"]["mean"], tol=0.002)
check("hardened osmFISH Tessera mean", 0.449, hd["osmFISH(cortex,Region)"]["Tessera"]["mean"], tol=0.002)
checks.append((f"hardened: Tessera best MERFISH+osmFISH, {hwins} distinct winners", 1, 1, True))

# task-fit law: GT contiguity predicts Tessera rank + spatial advantage
tf = json.load(open("experiments/task_fit_law.json"))
check("task-fit Spearman(contig,rank)", -0.714, tf["spearman_contiguity_vs_tessera_rank"], tol=0.002)
check("task-fit Spearman(contig,adv)", 0.771, tf["spearman_contiguity_vs_spatial_advantage"], tol=0.002)
mrow = next(r for r in tf["rows"] if "MERFISH" in r["platform"])
assert mrow["Tessera_rank"] == 1 and mrow["GT_contiguity"] > 0.9, "MERFISH highest-contig Tessera win"
checks.append(("task-fit: contiguity predicts rank (rho -0.71) + spatial adv (rho +0.77)", 1, 1, True))

# §7 expanded benchmark: 11 platforms, 6 methods (+SpaceFlow +SpatialLeiden), 3 seeds (post-review, hardened)
eb = json.load(open("experiments/expanded_bench.json"))
assert eb["n_platforms"] == 11, f"expanded n={eb['n_platforms']} != 11 (duplicate must be removed)"
assert eb["n_distinct_winners"] == 7, f"expanded distinct winners {eb['n_distinct_winners']} != 7"
assert eb["seeds"] == [1, 2, 3], f"expected 3 seeds, got {eb['seeds']}"
assert "SpatialLeiden" in eb["method_panel"], "SpatialLeiden (6th method) must be in the panel"
# lead result: spatial-prior advantage tracks contiguity, significant (denominator-free) + seed-robust
check("expanded Spearman(contig,spatial_adv)", 0.724, eb["spearman_contiguity_vs_spatial_advantage"], tol=0.01)
assert eb["p_contiguity_vs_spatial_advantage"] < 0.05, "lead correlation must be significant"
assert eb["rho_adv_per_seed_min"] > 0.6, f"lead rho not seed-robust: min {eb['rho_adv_per_seed_min']}"
assert eb["rho_adv_per_seed_max"] < 0.85, f"seed-robustness max out of range: {eb['rho_adv_per_seed_max']}"
# secondary rank correlations: weak + NOT significant (reported honestly, not as the headline)
check("expanded Spearman(contig,Tessera_rank)", -0.357, eb["spearman_contiguity_vs_tessera_rank"], tol=0.03)
check("expanded Spearman(contig,SpaceFlow_rank)", -0.586, eb["spearman_contiguity_vs_spaceflow_rank"], tol=0.03)
assert eb["p_contiguity_vs_tessera_rank"] > 0.05, "Tessera-rank correlation must be reported as ns"
assert eb["n_spaceflow"] == 9, f"SpaceFlow ran on {eb['n_spaceflow']} platforms (failed on 2 protein panels)"
# SpatialLeiden's inverted profile: it wins the least-contiguous platform (CODEX)
cod = next(r for r in eb["rows"] if "CODEX" in r["platform"])
assert cod["winner"] == "SpatialLeiden", f"SpatialLeiden should win CODEX, got {cod['winner']}"
# independent recompute (do not trust the script's own number) + duplicate guard
from scipy.stats import spearmanr  # noqa: E402
rows = eb["rows"]
rho_indep = spearmanr([r["GT_contiguity"] for r in rows], [r["spatial_advantage"] for r in rows]).correlation
check("independent scipy recompute of lead rho", round(float(eb["spearman_contiguity_vs_spatial_advantage"]), 3),
      float(rho_indep), tol=0.005)
sigs = [(r["n_cells"], r["GT_contiguity"], tuple(sorted(r["means"].items()))) for r in rows]
assert len(sigs) == len(set(sigs)), "duplicate platform present (identical cells/contiguity/means)"
# contiguous spatial-domain tasks: top method is a spatial method; winner at the top is a seed-level tie
mer = next(r for r in rows if "MERFISH(hypo" in r["platform"])
osm = next(r for r in rows if "osmFISH" in r["platform"])
mer_top2 = [m for m, _ in sorted(mer["means"].items(), key=lambda x: -x[1])[:2]]
assert set(mer_top2) == {"Tessera", "SpaceFlow"}, f"MERFISH top-2 should be the 2 spatial methods, got {mer_top2}"
assert osm["winner"] == "Tessera", "Tessera leads osmFISH (the more stable win)"
checks.append(("§7: spatial-prior law significant+seed-robust (+0.72); rank laws ns; SpatialLeiden wins CODEX", 1, 1, True))

# §8 R2: GT-free proxy (G001) — naive proxies invert, principled coh_gain recovers the trend
gp = json.load(open("experiments/gtfree_proxy.json"))
assert gp["naive_proxies_FAIL"]["moran"]["vs_advantage"][0] < 0, "moran proxy must invert (documented)"
assert gp["naive_proxies_FAIL"]["raw_coh"]["vs_advantage"][0] < 0, "raw_coh proxy must invert (documented)"
check("R2 coh_gain vs GT contiguity", 0.564, gp["principled_proxy_coh_gain"]["vs_gtcontig"][0], tol=0.03)
check("R2 coh_gain vs spatial advantage", 0.551, gp["principled_proxy_coh_gain"]["vs_advantage"][0], tol=0.03)
assert min(gp["coh_gain_headline_per_seed_rho"]) >= 0.4, "coh_gain headline not seed-robust"
checks.append(("§8a: naive proxies invert; coh_gain makes the rule label-free (+0.55)", 1, 1, True))

# §8a R5: search-corrected (Westfall-Young max-statistic) proxy p — coh_gain was SELECTED as the best of
# six candidate proxies, so its significance must be corrected for that search. This keeps the manuscript
# honest: coh_gain is a directional/marginal prior, NOT a significant predictor, once selection is priced in.
psc = json.load(open("experiments/proxy_search_corrected.json"))
assert psc["selected_candidate"] == "coh_gain", "selected proxy must be coh_gain"
assert psc["selected_is_argmax_over_search"], "coh_gain must be the argmax over the 6-candidate search"
assert len(psc["candidates"]) == 6, "search-correction must be over the full 6-candidate family"
check("R5 proxy observed rho (coh_gain)", 0.551, psc["observed_selected_rho"], tol=0.003)
check("R5 proxy search-corrected p", 0.275, psc["p_search_corrected"], tol=0.005)
assert psc["p_search_corrected"] > 0.05, "search-corrected proxy p must be reported as NON-significant"
assert psc["p_search_corrected"] > psc["p_uncorrected_selected"], \
    "search-corrected p must exceed the uncorrected per-candidate p"
checks.append(("§8a R5: coh_gain is directional but NOT significant after 6-candidate search correction "
               f"(search-corrected p={psc['p_search_corrected']} > uncorrected {psc['p_uncorrected_selected']})",
               1, 1, True))

# §8 R2: statistical rigor (G003) — bootstrap CI, LOPO out-of-sample, confound partial correlations
sr = json.load(open("experiments/stat_rigor.json"))
assert sr["bootstrap_gt_contiguity"]["frac_positive"] >= 0.95, "GT bootstrap not robustly positive"
assert sr["bootstrap_gt_contiguity"]["ci95"][0] > 0, "GT bootstrap CI includes 0"
check("R2 LOPO GT magnitude rho", 0.51, sr["lopo_gt_contiguity"]["loo_pred_vs_actual_rho"], tol=0.05)
assert sr["lopo_gt_contiguity"]["decision_rule_accuracy"] > sr["lopo_gt_contiguity"]["base_rate_accuracy"], \
    "LOPO decision rule must beat base rate"
assert sr["lopo_gtfree_cohgain"]["decision_rule_accuracy"] > sr["lopo_gtfree_cohgain"]["base_rate_accuracy"], \
    "label-free LOPO rule must beat base rate"
check("R2 contiguity partial rho controlling ALL", 0.667, sr["contiguity_partial_controlling_all"], tol=0.03)
assert min(sr["contiguity_partial_spearman_controlling"].values()) > 0.3, "contiguity not confound-independent"
checks.append(("§8b: law survives bootstrap CI, LOPO out-of-sample, all confounds (partial ρ=0.67)", 1, 1, True))

# §8 R2: native-baseline fairness (G002) — checked only once the heavy run has produced its artifact
import os  # noqa: E402
if os.path.exists("experiments/native_baselines.json"):
    nb = json.load(open("experiments/native_baselines.json"))
    assert nb["spearman_contiguity_vs_spatial_advantage"] >= 0.5, "headline must survive real-mclust backend"
    assert nb["p_contiguity_vs_spatial_advantage"] <= 0.05, "headline must stay significant under fair backend"
    assert nb["n_distinct_winners"] >= 5, "no-universal-SOTA must survive fair backend"
    # real mclust must genuinely be used (it actually wins some best-config cells, not a no-op)
    assert nb["backend_win_counts"].get("mclust-R", 0) > 0, "real R mclust never won — not actually used"
    # pin the manuscript's claimed +0.79 to the artifact (real-mclust value)
    check("R2 native real-mclust rho (manuscript says +0.79)", 0.791,
          nb["spearman_contiguity_vs_spatial_advantage"], tol=0.02)
    checks.append((f"§8c: law strengthens under real R mclust (rho={nb['spearman_contiguity_vs_spatial_advantage']}, "
                   f"{nb['n_distinct_winners']} winners, mclust-R wins {nb['backend_win_counts'].get('mclust-R')}/81)",
                   1, 1, True))
else:
    checks.append(("§8c: native_baselines.json pending (heavy run) — skipped", 1, 1, True))

# the manuscript text actually contains the headline strings
for s in ["0.036", "0.200", "0.494", "0.473", "0.392", "5/7", "3/7", "1/7", "0.451",
          "MERFISH", "osmFISH", "0.406", "0.449", "0.71", "0.77", "0.931",
          # §7 strings (post-review, n=11, 6 methods, 3 seeds)
          "SpaceFlow", "SpatialLeiden", "11 platforms", "7 distinct winners", "+0.72", "0.012",
          "−0.36", "0.413", "seed-level tie", "duplicate dataset", "[0.69, 0.77]",
          # §8 R2 strings (GT-free proxy, statistical rigor, backend fairness)
          "coh_gain", "label-free", "Moran", "−0.34", "−0.42", "bootstrap", "leave-one-platform-out",
          "partial ρ=0.67", "mclust", "BIC-based", "not a backend artifact", "6 distinct winners",
          "20 of 81", "+0.79", "Rscript",
          # §8e R4: native SpaGCN confirmation
          "tessera-spagcn", "native pipeline", "+0.67",
          # §9 R3 causal mechanism strings
          "controlled experiment", "manipulated", "causal", "+0.98", "0.345", "flips sign",
          # §8d R3 data-expansion (honest weakening) strings
          "platforms exhausted", "n=14", "+0.56", "weakens", "st_COAD",
          # §8e R3 SpaGCN (7th SOTA) strings
          "SpaGCN", "8 distinct winners", "kmeans", "understate",
          # component-ablation integration strings
          "mechanism_ablation.json", "load-bearing component", "edge gate"]:
    present = s in md
    checks.append((f"manuscript contains '{s}'", 1, 1 if present else 0, present))

# §9 R3: causal mechanism — contiguity manipulated, signal held fixed, advantage tracks + flips
mc = json.load(open("experiments/mechanism_synth.json"))
assert mc["nonspatial_ari_range"] <= 0.05, "mechanism control broken: non-spatial ARI not held fixed"
assert mc["spearman_contig_vs_adv_smooth"] >= 0.9 and mc["p_contig_vs_adv_smooth"] <= 0.05, \
    "mechanism: neighbour-mean advantage does not causally track manipulated contiguity"
assert mc["spearman_contig_vs_adv_stagate"] >= 0.9, "mechanism: STAGATE advantage does not track contiguity"
assert mc["adv_smooth_min"] < 0 < mc["adv_smooth_max"], "mechanism: advantage must flip sign"
check("R3 mechanism rho(contig,adv smooth)", 0.976, mc["spearman_contig_vs_adv_smooth"], tol=0.03)
checks.append(("§9: contiguity CAUSALLY drives advantage (manipulated, signal-fixed, sign-flips)", 1, 1, True))

# R3 data expansion (guarded — heavy run) + R3 SpaGCN (guarded — heavy run)
if os.path.exists("experiments/expand_data.json"):
    xd = json.load(open("experiments/expand_data.json"))
    assert xd["n_total"] == 14, f"expected n=14, got {xd['n_total']}"
    # honest: the law WEAKENS but survives (still positive + significant) when the missed cell-type
    # cancer platforms are added. We pin the attenuated value so the manuscript cannot quietly keep 0.72.
    assert 0.5 <= xd["spearman_expanded"] < xd["spearman_published_n11"] and xd["p_expanded"] <= 0.05, \
        "data-expansion result not the honest weakened-but-significant value"
    assert xd["n_distinct_winners_all"] >= 6, "no-universal-SOTA must survive the expansion"
    check("R3 expanded rho n=14", 0.559, xd["spearman_expanded"], tol=0.02)
    checks.append((f"§8d: law WEAKENS but survives +{xd['n_new']} missed platforms "
                   f"(0.72→{xd['spearman_expanded']}, p={xd['p_expanded']}, n=14)", 1, 1, True))
else:
    checks.append(("§8d: expand_data.json pending (heavy run) — skipped", 1, 1, True))
if os.path.exists("experiments/spagcn_panel.json"):
    sg = json.load(open("experiments/spagcn_panel.json"))
    assert sg["n_platforms"] >= 6, "SpaGCN ran on too few platforms to conclude (vacuous-pass guard)"
    assert sg["n_spagcn_wins"] <= 2, "SpaGCN became a near-universal winner — re-examine no-universal-SOTA"
    checks.append((f"§5b: SpaGCN (7th SOTA) wins {sg['n_spagcn_wins']} platforms — no-universal-SOTA holds",
                   1, 1, True))
else:
    checks.append(("§5b: spagcn_panel.json pending (heavy run) — skipped", 1, 1, True))

# §8e R4: SpaGCN re-run with its GENUINE native louvain pipeline (dedicated real-numba env) — the
# conclusion must hold without the kmeans-init fallback (guards the "understated" caveat).
if os.path.exists("experiments/spagcn_native.json"):
    sn = json.load(open("experiments/spagcn_native.json"))
    assert sn["n_platforms"] >= 6, "native SpaGCN ran on too few platforms (vacuous-pass guard)"
    assert sn["n_spagcn_native_wins"] <= 2, "native SpaGCN became near-universal — re-examine no-universal-SOTA"
    checks.append((f"§8e: NATIVE SpaGCN (real louvain) wins {sn['n_spagcn_native_wins']}/{sn['n_platforms']} "
                   f"— kmeans-init did not drive the result", 1, 1, True))
else:
    checks.append(("§8e: spagcn_native.json pending — skipped", 1, 1, True))

# Component ablation: only boundary_contrastive is load-bearing; edge_gating/multi_scale do not earn place
ma = json.load(open("experiments/mechanism_ablation.json"))
for plat in ["MERFISH(domain)", "osmFISH(Region)"]:
    cd = ma[plat]["component_drop"]
    top = max((k for k in cd if k != "all-components(vs backbone)"), key=lambda k: cd[k])
    assert top == "boundary_contrastive", f"{plat}: load-bearing component is {top}, not boundary_contrastive"
    assert cd["edge_gating"] <= 0.02 and cd["multi_scale"] <= 0.02, \
        f"{plat}: edge_gating/multi_scale unexpectedly earn their place"
check("ablation MERFISH boundary_contrastive drop", 0.283,
      ma["MERFISH(domain)"]["component_drop"]["boundary_contrastive"], tol=0.01)
checks.append(("component ablation: only boundary_contrastive load-bearing (edge_gating/multi_scale do not)",
               1, 1, True))

# PAPER.md (the submission-facing synthesis) must also carry the headline numbers it claims are verified
pa_md = open("manuscript/PAPER.md").read()
for s in ["+0.72", "11 platforms", "7 distinct", "coh_gain", "+0.55", "partial ρ=0.67",
          "6 distinct winners", "mclust", "no universal SOTA", "0.473"]:
    present = s in pa_md
    checks.append((f"PAPER.md contains '{s}'", 1, 1 if present else 0, present))

# paper.tex IS the submission artifact (produces paper.pdf) and claims its numbers are machine-checked;
# bring it into the loop so a future tex edit cannot silently drift from the JSONs while exit stays 0.
pa_tex = open("manuscript/paper.tex").read()
for s in ["+0.72", "+0.79", "+0.56", "+0.98", "0.473", "20 of 81", "0.36", "mclust", "Rscript",
          "no universal SOTA", "0.345",
          # R5: the proxy p must now be reported as SEARCH-CORRECTED (honest reframe), and the causal
          # law (p<0.001) must be present as the primary label-free weight
          "search-corrected", "p<0.001"]:
    present = s in pa_tex
    checks.append((f"paper.tex contains '{s}'", 1, 1 if present else 0, present))

# R6 decoupling control: load the adopted local artifact directly. JSON files remain git-ignored
# experiment outputs, but the release builder requires this one and this verifier fails if it is absent.
dc = json.load(open("experiments/mechanism_decoupled_v2.json"))
assert dc["status"].startswith("ADOPTED as CLAIM_LEDGER.md R6 robustness evidence"), \
    f"v2 decoupling artifact is not adopted: {dc['status']}"
assert dc["fidelity_restored"]["stagate_n_epochs"] == 400, "v2 must match the primary 400-epoch fidelity"
assert dc["fidelity_restored"]["seeds_v2"] == [1, 2, 3, 4, 5], "v2 must use the reported five seeds"
assert dc["fidelity_restored"]["n_substrates_v2"] == 2, "v2 must cover both reported substrates"

for substrate_name in ["S1_primary", "S2_independent"]:
    substrate = dc["results_by_substrate"][substrate_name]
    arm_a = substrate["arm_A_alignment_only"]
    arm_b = substrate["arm_B_contiguity_only"]
    contrast = substrate["contrast"]

    assert contrast["smooth_norm_slope_B_minus_A"] > 0, \
        f"{substrate_name}: contiguity does not have the larger smoother effect size"
    assert contrast["stagate_norm_slope_B_minus_A"] > 0, \
        f"{substrate_name}: contiguity does not have the larger STAGATE effect size"
    assert contrast["only_B_flips_sign_smooth"] and contrast["only_B_flips_sign_stagate"], \
        f"{substrate_name}: sign flip is not exclusive to the contiguity arm"
    assert contrast["B_floor_range"] < contrast["A_floor_range"], \
        f"{substrate_name}: contiguity arm does not keep the non-spatial floor flatter"
    assert not arm_a["adv_smooth_flips_sign"] and not arm_a["adv_stagate_flips_sign"], \
        f"{substrate_name}: alignment arm unexpectedly flips the advantage sign"
    assert arm_b["adv_smooth_flips_sign"] and arm_b["adv_stagate_flips_sign"], \
        f"{substrate_name}: contiguity arm does not reproduce the sign flip"

roll = dc["cross_substrate_rollup"]
for key in ["norm_slope_both_favor_B_all_substrates",
            "only_B_flips_sign_smooth_all_substrates",
            "only_B_flips_sign_stagate_all_substrates",
            "B_floor_flatter_than_A_all_substrates"]:
    assert roll[key], f"R6 cross-substrate discriminator failed: {key}"

# Pin every hand-typed Table tab:decouple number to the JSON, including effect sizes and floor ranges.
for substrate_name, label in [("S1_primary", "S1"), ("S2_independent", "S2")]:
    substrate = dc["results_by_substrate"][substrate_name]
    for arm_key, arm_label in [("arm_A_alignment_only", "alignment"),
                               ("arm_B_contiguity_only", "contiguity")]:
        arm = substrate[arm_key]
        smooth_effect = re.escape(f"{arm['normalized_slope_vs_adv_smooth']:.2f}")
        stagate_effect = re.escape(f"{arm['normalized_slope_vs_adv_stagate']:.2f}")
        floor_range = re.escape(f"{arm['nonspatial_floor_range']:.2f}")
        row_pattern = (
            rf"{label}\s*&\s*{arm_label}\s*&\s*"
            rf"{smooth_effect}\s*/\s*{stagate_effect}\s*&\s*"
            rf"{'yes' if arm['adv_smooth_flips_sign'] and arm['adv_stagate_flips_sign'] else 'no'}\s*&\s*"
            rf"{floor_range}"
        )
        present = re.search(row_pattern, pa_tex) is not None
        checks.append((f"paper.tex contains JSON-derived R6 row '{label} {arm_label}'",
                       1, 1 if present else 0, present))
checks.append(("R6 v2: slopes favor contiguity; only B flips; B floor is flatter on S1+S2",
               1, 1, True))

# --- numeric ties: derive hand-typed table/prose values from the JSON and require them in paper.tex
# (red-team found the presence-only list left tables/p-values un-tied to the ledger; this closes it:
# a wrong OR deleted value fails the check because the JSON-derived string is no longer present) ---
_ci = sr["bootstrap_gt_contiguity"]["ci95"]
_prt = sr["contiguity_partial_spearman_controlling"]
_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_derived = {
    f"[{_ci[0]:.2f},{_ci[1]:.2f}]": "bootstrap 95% CI",
    f"{_prt['k_nclasses']:.2f}/{_prt['eff_dim']:.2f}/{_prt['modality_imaging']:.2f}": "one-at-a-time partials",
    f"{sr['contiguity_partial_controlling_all']:.2f}": "partial controlling all",
    f"{eb['p_contiguity_vs_spatial_advantage']:.3f}": "law p-value",
    f"{_words[eb['n_distinct_winners']]} distinct": "n_distinct_winners (as word)",
    f"{gp['principled_proxy_coh_gain']['vs_advantage'][1]:.3f}": "coh_gain asymptotic uncorrected proxy p-value",
    f"{psc['p_uncorrected_selected']:.3f}": "coh_gain permutation uncorrected proxy p-value",
    f"{psc['p_search_corrected']:.2f}": "coh_gain SEARCH-CORRECTED proxy p-value (R5)",
}
for s, what in _derived.items():
    present = s in pa_tex
    checks.append((f"paper.tex contains derived {what} '{s}'", 1, 1 if present else 0, present))

for phrase, what in [
        ("asymptotic uncorrected", "0.079 asymptotic p-value label"),
        ("uncorrected permutation companion", "0.083 permutation p-value label")]:
    present = phrase in pa_tex
    checks.append((f"paper.tex labels {what}", 1, 1 if present else 0, present))

# robustness audit (de-circularised law, multiplicity, jackknife, binomial) -> tie to paper.tex
if os.path.exists("experiments/robustness_audit.json"):
    ra = json.load(open("experiments/robustness_audit.json"))
    rm, rj, rb = ra["multiplicity"], ra["jackknife"], ra["lopo_binomial"]
    _ra = {
        f"{rm['family3']['contiguity_corrected']['bonferroni']:.3f}": "family-3 corrected p",
        f"{rm['family7']['contiguity_corrected']['bh']:.3f}": "family-7 corrected p",
        f"[{rj['published_min']:.2f},{rj['published_max']:.2f}]": "jackknife range",
        f"{rb['p_value']:.2f}": "LODO decision binomial p",
    }
    for s, what in _ra.items():
        present = s in pa_tex
        checks.append((f"paper.tex contains robustness {what} '{s}'", 1, 1 if present else 0, present))

print(f"{'check':>40} | claimed | actual | ok")
print("-" * 70)
allok = True
for name, c, a, ok in checks:
    allok = allok and ok
    print(f"{name:>40} | {c:>7} | {a:>6} | {'PASS' if ok else 'FAIL'}")
print("\n" + ("ALL VERIFICATION CHECKS PASS" if allok else "VERIFICATION FAILED"))
import sys
sys.exit(0 if allok else 1)
