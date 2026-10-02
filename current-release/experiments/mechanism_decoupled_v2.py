"""FULL-FIDELITY re-run of the decoupled 2-arm follow-up (`mechanism_decoupled.py`).

Same experiment, same arm definitions, but restored to the SAME compute fidelity as the paper's
primary headline result `mechanism_synth.py` (STAGATE 400 epochs, GPU, no thread cap), plus:
  * denser sweep (11 levels, 0.1 spacing) so rho is not read off only 6 saturated points,
  * a SECOND independent substrate (true replication -- primary sweep used one substrate),
  * 5 seeds instead of 3 (cheap on GPU; tightens per-level advantage estimates).

Arm definitions are copied VERBATIM from mechanism_decoupled.py -- nothing about WHAT is measured
changes; only the fidelity/coverage of the measurement changes.

  Arm A (alignment only): coords + labels NEVER touched (contiguity held fixed across the arm). A
    fraction `a` of cells has its EXPRESSION vector swapped with another cell's -> erodes
    expression<->position alignment without moving a label.

  Arm B (contiguity only): coords NEVER touched; expression ALWAYS regenerated from whatever label
    now sits at a position (domain-mean + fresh iid noise) -> alignment perfect by construction. A
    fraction `c` of positions has its label overwritten with a random domain id -> breaks label
    spatial contiguity without touching internal label->expression consistency.

Beyond v1's summary (rho + normalized_slope per arm), this reports two honest discriminators that
v1's naive abs(rho) verdict ignored:
  1. SIGN-FLIP: does the advantage cross from + to - across the arm? The paper's headline
     phenomenon is a *sign flip* (spatial prior HELPS at high contiguity, HURTS at low). Report
     adv min/max per arm -- an arm that only decays a positive advantage toward 0 is qualitatively
     different from one that flips sign.
  2. FLOOR CONTROL: range of the non-spatial ARI floor across the arm. Arm B regenerates the signal
     each level so the floor should stay ~flat (clean "only-spatial-arrangement-changed"
     manipulation); Arm A's expression-swap also destroys the recoverable signal, so its floor
     collapses (confounded -- degrading alignment also removes signal for everyone).

Writes experiments/mechanism_decoupled_v2.json ONLY. Does NOT touch mechanism_synth.json or
mechanism_decoupled.json (v1 kept intact). The generated artifact is adopted as the R6 robustness
evidence in CLAIM_LEDGER.md and manuscript/paper.tex; the scientific claim itself remains LOCKED
under the ledger's global honest-claims policy.
"""

import sys
from _roots import data_root, external_root, spatial_omics_root

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import anndata as ad  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from scipy.stats import spearmanr, linregress  # noqa: E402

sys.path.insert(0, "src")
from tessera_st.data.synthetic import make_tessellation  # noqa: E402

EXT = str(external_root())
sys.path.insert(0, f"{EXT}/STAGATE")

# FULL FIDELITY: use the GPU and full threading, exactly like mechanism_synth.py (the primary).
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 400  # restored from v1's reduced 120 -> matches mechanism_synth.py primary result

LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # dense 0.1-spacing (v1 had 6)
SEEDS = [1, 2, 3, 4, 5]  # v1/primary used 3; 5 is cheap on GPU and tightens per-level means

# Per-arm checkpoint so an interrupted run resumes instead of restarting from scratch. Results are
# fully deterministic (fixed seeds), so a resumed run is byte-identical to an uninterrupted one.
CKPT = "experiments/mechanism_decoupled_v2_ckpt.json"
try:
    _ckpt = json.load(open(CKPT))
except (FileNotFoundError, ValueError):
    _ckpt = {}


def _save_ckpt():
    json.dump(_ckpt, open(CKPT, "w"))

# Substrate 1: EXACT primary params (n_side=34,...). Substrate 2: an independent second substrate
# in the same "space is necessary" regime (domain_signal < iid_noise so the sign-flip is visible),
# with different grid size / gene count / domain count / signal -> true replication, not a re-seed.
SUBSTRATES = {
    "S1_primary": dict(n_side=34, n_genes=40, n_domains=5, domain_signal=0.3, iid_noise=1.0,
                       small_domain=True),
    "S2_independent": dict(n_side=40, n_genes=60, n_domains=6, domain_signal=0.35, iid_noise=1.0,
                           small_domain=True),
}


def knn_idx(coords, k=6):
    return NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)[1][:, 1:]


def contiguity(coords, labels, idx):
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


def stagate_emb(expr, coords, seed):
    import STAGATE_pyG as ST
    a = ad.AnnData(X=expr.astype(np.float32)); a.obsm["spatial"] = coords
    ST.Cal_Spatial_Net(a, k_cutoff=6, model="KNN", verbose=False); torch.manual_seed(seed)
    return np.asarray(ST.train_STAGATE(a, n_epochs=N_EPOCHS, random_seed=seed, device=dev,
                                       verbose=False).obsm["STAGATE"])


def scramble_expr_only(expr, frac, rng):
    """Arm A knob (verbatim from v1): permute expression among a random `frac` of cells; coords &
    labels untouched by construction. Returns (perturbed_expr, aligned_mask)."""
    n = expr.shape[0]
    m = int(round(frac * n))
    aligned = np.ones(n, dtype=bool)
    if m < 2:
        return expr.copy(), aligned
    sel = rng.choice(n, m, replace=False)
    perm = rng.permutation(sel)
    out = expr.copy()
    out[sel] = expr[perm]
    aligned[sel] = perm == sel
    return out, aligned


def scramble_labels_regenerate(labels, domain_mean, frac, rng, n_dom, n_genes, iid_noise):
    """Arm B knob (verbatim from v1): overwrite a random `frac` of positions' labels with an
    independent domain id (breaks contiguity), regenerate that position's expression fresh from
    domain_mean[new_label] + iid noise (alignment stays perfect). Coords never touched."""
    n = labels.shape[0]
    new_labels = labels.copy()
    m = int(round(frac * n))
    if m >= 1:
        sel = rng.choice(n, m, replace=False)
        new_labels[sel] = rng.integers(0, n_dom, size=m)
    expr = domain_mean[new_labels] + rng.normal(0.0, iid_noise, size=(n, n_genes))
    return new_labels, expr


def evaluate(expr, coords, labels, seed):
    """Same evaluation pipeline as mechanism_synth.py / v1: standardize, PCA, neighbour-mean prior,
    STAGATE prior, ARI advantage for each."""
    k = int(labels.max()) + 1
    mu, sd = expr.mean(0), expr.std(0); sd[sd == 0] = 1
    Z = np.clip((expr - mu) / sd, -10, 10)
    idx = knn_idx(coords, 6)
    Zpca = PCA(30, random_state=seed).fit_transform(Z)
    nbr = Z[idx].mean(1)
    nbr_pca = PCA(30, random_state=seed).fit_transform(nbr)
    a_nonsp = best_ari(Zpca, labels, k, seed)
    a_smooth = best_ari(nbr_pca, labels, k, seed)
    try:
        a_stag = best_ari(stagate_emb(Z, coords, seed), labels, k, seed)
    except Exception as e:
        print("  STAGATE fail", str(e)[:80]); a_stag = float("nan")
    return idx, a_nonsp, a_smooth, a_stag


def run_arm(arm_name, substrate, sname):
    key = f"{sname}::{arm_name}"
    if key in _ckpt:
        print(f"  [{arm_name}] RESUMED from checkpoint ({len(_ckpt[key])} levels)", flush=True)
        return _ckpt[key]
    rows = []
    for lvl in LEVELS:
        per = {"contig": [], "align": [], "nonsp": [], "smooth": [], "stagate": [],
               "adv_smooth": [], "adv_stagate": [], "resid_std": []}
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 10_000 + int(round(lvl * 1000)))
            sl = make_tessellation(seed=seed, **substrate)
            coords = sl.coords  # NEVER touched in either arm
            if arm_name == "A_alignment_only":
                expr, aligned_mask = scramble_expr_only(sl.expr.astype(np.float64), lvl, rng)
                labels = sl.labels  # untouched -> contiguity ~constant across this arm
                align_score = float(aligned_mask.mean())
                resid_std = float("nan")
            else:  # B_contiguity_only
                n_dom = int(sl.labels.max()) + 1
                domain_mean = np.stack([sl.expr[sl.labels == d].mean(0) for d in range(n_dom)])
                labels, expr = scramble_labels_regenerate(
                    sl.labels, domain_mean, lvl, rng, n_dom, substrate["n_genes"],
                    substrate["iid_noise"])
                expr = expr.astype(np.float64)
                align_score = 1.0
                resid_std = float((expr - domain_mean[labels]).std())

            idx, a_nonsp, a_smooth, a_stag = evaluate(expr, coords, labels, seed)
            cont = contiguity(coords, labels, idx)
            per["contig"].append(cont); per["align"].append(align_score)
            per["nonsp"].append(a_nonsp); per["smooth"].append(a_smooth); per["stagate"].append(a_stag)
            per["adv_smooth"].append(a_smooth - a_nonsp)
            per["adv_stagate"].append(a_stag - a_nonsp)
            per["resid_std"].append(resid_std)
        rec = {"level": lvl,
               "contiguity": round(float(np.mean(per["contig"])), 3),
               "alignment": round(float(np.mean(per["align"])), 3),
               "nonspatial_ari": round(float(np.mean(per["nonsp"])), 3),
               "smooth_ari": round(float(np.mean(per["smooth"])), 3),
               "stagate_ari": round(float(np.nanmean(per["stagate"])), 3),
               "adv_smooth": round(float(np.mean(per["adv_smooth"])), 3),
               "adv_stagate": round(float(np.nanmean(per["adv_stagate"])), 3),
               "expr_label_resid_std": (round(float(np.mean(per["resid_std"])), 3)
                                         if arm_name != "A_alignment_only" else None)}
        rows.append(rec)
        print(f"  [{arm_name}] lvl {lvl:.2f} | contig {rec['contiguity']:.3f} | align "
              f"{rec['alignment']:.3f} | nonsp {rec['nonspatial_ari']:+.3f} | adv(smooth) "
              f"{rec['adv_smooth']:+.3f} | adv(STAGATE) {rec['adv_stagate']:+.3f}", flush=True)
    _ckpt[key] = rows
    _save_ckpt()
    return rows


def arm_summary(rows, x_key, arm_label):
    x = np.array([r[x_key] for r in rows])
    adv_s = np.array([r["adv_smooth"] for r in rows])
    adv_g = np.array([r["adv_stagate"] for r in rows])
    nonsp = np.array([r["nonspatial_ari"] for r in rows])
    rho_s = spearmanr(x, adv_s); rho_g = spearmanr(x, adv_g)
    ls_s = linregress(x, adv_s); ls_g = linregress(x, adv_g)
    xr = float(x.max() - x.min())
    return {
        "arm": arm_label, "x": x_key,
        f"{x_key}_range": [round(float(x.min()), 3), round(float(x.max()), 3)],
        "spearman_vs_adv_smooth": round(float(rho_s.correlation), 3),
        "p_vs_adv_smooth": round(float(rho_s.pvalue), 4),
        "spearman_vs_adv_stagate": round(float(rho_g.correlation), 3),
        "p_vs_adv_stagate": round(float(rho_g.pvalue), 4),
        "slope_vs_adv_smooth": round(float(ls_s.slope), 4),
        "slope_vs_adv_stagate": round(float(ls_g.slope), 4),
        "normalized_slope_vs_adv_smooth": round(float(ls_s.slope * xr), 4),
        "normalized_slope_vs_adv_stagate": round(float(ls_g.slope * xr), 4),
        # honest discriminators v1's verdict ignored:
        "adv_smooth_min": round(float(adv_s.min()), 3),
        "adv_smooth_max": round(float(adv_s.max()), 3),
        "adv_stagate_min": round(float(adv_g.min()), 3),
        "adv_stagate_max": round(float(adv_g.max()), 3),
        "adv_smooth_flips_sign": bool(adv_s.min() < 0 < adv_s.max()),
        "adv_stagate_flips_sign": bool(adv_g.min() < 0 < adv_g.max()),
        "nonspatial_floor_range": round(float(nonsp.max() - nonsp.min()), 3),
    }


def run_substrate(sname, substrate):
    print(f"\n########## SUBSTRATE {sname}: {substrate} ##########", flush=True)
    print(f"===== {sname} Arm A: alignment-only (contiguity held fixed) =====", flush=True)
    rows_a = run_arm("A_alignment_only", substrate, sname)
    print(f"===== {sname} Arm B: contiguity-only (alignment held fixed) =====", flush=True)
    rows_b = run_arm("B_contiguity_only", substrate, sname)

    sum_a = arm_summary(rows_a, "alignment", "A_alignment_only")
    sum_b = arm_summary(rows_b, "contiguity", "B_contiguity_only")

    cont_a = np.array([r["contiguity"] for r in rows_a])
    contiguity_control_range_a = round(float(cont_a.max() - cont_a.min()), 3)
    align_b = np.array([r["alignment"] for r in rows_b])
    alignment_control_range_b = round(float(align_b.max() - align_b.min()), 3)

    contrast = {
        "smooth_abs_rho_B_minus_A": round(abs(sum_b["spearman_vs_adv_smooth"])
                                          - abs(sum_a["spearman_vs_adv_smooth"]), 3),
        "stagate_abs_rho_B_minus_A": round(abs(sum_b["spearman_vs_adv_stagate"])
                                           - abs(sum_a["spearman_vs_adv_stagate"]), 3),
        "smooth_norm_slope_B_minus_A": round(sum_b["normalized_slope_vs_adv_smooth"]
                                             - sum_a["normalized_slope_vs_adv_smooth"], 4),
        "stagate_norm_slope_B_minus_A": round(sum_b["normalized_slope_vs_adv_stagate"]
                                              - sum_a["normalized_slope_vs_adv_stagate"], 4),
        # only Arm B (contiguity) reproducing the headline sign-flip is the qualitative separator
        "only_B_flips_sign_smooth": bool(sum_b["adv_smooth_flips_sign"]
                                         and not sum_a["adv_smooth_flips_sign"]),
        "only_B_flips_sign_stagate": bool(sum_b["adv_stagate_flips_sign"]
                                          and not sum_a["adv_stagate_flips_sign"]),
        # clean-manipulation check: is B's floor flatter than A's? (A's expr-swap also kills signal)
        "A_floor_range": sum_a["nonspatial_floor_range"],
        "B_floor_range": sum_b["nonspatial_floor_range"],
    }

    both_favor_b = (abs(sum_b["spearman_vs_adv_smooth"]) > abs(sum_a["spearman_vs_adv_smooth"])
                    and abs(sum_b["spearman_vs_adv_stagate"]) > abs(sum_a["spearman_vs_adv_stagate"]))
    both_favor_a = (abs(sum_a["spearman_vs_adv_smooth"]) > abs(sum_b["spearman_vs_adv_smooth"])
                    and abs(sum_a["spearman_vs_adv_stagate"]) > abs(sum_b["spearman_vs_adv_stagate"]))
    norm_slope_favors_b = (sum_b["normalized_slope_vs_adv_smooth"] > sum_a["normalized_slope_vs_adv_smooth"]
                           and sum_b["normalized_slope_vs_adv_stagate"] > sum_a["normalized_slope_vs_adv_stagate"])

    print(f"\n----- {sname} DECOUPLED CONTRAST -----")
    print(f"Arm A (align):   rho_smooth={sum_a['spearman_vs_adv_smooth']:+.3f} "
          f"rho_STAGATE={sum_a['spearman_vs_adv_stagate']:+.3f} | "
          f"nslope_smooth={sum_a['normalized_slope_vs_adv_smooth']:+.3f} "
          f"nslope_STAGATE={sum_a['normalized_slope_vs_adv_stagate']:+.3f} | "
          f"adv_smooth[{sum_a['adv_smooth_min']:+.3f},{sum_a['adv_smooth_max']:+.3f}] "
          f"flip={sum_a['adv_smooth_flips_sign']} | floor_range={sum_a['nonspatial_floor_range']:.3f}")
    print(f"Arm B (contig):  rho_smooth={sum_b['spearman_vs_adv_smooth']:+.3f} "
          f"rho_STAGATE={sum_b['spearman_vs_adv_stagate']:+.3f} | "
          f"nslope_smooth={sum_b['normalized_slope_vs_adv_smooth']:+.3f} "
          f"nslope_STAGATE={sum_b['normalized_slope_vs_adv_stagate']:+.3f} | "
          f"adv_smooth[{sum_b['adv_smooth_min']:+.3f},{sum_b['adv_smooth_max']:+.3f}] "
          f"flip={sum_b['adv_smooth_flips_sign']} | floor_range={sum_b['nonspatial_floor_range']:.3f}")
    print(f"  abs(rho) both favor B: {both_favor_b} | both favor A: {both_favor_a} | "
          f"norm_slope both favor B: {norm_slope_favors_b} | "
          f"only-B-flips-sign(smooth): {contrast['only_B_flips_sign_smooth']}")

    return {
        "substrate": {**substrate, "levels": LEVELS, "seeds": SEEDS, "stagate_n_epochs": N_EPOCHS},
        "arm_A_alignment_only": {"rows": rows_a, **sum_a,
                                 "contiguity_control_range": contiguity_control_range_a},
        "arm_B_contiguity_only": {"rows": rows_b, **sum_b,
                                  "alignment_control_range": alignment_control_range_b},
        "contrast": contrast,
        "abs_rho_both_favor_B": both_favor_b,
        "abs_rho_both_favor_A": both_favor_a,
        "norm_slope_both_favor_B": norm_slope_favors_b,
    }


t0 = time.time()
results = {}
for sname, substrate in SUBSTRATES.items():
    results[sname] = run_substrate(sname, substrate)

# cross-substrate roll-up of the honest discriminators
roll = {
    "norm_slope_both_favor_B_all_substrates": all(results[s]["norm_slope_both_favor_B"]
                                                  for s in SUBSTRATES),
    "only_B_flips_sign_smooth_all_substrates": all(results[s]["contrast"]["only_B_flips_sign_smooth"]
                                                   for s in SUBSTRATES),
    "only_B_flips_sign_stagate_all_substrates": all(results[s]["contrast"]["only_B_flips_sign_stagate"]
                                                    for s in SUBSTRATES),
    "B_floor_flatter_than_A_all_substrates": all(
        results[s]["contrast"]["B_floor_range"] < results[s]["contrast"]["A_floor_range"]
        for s in SUBSTRATES),
    "abs_rho_both_favor_B_all_substrates": all(results[s]["abs_rho_both_favor_B"]
                                               for s in SUBSTRATES),
    "abs_rho_both_favor_A_all_substrates": all(results[s]["abs_rho_both_favor_A"]
                                               for s in SUBSTRATES),
}

out = {
    "source_script_reused": ("experiments/mechanism_decoupled.py (v1, arm defs verbatim) at the "
                             "fidelity of experiments/mechanism_synth.py (STAGATE 400 epochs, GPU, "
                             "no thread cap)"),
    "fidelity_restored": {"stagate_n_epochs": N_EPOCHS, "stagate_n_epochs_v1": 120,
                          "device": str(dev), "torch_num_threads": torch.get_num_threads(),
                          "levels_v2": LEVELS, "levels_v1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                          "seeds_v2": SEEDS, "seeds_v1": [1, 2, 3],
                          "n_substrates_v2": len(SUBSTRATES), "n_substrates_v1": 1},
    "results_by_substrate": results,
    "cross_substrate_rollup": roll,
    "reading_note": ("abs(rho) is a poor discriminator here: both arms are monotone by "
                     "construction so both rho saturate near 1. The load-bearing comparisons are "
                     "(a) normalized_slope [effect size], (b) whether the advantage FLIPS SIGN "
                     "(the paper's headline phenomenon), and (c) whether the non-spatial floor is "
                     "held flat [clean manipulation] -- Arm A's expr-swap also destroys the "
                     "recoverable signal, so its shrinking advantage is partly generic signal loss."),
    "runtime_sec": round(time.time() - t0, 1),
    "status": ("ADOPTED as CLAIM_LEDGER.md R6 robustness evidence; causal claim remains LOCKED "
               "under the global honest-claims policy"),
}
json.dump(out, open("experiments/mechanism_decoupled_v2.json", "w"), indent=2)
print(f"\nwrote experiments/mechanism_decoupled_v2.json  ({out['runtime_sec']}s)")
print("cross-substrate rollup:", json.dumps(roll, indent=2))
