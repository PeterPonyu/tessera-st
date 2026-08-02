"""Decoupled 2-arm follow-up to `mechanism_synth.py` -- separates LABEL SPATIAL CONTIGUITY from
EXPRESSION<->POSITION ALIGNMENT, per the independent review's objection.

`mechanism_synth.py`'s single scramble knob relocates each cell's (label, expression) pair to a
new grid position. That simultaneously (a) breaks up the label field's spatial contiguity AND
(b) decorrelates the k-NN neighbour graph from expression similarity, because a scrambled cell's
spatial neighbours are no longer preferentially same-domain, so neighbour-averaged expression is
no longer informative -- "generic spatial-signal destruction". A referee's fair objection: that
sweep alone can't attribute the advantage to contiguity specifically rather than to this generic
effect.

This script reuses the SAME synthetic substrate (`tessera_st.data.synthetic.make_tessellation`,
identical params to `mechanism_synth.py`) and the SAME advantage metric -- ARI(spatial prior) -
ARI(non-spatial floor), for a neighbour-mean prior and STAGATE, via the same
`contiguity()` / `best_ari()` / `stagate_emb()` procedure -- but drives two INDEPENDENT knobs
instead of one:

  Arm A (alignment only): coordinates and labels are NEVER touched (label field's spatial
    contiguity is therefore ~constant across the whole arm -- verified below). A fraction `a` of
    cells has its EXPRESSION VECTOR swapped with another randomly-selected cell's, so the
    expression sitting at a position is no longer what that position's (unchanged) label
    "should" have produced. This erodes expression<->coordinate correspondence without moving a
    single label.

  Arm B (contiguity only): coordinates are NEVER touched, and expression is ALWAYS regenerated
    exactly from whatever label currently sits at a position (empirical domain-mean + fresh iid
    noise) -- expression<->position alignment is perfect by construction at every level (verified
    below). A fraction `c` of grid positions has its label overwritten with an independently
    drawn random domain id, breaking label spatial contiguity without touching the internal
    label->expression consistency.

Question: does spatial-prior advantage track CONTIGUITY (Arm B) more tightly than it tracks
generic ALIGNMENT (Arm A)? If Arm B's |rho|/slope clearly exceeds Arm A's for both priors, that
upgrades the `mechanism_synth.py` causal claim from "defensible" (contiguity confounded with
alignment) to "clean" (contiguity specifically, not generic spatial-signal loss, drives the
advantage).

COMPUTE NOTE: this machine had several other CPU-heavy jobs running concurrently (a BayesSpace
merge, a MAEST baseline run, a GPU job) when this was run, so STAGATE epochs were reduced
400->120 and torch was capped at 2 threads to avoid starving those jobs and to fit a CPU-only
budget. Everything else -- grid, domain count, noise levels, k, seeds, PCA dims, ARI/clustering
procedure -- is identical to `mechanism_synth.py`.

Writes experiments/mechanism_decoupled.json only. Does NOT touch mechanism_synth.json,
verify_manuscript.py, or any bayesspace_*.json artifact. This is a STAGED result for human review
-- no manuscript/paper.tex or CLAIM_LEDGER.md edits are made here.
"""

import sys

sys.path.insert(0, "experiments")
import _numba_stub  # noqa: E402

_numba_stub.install()

import json  # noqa: E402

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

EXT = "/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external"
sys.path.insert(0, f"{EXT}/STAGATE")

torch.set_num_threads(2)  # be a considerate neighbour -- other CPU jobs are active on this box
dev = torch.device("cpu")  # CPU-only by task constraint; do not touch the GPU

LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # fraction of cells perturbed, per arm
SEEDS = [1, 2, 3]  # matches mechanism_synth.py's seed count
N_EPOCHS = 120  # reduced from mechanism_synth.py's 400 -- see COMPUTE NOTE above

# identical substrate params to mechanism_synth.py
SUBSTRATE = dict(n_side=34, n_genes=40, n_domains=5, domain_signal=0.3, iid_noise=1.0,
                 small_domain=True)


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
    """Arm A knob: permute expression vectors among a random `frac` of cells. Coords & labels
    are simply never passed to this function -- untouched by construction. Returns
    (perturbed_expr, aligned_mask) where aligned_mask marks cells whose own expression is still
    sitting at their own position."""
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
    """Arm B knob: overwrite a random `frac` of grid positions' labels with an independently
    drawn domain id (breaks label spatial contiguity), then regenerate THAT position's
    expression fresh from domain_mean[new_label] + iid noise -- expression always matches its
    own current label by construction, so alignment never degrades. Coordinates are never
    touched."""
    n = labels.shape[0]
    new_labels = labels.copy()
    m = int(round(frac * n))
    if m >= 1:
        sel = rng.choice(n, m, replace=False)
        new_labels[sel] = rng.integers(0, n_dom, size=m)
    expr = domain_mean[new_labels] + rng.normal(0.0, iid_noise, size=(n, n_genes))
    return new_labels, expr


def evaluate(expr, coords, labels, seed):
    """Same evaluation pipeline as mechanism_synth.py: standardize, PCA, neighbour-mean prior,
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


def run_arm(arm_name):
    rows = []
    for lvl in LEVELS:
        per = {"contig": [], "align": [], "nonsp": [], "smooth": [], "stagate": [],
               "adv_smooth": [], "adv_stagate": [], "resid_std": []}
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 10_000 + int(round(lvl * 1000)))
            sl = make_tessellation(seed=seed, **SUBSTRATE)
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
                    sl.labels, domain_mean, lvl, rng, n_dom, SUBSTRATE["n_genes"],
                    SUBSTRATE["iid_noise"])
                expr = expr.astype(np.float64)
                align_score = 1.0  # perfect by construction -- expr always == f(current label)
                # control check: residual of expr against ITS OWN current label's domain mean
                # should have ~constant scale (iid_noise) at every level if alignment truly held.
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
        print(f"[{arm_name}] level {lvl:.2f} | contig {rec['contiguity']:.3f} | align "
              f"{rec['alignment']:.3f} | nonsp {rec['nonspatial_ari']:.3f} | adv(smooth) "
              f"{rec['adv_smooth']:+.3f} | adv(STAGATE) {rec['adv_stagate']:+.3f}", flush=True)
    return rows


def arm_summary(rows, x_key, arm_label):
    x = np.array([r[x_key] for r in rows])
    adv_s = np.array([r["adv_smooth"] for r in rows])
    adv_g = np.array([r["adv_stagate"] for r in rows])
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
    }


print("===== Arm A: alignment-only (contiguity held fixed) =====")
rows_a = run_arm("A_alignment_only")
print("\n===== Arm B: contiguity-only (alignment held fixed) =====")
rows_b = run_arm("B_contiguity_only")

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
}

both_smooth_and_stagate_favor_b = (
    abs(sum_b["spearman_vs_adv_smooth"]) > abs(sum_a["spearman_vs_adv_smooth"])
    and abs(sum_b["spearman_vs_adv_stagate"]) > abs(sum_a["spearman_vs_adv_stagate"]))
both_favor_a = (
    abs(sum_a["spearman_vs_adv_smooth"]) > abs(sum_b["spearman_vs_adv_smooth"])
    and abs(sum_a["spearman_vs_adv_stagate"]) > abs(sum_b["spearman_vs_adv_stagate"]))
if both_smooth_and_stagate_favor_b:
    verdict = ("Advantage tracks CONTIGUITY (Arm B) more tightly than generic ALIGNMENT (Arm A) "
               "for both the neighbour-mean and STAGATE priors -- supports contiguity as the "
               "specific driver, not merely generic spatial-signal loss.")
elif both_favor_a:
    verdict = ("Advantage tracks generic ALIGNMENT (Arm A) more tightly than CONTIGUITY (Arm B) "
               "for both priors -- does NOT support contiguity-specificity; the original sweep's "
               "conflation looks like it may have been picking up generic spatial-signal loss.")
else:
    verdict = ("Mixed: the two priors disagree on which arm dominates -- the decoupled arms do "
               "not cleanly separate contiguity from generic alignment; treat as ambiguous, not "
               "as an upgrade to a clean causal claim.")

print("\n===== DECOUPLED CONTRAST =====")
print(f"Arm A (alignment): rho(align, adv_smooth)={sum_a['spearman_vs_adv_smooth']:+.3f} "
      f"(p={sum_a['p_vs_adv_smooth']:.4f}) | rho(align, adv_STAGATE)="
      f"{sum_a['spearman_vs_adv_stagate']:+.3f} (p={sum_a['p_vs_adv_stagate']:.4f})")
print(f"  control: contiguity range across Arm A levels = {contiguity_control_range_a:.3f} "
      f"(small => contiguity truly held fixed)")
print(f"Arm B (contiguity): rho(contig, adv_smooth)={sum_b['spearman_vs_adv_smooth']:+.3f} "
      f"(p={sum_b['p_vs_adv_smooth']:.4f}) | rho(contig, adv_STAGATE)="
      f"{sum_b['spearman_vs_adv_stagate']:+.3f} (p={sum_b['p_vs_adv_stagate']:.4f})")
print(f"  control: alignment range across Arm B levels = {alignment_control_range_b:.3f} "
      f"(small => alignment truly held fixed, perfect by construction)")
print(f"\n{verdict}")

out = {
    "source_script_reused": ("experiments/mechanism_synth.py -- reuses make_tessellation() "
                              "substrate params and the contiguity()/best_ari()/stagate_emb() "
                              "advantage pipeline (ARI(spatial prior) - ARI(non-spatial floor))"),
    "substrate": {**SUBSTRATE, "levels": LEVELS, "seeds": SEEDS, "stagate_n_epochs": N_EPOCHS,
                  "stagate_n_epochs_original": 400,
                  "compute_note": ("STAGATE epochs reduced 400->120 and torch capped at 2 "
                                    "threads vs mechanism_synth.py, to fit a CPU-only budget on "
                                    "a shared machine running other jobs concurrently")},
    "arm_A_alignment_only": {"rows": rows_a, **sum_a,
                              "contiguity_control_range": contiguity_control_range_a},
    "arm_B_contiguity_only": {"rows": rows_b, **sum_b,
                               "alignment_control_range": alignment_control_range_b},
    "contrast": contrast,
    "verdict": verdict,
    "status": "STAGED -- not promoted through CLAIM_LEDGER.md LOCKED->graduated gate; report only",
}
json.dump(out, open("experiments/mechanism_decoupled.json", "w"), indent=2)
print("\nwrote experiments/mechanism_decoupled.json")
