# Round-3 revision protocol: annotation diagnostics, matched attribution on three encoders, and prospective tests

**Status: FROZEN.**
- **Freeze record.** This version is fixed by the commit titled "Freeze PROTOCOL-R3", made before any new cohort was downloaded. Whether it is also deposited publicly is the author's decision. If it is, the DOI and publication time are added below as a dated note, without any other change.
- **What was run before the freeze.** Only calibration analyses on the 24 datasets examined in earlier rounds (WP1, WP2 control selection, WP3); section 10 records them.
- **Changes after the freeze.** These are dated deviations in section 11.

## 1. Questions

1. **Diagnostics.** How strongly does a reference annotation's construction favor smoothing? This is measured from labels and coordinates, and for one diagnostic also from each observation's own expression.
2. **Prediction.** Do the diagnostics predict, before any model is fitted, whether smoothing helps a target and at what neighborhood scale?
3. **Attribution.** For three encoders with different mechanisms (Tessera-ST, GraphST, STAGATE), is the gain over expression-only clustering resolvably larger than a parameter-free propagation on the encoder's own graph and depth?
4. **Mechanism.** Is the gain carried by the component that couples neighbors?

## 2. Label diagnostics (code `diagnostics_r3.py`, tests `test_r3.py`)

All pairs are within one coordinate frame. Labels are integers, with −1 for missing.

### 2.1 Neighbor lists

- Each observation's k nearest other observations in its frame; the query is removed by index.
- Lists are never padded. A scale k is reported only if frames with at least k + 1 observations hold 90% or more of the labelled observations.
- k ∈ {6, 10, 15, 30, 50, 100, 200}.

### 2.2 Homophily and label informativeness

- **Edge homophily** h_edge(k) is the fraction of labelled directed edges whose endpoints share a label.
- **Adjusted homophily.** h_adj = (h_edge − chance) / (1 − chance). Within each frame, chance = Σ_c p_c², with p_c the class shares of endpoint labels on the symmetrized edge multiset (degree-weighted). Frames are combined with edge weights.
- **Label informativeness.** I(y_ξ; y_η) / H(y_ξ) over random edges of the symmetrized multiset, conditioned on frame.

### 2.3 Correlation length

- **Shells.** Ranks are grouped into {1}, {2–3}, {4–6}, {7–10}, {11–15}, {16–30}, {31–50}, {51–100}, {101–200}, each used only if 90% of labelled observations have that many neighbors.
- **Excess per shell.** (same − chance) / (1 − chance), where chance is the anchor frame's Σ_c π_c² over class frequencies π_c.
- **Length in neighbors.** ℓ_k is the rank where the excess first falls to 1/e of the first-shell value, interpolated in log rank between geometric shell midpoints.
  - If the excess never falls that far, ℓ_k is censored at the last shell.
  - If the first-shell excess is below 0.05, ℓ_k = 0.
- **Length in distance.** ℓ_d1 is the matching median pair distance in median nearest-neighbor distances (d1). ℓ_µm is given only where the pitch is documented: Visium 100 µm, Spatial Transcriptomics 200 µm, or coordinates stated in µm.

### 2.4 Predictability

- **Model.** Multinomial logistic regression (C = 1, class-balanced, standardized on the training fold).
- **Folds.** Five-fold cross-validation over square spatial blocks of side 10·d1, shuffled round-robin with seed 0. With fewer than 5 blocks but 3 or more frames, frames are the folds; otherwise the value is not estimable.
- **Reported.** κ_self, its balanced-accuracy analogue and chance level, and the random-fold value as a leakage-prone bound.
- **Neighborhood information gain.** max_k (κ_k − κ_self), where κ_k comes from the same model with the k-neighbor mean appended, together with its arg max.

### 2.5 Smoothing outcomes

- **Shared settings.** The shared 50-component input, K-means with 10 initializations at the reference class count K, seeds 1–3, and ARI on labelled observations.
- **Representations scored:**
  - PCA50;
  - directed k-neighbor means under the frozen round-1 neighbor rule;
  - the same-graph mean on the Tessera-ST graph;
  - saved 120-epoch Tessera-ST embeddings.
- **Derived quantities:** deployable gain g6, label-selected gain g* and its scale k*, and regret.
- **Reproduction check.** Recomputed values with saved round-1 counterparts must agree within 0.001 ARI. In calibration, 387 of 390 agreed to 1e-9; three differed by at most 0.0006 because K-means ran with one thread instead of several.

## 3. Datasets, units and roles

### 3.1 Calibration (seen before round 3)

- NSCLC regions and cell types (5 patients).
- BCC cell types (4 patients).
- Breast Visium.
- Colorectal TVA.
- MIBI and Open-ST (development).
- HER2ST all labels, named compartments and tumor/non-tumor (8 patients).
- Three anatomical tissues (development).
- The colorectal pilot (3 patients), reported separately.

### 3.2 New cohorts (downloaded after the freeze; units are patients or animals)

**Mouse spinal cord MERFISH** (Wang et al. 2026, PNAS 123:e2623177123; Zenodo 18039571).
- **Units, frames and targets.** The unit is the mouse and the frame is the section; the cell-type labels at each deposited level are targets in every branch.
- **Branches.** The deposited file decides the branch, recorded at preparation:
  - **A:** per-cell lamina labels exist, and they are a target;
  - **B:** per-cell relative depth from the authors' traced dorsal surfaces exists, and is binned into laminae at boundaries stated in the paper (if the paper states none numerically, branch B is not used);
  - **C:** neither exists, and no lamina target is built.
- **Animals included.** The healthy-atlas mice are the prospective units. Sham and SNI animals are used only for the descriptive transfer analysis.

**Colorectal Visium** (Valdeolivas et al. 2024; Zenodo 7760264).
- The four patients not analyzed before, one section per patient, chosen by the pilot's smallest-archive rule.
- Target: pathologist spot labels as released.
- The pathology archive holding label files for all patients was downloaded on 11 Oct 2026 for the pilot. The pilot code extracted only its three sections, and the new patients' count matrices were never downloaded.

**Breast Visium** (Wu et al. 2021; Zenodo 4739739; 6 patients).
- **All labels:** the released per-spot classification.
- **Tumor/non-tumor:** classes whose names contain "cancer", "DCIS" or "Invasive" against all others.
- The class vocabularies were seen in a schema-level metadata inspection on 17 Nov 2026; no scores were computed.
- Wu replaces the LIBD hippocampus cohort. That cohort's released domains are expression-derived (PRECAST clusters), and its manual annotation covers only a subset of capture areas.

### 3.3 Excluded and bounded

- **Excluded from independent tests:** the whole-brain atlas, used in an earlier development panel, whose registration used cell-type landmarks.
- **Bounding.** At most 6,000 observations per unit: whole frames are visited in sha256 order of the frame identifier and admitted while the total stays at or below 6,000; a single oversized first frame is subsampled with RandomState(0). The rule reads no labels.

## 4. Construction-scale experiment (WP1b)

**Rebuilt targets.** For each NSCLC patient and k_c ∈ {10, 25, 50, 100, 200, 400}, capped at frame size − 1:
- compute each cell's frequencies of the supplied cell types among its k_c nearest other cells in the frame;
- cluster them by K-means (10 initializations, random state 0) at the patient's region class count;
- the result is the rebuilt target.

On each rebuilt target, section 2 is computed, with Tessera-ST scored from its saved embeddings.

## 5. Arms and controls (WP3, WP4)

- **Parameter-free counterparts on each encoder's own graph:**
  - the same-graph mean;
  - P^L x, with P the row-normalized incoming-mean operator and L the encoder's message-passing depth (Tessera-ST 3; GraphST and STAGATE as recorded by their adapters);
  - L ∈ {1, 2, 4, 8} as a profile.
- **Tessera-ST arms:**
  - released;
  - no gate;
  - no contrastive term;
  - self-only;
  - permuted coordinates without the contrastive term;
  - spatial positives without message passing (self-edge forward pass, contrastive term on the six-neighbor multiset with uniform weights).
- **GraphST** (common PCA input primary, native preprocessing secondary):
  - released;
  - tuned: graph {3, 6, 15, 30} × epochs {600, 1200}, with the patient-excluded rule of the round-1 controls;
  - no contrastive term;
  - identity encoder adjacency with the spatial readout kept;
  - permuted coordinates.
- **STAGATE:**
  - released;
  - tuned: graph {default radius, 6, 15, 30} × epochs {500, 1000};
  - uniform attention;
  - identity graph;
  - permuted coordinates.
- **Seeds:** 3 for grids; 5 for the selected configuration and every arm.
- **Clustering:** K-means at K is primary, and each control is selected under the rule it is scored with. Leiden (15-neighbor graph, resolution bisected to K) is secondary, with Leiden-selected controls. Silhouette-selected K is a sensitivity analysis.
- **Where the tuned configuration is selected:** within each new cohort, on its other units. The colorectal selection also uses the three pilot patients.

## 6. Inference

### 6.1 Units, margins and tests

- **Unit.** The patient or animal. Seeds are averaged within a unit, and same-seed GPU repeats estimate computational noise.
- **SESOI (smallest effect size of interest).** 10% of the cohort's spatial headroom (mean over units of the best of the 30 label-selected controls minus PCA50), floored at 0.01 ARI. Calibration values: NSCLC regions 0.029, BCC cell types 0.010, HER2ST all labels 0.010, HER2ST tumor/non-tumor 0.012. New cohorts use the same formula.
- **Decision for a difference d (encoder minus counterpart), from a 95% t-interval over units:**

| Interval position | Verdict |
|---|---|
| Lower bound > +SESOI | Encoder superior |
| Upper bound < −SESOI | Counterpart superior |
| Otherwise | No resolvable difference (interval reported) |

### 6.2 Power and multiplicity

**Power.** At the planned units, the calibration spread of Tessera-ST minus its same-graph mean (and minus P³x) gives these expected 95% half-widths:

| Planned units | Expected half-width |
|---|---|
| Spinal cord, 10 mice | 0.060 |
| Colorectal, 4 new patients | 0.133 |
| Colorectal, all 7 patients | 0.077 |
| Wu breast, 6 patients | 0.088 |
| Pooled new cancer units, 10 | 0.060 |
| Pooled new units, 20 | 0.039 |

Every half-width exceeds every SESOI, so **equivalence at SESOI is not reachable with these units and is not claimed**. P3 can establish superiority of either side, and otherwise reports intervals.

**Multiplicity.** Holm correction within P1 (four comparisons) and within P3 (three encoders × two counterparts, per target class).

## 7. Primary predictions, tested on the new cohorts

### P1. Diagnostic scale rule (rule chosen on the calibration targets; section 10)

- **When to smooth.** Smooth if the neighborhood information gain exceeds 0.05; when the gain is not estimable, smooth if h_adj(6) exceeds 0.5.
- **At what scale.** The k-grid value nearest max(ℓ_k, 6). Censored lengths map to the largest available k.
- **Otherwise** no smoothing (PCA50).
- **Regret** = best attainable ARI over PCA50 and the k-neighbor means, minus the ARI at the rule's choice. Unit-level regret is the mean over the unit's targets.
- **Test.** Over all new units, one-sided Wilcoxon signed-rank tests compare the rule with each fixed choice (no smoothing; k = 6; k = 30; k = 200), Holm-adjusted. P1 holds if all four adjusted p-values are below 0.05.

### P2. Construction scale (section 4, on the five NSCLC patients)

- (i) ℓ_k rises with k_c, with Spearman ρ ≥ 0.9 in at least 4 of 5 patients.
- (ii) k* rises with k_c, with ρ ≥ 0.8 in at least 4 of 5.
- (iii) For every k_c, the patient mean of Tessera-ST minus the same-graph mean is at most +0.029, the NSCLC SESOI.

### P3. Attribution per encoder

- **Comparisons.** For Tessera-ST, GraphST and STAGATE at their patient-excluded tuned configurations, each encoder is compared with two counterparts on its own graph: the depth-matched propagation and the same-graph mean.
- **Decision.** The rule of section 6, with Holm adjustment.
- **Target classes,** decided separately:
  - region targets: pathologist, anatomical, geometry-derived or computed;
  - cell-type targets: expression-assigned.
- **Units.** The new units are primary. Calibration cohorts are reported descriptively.

## 8. Secondary analyses (descriptive)

- A magnitude model fitted on the calibration targets predicts g* in the new datasets. It is compared with two baselines: provenance class only, and K plus class balance only.
- Per-class F1 after Hungarian matching.
- Agreement within two median neighbor distances of a label boundary.
- Tumor-margin bands (HER2ST, colorectal, Wu), with the edge-weighted mean as an edge-aware control.
- A target-conditional label-free scale rule (unsupervised clusters at silhouette-selected K, with a no-smoothing option). An earlier development proxy built from unsupervised-cluster contiguity had the opposite sign, and this is disclosed.
- Leiden replication of P3.
- Spinal transfer from sham to SNI.
- Ten-seed Tessera-ST refits on the bounded calibration objects, as computational noise.

## 9. Order of operations

1. Commit this file ("Freeze PROTOCOL-R3").
2. Optional public deposit (author's decision).
3. Download, inspect and prepare the new cohorts.
4. WP1 diagnostics, WP3 arms and WP4 encoders on the new cohorts.
5. WP1b.
6. Run `analyze_r3.py`, which evaluates P1–P3 exactly as written.

## 10. Calibration record (seen data; exploratory)

Source: results/revision-r3/W1, W2, W3 and freeze.

- **Diagnostics against label-selected gain (45 targets).** h_adj(6) ρ = 0.76; label informativeness 0.74; information gain 0.74; κ_self −0.47; ℓ_k 0.32. No diagnostic predicts Tessera-ST minus the same-graph mean (|ρ| ≤ 0.3).
- **Regret on calibration targets** (mean / median):

| Choice | Mean | Median |
|---|---|---|
| P1 rule | 0.041 | 0.013 |
| No smoothing | 0.094 | 0.043 |
| k = 6 | 0.071 | 0.046 |
| k = 30 | 0.075 | 0.052 |
| k = 200 | 0.107 | 0.104 |

  The thresholds (0.05; fallback 0.5) were chosen on these targets.
- **Control selection under each rule** (Tessera-ST minus the selected control, Leiden-selected and Leiden-scored): NSCLC −0.154 (4 of 5 below), HER2ST all labels −0.049 (5 of 8), BCC +0.027 (2 of 4), HER2ST tumor/non-tumor +0.051 (2 of 8).
- **Spatial positives without message passing.** The median of Tessera-ST minus the arm is −0.007 over 45 targets. On NSCLC regions the arm recovers 56–89% of Tessera-ST's gain over PCA50.
- **Depth-matched propagation (L = 3).** The median of Tessera-ST minus P³x is −0.017; Tessera-ST is below it on 25 of 45 targets.

## 11. Deviations

None.
