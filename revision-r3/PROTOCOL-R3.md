# Round-3 revision protocol: annotation diagnostics, matched attribution on three encoders, and prospective tests

**Status: DRAFT.**
- **When it freezes:** this file is frozen by the commit titled "Freeze PROTOCOL-R3" and, if the author approves, deposited publicly before any new cohort is downloaded.
- **What it covers:** sections 1–4 define quantities already used in the calibration analyses on the 24 existing datasets (WP1, WP1b, WP2).
- **Values filled at the freeze:** the effect-size margin and the power check take numbers that are written into section 6 at the freeze.
- **Changes afterwards:** a change after the freeze is a dated deviation in section 9.

## 1. Questions

1. **Annotation diagnostics.** How strongly does a reference annotation's construction favor smoothing? This is measured from the labels and coordinates, plus the observations' own expression for one diagnostic.
2. **Prediction.** Do those diagnostics predict, before any model is fitted, whether smoothing helps a target and at what neighborhood scale?
3. **Attribution.** For three encoders with different mechanisms (Tessera-ST, GraphST, STAGATE), is the gain over expression-only clustering equivalent to, or below, a parameter-free propagation on the encoder's own graph and depth?
4. **Components.** Is the gain removed with the component that couples neighbors, and is it unchanged when gates or attention are removed?

## 2. Label diagnostics (code: `diagnostics_r3.py`)

All pairs are within one coordinate frame. Labels are integers, with −1 meaning missing.

### 2.1 Neighbor lists

- The k nearest other observations within the frame; the query is removed by index.
- Lists are never padded. A scale k is reported for a dataset only if frames with at least k + 1 observations hold 90% or more of its labelled observations.
- Scales are k ∈ {6, 10, 15, 30, 50, 100, 200}.

### 2.2 Edge homophily, adjusted homophily and label informativeness

- **Edges.** Directed edges run from each labelled observation to its labelled neighbors among the first k.
- **Edge homophily** h_edge is the fraction of edges whose endpoints share a label.
- **Chance term.** Within each frame, take class shares p_c over endpoint labels of the symmetrized edge multiset; these are degree-weighted shares (Platonov et al.). The frame's chance is Σ_c p_c². The dataset chance is the edge-weighted mean over frames.
- **Adjusted homophily.** h_adj = (h_edge − chance) / (1 − chance).
- **Label informativeness (LI).** I(y_ξ; y_η) / H(y_ξ) over the endpoint labels of a random edge of the symmetrized multiset, conditioned on frame (an edge-weighted mean of I over an edge-weighted mean of H).

### 2.3 Correlation length

- **Rank shells.** Neighbor ranks are grouped into shells {1}, {2–3}, {4–6}, {7–10}, {11–15}, {16–30}, {31–50}, {51–100}, {101–200}. A shell is used only if 90% of labelled observations have that many neighbors in their frame.
- **Excess per shell.** (same-label fraction − chance) / (1 − chance), where chance is the mean, over the shell's pairs, of the anchor frame's Σ_c π_c², with π_c the frame's class frequencies.
- **Length in neighbors.** ℓ_k is the rank at which the excess first falls to 1/e of the first-shell excess, interpolated linearly in log rank between geometric shell midpoints.
  - If the excess never falls that far, ℓ_k is censored at the last shell.
  - If the first-shell excess is below 0.05, ℓ_k = 0 (no spatial coherence).
- **Length in distance.** ℓ_d1 is the matching median pair distance in units of the dataset's median nearest-neighbor distance d1. ℓ_µm = ℓ_d1 · d1 · (µm per coordinate unit), reported only where the platform pitch is documented: Visium 100 µm, Spatial Transcriptomics 200 µm, and new cohorts whose coordinates are stated in µm.

### 2.4 Own-expression predictability

- **Classifier.** Multinomial logistic regression (C = 1, class-balanced, standardized features fit on the training fold).
- **Folds.** Spatially blocked 5-fold cross-validation. Square blocks have side 10·d1 per frame, and blocks are assigned to folds round-robin after a seeded shuffle (seed 0).
  - With fewer than 5 blocks but at least 3 frames, frames are the folds; otherwise the quantity is not estimable.
  - Random 5-fold values are reported alongside as a leakage-prone upper bound.
- **Reported:** balanced accuracy, its chance level (1 / number of classes present) and Cohen's κ (`kappa_self`).
- **Neighborhood information gain.** The same classifier with the k-neighbor mean appended; κ_k − κ_self. Report its maximum over k and the k at which it occurs.

### 2.5 Smoothing outcomes

All outcomes use the shared 50-component input, K-means with 10 initializations at the reference class count K, seeds 1–3, and ARI on labelled observations. Each value is the mean over seeds.
- **PCA50.**
- **Directed k-neighbor means** for each k, using the frozen neighbor rule of the round-1 controls.
- **Same-graph mean** on the Tessera-ST graph.
- **Tessera-ST,** from saved 120-epoch embeddings.

Derived quantities:
- **Deployable gain** g6 = ARI(same-graph mean) − ARI(PCA50).
- **Label-selected gain** g* = max_k ARI(k-mean) − ARI(PCA50), with k* the arg max.
- **Tessera-ST minus same-graph mean.**

Recomputed values with saved round-1 counterparts must equal them to within 1e-9 before anything is written.

## 3. Datasets, units and roles

- **Existing calibration datasets (24).**
  - 5 NSCLC patients, with regions (computed niches) and cell types (expression-assigned).
  - 4 BCC patients (expression-assigned cell types).
  - Breast Visium (dominant category), colorectal TVA (expression-assigned subpopulations).
  - MIBI and Open-ST: development data.
  - 8 HER2ST patients with three pathologist targets: all labels, named compartments, tumor/non-tumor.
  - Three anatomical tissues: development data.
  - The 3-patient colorectal pilot is reported separately.
- **New core cohorts,** downloaded only after the freeze:
  - **Mouse spinal cord MERFISH** (Zenodo 18039571; 10 healthy mice, plus sham and SNI animals). The unit is the mouse; the frame is the section. Targets come from one pre-specified branch:
    - **A:** per-cell lamina labels deposited, which are then used directly;
    - **B:** per-cell relative depth from the authors' traced dorsal surfaces deposited, which is binned into laminae at the boundaries the paper reports;
    - **C:** neither, in which case the deposited level-1 and level-2 cell-type labels are the targets and laminae are not constructed.

    In every branch the cell-type labels are targets.
  - **Colorectal Visium** (Zenodo 7760264): the 4 patients not analyzed before, one section per patient, chosen by the pilot's rule (smallest archive). Pathologist spot labels. Reported separately from the 3 pilot patients.
  - **Breast Visium** (Wu et al. 2021, Zenodo 4739739): 6 patients with per-spot pathologist classification. It replaces the LIBD hippocampus cohort, whose released domains are expression-derived (PRECAST) and whose manual annotation covers only a subset of capture areas.
- **Excluded from independent tests:** the whole-brain atlas, used in an earlier development panel; its registration also used cell-type landmarks.
- **Bounding.** Each dataset holds at most 6,000 observations, chosen by a label-blind rule recorded with the input: whole frames in a fixed hash order, then a seeded subsample of a single oversized frame.

## 4. Construction-scale experiment (WP1b; designed before it is run)

- **Rebuilt targets.** For each NSCLC patient and k_c ∈ {10, 25, 50, 100, 200, 400}, capped at frame size − 1, compute each cell's frequencies of the supplied cell types among its k_c nearest neighbors within the frame. Cluster them by K-means (10 initializations, random state 0) at the patient's region class count. The result is the rebuilt target.
- **Measured on each rebuilt target:** the section 2 diagnostics, g*, k*, and Tessera-ST (saved embeddings, seeds 1–3) minus the same-graph mean.
- **Predictions:**
  - (i) ℓ_k rises with k_c (Spearman ρ ≥ 0.9 in at least 4 of 5 patients);
  - (ii) k* rises with k_c (ρ ≥ 0.8 in at least 4 of 5);
  - (iii) at no k_c does Tessera-ST minus the same-graph mean exceed the margin of section 6.

## 5. Arms and controls (WP3, WP4)

- **Parameter-free counterparts on each encoder's own graph:**
  - the same-graph mean;
  - propagation P^L x, with P the row-normalized operator of that graph and L the encoder's message-passing depth (Tessera-ST 3; STAGATE and GraphST as implemented);
  - a scale profile with L ∈ {1, 2, 4, 8}.
- **Tessera-ST arms:**
  - released settings;
  - no gate;
  - no contrastive term;
  - self-only;
  - permuted coordinates without the contrastive term;
  - spatial positives without message passing: the forward pass on self edges, with the contrastive term on the six-neighbor multiset at uniform weights.
- **GraphST** (common PCA input primary; native preprocessing secondary):
  - released;
  - tuned: graph {3, 6, 15, 30} × epochs {600, 1200}, with the patient-excluded selection of the round-1 controls;
  - no contrastive term;
  - identity encoder adjacency, with the spatial readout kept;
  - permuted coordinates.
- **STAGATE:**
  - released;
  - tuned: graph {default radius, 6, 15, 30} × epochs {500, 1000};
  - uniform attention;
  - identity graph;
  - permuted coordinates.
- **Seeds:** 3 for tuning grids; 5 for the selected configuration and for every arm.
- **Clustering:**
  - Primary: K-means at K.
  - Secondary: Leiden, with a 15-neighbor graph and resolution bisected to K.
  - Sensitivity: silhouette-selected K.

## 6. Inference

- **Unit.** The patient or animal. Seeds are averaged within a unit before any test, and same-seed GPU repeats estimate computational noise.
- **Smallest effect size of interest (SESOI).** 10% of the cohort's spatial headroom: the mean over units of the best label-selected control minus PCA50, with a floor of 0.01 ARI. Sensitivity analyses use 0.02 and 0.05 ARI.
- **Equivalence.** Two one-sided tests at α = 0.05 on unit-level differences, with cohort as a fixed effect. A superiority claim needs the lower 95% bound above +SESOI.
- **Multiplicity.** Holm correction across the three primary predictions.
- **Power.** Unit-level bootstrap of the calibration datasets at the planned unit counts gives the expected interval half-widths; tests that cannot reach SESOI are declared descriptive in advance.
- **Values fixed at the freeze:** the SESOI per cohort and the power table.

## 7. Primary predictions on the new cohorts (frozen at the freeze)

- **P1. Scale prediction.** For every target in the prospective cohorts, predict k̂ as the scale grid value nearest ℓ_k, or "no smoothing" when ℓ_k = 0. Success requires |log2 k̂ − log2 k*|, or both no smoothing, within 1 in at least 80% of units, for each target type.
- **P2.** The construction-scale predictions of section 4.
- **P3. Attribution per encoder.** Encoder minus its depth-matched propagation, and encoder minus its same-graph mean, both judged by the equivalence rule of section 6, reported separately for each encoder and each target type.

## 8. Secondary analyses (descriptive)

- A magnitude model fitted on the 24 calibration datasets predicts g* in new datasets. It is compared with two baselines: provenance class only, and K plus class balance only.
- Per-class F1 after Hungarian matching.
- Agreement within two median neighbor distances of a label boundary.
- Tumor-margin bands, with the edge-weighted mean as an edge-aware control.
- A target-conditional label-free scale rule. An earlier development version of a label-free proxy had the opposite sign, and this is disclosed.
- Leiden replication.
- Spinal transfer from sham to SNI.

## 9. Deviations

None yet.
