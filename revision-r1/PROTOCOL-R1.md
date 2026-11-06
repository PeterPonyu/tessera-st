# Revision R1 analysis protocol

Locked before any revision fit. This protocol specifies the analyses added in response to the journal review of the controls audit. The frozen estimator, preprocessing, graph rule, control grid and scoring code are imported unchanged; no existing result is modified. New outputs are written only under the project directory `results/revision-r1/`.

## Environment and devices

- GPU fits: local NVIDIA RTX 5090 Laptop GPU, PyTorch 2.12.1 with CUDA 13.0, scikit-learn 1.6.1. The original CUDA 12.1 host was unreachable when this protocol was locked, so every GPU comparison is made against a 120-epoch reference refitted in this same environment, never against the original table values.
- HER2ST breast fits: CPU with one thread (`torch.set_num_threads(1)`), so that changes of cluster-head size and training budget are not confounded with device or thread nondeterminism.
- Seeds 1, 2 and 3 everywhere. K-means uses ten initializations and `random_state` equal to the seed, as before.

## Training core

`train_core.py` re-implements the frozen training loop with the training budget, graph neighbours, contrastive weight, contrastive temperature, cluster-head size and evaluation checkpoints exposed. At the original settings it must reproduce the frozen implementation exactly on CPU with one thread (test `test_core_matches_frozen`). A checkpoint taken at epoch 120 within a longer run must equal a 120-epoch run (test `test_checkpoint_equals_short_run`).

## E1. Training budget

- Objects: the 13 bounded objects (frozen derived inputs), the 8 HER2ST patients (frozen loader and preprocessing; fixed two-cluster head) and the 3 anatomical sections (frozen breadth loader).
- Arms: Tessera, NoGate, NoContrast, SelfOnly and PermutedXY-NoContrast, each trained to 400 epochs.
- Checkpoints: epochs 40, 80, 120, 160, 200, 300 and 400 (embedding, last-layer gates, losses).
- Readouts per checkpoint: ARI and NMI at the primary cluster number; seam AUROC of one minus the last-layer gate where gates exist. For HER2ST the all-label target is primary and compartments and tumor boundary are secondary. For lung, cell-type ARI at the cell-type cluster number is secondary.
- Contrasts: against PCA50, the same-graph mean and the patient-excluded tuned control (deterministic; recomputed here and checked against the saved values), at (i) 120 epochs, (ii) 200 epochs, (iii) the epoch selected on the other patients of the cohort, and (iv) the epoch selected on the test labels (an optimistic bound, never used for inference).
- E1b: leave-one-patient-out NSCLC transfer with the source-only transform, Tessera at 120 and 200 epochs, three seeds, nearest source-centroid readout as before.

## E2. Patient-excluded tuning of Tessera

- Grid (24 configurations): graph neighbours k in {6, 15, 30}; contrastive weight in {0.5, 1.0}; contrastive temperature in {0.2, 0.5}; epochs in {120, 200}, with 120 read from a checkpoint of each 200-epoch run. The released code uses one edge set for message passing and contrastive positives, so any effect of k is a joint effect of both.
- Cohorts and targets: NSCLC regions (5 patients), BCC cell types (4), HER2ST all labels and tumor boundary (8; fixed two-cluster head, CPU, one thread).
- Selection: for each held-out patient, the configuration with the highest mean ARI over the other patients of the cohort (seed means). The held-out patient's scores are never read during its own selection (test `test_selection_excludes_held_out`).
- Reported: nested Tessera minus the nested control (primary); nested Tessera minus the same-graph mean built on the selected k; nested Tessera minus the released configuration refitted here (tuned versus untuned); NoGate and NoContrast at each fold's selected configuration; a matched-scale control selected from the saved control grid restricted to k at most 30; the label-selected best of 24 Tessera configurations against the label-selected best of the 30 saved control configurations (descriptive; the larger and noisier grid gains more from label selection); and the configuration selected in each fold.

## E3. Cluster number and clustering geometry with fixed representations

- Representations, each built once and reused for every K: PCA50; the same-graph mean; each patient's patient-excluded control as selected at the primary K; the saved Tessera seeds (bounded objects) or the 120-epoch fixed-head fits (HER2ST); and the E1 Tessera embeddings at 200 epochs.
- Targets: lung regions and lung cell types; BCC cell types; breast Visium, colorectal TVA, MIBI and Open-ST; HER2ST all labels, named compartments and tumor boundary; anatomical sections (secondary).
- K grid: every integer from 2 to max(2 K_ref, 10), capped at 40.
- Clustering rules: Euclidean K-means and cosine K-means (rows scaled to unit norm) are co-primary. Leiden on a 15-neighbour graph of each representation, with resolution bisected to each K within K_ref plus or minus 2, is secondary (lung regions and HER2ST all labels).
- Shared label-free K per object and target: the K maximizing the mean silhouette across the fixed representations (Euclidean, Euclidean K-means partitions at seed 1, at most 3,000 observations sampled with seed 0), searched over 3 to 2 K_ref, or 2 to 6 for two-class targets. Per-representation choices are reported descriptively.
- Breast harmonization: refit Tessera on CPU with one thread both with the original per-K cluster head and with the fixed two-cluster head, in this environment; all targets and K are clustered from the fixed-head embedding. The harmonized analysis is primary; any patient whose sign against the tuned control differs between the two head sizes is listed with both values.
- Secondary sensitivity: the released majority-vote label refinement (six neighbours, two rounds, frame-restricted) applied identically to every arm's partition at K_ref.
- Tests: representations are byte-identical across K (`test_representations_fixed`); Euclidean K-means on the saved PCA50 embedding reproduces the saved primary ARI (`test_pca50_reproduces`); on CPU the cluster-head size changes the fitted embedding, and on CUDA it does not (`test_head_size_effect`).

## E4. Native external methods

- Required: GraphST and BANKSY in their native implementations. Best effort: CASTLE. Each method uses its documented preprocessing on the same retained cells and log-normalized expression; coordinates are offset per frame so that no graph edge crosses a frame (test `test_no_cross_frame_edges`). On NSCLC, CASTLE is run with the supplied cell types and with its expression-similarity fallback; the cell-type-aware run is partly circular because the region labels were built from those cell types.
- Clustering: matched K-means at K_ref on each method's embedding and each method's native clustering at K_ref are co-primary.
- Cohorts: NSCLC regions and cell types; HER2ST all labels and tumor boundary. Contrasts against PCA50, the same-graph mean and the tuned control.
- A method that cannot be run by the stated cut-off is reported as not evaluated, with the reason. Third-party code is installed from source with its commit recorded and is not copied into this repository.

## Statistics

Patients are the inference unit. Paired patient contrasts are summarized by their mean, the number of patients above and below zero, exact two-sided sign-flip p-values and, for the eight-patient cohort, 10,000-draw patient percentile bootstrap intervals with a fixed seed. With four or five patients the smallest attainable p-value is 0.125 or 0.0625, so these values are descriptive. A relevance margin of 0.02 ARI is used throughout. A majority means more than half of the cohort's patients.

## Decision rules and pre-written wording

Wording in brackets is filled from the ledgers; outcomes are reported whatever they show.

DR-1 (ordering under balanced tuning; E2 nested). For each cohort and target (NSCLC regions, BCC cell types, HER2ST all labels, HER2ST tumor boundary) and each comparator (patient-excluded control; same-graph mean on the selected k):
- Gap retained if the mean contrast is at most -0.02 and a majority of patients are below zero: "After the same patient-excluded tuning, the neighborhood control still exceeded Tessera-ST (mean difference [d]; [n] of [N] patients)."
- Reversed if the mean contrast is at least +0.02 and a majority are above zero: "After patient-excluded tuning, Tessera-ST exceeded the neighborhood control (mean difference [d]; [n] of [N] patients), so the averaging account applies to the released configuration only."
- Gap closed otherwise: "Patient-excluded tuning of Tessera-ST closed the gap to the neighborhood control (mean difference [d]), whereas the released configuration had not."
- Mechanism sentence, always: "At the selected configurations, removing the gates changed agreement by [g] and removing the contrastive term by [c]." If a closure or reversal occurs only at k of 15 or more and NoGate matches Tessera at those configurations, the gain is attributed to smoothing and contrast, not to the learned gates.
- E1 and transfer results are reported beside DR-1 and do not trigger it.

DR-2 (cluster number). "Robust to cluster number" is written for a cohort only if, under both Euclidean and cosine K-means, the mean contrast against the tuned control keeps its K_ref sign with a sign majority at every K within K_ref plus or minus 2 and at the shared label-free K: "The ordering held at every cluster number within two of the reference class count and at the label-free cluster number, under both Euclidean and cosine clustering." Otherwise: "The ordering held for [K range] under [rule]; outside it [describe]."

DR-3 (training budget). Paired 200-epoch minus 120-epoch Tessera ARI within each cohort. If the mean absolute change is at least 0.02 with a sign majority, the Abstract states the sensitivity: "Training to the released default of 200 epochs changed agreement by [d] ([n] of [N] patients)." Otherwise: "Training to the released default of 200 epochs changed agreement by [d] on average, within the 0.02 margin."

DR-4 (native methods). GraphST or CASTLE is described as also accounted for by averaging on these cohorts only if, under both co-primary clustering rules, it falls below the tuned control by at least 0.02 in a majority of lung and of HER2ST patients: "Native [method] also fell below the tuned neighborhood control in [n] of 5 lung and [m] of 8 breast patients under both clustering rules." Otherwise: "Native [method] was not consistently below the tuned neighborhood control; the averaging account is shown for Tessera-ST only [or: depends on the clustering rule]."

## Outputs

Each analysis writes a run lock (this protocol's SHA-256, code hashes, environment), one JSON record per fit and tidy CSV tables under `results/revision-r1/<E#>/`. Manuscript numbers are generated from these CSV tables only.

## Amendment A1 (5 November 2026): wider encoder graphs in E2

Written after the E2 results and an independent critique had been seen, and before any of the fits below. Reason: every NSCLC fold of E2 selected the largest graph offered to the encoder (k = 30), whereas the neighbourhood controls could use up to 200 neighbours, so the E2 grid itself may have limited the encoder.

- Extension: k in {50, 100}, crossed with the E2 contrastive weights, temperatures and epochs (16 further configurations; 40 in total). k = 200 was attempted and does not fit in the 24 GB of GPU memory for a 6,000-cell lung section with the released training code (full-batch training with ten contrastive negatives per edge); the training code is not changed to make it fit.
- Cohorts and devices: NSCLC regions on CUDA; HER2ST all labels and tumor boundary on CPU with one thread and the fixed two-cluster head, as in E2. BCC is not extended, because larger neighbourhoods lowered BCC cell-type agreement for every control family in the original analysis.
- Selection, comparators and wording: as in E2 and DR-1, applied to the 40-configuration grid. The matched-scale control is selected from the saved control grid restricted to k at most 100. NoGate and NoContrast are fitted at each fold's selected configuration, reusing E2 fits where the configuration is unchanged.
- Status: exploratory, because it was specified after the E2 results were known. The pre-specified 24-configuration results remain the primary E2 report, and the extension is reported beside them.
