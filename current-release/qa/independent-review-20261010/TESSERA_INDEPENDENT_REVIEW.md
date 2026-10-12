**VERDICT: MAJOR REVISION.** The study design is honest and much better controlled than most method papers. But the manuscript under-reports what its own data show, misreads the permutation control, and confounds the target type with the dataset. It also leaves out spatial baselines that already sit on this machine.

# Review: "Evaluating Tessera in tumor-associated spatial tissues"

## 1. Summary

The paper asks when a fixed, previously released gated graph encoder (Tessera) improves unsupervised partition agreement over expression-only and simple spatial controls. It uses 13 objects: 5 NSCLC CosMx patients (target: niche "regions"), 4 BCC CosMx patients (cell types), breast Visium (dominant Xenium category), colorectal TVA (34 lv2 subpopulations), MIBI and an Open-ST lymph-node subset. Each object is capped at 6,000 cells. Nine arms × 3 seeds all receive the same PCA50 input and are scored with KMeans at the reference K using ARI. Lung and BCC differences are summarised by a patient-level bootstrap. A five-fold NSCLC leave-one-patient-out transfer compares PCA50, self-only and Tessera using nearest source centroids.

Main claims:
- Tessera beats PCA50 on lung regions (ΔARI +0.097, CI +0.016 to +0.158) but ties the neighbour mean (+0.002).
- Tessera loses on BCC (−0.026), breast, HNSCC and MIBI, and wins on TVA (5/13 objects favour it).
- The benefit is target-specific, and region transfer improves while cell-type transfer worsens.

## 2. Logic chain audit

**Question → hypothesis.** No falsifiable hypothesis about Tessera is ever stated. The code shows what the components were built to do: the gate is the "anti-over-smoothing core… a gate near 0 means this edge crosses a boundary" (`source/src/tessera_st/model/gating.py`), and the earlier project's `CLAIM_LEDGER.md` C1–C4 were gating, fusion, contrastive and calibration claims. The paper never says which of these it is testing. Instead it lists ~100 negative disclaimers ("not calibrated", "not pathology"). The Introduction should state the component hypotheses and the decision rule. Results can then say plainly which ones failed.

**Wrong primary comparator.** The abstract leads with Δ vs PCA50, the weakest control. On lung regions PCA50 + KMeans scores near chance (0.039–0.059 in 3/5 patients), so any smoothing "wins". The scientifically relevant contrast, Tessera vs neighbour mean, is +0.002 (CI −0.050 to +0.039) for NSCLC. For BCC it is −0.008 (−0.037 to +0.028), a number that sits in `ledgers/manuscript_numbers.json` but appears nowhere in the paper. Both contrasts should be co-primary.

**Circular lung target, never tested at matched scale.** The regions are Mclust on cell-type frequencies over 200 neighbours, computed on the whole section (§2.1). All spatial arms use 6-NN, and no arm uses the 200-NN scale or composition features. The paper acknowledges the circularity but still uses the target as its headline positive result. A composition (CellCharter/MENDER-style) control at k≈200 would show directly how much of this target any method can recover "for free".

**"Target dependence" is confounded with dataset.** Region-vs-cell-type is perfectly aligned with dataset, panel (960 vs 3,000 features), K (5–7 vs 11–12) and tissue. The only de-confounded evidence is the within-lung swap in Supp. Fig. S4: the same predictions scored against cell types. There PCA50 averages 0.312 and Tessera 0.274, and Tessera is worse in 3/5 patients. This analysis belongs in the main text, but as it stands it is mis-specified: it scores K=5–7 partitions against 21–22 cell types (`run_experiments.py`, `scores(... secondary)`). It must be re-clustered at the cell-type K.

**Ablation findings the text does not state** (Tables S1–S2, `all_primary_scores.csv`):
- **No-gate ≥ full Tessera in all 5 lung patients** (mean +0.019; Lung5 0.288 vs 0.231). No-fusion is also ≥ full in 4/5.
- **Gate seam AUROC is at or below chance**: Lung5 0.496, Lung6 0.385, Lung9 0.475, Lung12 0.495, Lung13 0.548, breast 0.467, TVA 0.444, MIBI 0.420. Fig. 6(c) labels this "descriptive gate separation"; the data show none.
- **Gates have collapsed.** Fig. 14(b, right) shows nearly all last-layer gates at 0–0.05 with identical same- and different-region distributions. Under the normalised operator (Eq. 7) this is close to an ungated mean.
- **Message passing alone adds nothing.** In lung, no-contrast ≈ self-only ≈ PCA50 (Lung5: 0.060 / 0.062 / 0.059). The entire lung gain comes from the neighbour-positive contrastive term.

Taken together, the namesake component is inert or harmful and boundary claim C1 is falsified. That is the paper's clearest mechanistic result. The abstract instead says only that outcomes "depend on the target and the use of geometry" — an underclaim of a negative finding.

**Permutation control misread** (abstract; §3.8; Fig. 13(b) "Effect of observed geometry"). Permuting XY while keeping the contrastive loss turns the positive pairs into random cell pairs. The embedding collapses: permuted seed-pair ARI is 0.01–0.28, and breast ARI drops to 0.025. "Measured minus permuted > 0" therefore shows that random positives are harmful, not that measured geometry helps. The correct reference is self-only. Against it, Tessera is *worse* in all 4 BCC patients and on breast (0.154 vs 0.198), MIBI and HNSCC. In other words geometry hurts in 8/13 objects, which is the opposite of what Fig. 13(b) suggests.

**Lung6 "negative case" is a sampling artefact.** Lung6 keeps two FOVs that are 98% "tumor interior" and 97% "macrophages". The region task reduces to telling two FOVs apart, with K=5 forced by classes of 68, 62 and 5 cells. Fig. 7(b) shows KMeans splitting a homogeneous field. Lung5's K=7 likewise includes a single-cell "neutrophils" class. The paper discusses none of this.

**Transfer.** There is no neighbour-mean arm, so the region gain cannot be attributed to Tessera. Region accuracy is lower for Tessera in L9 (0.181 vs 0.226) and L12 (0.180 vs 0.212), and the mean is driven by Lung6 (+0.217). The robust result is the opposite one: Tessera is worse on cell-type accuracy *and* balanced accuracy in 5/5 patients (Table S5). It should lead the transfer section.

**Abstract/Discussion support.** The claim that "positive lung and TVA cases… establish the task-specific behavior of the implemented model" (§4, last paragraph) is partly unsupported:
- The lung gain belongs to smoothing.
- TVA is the only object where Tessera beats the neighbour mean by a margin (0.149 vs 0.114). That is one section, K=34, five FOVs (379–2,301 cells), with compartment ARI ≈ 0.03–0.05 for every arm.

A supportable conclusion: "Neighbour-positive contrastive smoothing, not gating, drives agreement with spatially constructed niche labels. It hurts molecular cell-type targets, including under patient transfer."

**Flow.** Several results are introduced without motivation: the entropy–seam scatter (Fig. 10c), ranked cluster sizes (10f), the degree plot (14c), and stability vs agreement (13c). Fig. 13(d) (reference edge homogeneity vs Δ) is the one hypothesis-generating panel, but it is never discussed, and Lung6 (homogeneity 0.992, Δ < 0) contradicts the pattern it implies. §3.1 ("A completed, target-explicit expansion") reports process, not a finding.

## 3. Methods and statistics

**Sampling.** Each lung "patient" is 1–2 FOVs, i.e. 3.9–5.9% of eligible cells (Lung5 5,945/100,335; Lung13 3,853/81,236). BCC C is 5.6% and TVA 4.8%. The 6,000-cell cap is justified as "tractable", yet the paper's own Fig. 14 shows fits of 0.78–1.85 s and ≤0.85 GiB. Full sections (≈100k cells) fit easily on the local RTX 5090 (24 GB). The cap is unjustified, and it is the main threat to the "patient-level" framing.

**Open-ST is doubly thinned.** It is a random 6,000 of an already-subsampled 15,000 (`prepare_data.py`, single-frame `RandomState(0)` branch). The full 73,665-cell section 26 is on disk (`spatial-omics-reform/data/processed/openst_hnscc_sec26_dom`). Random thinning makes 6-NN edges non-physical and biases against every spatial arm. Breast Visium is restricted to 307 genes.

**K = number of observed classes, including singletons.** Classes under 1% prevalence should be excluded from K (or reported both ways), and a K-free metric added. Also note that KMeans runs on unnormalised z although the contrastive loss uses cosine geometry.

**Seeds.** For PCA50 the three seeds vary only KMeans initialisation (seed-pair ARI 0.97–1.00 in lung). For Tessera, seed-pair ARI is only **0.38–0.73** (`seed_stability.csv`), so partitions change substantially between seeds while ARI SDs look small. Three seeds are not enough, and "paired by seed" contrasts pair unrelated sources of randomness.

**Patient inference.** A percentile bootstrap over n=4 or 5 is badly behaved; BCC has only 35 distinct resample multisets. Exact sign-flip tests give two-sided p = 0.125 for both NSCLC (4/5 positive) and BCC (4/4 negative). The minimum attainable p is 0.0625 (n=5) and 0.125 (n=4). The CI excluding zero for BCC (−0.044 to −0.008) is therefore misleading. Report sign counts, exact p-values and Tessera−neighbour-mean contrasts. BCC A (−0.001 against a seed SD of 0.013) is a tie, so "3/4 beyond seed noise" is the honest count.

**Baseline fairness.** The neighbour mean is a single, untuned, self-excluding k=6 average (Eq. 12). No BANKSY-style self+λ·neighbour mix, no radius/k sweep, and no composition baseline. None of BANKSY, GraphST, STAGATE, SpaGCN, CellCharter, SEDR or MENDER is run. The Discussion says they "would require their own validated assay-specific runs", but local runners exist under `/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external/` (`run_banksy_domain.py`, `run_graphst_domain.py`, `run_stagate_domain.py`, `run_spagcn_domain.py`, `run_sedr_domain.py`, `run_niche_cellcharter.py`, `run_niche_banksy_agf.py`). SpaGCN and cellcharter 0.3.7 are installed in the `nfspatial` and `dl` conda envs, and source checkouts of GraphST, STAGATE and SEDR are present. Tessera itself is also untuned (k, epochs, contrastive weight). That is fair only if the baselines are equally untuned, which argues for a nested-selection design for all arms.

**Leakage.** None from labels in the within-object fits beyond oracle K, which all arms share. The target construction is the circularity issue described above.

**Ill-defined or ambiguous items:**
- "Raw ARI" ("raw" means unrefined, not unadjusted).
- Predicted edge homogeneity for self-only is trivially 1.0 on a self-loop graph (Fig. 6d).
- Fig. 7 ARIs are seed-1 values (L5 0.25), while Table 2 reports means (0.231).
- "No fusion is the historical run key NoScale" (Table S2) is an internal key.
- §2.3 refers to a "12-GiB remote device".

## 4. Figures

**Global problems:**
- Orange encodes both "BCC cohort" (Figs 5, 13d) and "Tessera arm" (Figs 10, 12, 13a, 14).
- Panel letters label rows, not plots (Figs 7, 12, 14), so "Fig. 14(b)" refers to two different plots.
- Map ARIs are seed-1 values; Table 2 reports means.
- Every map omits the neighbour-mean arm, the decisive comparator.
- Of 18 figures, 7 are conceptual, cost or inspection figures carrying no quantitative claim.

**Fig 1 (workflow).** Decorative mock plots in (f) look like data. Panels (a) and (e) overlap with Fig. 4. *Redesign:* merge Figs 1 and 4 into one design panel: a dataset × {target type, K, platform, patients, FOVs retained} matrix, the arms, and within-object vs LOPO flows.

**Fig 2 (cohort).** Repeats Table 1 (modelled, frames, classes). The only new information is the PCA variance (7.5–18%; MIBI 1.0). *Redesign:* add "% eligible retained", "FOVs retained/total" and "PCA variance" to Table 1, then drop the figure or move it to the SI.

**Fig 3 (architecture).** Every tensor is the same tile glyph. The gate-normalised operator, which is central to the interpretation, is not depicted, and the in-figure text ("Seam score ≠ calibrated error") is a disclaimer. *Redesign:* a standard block diagram with equation numbers; move to SI if Tessera has been published before.

**Fig 4 (target lineage).** Mostly words and coloured dots. Its claims fit in one sentence. Delete it (merge into Fig 1).

**Fig 5 (comparison).**
- (a): an absolute-ARI sequential heatmap. Between-object variance dominates, so the within-row contrasts that matter are invisible. PCA30 duplicates PCA50, and the column order mixes controls with ablations.
- (c, d): points are unlabelled, and (d) changes comparator (NMI vs PCA) and duplicates Table S3.

*Redesign:* (a) diverging heatmap of Δ vs PCA50 and vs neighbour mean, with PCA30 removed. (b) per-patient paired dot plot showing both contrasts with sign counts and exact p. (c) dumbbell plot of PCA → NM → Tessera per object.

**Fig 6 (ablations).**
- (b), (e) and (f) re-plot the No-contrast, Self-only and No-gate columns of (a): triple redundancy.
- (c) shows AUROC ≈ 0.5 under a neutral title, with objects as numbers 1–13.
- (d) puts self-only at x = 1.0, an artefact of the self-loop graph.
- The footnote "control in (a), full model in (b)" is confusing.

*Redesign:* (a) heatmap; (b) gate AUROC by named object against a trivial |Δx| edge baseline, titled with the finding; (c) gate histograms (moved from Fig 14b) for 3–4 objects.

**Fig 7 (lung maps).** Hungarian matching assigns whole clusters to near-empty classes: Lung5 "neutrophils" (n=1) appears as large teal patches in the Tessera map, and Lung6 "immune" (n=62) and "myeloid-enriched stroma" (n=5) appear as large regions. These are false visual claims. Several of the 10 classes are similar purples. The neighbour-mean column is missing, and the seed is not stated. *Redesign:* add the NM column and an agree/disagree map, grey out classes under 1%, state "seed 1", and discuss the single-region Lung6 field explicitly.

**Fig 8 (BCC maps).** Six panels per row at about 1.1 in wide are illegible in print. The 12-class palette has near-identical oranges and greens, and large markers overplot. Patient B Tessera shows horizontal banding and Patient C a left–right gradient, which may be a learned spatial-gradient artefact; it is not addressed. *Redesign:* two patients in the main text with four columns (Ref/PCA/NM/Tessera); others to SI; add confusion matrices showing which cell types merge.

**Fig 9 (breast H&E).** Row (a) lacks PCA50, and row (b) lacks the reference, so no panel compares prediction with reference in the ROI. Fourteen categories at Visium spot size cannot be read. *Redesign:* an ROI row with Ref/PCA/NM/Tessera, collapsed to about 5 coarse classes; otherwise move to SI.

**Fig 10 (breast mixtures).**
- (a) is useful: PCA beats Tessera at every purity threshold.
- (c), the largest panel, shows a null ρ = −0.09, whose sign is the opposite of what a boundary score should give.
- (d) has no scale bar and no H&E.
- (e) is a bar chart of three numbers.
- (f) is marginal.

*Redesign:* keep (a) annotated with n, plus (b); delete (e); move (c, f) to SI.

**Fig 11 (CRC proteins).** "DAPI protein" is wrong: DAPI is a nuclear stain. Colourbar ranges differ, and the grey scale bars are low contrast. Modal lv2 → compartment mapping inflates apparent agreement. No quantitative claim is attached. *Redesign:* quantify (e.g., PanCK/CD45 per predicted cluster, or compartment ARI) or move to SI.

**Fig 12 (transfer).**
- (c-left) is a panel of zeros.
- There is no NM arm and no paired lines.
- The 0–1 y-axes compress the differences.
- Row labels duplicate panel titles.

*Redesign:* a 2×2 of region and cell-type balanced accuracy as paired per-patient lines, with Δ panels vs PCA and NM; macro-F1 to SI.

**Fig 13 (stability).**
- (a) is informative, but its numbers (Tessera 0.38–0.73) never appear in the text.
- (b) uses numeric object IDs and supports a mistaken inference (see §2).
- (c) is 39 unlabelled points.
- (d) has no colour legend.

*Redesign:* keep (a); replace (b) with measured-minus-self-only; delete (c); label (d) and discuss Lung6.

**Fig 14 (cost/mechanism).**
- The cost panels (~1 s fits) need one sentence, not a figure.
- Mean in-degree is exactly 12 by construction, so (c-left) carries no information.
- Row labels collide with the y-axis labels.
- The gate histogram is the paper's key mechanistic evidence and is buried here.

Move the gate histogram to Fig 6 and the rest to SI.

**S1 (Open-ST).** Shows ARI 0.30/0.23 (seed 1) while the text gives 0.271/0.219. The "26; n=6,000" label is cryptic. The random thinning is visible and should be flagged.

**S2 (MIBI).** No ARI annotations, unlike the other maps. Acceptable as SI.

**S3 (RNA markers).** Magma black-at-zero on white makes zeros dominate. Not linked to any model output. Delete or pair with the partitions.

**S4 (secondary targets).** The most important supplementary figure, because it is the de-confounded target test. Recompute at cell-type K and promote to the main text.

**Recommended consolidation:** about 8 main figures — design, comparison, ablation+gate, lung maps, BCC maps, transfer, stability, breast purity. Move the rest to SI.

## 5. Writing

- **Defensive density.** About 104 negation or disclaimer constructions ("is not", "rather than", "neither") and "genuine" 7× in about 9,000 words. Captions end with audit statements such as "Conceptual glyphs are not synthetic evidence" (Fig 4) and "not measured tissue, third-party logos or new architecture claims" (Fig 1). Delete them; state the limits once, in a Limitations paragraph.
- **Internal process text leaking into the manuscript:**
  - Data availability: "Prior manuscript packages and the Frontiers V3/V9 files are unchanged. Completion of local fits and document QA is not submission approval…"
  - SI: "Scientific submission remains on hold pending explicit author assessment."
  - SI: "33 DOI-bearing references were checked against Crossref", while the reference list has 27 entries.

  Remove all of these.
- **Process-centred titles.**
  - "Actual new model inputs" (Table 1) → "Datasets analysed".
  - "Actual component and geometry controls" (Fig 6) → "Ablations: the contrastive term, not gating, drives lung gains".
  - "A completed, target-explicit expansion" (§3.1) → "Data and targets".
- **Terms used before definition:**
  - In the Abstract: "coordinate-frame identity", "unrefined assignments", "bounded assay objects".
  - In the Introduction: "seam score", "display field", "anchor", "frames", "reference-conditioned".
  - "HNSCC subset" should be "Open-ST metastatic lymph node (HNSCC)".
- **Abstract rewrite example.** "A fixed whole-field selection produces 64,990 observations and 351 method–seed fits. All arms receive the same molecular representation, retain coordinate-frame identity and return unrefined assignments." → "All methods share one PCA input and K-means assignment; we report every arm and seed."
- **Methods details belonging in the main text vs SI.** The last Introduction paragraph (normalised aggregation, untrained soft head) is Methods material. Eqs 25–32 describe display mechanics (histogram densities, ROI boxes, Hungarian colouring, colour clipping), and the SHA256 field ordering is reproducibility detail; all of these belong in the SI. Of the 34 equations, about 12 carry the science.
- **Buried question.** The research question appears in paragraph 4 of the Introduction. Move it to the end of paragraph 1.
- **Missing citation.** The original Tessera work is never cited except as a GitHub/Zenodo link in Data availability.

## 6. Further experiments (prioritised; ★ = top 3 feasible locally now)

**★E1. Spatial-scale and BANKSY-style controls with nested selection.**
- *Hypothesis:* a tuned neighbour mean matches or beats Tessera on lung, and λ→0 (≈PCA) wins on BCC.
- *Design:* k ∈ {6, 10, 15, 30, 50, 100, 200} and radius variants; self-inclusion; BANKSY-style [x, √λ·mean_k(x)] with λ ∈ {0, 0.2, 0.5, 0.8}; composition features (200-NN frequencies of PCA-KMeans clusters, CellCharter/MENDER-like). Select (k, λ) by leave-one-patient-out inside the lung and BCC cohorts, then score the held-out patient.
- *Statistic:* per-patient Δ vs Tessera, sign count, exact sign-flip p.
- *Success / falsification:* the claim that Tessera adds nothing beyond smoothing is supported if nested NM/BANKSY ≥ Tessera in ≥4/5 lung patients.
- *Data:* `data/*/input.npz` (x, coords, field), `results/*/graph.npz`. Reuse `neighbor_embedding` in `source/scripts/run_experiments.py`.
- *Compute / status:* CPU, under 30 min. Offline, now.

**★E2. Re-scoring saved embeddings: K, target swap and gate baseline.**
- *Hypotheses:* (a) the Lung6 reversal is a K/FOV artefact; (b) Tessera loses on lung cell types at the proper K; (c) a trivial |x_i − x_j| edge score beats the gate's AUROC.
- *Design:* re-run KMeans on the saved `results/*/<arm>-seed*.npz` embeddings with (a) K = classes ≥1% prevalence and (b) K = 21–22 against `secondary_name`. Add a K-free metric (best-match F1 or AMI over K ∈ [K−2, K+4]). Compute AUROC for |Δ PCA| and for last-layer gates, and gate-histogram summaries for all objects.
- *Success:* (b) Tessera < PCA in ≥4/5 patients; (c) |Δx| AUROC > gate AUROC in ≥10/13 objects. Either outcome is reportable.
- *Compute / status:* CPU, minutes. Offline, now.

**★E3. Whole-section runs and FOV-resampling stability of the BCC reversal.**
- *Hypothesis:* the signs survive whole-section evaluation and alternative FOV draws.
- *Design:* (i) all eligible cells per lung section (≈80–100k), BCC patient (22k–94k), CRC 232 (123,600) and Open-ST section 26 (73,665; `openst_hnscc_sec26_dom/anndata.h5ad`). (ii) For BCC, 10 disjoint 6,000-cell whole-FOV packs per patient (vary the hash salt in `selection()`) × 5 seeds × {PCA50, NM, Tessera, NoGate}.
- *Statistic:* fraction of packs with Δ < 0 per patient; whole-section Δ per patient.
- *Success:* Δ < 0 in ≥80% of packs for ≥3/4 patients.
- *Data:* `cosmx_lung.zip`, `bcc_export/`, `crc_cosmx/232.h5ad` via `prepare_data.py` with the CAP raised.
- *Compute:* full-batch GNN on 100k nodes and ~1.2M edges is under 2 GB and about 1 min per fit. Roughly 2–4 GPU-h for (i) and about 20 min for (ii).
- *Status:* offline, now.

**E4. Established spatial methods on identical inputs.**
- *Methods:* BANKSY, GraphST, STAGATE, SpaGCN, CellCharter, SEDR.
- *Design:* native clustering plus KMeans at K on each embedding; 13 objects × 3 seeds.
- *Statistic:* Δ vs Tessera and vs nested NM.
- *Code:* `spatial-omics-reform/external/run_*_domain.py`, `run_niche_cellcharter.py`. Envs `nfspatial` and `dl` have SpaGCN, cellcharter and torch_geometric; GraphST, STAGATE and SEDR are source checkouts whose dependencies need checking.
- *Compute / status:* about 4–8 GPU-h. Feasible offline, subject to an env check.

**E5. Transfer with spatial controls and technical-replicate targets.**
- *Design:* add NM, BANKSY-style and NoGate arms to `run_transfer.py`. Add replicate sections Lung5_Rep1/Rep3 and Lung9_Rep2 (present in `cosmx_lung.zip`) as extra target sections to measure within-patient section variance.
- *Statistic:* per-patient balanced-accuracy Δ.
- *Compute / status:* about 30 GPU-min. Offline, now.

**E6. A correct geometry null.**
- *Design:* (a) a random degree-matched graph *without* the contrastive term; (b) spatially structured permutation (FOV rotation, or toroidal shift of the label/expression field) that keeps autocorrelation. Compare measured vs self-only vs (a)/(b).
- *Compute / status:* minutes. Offline, now.

**E7. Gate collapse diagnosis.**
- *Design:* log the gate distribution per layer and epoch; test gate-bias initialisation and a temperature; and check whether a gate pre-trained on reference seams changes ARI. The last is diagnostic only, not a new method claim.
- *Compute / status:* about 1 GPU-h. Offline.

**E8. More independent patients.**
- *Design:* ≥10 annotated CosMx/Xenium NSCLC or BCC patients from new sources.
- *Status:* not offline. The local PDAC CosMx `gse277782` (3 patients, cell types plus a "Type" region label) has no coordinates on disk; the coordinates would need downloading.

## 7. Verdict and ranked revision items

**Major revision.**

**Major items:**
1. Report the ablation/gate findings plainly: no-gate ≥ full in 5/5 lung patients, gate AUROC 0.385–0.548, collapsed gates, and no-contrast ≈ PCA. Rewrite the abstract and conclusions around "contrastive neighbour smoothing, not gating".
2. Make Tessera−neighbour-mean co-primary. Run E1 with nested selection, and E4 or at least BANKSY/CellCharter/GraphST.
3. Fix the permutation inference (§2): use self-only as the reference and drop "dependence on geometry" as a positive claim.
4. De-confound target from dataset (E2b) and promote S4.
5. Replace the n=4/5 percentile CIs with sign counts and exact p (0.125 for both cohorts), and add more seeds plus FOV-pack resampling (E3).
6. Remove or justify the 6,000-cell cap: the paper's own cost data show it is unnecessary. Stop random-thinning Open-ST.
7. Fix K: exclude singleton classes (Lung5 n=1, Lung6 n=5) or report both ways; disclose that Lung6's region task is FOV separation.
8. Transfer: add an NM arm and lead with the 5/5 cell-type loss.

**Minor items:**

9. Consolidate the 18 figures to about 8 main figures. Fix the Fig 6(d) self-only artefact, the Fig 7 colour matching to empty classes, "DAPI protein", seed-1 vs mean ARIs, numeric object IDs and the orange colour collision.
10. Remove internal-process text and defensive captions, and define terms before use.
11. Move display-mechanics equations and hashing detail to the SI, and cite the original Tessera work.
12. Reconcile the 27 vs 33 reference count and rename "HNSCC subset".

**Open questions for the author (unscored):**
- Are the Patient B and C banding patterns in Fig 8 caused by coordinate- or illumination-correlated technical signal?
- Was `PROTOCOL.md` time-stamped externally before fitting?
- Were Tessera's hyperparameters ever tuned on any object used here (for example the earlier 15k HNSCC object)?

Key paths:
- Manuscript: `/home/zeyufu/Desktop/singlecell-genomics-research/research/revisions/tessera-20261008-expanded/source/manuscript/paper.tex`
- Ledgers: `/home/zeyufu/Desktop/singlecell-genomics-research/research/revisions/tessera-20261008-expanded/source/manuscript/ledgers/` (`all_primary_scores.csv`, `seed_stability.csv`, `manuscript_numbers.json`)
- Runner: `/home/zeyufu/Desktop/singlecell-genomics-research/research/revisions/tessera-20261008-expanded/source/scripts/run_experiments.py`
- Data preparation: `/home/zeyufu/Desktop/singlecell-genomics-research/research/revisions/tessera-20261008-expanded/source/scripts/prepare_data.py`
- Baseline runners: `/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external/`
- Rendered atlas pages: `/tmp/review_tess/atlas-01.png` … `atlas-18.png`