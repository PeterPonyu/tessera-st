# Tessera tumor-associated expansion — frozen before fitting

Protocol: `tessera-tumor-expansion-20261008-v1`. This is a new, local
experiment, not a replacement for the October 2 release or the earlier
typography revision. No submission or external deposit is authorized.

## Questions and targets

1. Does the unchanged Tessera encoder improve raw partition agreement over
   an expression-only control on tumor-associated spatial assays?
2. Which of its three trained components contributes under matched inputs?
3. Does the signal survive spatial-coordinate permutation, and can a frozen
   source-trained representation transfer to a held-out NSCLC patient?
4. Can predictions be inspected in genuine, registered tissue images without
   presenting transcriptomic annotations as independent pathology?

The primary NSCLC target is the supplied **spatial-region** annotation
(`label`), computed by Mclust on 200-neighbor cell-type frequencies in the
redistributed benchmark. Its construction contains a spatial prior, so
agreement is not independent tissue validation. Its supplied cell-type
annotation is a separate secondary target.
BCC uses supplied InSituType cell-type labels. Breast uses the dominant
Xenium-derived category in each measured Visium spot; mixed spots remain
mixed and are not called malignant ground truth. Colorectal section 232 is
source-coded TVA, not automatically invasive CRC; `lv2` is a source molecular
annotation. Open-ST uses its original supplied annotations; MIBI uses `Cluster`.
Targets are heterogeneous; no cell-pooled cross-assay significance test.

## Fixed inputs and selection

- Five NSCLC sections, one per source patient: Lung5_Rep2, Lung6,
  Lung9_Rep1, Lung12, Lung13. The other technical/section replicates do not
  increase the patient count.
- BCC PatientA–D, using Patient_ID, not run names as donors.
- One registered breast Visium object; one colorectal TVA section.
- Original MIBI three-field object and a bounded Open-ST section-26 subset.
- At most 6,000 observations per object. All breast and MIBI observations
  are retained after numerical QC. For multiframe objects, whole fields are
  visited in SHA256(`selection-v1|field-id`) order and admitted if they fit
  the remaining budget. No labels, fitted results or visual attractiveness
  enter selection. BCC conditions are visited round-robin so experimental
  condition, not cell annotation, can stratify the selection. A single
  oversized frame uses RandomState(0) sorted row sampling. No partial
  sampling of a field is used when other complete fields fit.
- QC: source selected/fil or not-low-quality flag, finite coordinates and
  nonnegative finite features, nonzero row sum. Missing annotations are
  retained as model inputs and excluded only from the corresponding score.
- NSCLC frame IDs come from the archived FOV-prefixed observation IDs;
  this convention and its limitations are reported. BCC composite frame
  includes run, patient and fov; colorectal includes section and fov.
- Counts: library-size normalize to 10,000, log1p. Noninteger source data
  are retained without a second log transform. Remove explicitly named
  negative/control features; select up to 3,000 highest-variance features
  without labels; standardize, clip at +/-10, PCA50 (or available rank).
  Identical saved PCA inputs are used by all primary arms. No label-driven
  dimension count. Original PCA30 comparison is additionally reported.

## Arms, training and scores

All seeds are 1, 2, 3; KMeans has n_init=10 and uses the number of observed
reference classes. This primary partition task is **reference-conditioned**.
No spatial label refinement and no checkpoint/seed/field selection.

1. PCA50 + KMeans (matched primary control).
2. PCA30 + KMeans (legacy-style expression control).
3. Within-field neighbor mean of PCA50 + KMeans (simple spatial control).
4. Unchanged full Tessera.
5. No edge gating.
6. No scale fusion.
7. No contrastive term.
8. Self-only encoder: same layers/fusion/decoder, self edges only,
   reconstruction only; an expression-only neural structural control,
   not a claim of exactly equal trainable parameter count.
9. Full Tessera with coordinates permuted independently within every field.

Tessera arms use hidden64/embed32/3 layers/dropout0.1, Adam lr0.001,
weight decay0.0001, 120 epochs, reconstruction weight1, contrastive weight0.5,
temperature0.2 and 10 negatives. The existing negative sampler can include
self or neighbor collisions; this is disclosed, not silently repaired.
The untrained legacy soft head is not an uncertainty estimator. Gate-derived
boundary values are descriptive; no calibrated confidence or pathology
boundary accuracy is claimed. No new architecture is introduced.

Primary: raw ARI and paired difference from matched PCA50. Secondary:
NMI, seed stability (pairwise ARI), labeled within-field edge homogeneity,
reference seam AUROC for gate scores where both seam classes exist,
runtime and GPU peak allocated memory. Seam targets are derived from the
same reference annotations, not independent physical boundaries.

Report every completed arm and failure. Seeds are optimization replicates.
For the five NSCLC patients, a percentile bootstrap of patient-level,
seed-mean differences uses B=10,000 and RandomState(20261008); n=5 remains
small. BCC n=4 is reported separately; fields/cells are not patients.
No significance claims from three seeds and no universal-superiority gate.

## Held-out-patient extension

Five leave-one-patient-out folds on the same retained NSCLC inputs. Source
patients alone fit gene scaling, PCA50 and the neural encoder; source graphs
are disconnected by patient/FOV. Frozen source transformations and weights
are applied to the target. Compare PCA50, self-only and full Tessera at all
three seeds, with source-label nearest centroids and no target class-count
or label tuning. Report overall accuracy, balanced accuracy on all observed
target classes, supported-class macro F1 and target categories absent from
source. This is source-trained inductive representation transfer, **not**
zero-shot inference of a pretrained foundation model.

## Visual evidence and provenance

Use genuine local breast tissue image with published scales and verified
barcode-position agreement. Show full tissue, actual measured spots and a
geometry-selected ROI (median-coordinate center, fixed fraction of extent).
Scale bar derives from the source spot diameter and registration metadata.
No synthetic H&E, invented coordinates or artificial pathology labels.
NSCLC/BCC/TVA maps are measured-coordinate maps, not microscope photographs.
RNA markers are same-modality description, not independent validation;
available colorectal protein measurements are separately identified.

All selected IDs, input hashes, fitted labels/embeddings, histories, graph
audits, environment versions and numerical tables must be retained. The
previous release and Frontiers V3/V9 manuscript are read-only references.
Final deliverables: expanded manuscript, supporting information, figure
atlas, editable native SVG schematic, reproducibility bundle and an honest
remaining-validation report. Technical completion is not submission approval.
