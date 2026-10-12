# External pathologist-boundary protocol

Locked 11 October 2026, before any HER2ST count matrix or spot-label file was
downloaded or opened, and before any model was fit. Filename listings and the
published data record were inspected; spot-level labels and scores were not.

## Question

On tissue regions drawn by a pathologist, does the released Tessera encoder,
fit at its frozen hyperparameters, exceed a spatial control whose scale is
chosen without that patient's labels?

This is the comparison named in the manuscript as the next decisive test. It
is not a clinical validation, a calibration study, or a claim of universal
superiority.

## Cohort

- Article: Andersson et al., Nature Communications 12, 6012 (2021),
  DOI 10.1038/s41467-021-26271-2. The Zenodo record names the preprint
  10.1101/2020.07.14.200600 as its source publication.
- Expression and spot labels: the public `almaan/her2st` repository
  (`master` tree inspected 11 October 2026). Counts are
  `data/ST-cnts/{section}.tsv.gz`. Pathologist spot tables are
  `data/ST-pat/lbl/{section}_labeled_coordinates.tsv`. The repository README
  in that label directory states that the labels were taken from the
  pathologist's annotated H&E images.
- Data archive for the same study: 10.5281/zenodo.4751624. Images are not
  downloaded. Expression-cluster tables under `res/ST-cluster` are not targets.
- The published annotation is morphological (in situ cancer, invasive cancer,
  adipose tissue, immune infiltrate, connective tissue). Every non-missing
  label string is retained. Classes are not dropped or merged after scores
  are seen.
- Repository label files: A1, B1, C1, D1, E1, F1, G2, H1, and J1. J1 has no
  count matrix in `data/ST-cnts` and is excluded for that reason. The included
  sections are the label files that also have a count matrix: A1, B1, C1, D1,
  E1, F1, G2, H1. Patient identity is the leading letter. One labeled section
  is expected per patient.
- These eight patients are not among the 13 manuscript objects. DLPFC /
  spatialLIBD is not eligible: it is the package's real-data development path.

## Geometry and boundary definition

- Coordinates are `pixel_x` and `pixel_y` from the pathologist spot table,
  the image geometry of the annotation.
- The count-matrix spot index is the rounded array coordinate
  `round(x)xround(y)` from the pathologist table. That is the join key.
  Rows without finite `x`, `y`, `pixel_x` and `pixel_y` are not joinable and
  are removed before the match rate is calculated. At least 90% of the
  remaining labeled spots must match a count-matrix index exactly, and the
  keys must be unique. Otherwise the run stops. Raw label-row names and
  rounded pixel coordinates are not the count index; both matched none of the
  A1 count spots in an identifier check done before any fit. There is no
  fuzzy spatial join and no pixel tolerance. The k-nearest-neighbor graph
  still uses `pixel_x` and `pixel_y`.
- Spots the pathologist table does not list are left out of the graph as well
  as the score. Invalid labels use the existing `encode` exclusions
  (blank, nan, none, unknown, na).
- A section with fewer than 50 valid spots or fewer than 2 valid classes is
  excluded before fitting and listed.
- A reference boundary edge is a directed edge in the frozen k=6 Tessera graph
  whose endpoints both have valid labels and those labels differ. No image is
  re-registered in this run.

## Frozen procedures

- Preprocessing copies `prepare_data.finish`: drop all-zero rows, drop
  negative-control feature names, library-size log normalization only when the
  matrix is a count matrix, 3,000 highest-variance features, z-score clipped
  at 10, PCA with 50 components, `svd_solver='randomized'`, `random_state=0`.
- Tessera is `run_experiments.train_net` with arm `Tessera`, seeds 1, 2 and 3,
  120 epochs and the locked model settings. Hyperparameters are not searched.
  The lung checkpoint is not reused. Device is CPU.
- Cluster number K is the number of valid pathologist classes on that section.
  Adjusted Rand index uses every valid spot. K-means uses `n_init=10` and the
  same seed.
- The spatial-control grid is `run_followup.configs`. Selection copies
  `scale_controls`: PCA50 is excluded from selection; the held-out patient's
  labels are not used; the chosen configuration is the `idxmax` of the
  other-patient mean ARI. Ties keep the earliest label in sorted configuration
  order.
- The primary contrast is the patient's three-seed mean Tessera ARI minus the
  three-seed mean ARI of that selected control. PCA50 is reported separately.
  The sign-flip test is the existing two-sided enumeration of all \(2^P\)
  assignments. Its smallest attainable value is \(2/2^P\). It is descriptive.
  Folds overlap.
- The boundary contrast, also three-seed means, is gate seam AUROC versus
  Euclidean PCA-feature difference on the same edges, using the rescore
  definition. The oracle best control on the held-out patient is stored and
  is not a result.
- Patient G is included in the primary analysis because G2 is the label file
  with a count matrix. The annotated-image file in the same repository is
  named G1, not G2. A prespecified sensitivity repeats selection and the
  sign-flip on the other seven patients only. It is not used to choose the
  primary.

## What this run cannot clear

Completion of this protocol does not lift submission. The following remain
even if every patient favors one method:

- One pathologist, no second blinded annotation and no adjudication.
- H&E registration is inherited from the public spot table and is not rechecked.
- The G1 image name and G2 label name are unresolved.
- Leave-one-patient-out folds overlap, and eight patients are a small sample.
- No author-locked effect margin or prospective sample size.
- K is label-informed. Tessera is refit on each section; only hyperparameters
  are frozen.
- Calibration and abstention are not tested.
- The platform is the original spatial-transcriptomics array, not a new
  CosMx, Xenium or Visium cohort.
- The compiled manuscript is a separate document and is not changed by this
  protocol file.
