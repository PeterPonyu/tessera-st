# HER2ST external pathologist-label comparison (first locked pass)

This file records the first locked pass. The paper reports the final pass, which
scored every arm together (`../validation-extension/RESULT_EXTENSION.md`).
Tessera's per-patient scores differ slightly between passes because its
optimization is not bit-reproducible; the controls are deterministic.

11 October 2026. Protocol: `PROTOCOL.md`. Scores: `results/summary.json`,
`results/primary_patients.csv`, `results/scale_controls.csv`,
`results/tessera_scores.csv`, `results/boundary_edges.csv`.

The released Tessera encoder was fit only on this cohort, at the frozen
hyperparameters in `source/scripts/run_experiments.py`. No manuscript object
was refit. The spatial control for each patient was chosen with that patient's
labels held out, using the same grid and `idxmax` rule as the lung/BCC scale
follow-up. Adjusted Rand index values below are three-seed means.

## Cohort

Eight HER2-positive breast tumors from Andersson et al., Nature Communications
12, 6012 (2021), DOI 10.1038/s41467-021-26271-2. Spot labels are the public
pathologist tables in `almaan/her2st` (`data/ST-pat/lbl`). Expression is the
matching count matrix. Data record 10.5281/zenodo.4751624. Section J1 has a
label file and no count matrix, so it was excluded before fitting.

These patients are outside the 13 manuscript objects. DLPFC was not used.
The join key is the rounded array coordinate, which is the count-matrix index.
Image pixel coordinates are the neighbor geometry. Match rates on
finite-coordinate spots were 0.990 to 1.000. H&E images were not downloaded,
and registration was not rechecked. The repository names patient G's annotated
image `G1` and the label file `G2`.

Count matrices were integer counts and were library-size log-normalized.
Each file was under 2 MB. ClashParty port 7890 was closed, so these small
GitHub objects were fetched directly. The 592 MB image archive was not requested.

## Primary result

The nested spatial control was higher than Tessera in 6 of 8 patients.
Mean Tessera minus control was −0.023. The two-sided sign-flip value was
0.102 (26/256). Folds overlap, so this is a descriptive count.

| Patient | Section | Spots | K | Control selected on the other patients | Tessera | Control | PCA50 | Tessera − control | Tessera − PCA50 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| A | A1 | 346 | 6 | Neighbor mean, k=6 | 0.104 | 0.183 | 0.216 | −0.080 | −0.113 |
| B | B1 | 295 | 5 | Neighbor mean, k=6 | 0.268 | 0.281 | 0.245 | −0.014 | +0.023 |
| C | C1 | 176 | 4 | BANKSY-style, k=10, λ=0.8 | 0.141 | 0.180 | 0.145 | −0.039 | −0.004 |
| D | D1 | 306 | 4 | Neighbor mean, k=10 | 0.181 | 0.221 | 0.223 | −0.040 | −0.041 |
| E | E1 | 587 | 4 | BANKSY-style, k=10, λ=0.8 | 0.018 | 0.008 | 0.058 | +0.010 | −0.040 |
| F | F1 | 691 | 4 | Neighbor mean, k=6 | 0.047 | 0.053 | 0.071 | −0.006 | −0.024 |
| G | G2 | 467 | 7 | Neighbor mean, k=6 | 0.168 | 0.219 | 0.176 | −0.051 | −0.008 |
| H | H1 | 613 | 7 | BANKSY-style, k=6, λ=0.8 | 0.299 | 0.264 | 0.219 | +0.035 | +0.080 |

Tessera was below PCA50 in 6 of 8 patients (mean −0.016, sign-flip 0.445).
Patient E's positive control contrast is a 0.010 gap between two near-zero
scores. Patient H is the only section where Tessera's absolute agreement is
both higher than the nested control and above 0.2.

Repeating selection without patient G, because of the G1/G2 filename mismatch,
left the control ahead in 5 of 7 patients (mean −0.017, sign-flip 0.281).

The saved selection was recomputed from `scale_controls.csv` after the run.
Held-out labels were not required to recover the recorded control names, and
the patient means match the saved ARI columns.

## Boundary edges

On the frozen k=6 graph, the last-layer gate had a higher seam AUROC than a
PCA-feature difference in 3 of 8 patients (E, F, and H). Patient E's gate
AUROC was 0.409. Patient F's was 0.507. Patient H's was 0.542 against an
expression-difference AUROC of 0.522. In the other five patients the
expression difference was higher.

## Target composition

The locked rule kept every label string outside the existing missing-value
exclusions. The tables include `undetermined` in every section (5/346 in A1
through 83/613 in H1) and `breast glands` in B1, G2, and H1. Those strings
were not removed after the scores were known. A later exclusion would be a
different analysis.

## Limits

These limits are stated in the paper's Discussion.

- The external comparison does not show a learned spatial gain over the
  nested neighborhood control.
- The labels are one pathologist's published H&E regions. There is no second
  blinded annotation, no adjudication, and no rechecked registration.
- Patient G's image filename and label filename disagree. The seven-patient
  sensitivity has the same direction.
- Eight patients and overlapping selection folds remain a small, dependent
  sample. No author-locked effect margin was set.
- Cluster number K is the number of label classes on that section. Tessera
  was refit on each section; the lung weights were not reused.
- `undetermined` and `breast glands` are inside the scored target.
- Gate scores do not provide a boundary detector beyond an expression difference.
- Calibration and abstention were not run.
- The assay is the original spatial-transcriptomics array. Absolute adjusted
  Rand index stays between 0.018 and 0.299.
- The compiled manuscript still describes this comparison as the next test.
  Its PDF, figures, and historical receipts were not changed. The frozen
  regression suite passed 52 tests after this run.
