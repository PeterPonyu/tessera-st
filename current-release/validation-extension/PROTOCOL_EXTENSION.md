# Extension protocol: weighting ablation, seam statistics and tissue breadth

Locked 11 October 2026 before any analysis below was run. It follows the
HER2ST comparison (`validation-external/`), whose primary and enhanced results
had already been seen. The analyses here were therefore chosen after those
results and are not an untouched confirmation. They test whether the
expression-weighted finding survives its obvious controls, and whether the
ordering of Tessera and a tuned neighborhood control holds outside tumors.

No estimator, hyperparameter, graph rule or preprocessing step is changed.
Everything runs on CPU with the frozen code in `source/`.

## Common definitions

- Representation input: the 50-component PCA from the frozen preprocessing
  (`prepare_data.finish`; reproduced in
  `validation-external/run_her2st_validation.preprocess`).
- Graph: `run_experiments.graph(coords, fields, "Tessera", 1)`, the symmetrized
  six-neighbor graph within each frame. Edge `e` carries source `j` into
  destination `i`.
- Edge distance `d_e`: Euclidean distance between the PCA inputs of `j` and `i`.
- Weighted mean: `x~_i = sum_e w_e x_j / sum_e w_e` over edges into `i`; there is
  no self term.
  - ExpressionWeighted (E6): `w_e = exp(-d_e / median(d))`, median over all
    edges of the object.
  - UniformSameGraph (U6): `w_e = 1` on the identical graph.
  - WeightShuffled (S6): E6 weights permuted across edges, seeds 1-3.
  - ShuffledCoords (C6): coordinates permuted within each frame (seeds 1-3),
    the graph rebuilt from the permuted coordinates, then E6 weights.
- Clustering and score: K-means, `n_init=10`, seeds 1-3, K = number of valid
  classes of the target; adjusted Rand index (ARI) over valid observations,
  three-seed mean.
- Grid and nested control: `run_followup.configs()`; PCA50 is not eligible;
  the control for a held-out unit is the configuration with the highest mean ARI
  over the other units (`idxmax`; ties keep sorted order).
- Tessera: `run_experiments.train_net(..., "Tessera", seed, "cpu")`, seeds 1-3,
  frozen hyperparameters.

## X1. HER2ST hardening (same eight patients, same joins)

Targets, all scored on the same joined observations, with unassigned
(`undetermined`) observations kept in the graph and the PCA:

- `all`: every label string, as in the primary comparison.
- `compartments`: `undetermined` invalid.
- `boundary_T1`: invasive cancer and cancer in situ versus adipose tissue,
  connective tissue, immune infiltrate and breast glands (the locked enhanced
  target).
- `boundary_T2`: invasive cancer versus the same other tissues; cancer in situ
  and `undetermined` are invalid.
- `boundary_T3`: invasive cancer and cancer in situ versus adipose tissue,
  connective tissue and immune infiltrate; breast glands and `undetermined`
  are invalid.

Analyses:

1. Section identity for patient G: join rate of the labeled spots to the count
   matrix of each of that patient's two public sections (array coordinates).
   Data-integrity check only.
2. Weighting ablation: E6, U6, S6 and C6, plus PCA50, Tessera, and the nested
   control, on every target.
3. Edge-level seam AUROC for `compartments` and `boundary_T1`: a seam is a graph
   edge whose two valid endpoints have different labels. Scores are the learned
   last-layer gate (as `1 - gate`, mean of seeds, from fits at the target's K),
   the raw PCA-feature difference, the E6-smoothed difference
   `||x~_i - x~_j||`, and the U6-smoothed difference. Per-patient AUROC.
   Patient-level 10,000-resample percentile bootstrap of the mean of each score
   and of the pairwise differences (smoothed minus gate, raw minus gate,
   smoothed minus raw). Descriptive with eight patients.
4. Partition tables for map figures: reference, Tessera (seed 1), selected
   control (seed 1), E6 (seed 1), U6 (seed 1), PCA50 (seed 1), for `all`,
   `compartments` and `boundary_T1`.
5. Reproducibility: the refit Tessera ARI at each previously scored target
   must equal the earlier value; any mismatch is reported.

Prespecified reading:

- H1 (weighting matters). E6 exceeds U6 in at least 6 of 8 patients on
  `boundary_T1`, and E6 exceeds both S6 and C6 in at least 6 of 8. If not, the
  weighting is not credited.
- H2 (gate does not detect the boundary). The learned gate does not exceed the
  raw PCA-feature difference in at least 5 of 8 patients on `boundary_T1`.
- Target sensitivity: the sign of E6 minus the nested control on `boundary_T2`
  and `boundary_T3` is reported next to `boundary_T1`. No variant is selected.

## X2. Tissue breadth (single sections; same frozen pipeline)

Objects, labels and sources. Each is a public release; each tissue was also used
when the encoder was first benchmarked, so these test breadth and are not
untouched.

| Object | Tissue | Label column | Source |
|---|---|---|---|
| DLPFC section 151673 | human dorsolateral prefrontal cortex, Visium | `Region` (layers 1-6, white matter) | Maynard et al. 2021, doi:10.1038/s41593-020-00787-0 |
| STARmap | mouse primary visual cortex | `Region` (layers 1-6, corpus callosum, hippocampus) | Wang et al. 2018, doi:10.1126/science.aat5691 |
| MERFISH hypothalamus | mouse preoptic hypothalamus | `domain` (eight nuclei/regions) | Moffitt et al. 2018, doi:10.1126/science.aau5324 |

Not included, and why: human breast Visium with regional annotation, Slide-seqV2
hippocampus (both flagged in the local data registry as not cleared for
paper-facing claims); CODEX spleen (no stated reuse license); osmFISH (the file
is absent); seqFISH, IMC and other cell-type targets (a different question).

Rules: all three have fewer than 6,000 observations, so no subsampling.
Preprocessing is the frozen rule (library-size log normalization only when the
matrix is integer counts; the MERFISH matrix is volume-normalized and is used as
supplied). Invalid labels (missing) are excluded from the score and kept in the
graph. K is the number of valid classes. One frame per object.

Arms: PCA50, the full grid, U6, E6, S6, C6, Tessera. The nested control uses
leave-one-object-out selection across the three objects (two objects select).
The oracle-best grid configuration is stored and is not for inference.
GT contiguity = reference-edge homogeneity on the six-neighbor graph.

Reading: no threshold. The ordering of Tessera, nested control, oracle-best
control and E6 is reported per object. With three objects there is no
sign-flip statistic.

## X3. Development objects (13 manuscript objects)

E6, S6 and C6 on the 13 saved inputs at the primary K. The U6 comparator is the
existing `NeighborMean` arm, which is the same symmetrized graph. Tessera, PCA50,
`NeighborMean` and the nested control for lung and BCC are read from the frozen
ledgers.

Reading: per cohort, report counts of objects where E6 exceeds `NeighborMean`,
S6 and C6, and where E6 exceeds the nested control (lung, BCC).

## Reporting rules

Every arm, target and object is reported. Absence of a favorable result is
reported as such. The extension adds no claim about fine-tuned encoders, clinical
use or other methods.
