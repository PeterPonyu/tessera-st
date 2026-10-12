# Extension results: weighting controls, seam statistics and tissue breadth

11 October 2026. Protocol: `PROTOCOL_EXTENSION.md` (locked before these runs).
Runner: `run_extension.py`. Receipts: `results/run-hardening.json`,
`results/run-dev.json`, `results/run-breadth.json`, each binding the protocol,
runner and frozen-code hashes. Every number below is read from those receipts
and appears in the paper through `source/manuscript/ledgers/external_numbers.tex`.

## Correction to the earlier weighting report

The earlier enhanced analysis (`validation-external/ENHANCED_RESULT.md`) found an
expression-weighted neighbor mean above the tuned control in 7 of 8 patients on
the tumor boundary and presented weighting as the improvement. The same-graph
controls the protocol added do not support that reading:

- Weighted minus unweighted mean on the same graph: 4 patients above, 4 below
  (mean -0.012). Shuffling the weights across edges changed nothing (4 above).
  Shuffling coordinates removed the effect in 8 of 8 (mean +0.309).
- On the 13 bounded objects, weighted minus unweighted: 7 above, 6 below
  (mean -0.001).
- Prespecified H1 (weighted above unweighted, shuffled weights and shuffled
  coordinates in at least 6 of 8) was not met.

What remains is the same-graph mean itself.

## Independent cohort (eight patients; final pass)

| Target | Tessera below same-graph mean | Tessera below tuned control | Same-graph mean above tuned control |
|---|---|---|---|
| All labels | 7 of 8 (mean -0.047, exact p 0.0156, bootstrap -0.074 to -0.020) | 6 of 8 (mean -0.022, p 0.141) | 6 of 8 (mean +0.024) |
| Named compartments | 6 of 8 (mean -0.028) | 5 of 8 (mean -0.020) | 6 of 8 (mean +0.008) |
| Tumor boundary | 5 of 8 (mean -0.062) | 4 of 8 (mean -0.008) | 8 of 8 (mean +0.053, p 0.0078, bootstrap +0.023 to +0.088) |

Tessera was below PCA50 in 6 of 8 patients on all labels (mean -0.015). Boundary
variants (invasive cancer only; without breast glands) gave the same direction
(`results/hardening` and Supplementary Table S10).

Section identity for patient G: labels join their own section at 100.0% and the
neighboring public section at 84.2%.

## Seam statistics (tumor-boundary edges)

Gate AUROC 0.505 (patient bootstrap 0.462 to 0.544; above 0.5 in 4 of 8); plain
feature difference 0.520; smoothed feature difference 0.583 (0.513 to 0.650).
Smoothed minus gate +0.078 in 7 of 8 patients (0.037 to 0.118). Prespecified H2
(gate exceeds the feature difference in at most 3 of 8) was not met: the gate
exceeded it in 4, although neither score separated the boundary. Across all 24
objects the gate AUROC had median 0.497 and exceeded 0.5 in 11.

## Replicate fits

The frozen encoder was fitted in three passes (first locked comparison, an
identical rerun, the final pass). Per-patient scores differed by up to 0.052 ARI
(median 0.002) over 24 patient-target pairs; one pair changed side relative to
the tuned control and none relative to the same-graph mean. Controls and weighted
means are deterministic. `replicates/run1/` keeps the first pass.

## Three further tissues (single sections; all also used in the encoder's first benchmark)

| Tissue | Tessera | PCA50 | Same-graph mean | Tuned control (leave-one-tissue-out) | Best of 30, chosen with labels | Gate AUROC |
|---|---|---|---|---|---|---|
| Human cortex (Visium) | 0.416 | 0.190 | 0.253 | 0.286 | 0.341 | 0.457 |
| Mouse cortex (STARmap) | 0.505 | 0.163 | 0.491 | 0.568 | 0.606 | 0.480 |
| Hypothalamus (MERFISH) | 0.357 | 0.079 | 0.175 | 0.425 | 0.460 | 0.468 |

Tessera exceeded PCA50 in all three (+0.226 to +0.342) and the tuned control only
in the human cortex section (+0.131); only that section exceeded the best of 30
configurations (+0.075). Across all 24 objects Tessera exceeded the same-graph
mean in 12 and the best of 30 configurations in 2 (the cortex section and
colorectal TVA).

## Not done

Calibration and selective abstention, a second annotator, and registration
review were not run. Breast Visium with regional annotation and Slide-seqV2 are
flagged in the local data registry as not cleared for paper-facing claims;
CODEX spleen has no stated reuse license; the osmFISH file is absent.
