# Untouched colorectal pathology pilot — 11 October 2026

**The external-data gap is closed at pilot scale. The stronger blinded boundary
validation remains open.** A new three-patient colorectal Visium study supports
the same-graph neighborhood control and qualifies the universal chance-gate
interpretation. The released gate carries weak seam information here, without
demonstrated added value over a raw feature difference.

## Cohort and independence

Valdeolivas et al., npj Precision Oncology 8, 10 (2024),
[doi:10.1038/s41698-023-00488-4](https://doi.org/10.1038/s41698-023-00488-4),
[Zenodo 7760264](https://zenodo.org/records/7760264), CC-BY-4.0. Published
methods describe pathologist H&E annotation in QuPath and spot categorization
in Loupe. Targets here are those pathology CSVs, not molecular CMS labels,
deconvolution, or computed tumor-surrounding categories.

The pre-download protocol chose the three smallest distinct-patient sample
archives, one technical section per patient. All archives are publisher-MD5
and SHA-256 verified. Download volume was 172637586 bytes (164.64 MiB), through
ClashParty 7890 and the existing fetch gate. One truncated third archive was
resumed and validated; the partial body was never opened as a dataset.

The current Tessera source, protocols, run receipts and input metadata contained
none of the seven study patient IDs or this accession (`cohort-audit.json`).
This establishes separation within that recorded scope, not an audit of every
past human analysis. HER2ST and the earlier anatomical tissues were already used
and were rejected as fresh holdouts.

The pilot has 1934 tissue spots; 1931 have released labels. Every annotated tissue
barcode joined exactly to counts and coordinates. Three unlabeled spots remain
in the graph and preprocessing. H&E overlays passed the author-side check of
orientation, bounds and gross tissue registration. No new annotation was made,
no diagnosis was adjudicated, and no second reader or independent registration
review was available.

## Target correction and complete reporting

The original locked target keeps all released strings. Schema inspection
revealed `exclude`, a QC category, in 173 spots. The schema-triggered amendment
was written after the first PCA/mean scores, before any gate seam AUROC was
calculated. It preserves the original target and fits, and adds a sensitivity
that invalidates `exclude` only for scoring. The biological assessment uses the
1758 remaining labels; mixed tumor/stroma and immune-infiltration categories
stay distinct. These are broad morphological compartment seams, not a validated
binary tumor boundary. This is a disclosed sensitivity, not an entirely
prespecified biological confirmation.

## Results without the QC category

ARI values are three-seed means. Each method uses the same PCA and graph;
clustering resolution is the valid target's class count after fitting.

| Patient | Tissue / scored spots | PCA50 ARI | Same-graph mean ARI | Tessera ARI | NoGate ARI | Gate seam AUROC | Raw-feature AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A120838 | 328 / 240 | 0.271 | 0.223 | 0.206 | 0.211 | 0.507 | 0.499 |
| A595688 | 387 / 356 | 0.481 | 0.338 | 0.301 | 0.303 | 0.648 | 0.646 |
| A551763 | 1219 / 1162 | 0.074 | 0.025 | 0.001 | 0.000 | 0.546 | 0.575 |

- Tessera minus mean: **-0.026 ARI**, descriptive patient-bootstrap interval
  -0.037 to -0.017; negative in 3/3 patients. The +0.02 pilot relevance-margin
  criterion for a learned gain is not met. This is not an equivalence test.
- NoGate minus Tessera: +0.002 ARI, interval -0.001 to +0.005. This does not
  establish a useful gate contribution or prove exact equality.
- Gate AUROC: **0.567**, interval 0.507 to 0.648. It passes the pilot's
  above-chance bootstrap rule; with only three patients this is weak evidence.
- Raw-feature AUROC: 0.573. Gate minus raw: **-0.006**, interval -0.029 to
  +0.008. The gate-superiority criterion is not met.
- The mean-feature AUROC is 0.555; smoothing is not uniformly better for seam
  discrimination in this cohort. The earlier breast result is cohort-specific.
- PCA50 has the highest partition ARI of the four tested arms in all three
  patients. This pilot therefore does not establish that spatial averaging
  improves over expression alone; its advantage is over this released encoder.

The original locked target also has Tessera below the same-graph mean in 3/3
patients (-0.011 ARI), gate AUROC 0.565, and raw-feature AUROC 0.566. Both targets
are preserved in separate CSVs and supplementary tables; no favorable target
was selected.

Intervals use 10000 patient resamples, not edge or seed resamples. Three patients
produce unstable percentile intervals; the minimum attainable two-sided sign
flip is 0.25. There is no confirmatory population-level significance claim.
The 1000 label-shuffle iterations per section and target yield null medians
between 0.45 and 0.55 for every score. They are negative controls, not patient CIs.

## Computation, provenance and verification

Final fits: 18 (Tessera and NoGate, three seeds, three sections), frozen 120-epoch
estimator, one CPU thread, approximately **220 CPU seconds** of model fitting.
The fixed two-class auxiliary head is absent from the training objective; no
labels or class count entered fitting. No control scale was searched and no
checkpoint was transferred. This tests the released algorithm fitted at fixed
settings on new sections, not transport of previously trained lung weights.

The 4080 host was unreachable. Initial four-thread CPU fits were unexpectedly
slow (239-258 seconds per first fit); a timing-only one-thread check took 11
seconds. Partial runs are preserved separately, and all final models were fit
again with one thread. The resource decision preceded seam analysis and used
timing, not selection of favorable scores. Neither the original pilot source
nor the frozen Tessera estimator was changed for this invocation.

The separate read-only verifier passed before paper integration:

- 72 ARI rows and 18 unique-edge AUROC rows recomputed from saved predictions.
- Equivalent K-means partitions confirmed at every arm/seed/target.
- 180 independently recomputed sampled shuffle rows; all 18000 null rows are
  hash-bound, with 1000 distinct iterations per section/score/target.
- Raw ZIP CRC, publisher MD5, SHA-256, exact joins, finite features, graph
  invariants, protocol/runner/frozen-source/output bindings checked.
- All **618 pre-existing protected files unchanged** before document integration.
- Existing manuscript regression suite: **61 passed**, no failures or skips.

The integrated manuscript adds six regression checks for these receipts, QC
handling, patient units, paper values and interpretation. The final document
build passes **67 tests** with zero failures/errors/skips: manuscript 24 pages,
supporting information 14 pages and unchanged atlas 21 pages. All current
manuscript/supporting pages were inspected on current contact sheets, with the
new pilot pages also inspected at full size. The 21 prior figure PDFs and atlas
images remain unchanged. A clipped summary tick label was corrected and
re-viewed. `visual-review.json` states the limited author inspection scope.

`verification-final.json` records the separate post-integration recomputation;
`integration-manifest.json` lists intentional, backed-up document/build edits
among the protected baseline. Existing numerical ledgers, original receipts,
inputs, predictions and core estimator retain their hashes. The prior
pre-integration verifier source and receipt are preserved separately. Tests and
author visual inspection remain distinct from independent scientific review.

## Submission assessment

Closed: missing historical runners; absence of a fresh pathology-annotated
external study; absence of any H&E registration inspection for this new pilot;
source/output verification; same-graph and NoGate controls on the new sections.

Still open for the stronger boundary-validation claim: independent blinded
reader agreement/adjudication, independent registration assessment, and a
patient sample supporting a prespecified effect margin. Three cost-selected
patients and the schema-triggered sensitivity cannot establish that claim.
The public source alone does not document reader blinding to molecular data.

The current paper can support a scoped retrospective mechanism comparison:
simple averaging exceeds this released encoder in the examined cohorts, while
weak seam information does not establish useful gating. It cannot claim that
all gates are at chance, that the new cohort proves equivalence, or that the
boundary readout is clinically validated. Calibration/abstention is a separate
extension if uncertainty is claimed; no such validated claim is made here.
Author-side tests do not replace fresh scientific review of the revised scope.

The independent-validation HOLD is therefore **partly resolved and retained
for the stronger claim**, rather than attributed to missing external data.
