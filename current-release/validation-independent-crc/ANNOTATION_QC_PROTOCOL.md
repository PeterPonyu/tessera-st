# Schema-triggered annotation sensitivity

11 October 2026. The prelocked pilot is preserved unchanged. Opening the first
pathology CSV revealed an explicit `exclude` category (88/328 tissue spots).
Its presence was not evident in the archive listing. The first section's PCA
and uniform-mean ARI values have been printed; no learned-gate AUROC has been
calculated or inspected. The main fits are already running. Therefore this
document is a disclosed protocol amendment / sensitivity, not a claim that
the original target specification anticipated the schema.

Keep the original all-released-label endpoint and original fits/receipts for
traceability, including `exclude` as the original encoder recognizes it.
It must not be interpreted as a biological compartment or tissue boundary.

On the same saved inputs, embeddings and graphs, add a pathology-compartment
target that invalidates the exact string `exclude` (case/whitespace normalized),
plus the original missing-label exclusions. Keep those spots in preprocessing
and the graph. Preserve every other pathologist category, including mixed
tumor/stroma and graded immune infiltration categories; do not collapse them
into a binary tumor boundary. These are broad morphology seams.

For this sensitivity, re-cluster each already fitted embedding with K equal
to the remaining valid pathology classes (K-means n_init=10, seeds 1-3), with
no model refit or tuning. Recompute ARI and gate/raw/uniform-feature AUROC on
unique undirected edges with two valid endpoints. Use the same three-patient
bootstrap and 1000 complete-label-shuffle nulls as the original protocol.
The eligibility minima and thresholds stay unchanged. Report original and
QC-filtered values side by side; no target is selected for a favorable result.

For the submission assessment, only the target without the explicit QC
category is evidence about histological compartment boundaries. Annotation
provenance, mixed categories, single-reader status and registration limitations
still apply. A sensitivity does not make this pilot a pristine prespecified
biological confirmation.
