# Tessera untouched colorectal pathology pilot

Locked 11 October 2026 before downloading or opening any of this study's count
matrices, spot-label tables, images, or model scores. Public article methods,
archive filenames, sizes, checksums and patient mapping were inspected.

## Question and scope

Do learned gates discriminate independently assigned histological compartment
seams better than a raw expression difference, and does the frozen Tessera
encoder improve compartment agreement over an untuned mean on the same graph?
This is a small external mechanism check, not a clinical or equivalence claim.
The practical question is whether the learned component earns its inference
and fitting cost over a simple neighborhood operation.

HER2ST and the 13 primary objects are development evidence for this protocol.
The new cohort is Valdeolivas et al., npj Precision Oncology 8, 10 (2024),
doi:10.1038/s41698-023-00488-4. The public release is Zenodo 7760264,
doi:10.5281/zenodo.7760264, CC-BY-4.0. Published methods state that H&E was
annotated in QuPath and spots categorized by a pathologist in Loupe. Molecular
CMS labels, deconvolution and gene-derived tumor-surrounding categories are
ineligible targets. The release does not establish a second blinded annotator.

## Untouched-cohort and resource rules

Search the current Tessera protocols, source, input metadata and run receipts
for this study's seven patient IDs and accession. No new cohort is called
untouched if it already contributed fits, target selection or scale selection
to Tessera. Record the audit and its scope; do not claim an audit of all past
human activity or another project's data. CRC TVA232 in the primary run is a
different source object.

Choose one serial section per patient by smallest published archive size, then
the three smallest distinct-patient archives. This selection uses download cost
only, before reading labels or scores:

| Patient | Study name | Archive | Bytes |
|---|---|---|---:|
| A120838 | S4_Col_Sig | SN84_A120838_Rep1.zip | 53827477 |
| A595688 | S2_Col_R | SN124_A595688_Rep2.zip | 56124799 |
| A551763 | S1_Cec | SN124_A551763_Rep2.zip | 61982650 |

The pathology annotation archive is 702660 bytes. Total: 172637586 bytes,
approximately 164.64 MiB. Do not fetch whole-slide archives, deconvolution,
FASTQ or BAM. Reuse the existing dependencies and frozen estimator. Use
ClashParty 7890 and SC_FETCH_PROXY with research/pipelines/fetch_gate.py.
Require successful HEAD, exact Content-Length, each file below 1 GiB, and
publisher MD5 plus local SHA-256. Stop when the designated proxy or HEAD gate
fails. Starting the existing ClashParty core with TUN disabled does not change
the desktop/system proxy or the 7897 client configuration.

Only the count matrix, tissue positions, scalefactors, low/high-resolution H&E
and required pathology CSVs are extracted from the three sample archives.
Run CPU preprocessing immediately after extraction. Small CPU fits are allowed;
if fitting proves heavy, move compute to dl4080 without changing the estimator.
Raw downloads, inputs, predictions and receipts remain separate from historical
results. No existing receipt or matrix is overwritten.

## Eligibility and registration checks

Use released filtered counts, exact barcode joins and the released tissue pixel
coordinates. Preserve all tissue observations in preprocessing and graphs,
including unannotated spots. Exclude missing labels from scoring only. Require
unique barcodes in every table, finite coordinates, integer nonnegative counts,
at least 95% of annotated tissue spots matching counts and positions, at least
100 scored spots, at least two compartment classes, and at least 20 unique
undirected seam edges and 20 within-compartment edges. Fail closed on ambiguous
files, labels, patient identities or joins. No fuzzy coordinate matching.

Render the pathology labels over the archive's H&E with released scalefactors.
Inspect gross orientation, image bounds and tissue registration. This author
check does not validate the pathologist's diagnoses or replace independent
registration review. Report artifacts and ambiguous/mixed categories unchanged.

## Frozen comparisons and negative controls

Preprocessing copies prepare_data.finish via the existing external runner:
drop all-zero rows and negative-control genes; library-size normalization to
10000 and log1p for integer counts; top 3000 variable features; clipped z-score;
PCA50 with randomized SVD and random_state=0. No label enters feature selection.
Graph: frozen symmetrized six-neighbor Tessera graph, one frame per section.

Arms: PCA50; UniformSameGraph (the existing neighbor_embedding); Tessera;
NoGate. Tessera and NoGate use run_experiments.train_net, 120 epochs and all
frozen settings, seeds 1, 2 and 3. Use a fixed two-cluster auxiliary head in
every fit. The head is absent from the frozen training loss; no labels or class
count enter fitting. This choice is recorded as a pilot configuration and is
not represented as replaying any earlier run. No checkpoint transfer, feature
or scale tuning, test-label calibration, or selection of favorable seeds.

After fitting, evaluate all labeled compartments using K-means, K equal to the
number of valid released compartments, n_init=10, seeds 1-3. This secondary
ARI endpoint is label-informed in its resolution; boundary AUROC is the
primary endpoint and uses no K selection. Do not merge compartments after
seeing scores. A broad category seam is not automatically a tumor boundary.

For seam detection use one copy of each undirected k=6 edge with two valid
endpoints. Seam=unequal pathologist labels. The scores are 1 minus last gate
(average directed duplicates, then three seeds), raw PCA Euclidean difference,
and uniform-mean Euclidean difference, on exactly the same edges. Report
both per-patient and equally weighted patient means. Shuffle complete spot
labels 1000 times within each section to check the AUROC null; this is a
negative control, not a data split. No random train/test split.

## Metrics, uncertainty and stopping rules

Primary: per-patient gate seam AUROC and gate minus raw-feature AUROC.
Discrimination is not credited unless the 95% patient-bootstrap lower bound
exceeds 0.5. A learned-boundary advantage additionally requires the 95% lower
bound of gate minus raw-feature AUROC above zero. Use 10000 patient bootstrap
resamples with seed 20261011. With three patients these intervals are unstable;
also report the three individual values and the exact sign-flip minimum
attainable two-sided p=0.25. Never treat edges, seeds or serial sections as
independent patients. Label-shuffle intervals are descriptive null checks.

Secondary: three-seed mean ARI and Tessera minus UniformSameGraph; NoGate minus
Tessera; PCA50 reference; fit seconds, graph edges, observations and peak memory
when available. An author-chosen pilot relevance margin is 0.02 ARI. A learned
gain is credited only if its patient-bootstrap lower bound exceeds +0.02;
otherwise report it as unsupported, not equivalent or absent universally.
The margin is not an established biological/clinical standard. No superiority
claim based on oracle controls or post-hoc endpoint selection.

Stop after three eligible sections. Do not download more sections in response
to results. Report every arm and exclusion. If an eligibility check fails,
record the reason rather than substituting another patient after inspecting
outcomes. Calibration and selective abstention are out of this pilot's scope.

## Completion evidence and submission decision

Bind protocol, runner, frozen code, raw archives, input matrices, predictions,
score tables and figures by SHA-256. Recompute scores from saved predictions
and labels in a separate read-only verifier, test barcode/edge invariants and
negative-control behavior, and run the existing manuscript regression suite.
Keep original source/results/PDFs intact during the experiment. Write a separate
results report, blocker assessment and framework task board with evidence links.

This pilot can close absence of any new pathology-annotated external test if
provenance, eligibility and fits pass. It cannot by itself close second blinded
annotation/adjudication, independent registration review, or adequately powered
patient-level inference. Reassess whether calibration is a claim-dependent
extension rather than a mandatory submission blocker for this mechanism paper.
Do not equate author-side tests, package completeness, or small p-values with
independent scientific review. The final submission assessment must identify
actual remaining blockers and the narrower claims the evidence supports.

## Sources

- https://doi.org/10.1038/s41698-023-00488-4
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10781769/
- https://zenodo.org/records/7760264
- https://github.com/alberto-valdeolivas/ST_CRC_CMS
