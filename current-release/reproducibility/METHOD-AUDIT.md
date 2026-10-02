# Current method audit — 2026-09-23 integration

**SCIENTIFIC_SUBMISSION_HOLD**, with completed seven-object reanalysis under
`field-identified-fixed-backend-20260923-v1`. The old description below is an
archived pre-integration state and no longer describes the current run count.

## Input identity and spatial estimands

Eligibility requires a recorded field/section or a single registered Visium
spatial library. DLPFC uses `uns/spatial/151673`, MERFISH `slice_id`, MIBI
`library_id`, and STARmap/Open-ST/CODEX/Zhuang `section_id`. Source SHA-256,
observation indices/IDs, selected features, fields and labels are recorded. The
historical RandomState(0) 16,000-cell cap is retained before label filtering;
the pilot's top-class truncation is not used.

All spatial queries use min(6, field-size minus one), exclude self by row identity
and allow distinct cells with coincident coordinates. Singletons retain their
features under smoothing and do not contribute a neighbour-agreement term.
Contiguity is cell-weighted over labelled cells with labelled within-field
neighbours. The permutation baseline preserves class frequencies within fields.
CODEX therefore has corrected labelled-neighbour contiguity 0.120425 on 15,454
labelled cells, with one labelled singleton, rather than the earlier all-cell
diagnostic (~0.11795). Its model input still contains 16,000 cells.

The historical cross-field audit remains 13,208/19,854 MIBI links and
95,840/96,000 CODEX links. These are legacy counts, not new graphs. All repaired
stored spatial graphs have zero cross-field links.

## Methods and negative control

Seven procedures × seven objects × seeds [1,2,3] = **147 complete records**,
with no failed run. Procedures are expression-only PCA30--KMeans, neighbour-mean
PCA30--KMeans, BANKSY-style PCA20 augmentation, SpatialLeiden, Tessera, STAGATE
and SEDR. The last three use 120/600/200 epochs. Reference class count is shared;
the comparison is reference-conditioned, not wholly label-free.

STAGATE's injected graph is checked after actual PyG conversion. SEDR receives
a symmetric within-field graph and a fixed seeded within-field non-neighbour
mask. The adapter repairs the upstream candidate-index sampling issue, caps
negative counts at available non-neighbours and normalises within-field pairs.
This is a declared repaired adapter, not an untouched native implementation.

The expression-only fitting function receives features, K and seed, not coordinates.
Its primary predictions are never refined. Fixed two-round refinement for spatial
methods is a separate result, not selected using truth. Refined-control and
pooled-coordinate counterfactuals are visibly diagnostic only. Feature-space
links (SpatialLeiden) and Tessera feature-space contrastive negatives may span
fields; they do not assert physical distance across independent frames.

SpaceFlow, GraphST and SpaGCN remain `not_run` in explicit availability records.
Their new adapter entry points fail closed. The old `expanded_bench.py` is not
the new reanalysis entry point and its unsafe-loader guard is not bypassed.

## Statistical and mechanism scope

Average ranks in `stat_rigor.py` and the new summary pass row-permutation tests.
The historical ordinal-rank counterexample ranges 0.584748–0.956280; correct
mid-ranks yield 0.928385 invariantly. This diagnoses a bug on old outcomes; it is
not stronger corrected tissue evidence. New partial associations are descriptive.

Twenty generator endpoints (two substrates × five seeds × two endpoints) verify
that expression spatial autocorrelation and class balance move under label
regeneration. The fixed conditional expression–label rule is not fixed alignment.
The original manuscript retains specified-operation total effects and withdraws
independent-contiguity/clean-decoupling language. Historical LODO predicts
advantage >0.05 with overlapping folds; independent-binomial inference/power are
withdrawn, and proxy selection was not nested in its folds.

## Verification and remaining limits

The run `QA.json` records independent raw-HDF5 identity/metric checks and
147/147 prediction rescores. Manuscript ledgers bind tables, figures and sources.
The default rebuild reads these results without fitting models. Full historical
n11/n12/n14 replacement, three external adapters, eight objects' frame provenance
and legacy confidence/calibration remain unresolved. Both submission snapshots
are preserved. The current worker cannot attest visual inspection: `view_image`
is unavailable, so renders and machine layout checks are supplied for review.
The explicit current field is `visual_review=NOT_PERFORMED`, never visual PASS.

## Archived pre-integration audit

**HISTORICAL ONLY — superseded for the retained seven-object/seven-procedure scope.**
The following record describes the state before the 147 completed runs and their
independent rescore. Its blanket rerun-unresolved and no-new-estimate statements
are not current. Current remaining limits are enumerated immediately above.

> Historical status at that time: SCIENTIFIC_SUBMISSION_HOLD; coordinate/control reruns and confidence validation unresolved.
>
> MIBI: 13208 of 19854 original six-neighbour links cross local fields. CODEX: 95840 of 96000 original links cross 565 fields in the fixed 16000-cell subsample.
>
> The repaired core graph/refinement helpers preserve field identity (zero cross-field links on both existing coordinate inputs). External method adapters still require complete field-aware implementation and reruns. Legacy unsafe loaders fail closed.
>
> The nominally nonspatial historical floor entered spatial refinement. Candidate scoring excludes refinement for that control, records original seed identities, refuses incomplete panels and binds caches to input/code content.
>
> Significance and winner-count acceptance gates are removed. Negative/nonsignificant results are valid outcomes. Oracle selection against reference labels remains an evaluation ceiling, not deployable method selection.
>
> The soft confidence head has no loss/held-out calibration and is unrelated to reported KMeans labels. fit_predict now withholds confidence. Cluster correctness for ECE is repaired to compare mapped predictions with true labels; the old ECE and calibration-mechanism claims are withdrawn. No replacement ECE is supplied.
>
> The architecture depicts implemented outputs; the unsupported confidence-flag ablation bar is omitted without changing its archived source value. Synthetic causal claims remain generator-scoped and are not pooled with tissue evidence.
>
> The 11/12/14 panels overlap. CODEX niche labels are cluster-derived proxy labels. Absence of frame identifiers in other objects, including COAD/LIHC/OV, does not establish valid single-frame provenance.
>
> Retrospective power calculations are conditional planning, not preregistration. No data/model rerun or new scientific estimate is claimed.
>
> Run `python -m pytest -q reproducibility/tests/test_tessera_method_integrity.py` from the capsule root. Tests are behavior/identity checks, not scientific model reruns.
>
> The pre-build contract checks byte identity of bound evidence and code. It must not be interpreted as scientific clearance.
