# Tessera current author-review revision — 2026-09-23

Status: **SCIENTIFIC_SUBMISSION_HOLD**, with completed original-manuscript integration
of a seven-object field-identified reanalysis. This section supersedes the earlier
audit below. Earlier statements that no models were rerun are historical, not current.

The working face is `manuscript/paper.tex` and its compiled `manuscript/paper.pdf`.
The near-ready manuscript was revised in place, not replaced by a short candidate
report. `manuscript/submission_flat/` and `manuscript/submission_flat_candidate/`
remain untouched historical snapshots; they are not the current revised source.

## Completed scope

Protocol: `field-identified-fixed-backend-20260923-v1`.

- Seven objects: DLPFC, MERFISH, STARmap, MIBI, Open-ST, CODEX and Zhuang.
- Seven procedures: expression-only PCA--KMeans, neighbour mean, BANKSY-style,
  SpatialLeiden, Tessera, STAGATE and SEDR.
- Seeds 1, 2 and 3: **147/147 method--seed runs completed, no recorded failures**.
- Epoch budgets: Tessera 120, STAGATE 600, SEDR 200. All embedding methods use
  KMeans with ten initialisations; SpatialLeiden supplies community assignments.
- Spatial graphs/features/metrics/refinement preserve field identity. STAGATE's
  actual runtime graph is checked. SEDR receives explicitly within-field positive
  and non-neighbour reconstruction pairs.
- The primary expression-only function has no coordinate parameter and receives
  no refinement. Separate refined-control/pooling counterfactuals are diagnostic
  arms, not eligible controls or true-label oracle alternatives.
- Independent raw-HDF5 checks recover cells, fields, labels and contiguity;
  saved per-cell predictions reproduce all 147 ARIs.
- Average-rank partial correlations are row-order invariant. The original manuscript
  now retains specified-intervention total effects, not independent contiguity.

Complete input/prediction/seed records are in
`experiments/generated/field-identified-fixed-backend-20260923-v1/`. Bundled manuscript
ledgers and figure-source bindings are in `manuscript/ledgers/revision_20260923/`.
No new estimate is mixed with historical n11/n12/n14 rows.

## Current build and verification

```bash
bash manuscript/scripts/rebuild_current.sh
python manuscript/verify_committed_claims.py
make -C manuscript verify
python -m pytest -q reproducibility/tests
```

The default build checks locked evidence/code, renders the new figures from bundled
ledgers and compiles with native `latexmk`. Retained historical PDF panels are bound
inputs. It does not train, download data, install dependencies, rerun historical
experiments or overwrite either submission snapshot. Existing Python
NumPy/SciPy/Matplotlib/PyMuPDF, TeX Live/latexmk, Poppler and Arial are used.

`release-current.json` and `release-spec.json` bind the same current face;
`attestation/current-revision-build.json` is its new build record. The older
`attestation/build-attestation.json` is a historical submission record only.
`attestation/historical-snapshots-20260923.json` records byte-for-byte comparison
of 180 files in both submission snapshots with the preintegration backup.
`attestation/revision-tests.xml` contains the 34-test regression result (zero
failures/errors/skips); its byte identity is bound by the current release. A default
rebuild checks that saved evidence and does not silently claim another test run.
The build's source checks do not silently refresh the method lock; that requires
the explicit reviewed-source command `python manuscript/scripts/bind_revision_contracts.py --bind-method`.

Optional independent rescore, using the existing raw public HDF5 files but no refits:

```bash
python experiments/verify_field_aware_panel_20260923.py
```

## Remaining blockers

- **SpaceFlow, GraphST and SpaGCN are not run** under the corrected protocol.
  SpaceFlow's global distance penalty still lacks a field-aware adapter, and
  `gudhi` is absent in this environment. No replacement scores are supplied.
- Frame provenance is not established here for seqFISH, osmFISH, BRCA, IMC,
  Slide-seqV2 and the COAD/LIHC/OV objects. Exclusions are identity-based,
  not outcome-based; the retained set is a convenience panel.
- Historical oracle correlations, backend/proxy analyses, winner/rank summaries
  and tissue ablations remain old-protocol records, not repaired evidence.
- Confidence/ECE claims remain withdrawn; the soft head is not trained or
  calibrated for KMeans assignments.
- Independent scientific review and visual review of the revised PDF remain.
  This worker's `view_image` calls are rejected by the environment. Page rendering
  and machine geometry checks cannot be called visual QA. Current contracts record
  **`visual_review=NOT_PERFORMED`**; page renders are available for human review.

Hashes and tests establish the stated implementation/result identity, not biological
validity, external validation or submission readiness.

## Archived pre-integration description

**HISTORICAL ONLY — superseded by the completed 2026-09-23 integration above.**
The quoted text below records the pre-integration state, not current instructions
or an additional current-status declaration. In particular, its blanket
"coordinate/control reruns ... unresolved" and "No data/model rerun" statements
are no longer true for the seven retained objects and seven procedures: their
147 runs, corrected expression-only comparator and prediction rescore are complete.
Current HOLD concerns the three explicitly unrun methods, eight unresolved object
identities, old-protocol dependent analyses and the scientific/visual review limits
listed under **Remaining blockers**, not an absence of the completed reanalysis.

### Archived package description (not current build instructions)

> Paper: Tessera.
>
> Run `bash manuscript/scripts/rebuild_current.sh` from the extracted package root. The default build verifies bound input/code identities and renders existing results and object assets. It does not train models or download assay data. Existing tools: Python (numpy, pandas, matplotlib, PyMuPDF, Pillow; RDKit for molecular 2D diagrams), R figure packages, TeX Live/latexmk and Arial. PyMOL is optional for 3D rerender via PYMOL_EXE.
>
> This package contains historical outputs for traceability. Historical PASS/status files do not certify this revision; the following method audit and current manifest take precedence. Figure/PDF reproducibility does not certify submission readiness.

### Archived method-integrity audit (pre-integration status)

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
