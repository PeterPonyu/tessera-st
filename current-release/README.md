# Tessera-ST: field-aware spatial evaluation audit

Author-review release, 2 October 2026. **SCIENTIFIC_SUBMISSION_HOLD**.
This is a protocol audit, not a universal contiguity law, a validated method
selector, calibrated confidence output, or a claim of superior architecture.

Code: https://github.com/PeterPonyu/tessera-st/tree/author-review-2026-10-02/current-release
Version archive: https://doi.org/10.5281/zenodo.23101191
Anonymous public availability is recorded separately in the publication
receipt; a DOI in source text alone does not prove publication.
Publication does not remove scientific HOLD.

## Current manuscript

`manuscript/paper.pdf` is the **current, venue-neutral single-column article**,
generated from the revised seven-object scientific source, not the historical
20 August 2026 paper. `manuscript/SI.pdf` is the matching readable supplement
(Tables S1–S5). `provenance/source-capsule-IEEE-paper.pdf` preserves the
previous double-column author-review face, not the current default.

## Evidence and boundaries

Seven objects (DLPFC, MERFISH, STARmap, MIBI, Open-ST, CODEX, Zhuang), seven
procedures, seeds 1/2/3: 147 saved method–seed runs. SpaceFlow, GraphST and
SpaGCN remain unavailable in the corrected protocol. Eight historical objects
with unresolved frame provenance are not filled with old scores. CODEX
SpatialLeiden is not equal-K matched; niches are cluster-derived proxies.
Three optimisation seeds are not independent tissue replicates.

`experiments/generated/field-identified-fixed-backend-20260923-v1/` contains
prediction arrays, selected row/field maps, exact source hashes, per-run records
and package identities. `manuscript/ledgers/revision_20260923/` binds the
current tables and figure inputs. Other ledgers and older experiment drivers
are historical context only: their original numbers are not corrected-panel
evidence. Synthetic effects are generator/intervention/scorer-specific and
are never pooled with measured tissue associations. The confidence head is
withheld and no replacement ECE claim is supplied.

## Reproduction levels

From this directory, with existing Python dependencies:

```bash
python verify_release.py
python -m pytest -q reproducibility/tests/test_rank_ties.py reproducibility/tests/test_supporting_information.py
bash manuscript/scripts/rebuild_current.sh
```

Verification rescales no score and refits no model: it checks checksums,
within-field saved graph edges and all 147 stored prediction ARIs, then checks
the manuscript claim contract. The document build consumes committed vector
figures and tables with TeX Live/latexmk and Poppler. Rendering from raw images
is a separate operation with third-party input prerequisites, not the default.
Arial is needed only for the optional scientific-figure redraw; its font file
is not bundled. No dependency is installed automatically.

For full tissue refits, obtain the exact upstream assays and STAGATE/SEDR
implementations in `DATA_SOURCES.md`, set `TESSERA_DATA_ROOT` to the source
workspace containing `data/` and `external/`, then use the field-aware runner.
With those external implementations present, run
`python -m pytest -q reproducibility/tests/test_field_aware_adapters.py`.
That upstream-dependent suite is separate from the default offline tests;
missing STAGATE is a missing prerequisite, not a silent test skip.
Its saved protocol contains original absolute paths as **historical run
provenance**, not portable commands. Original capsule contracts and source
hashes remain in `provenance/`; the current `RELEASE-MANIFEST.json` binds this
public face. An identity manifest certifies bytes, not scientific validity.

## Licensing and status

Author-owned code: MIT (`LICENSE`). Author-written manuscript, explanations,
tables and author-generated result aggregates: CC BY 4.0. Upstream assays,
labels, photographs, external code and model weights keep their original
conditions; see `THIRD_PARTY_NOTICES.md`. Third-party raw measurement matrices,
external implementation source and model weights are not included.

Local numerical and rendering checks are not independent scientific review,
external validation, author sign-off or journal submission. This release is
public for research inspection while scientific submission remains on HOLD.
