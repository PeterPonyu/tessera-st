# Compact GitHub source and documents

This is the compact distribution of **2026.10.11-controls-audit**, selected from the finalized complete Tessera archive with SHA256 `09c75b7a7927755e60d4097fbb85b3120ee792bc2f1372b0f2d1a14a5cdde680`. The complete archive is associated with software [DOI 10.5281/zenodo.23308077](https://doi.org/10.5281/zenodo.23308077); publication status follows the public record.

Current source, protocols, manuscript sources/PDFs, compact score tables, receipts and source/checksum instructions are included. Retained `.npz`, `.npy`, `.pt`, `.pth` and other large assay/model-state payloads are omitted. `COMPACT-DISTRIBUTION.json` lists every omission with its exact checksum. `provenance/full-package-MANIFEST.sha256` preserves the complete-archive file manifest.

`python portable/verify_manifest.py` checks this compact distribution's own manifest. `python portable/import_check.py` checks the packaged import closure in the documented environment. Full saved-result verification, primary-input preflight, the 82 scientific tests and refitting require the complete archive; they cannot be established by missing arrays in this compact copy. Extract the complete archive into a separate directory and follow its `README.md` and `REPRODUCE.md` for those commands.

The complete archive's source guide follows. Statements about included arrays or saved numerical replay in that guide refer to the complete archive.

---

# Tessera controls audit: analysis and editable sources

Software/source version: `2026.10.11-controls-audit`. The software DOI
`10.5281/zenodo.23308077` is reserved for this release and is not claimed to be
published by this package-preparation step. Manuscript/preprint identity is
separate from software archive identity. The evaluated original implementation is separately archived at
`10.5281/zenodo.23101191`; it does not archive these later controls and source
additions.

This standalone archive preserves the evaluated Tessera implementation, original
protocols and run locks, editable manuscript and figure sources, current primary
and patient-held-out transfer saved predictions, follow-up control evidence,
external HER2ST and extension result evidence, and the independent CRC pilot and
its disclosed annotation-QC sensitivity analysis. The numerical research outputs
and frozen protocols remain unchanged. Final document/figure sources and citation metadata are bound by the sealed
package manifest; historical document QA records describe their dated originals.

The main comparison concerns this released implementation and the supplied
controls. The reference annotations have distinct biological and computational
origins. Repeated fits, tissue sections and sensitivity analyses are not new
biological donors. Interpretation and retained-evidence limits are recorded in
`provenance/verification-scope.json` and the original protocols/results.

## Verify retained results after extracting anywhere

Use Python 3.13.7 with the versions in `requirements-analysis.txt`, or the validated
existing audit runtime recorded in `provenance/environment-observed.json`. From
the extracted package root:

```bash
PYTHONDONTWRITEBYTECODE=1 python portable/verify_manifest.py
PYTHONDONTWRITEBYTECODE=1 python portable/import_check.py
PYTHONDONTWRITEBYTECODE=1 python portable/verify_saved_results.py
PYTHONDONTWRITEBYTECODE=1 python source/scripts/recover_followup_runners.py --verify
```

These commands require no network, GPU, author-absolute paths, source symlinks or
external working-tree checkout. The verifier checks the saved predictions and
receipts directly and forbids neural and KMeans fitting. It reconstructs the 351
primary rows and 90 transfer target rows; checks follow-up table arithmetic and
39 saved gate profiles; checks HER2ST/extension selections and retained seed-1
partitions; and reconstructs CRC ARI, unique-edge AUROC, patient bootstrap and a
stated shuffled-label subset. The scripts print machine-readable evidence.

The source has two different reproduction levels. Direct saved-score checking is
available where the producers retained predictions. Historical HER2ST/extension
learned embeddings, remaining seed partitions, whole-section predictions and
geometry-null embeddings were not retained. Their source/receipt/table and
selection consistency checks do not establish an independent replay of those
fits. The original `validation-independent-crc/verify_pilot.py` additionally
checks large raw archives and author-local historical integration backups; the
portable command checks only the explicitly documented saved-evidence scope.

## Acquire exact upstream files separately

Large original assay archives are excluded. The primary derived inputs are
included and bound to their original input records. `provenance/upstream-inputs.json`
records their raw source hashes, exact HER2ST and CRC acquisition URLs, byte counts
and SHA256 checksums, and the publisher CRC MD5 checksums. Matching a publication
name alone is insufficient to identify equivalent input bytes.

For manually acquired files, verification and copying are explicit:

```bash
python portable/acquire.py her2st --source-dir /path/to/manual/her2st --destination /path/to/fresh/her2st
python portable/acquire.py crc --source-dir /path/to/manual/crc --destination /path/to/fresh/crc
```

To acquire from the recorded URLs instead, add `--download`. Optional
`--proxy http://host:port` uses a per-request proxy without any machine-specific
port restriction or global environment changes. Acquisition rejects byte or
checksum differences. J1 counts were unavailable in the historical HER2ST source
and J1 was excluded before fitting; the manifest keeps that exclusion and never
substitutes a different specimen.

`portable/run_analysis.py` provides configurable input and fresh-output locations
for the original preserved producers; see `--help` and `REPRODUCE.md`. The historical
workspace fetch gate is supplied only as checksum-bound provenance. The portable
runner resolves its helper within this archive and checks acquired inputs before
any experimental path is invoked.

## Environment, documents and full experimental reproduction

`environment.yml` and `requirements-analysis.txt` specify the observed supported
Python audit/import environment. The current tested PyTorch build is recorded in
`provenance/environment-observed.json`; the frozen original training environment
is preserved in the primary and transfer run locks. No fresh package installation
or full refit was performed during this packaging work.

Document reconstruction instructions remain in `EDITABLE_SOURCE_README.md`.
Document rebuilding requires the stated TeX/Poppler tools separately from the
Python analysis dependencies. Saved-result verification and document compilation
do not certify a new GPU refit. For a refit, follow each original protocol and
write to fresh output directories; CUDA reductions were not guaranteed bitwise
deterministic in the original runs.

Rights and source-specific reuse limits are documented in `LICENSING.md` and the
preserved source records. The release includes original source and developer
provenance; it does not relabel shared measurements or existing labels as
independent biological evidence.

The neural core and expanded analysis adapters have separate provenance.
`provenance/historical-core-versus-adapters.json` compares exact included source
hashes to the historical public software checkout. Matching encoder, model,
gating, loss and metrics code does not imply that every graph, preprocessing,
configuration or trainer byte matches that release. Scientific receipts identify
the expanded-analysis implementation actually evaluated here. The same-graph
HER2ST comparison is exploratory and remains an author/developer analysis.
No 10x breast H&E image/background is redistributed; the breast figure contains
author-generated coordinate maps. Source annotations retain their source terms.
The excluded image provenance and replacement scope are stated precisely in
`provenance/breast-image-source-and-rights.json` and `LICENSING.md`.
