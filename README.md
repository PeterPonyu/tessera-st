# Tessera-ST: matched neighbourhood controls and label diagnostics for spatial encoders

## Latest version

The analyses of the third review round are on the `revision-r3` branch and at the tag
[2026.11.17-revision-r3](https://github.com/PeterPonyu/tessera-st/tree/2026.11.17-revision-r3): label diagnostics of
reference annotations, niche labels rebuilt at known construction scales as a positive control, a prospective test in
three new cohorts, and GraphST, STAGATE and Tessera-ST each compared with parameter-free propagation on its own graph.
The README of that version (section "Round-3 analyses") gives the commands for each step.

- Code, derived inputs and result tables: [doi:10.5281/zenodo.24315838](https://doi.org/10.5281/zenodo.24315838)
  (version 2026.11.17-revision-r3, which keeps every file of the earlier versions).
- Analysis protocol for the new cohorts, deposited before their expression data were downloaded:
  [doi:10.5281/zenodo.24303149](https://doi.org/10.5281/zenodo.24303149).
- First-revision analyses: [doi:10.5281/zenodo.23999069](https://doi.org/10.5281/zenodo.23999069), tag
  [2026.11.05-revision-r1.3](https://github.com/PeterPonyu/tessera-st/tree/2026.11.05-revision-r1.3), branch `revision-r1`.

The sections below describe the original release of October 2026, which this branch still carries.

## Original release (October 2026)

### Manuscript documents of the original submission

The reviewed main Article, Supplementary Information and editable document sources are available in the [submission-2026-10-12 document release](https://github.com/PeterPonyu/tessera-st/releases/tag/submission-2026-10-12) and in [submission-2026-10-12/](submission-2026-10-12/README.md).

The software and saved-analysis release of that version is 2026.10.11-controls-audit, DOI [10.5281/zenodo.23308077](https://doi.org/10.5281/zenodo.23308077). It archives the frozen computational evidence and its earlier document sources. The later editorial revision has a separate GitHub document tag; it adds no model fits or biological samples. Earlier document files remain available for provenance.

Frozen research-source version **2026.10.11-controls-audit** is in [current-release/](current-release/). The earlier archived [inline manuscript](current-release/output/scientific-reports/scientific-reports-manuscript-inline.pdf), [journal manuscript](current-release/output/scientific-reports/scientific-reports-manuscript.pdf) and [supplement](current-release/output/scientific-reports/scientific-reports-supplement.pdf) report matched neighbourhood controls, component removals, patient-held-out transfer, exploratory HER2ST comparator extensions and the separate colorectal pilot. The author developed and maintains the evaluated implementation.

The root `src/` implementation is byte-identical to the current archive's implementation. The original neural core and the field-aware graph/data adapters have distinct provenance, documented in [the comparison record](current-release/provenance/historical-core-versus-adapters.json). The earlier public manuscript and implementation are archived at [DOI 10.5281/zenodo.23101191](https://doi.org/10.5281/zenodo.23101191); it does not archive later controls or document changes.

The complete versioned source and saved-evidence archive is associated with [software DOI 10.5281/zenodo.23308077](https://doi.org/10.5281/zenodo.23308077). This version is published, and every public file plus the reconstructed complete ZIP has been downloaded and hash-verified. The finalized complete ZIP has SHA256 `09c75b7a7927755e60d4097fbb85b3120ee792bc2f1372b0f2d1a14a5cdde680`.

This checkout and the compact source/document ZIP include current code, protocols, documents, score tables and receipts. Retained input/embedding arrays and model states are supplied in the complete archive. [Compact-distribution instructions](current-release/README.md) explain which checks run here and which require the complete archive. Saved-result verification and full refitting have distinct requirements and documented limits.

Author-owned code is MIT; author-owned research content is CC BY 4.0. [Licensing](LICENSING.md) and [source notices](NOTICE.md) preserve third-party terms. Breast displays use author-generated coordinate maps; no 10x breast H&E image/background is redistributed in the current release. The separate dataset-licensed CRC images retain their attribution.

The mutable compact checksum list omits the removed obsolete cover draft; the complete published archive retains its original manifest. `PUBLICATION_STATUS.json` records this housekeeping change separately from the frozen software and document releases.
