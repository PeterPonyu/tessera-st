# Tessera-ST: a gated graph encoder for spatial omics, and its controls audit

Tessera-ST is a self-supervised graph encoder for spatial molecular data (package `tessera_st`). Earlier releases call it Tessera. It is unrelated to other tools named Tessera, such as the tissue-tiling method of Stein et al. (2025), the multi-sample differential-expression model of Constantine et al. (2026) and the representation-learning framework of Dehghani Amirabad et al. (2026). The accompanying study tests how much of the encoder's spatial gain is explained by neighborhood averaging, using matched controls, component removals and patient-held-out evaluation.

## Reproduce this study

There are three levels of reproduction. Level 1 needs only the saved results. Levels 2 and 3 refit models.

**Getting the files.** Clone this repository. The GitHub copy omits large arrays (inputs, embeddings, model states). To get them:

- **Original submission:** download the complete archive at [doi:10.5281/zenodo.23308077](https://doi.org/10.5281/zenodo.23308077) and reassemble it as described in its `REASSEMBLE.md`.
- **Revision analyses:** download the revision archive listed in the manuscript's Code availability statement.

**Environment.** Python 3.13 with the versions in `current-release/requirements-analysis.txt`. PyTorch 2.12 with CUDA is used for the GPU steps; every step also runs on CPU, more slowly.

**Where the files go.** By default the scripts look for `data/`, `results/` and `validation/` next to this repository, and otherwise for the same folders under `current-release/`. Override with the environment variables `TESSERA_DATA`, `TESSERA_RESULTS`, `TESSERA_VALIDATION`, `TESSERA_R1_OUT` (revision outputs), `TESSERA_R1_SUMMARY` (revision summary tables) and `TESSERA_LAB_ROOT` (raw anatomical sections). The figure scripts honor the same variables, write to the folders given by `R1_PDF_OUT`, `R1_JPG_OUT` and `R1_QA_OUT`, and use Arial; if Arial is not installed system-wide, set `R1_FONT_DIR` to a folder containing the Arial TTF files.

### Level 1: regenerate every table and figure from saved results (minutes, no fitting)

```bash
python current-release/portable/verify_saved_results.py      # original tables: recomputed from saved predictions
python revision-r1/analyze_r1.py                              # revision tables, decision rules, manuscript numbers
python revision-r1/figures/build_r1_figures.py all            # every main and supplementary figure (needs matplotlib and PyMuPDF)
```

| Manuscript item | Produced by |
|---|---|
| Tables 1–3 and Supplementary Tables S1–S15 | the original analyses, checked by `verify_saved_results.py` |
| Supplementary Table S16 (settings) and S17 (data sources) | written in the manuscript source from the configuration and the source manifests |
| Supplementary Tables S18–S21 and Figures 5 and 6 | `revision-r1/analyze_r1.py` and `revision-r1/figures/build_r1_figures.py` |
| All other figures | `revision-r1/figures/build_r1_figures.py` from the saved results |

### Level 2: refit every analysis from the archived derived inputs (about 2 h on a laptop GPU)

The original analyses:

```bash
python current-release/portable/run_analysis.py primary  --device cuda --output-root OUT/primary
python current-release/portable/run_analysis.py transfer --device cuda --output-root OUT/transfer
python current-release/portable/run_analysis.py her2st   --raw-dir RAW/her2st --output-root OUT/her2st
python current-release/portable/run_analysis.py crc      --output-root OUT/crc
```

The analyses added during review follow `revision-r1/PROTOCOL-R1.md`. Run the tests before any fit:

```bash
python -m pytest revision-r1/test_r1.py                       # the training loop reproduces the original fits exactly on CPU
python revision-r1/run_budget.py gpu;    python revision-r1/run_budget.py her2st;      python revision-r1/run_budget.py transfer
python revision-r1/run_tuning.py grid-gpu; python revision-r1/run_tuning.py grid-her2st; python revision-r1/run_tuning.py mechanism
python revision-r1/run_k_sweep.py sweep; python revision-r1/run_k_sweep.py harmonize; python revision-r1/run_k_sweep.py refine
python revision-r1/native/prepare_inputs.py                   # then the native methods; see revision-r1/native/VERSIONS.md
python revision-r1/analyze_r1.py
```

Breast (HER2ST) fits run on one CPU thread, so they are deterministic. GPU fits are not bitwise deterministic, so refitted scores can differ slightly from the saved ones.

### Level 3: rebuild the derived inputs from the public sources

The public datasets and their repository identifiers are listed in the manuscript's Supplementary Table S17. Each acquired file is checked against the size and SHA-256 recorded in the archive:

```bash
python current-release/portable/acquire.py her2st --source-dir RAW/manual-her2st --destination RAW/her2st
python current-release/portable/acquire.py crc    --source-dir RAW/manual-crc    --destination RAW/crc
python current-release/portable/run_analysis.py prepare --lab-root RAW/sources --output-root OUT/inputs
```

## Current manuscript documents

The reviewed main Article, Supplementary Information and editable document sources are available in the [submission-2026-10-12 document release](https://github.com/PeterPonyu/tessera-st/releases/tag/submission-2026-10-12) and in [submission-2026-10-12/](submission-2026-10-12/README.md).

The software and saved-analysis release remains version 2026.10.11-controls-audit, DOI [10.5281/zenodo.23308077](https://doi.org/10.5281/zenodo.23308077). It archives the frozen computational evidence and its earlier document sources. The later editorial revision has a separate GitHub document tag; it adds no model fits or biological samples. Earlier document files remain available for provenance.

Frozen research-source version **2026.10.11-controls-audit** is in [current-release/](current-release/). The earlier archived [inline manuscript](current-release/output/scientific-reports/scientific-reports-manuscript-inline.pdf), [journal manuscript](current-release/output/scientific-reports/scientific-reports-manuscript.pdf) and [supplement](current-release/output/scientific-reports/scientific-reports-supplement.pdf) report matched neighbourhood controls, component removals, patient-held-out transfer, exploratory HER2ST comparator extensions and the separate colorectal pilot. The author developed and maintains the evaluated implementation.

The root `src/` implementation is byte-identical to the current archive's implementation. The original neural core and the field-aware graph/data adapters have distinct provenance, documented in [the comparison record](current-release/provenance/historical-core-versus-adapters.json). The earlier public manuscript and implementation are archived at [DOI 10.5281/zenodo.23101191](https://doi.org/10.5281/zenodo.23101191); it does not archive later controls or document changes.

The complete versioned source and saved-evidence archive is associated with [software DOI 10.5281/zenodo.23308077](https://doi.org/10.5281/zenodo.23308077). This version is published, and every public file plus the reconstructed complete ZIP has been downloaded and hash-verified. The finalized complete ZIP has SHA256 `09c75b7a7927755e60d4097fbb85b3120ee792bc2f1372b0f2d1a14a5cdde680`.

This checkout and the compact source/document ZIP include current code, protocols, documents, score tables and receipts. Retained input/embedding arrays and model states are supplied in the complete archive. [Compact-distribution instructions](current-release/README.md) explain which checks run here and which require the complete archive. Saved-result verification and full refitting have distinct requirements and documented limits.

Author-owned code is MIT; author-owned research content is CC BY 4.0. [Licensing](LICENSING.md) and [source notices](NOTICE.md) preserve third-party terms. Breast displays use author-generated coordinate maps; no 10x breast H&E image/background is redistributed in the current release. The separate dataset-licensed CRC images retain their attribution.

The mutable compact checksum list omits the removed obsolete cover draft; the complete published archive retains its original manifest. `PUBLICATION_STATUS.json` records this housekeeping change separately from the frozen software and document releases.
