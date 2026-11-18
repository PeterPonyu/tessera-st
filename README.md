# Tessera-ST: a gated graph encoder for spatial omics, with matched controls and label diagnostics

Tessera-ST is a self-supervised graph encoder for spatial molecular data (package `tessera_st`). Some files call it Tessera. It is unrelated to other tools named Tessera, such as the tissue-tiling method of Stein et al. (2025), the multi-sample differential-expression model of Constantine et al. (2026) and the representation-learning framework of Dehghani Amirabad et al. (2026). The accompanying study asks what a learned spatial encoder adds beyond averaging each observation with its neighbours, using label diagnostics of reference annotations, matched parameter-free controls, component removals and patient-held-out evaluation.

## Reproduce this study

There are three levels of reproduction. Level 1 needs only the saved results. Levels 2 and 3 refit models.

**Getting the files.** Clone this repository. The GitHub copy omits large arrays (inputs, embeddings, model states). To get them:

- **Archived files:** download the archive at [doi:10.5281/zenodo.24315838](https://doi.org/10.5281/zenodo.24315838), which holds every earlier file:
  - the code, the derived inputs of the prospective cohorts and every result table of the label diagnostics, construction-scale experiment, prospective test and encoder comparison (`revision-r3/`);
  - the code and result tables of the budget, tuning and cluster-number analyses (`revision-r1/`);
  - the complete archive of the calibration analyses (controls audit of Tessera-ST), reassembled as described in its `REASSEMBLE.md`.
- **Protocol:** the protocol for the prospective cohorts was deposited before their expression data were downloaded, at [doi:10.5281/zenodo.24303149](https://doi.org/10.5281/zenodo.24303149). See [Prospective analyses](#prospective-analyses) below.

**Environment.** Python 3.13 with the versions in `current-release/requirements-analysis.txt`. PyTorch 2.12 with CUDA is used for the GPU steps; every step also runs on CPU, more slowly. The encoder fits of the prospective analyses (GraphST, STAGATE and Tessera-ST, `revision-r3/remote/`) ran in a separate environment: Python 3.11.10 and PyTorch 2.5.1 with CUDA 12.4, in the container image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. `results/revision-r3/W4/BOX.md` in the archive and every fit record give the exact versions; their calibration, scoring and summaries ran in the Python 3.13 environment above.

**Where the files go.** By default the scripts look for `data/`, `results/` and `validation/` next to this repository, and otherwise for the same folders under `current-release/`. Override with the environment variables `TESSERA_DATA`, `TESSERA_RESULTS`, `TESSERA_VALIDATION`, `TESSERA_R1_OUT` (outputs of `revision-r1/`), `TESSERA_R1_SUMMARY` (its summary tables) and `TESSERA_LAB_ROOT` (raw anatomical sections). The figure scripts honor the same variables, write to the folders given by `R1_PDF_OUT`, `R1_JPG_OUT` and `R1_QA_OUT`, and use Arial; if Arial is not installed system-wide, set `R1_FONT_DIR` to a folder containing the Arial TTF files.

### Level 1: regenerate every table and figure from saved results (minutes, no fitting)

```bash
python current-release/portable/verify_saved_results.py      # original tables: recomputed from saved predictions
python revision-r1/analyze_r1.py                              # tables, decision rules and numbers of revision-r1/
python revision-r1/figures/build_r1_figures.py all            # every main and supplementary figure (needs matplotlib and PyMuPDF)
```

| Item (numbering of the controls-audit documents) | Produced by |
|---|---|
| Tables 1–3 and Supplementary Tables S1–S15 | the original analyses, checked by `verify_saved_results.py` |
| Supplementary Table S16 (settings) and S17 (data sources) | written from the configuration and the source manifests |
| Supplementary Tables S18–S25 and Figures 5 and 6 | `revision-r1/analyze_r1.py` and `revision-r1/figures/build_r1_figures.py` |
| All other figures | `revision-r1/figures/build_r1_figures.py` from the saved results |

### Level 2: refit the supported analyses from the archived derived inputs

The primary analyses of the controls audit take about 2 h on a laptop GPU; the budget, tuning and cluster-number analyses take longer. The graphs of 200 neighbors need a GPU with about 80 GB of memory, and the breast and colorectal fits need the raw downloads of Level 3. Some extension outputs cannot be refitted from scratch; they are kept with the archived results.

The primary analyses:

```bash
python current-release/portable/run_analysis.py primary  --device cuda --output-root OUT/primary
python current-release/portable/run_analysis.py transfer --device cuda --output-root OUT/transfer
python current-release/portable/run_analysis.py her2st   --raw-dir RAW/her2st --output-root OUT/her2st
python current-release/portable/run_analysis.py crc      --output-root OUT/crc
```

The budget, tuning and cluster-number analyses follow `revision-r1/PROTOCOL-R1.md`. Run the tests before any fit:

```bash
python -m pytest revision-r1/test_r1.py                       # the training loop reproduces the primary fits exactly on CPU
python revision-r1/run_budget.py gpu;    python revision-r1/run_budget.py her2st;      python revision-r1/run_budget.py transfer
python revision-r1/run_tuning.py grid-gpu; python revision-r1/run_tuning.py grid-her2st; python revision-r1/run_tuning.py mechanism
python revision-r1/run_k_sweep.py sweep; python revision-r1/run_k_sweep.py harmonize; python revision-r1/run_k_sweep.py refine
python revision-r1/native/prepare_inputs.py                   # then the native methods; see revision-r1/native/VERSIONS.md
python revision-r1/run_tuning.py grid-gpu-wide; python revision-r1/run_tuning.py grid-her2st-wide; python revision-r1/run_tuning.py mechanism-wide
python revision-r1/run_budget.py early; python revision-r1/run_budget.py her2st-seeds
python revision-r1/run_k_sweep.py agreement
python revision-r1/analyze_r1.py
```

The last three command lines run the exploratory extensions A1 (graphs of 50 and 100 neighbors) and A3 (earlier checkpoints, ten breast seeds) of the protocol, and A4 (partition agreement at the reference cluster number, from saved embeddings, about a minute); A1 and A3 take several more hours. Extension A2, the 200-neighbor lung graphs, needs a GPU with about 80 GB of memory. `revision-r1/remote/setup_box.sh` prepares such a machine from a commit of this repository and `revision-r1/remote/run_a2.sh` runs the modes `hwcheck-a2`, `grid-gpu-a2` and `mechanism-a2` of `run_tuning.py`; the machine and run used for the archived scores are recorded in `E2/remote-a2/` of the results archive.

Breast (HER2ST) fits run on one CPU thread, so they are deterministic. GPU fits are not bitwise deterministic, so refitted scores can differ slightly from the saved ones.

### Level 3: rebuild the derived inputs from the public sources

The public datasets and their repository identifiers are recorded in the source manifests of the archive. Each acquired file is checked against the size and SHA-256 recorded in the archive:

```bash
python current-release/portable/acquire.py her2st --source-dir RAW/manual-her2st --destination RAW/her2st
python current-release/portable/acquire.py crc    --source-dir RAW/manual-crc    --destination RAW/crc
python current-release/portable/run_analysis.py prepare --lab-root RAW/sources --output-root OUT/inputs
```

## Prospective analyses

The label diagnostics, the construction-scale experiment, the prospective test and the encoder comparison are in
`revision-r3/`. The protocol is `revision-r3/PROTOCOL-R3.md`, byte-identical to its deposit, and every
departure from it is listed in `revision-r3/PROTOCOL-R3-DEVIATIONS.md`. Results are written under
`results/revision-r3/` (override with `TESSERA_R3_OUT`).

1. **Calibration on the 24 calibration datasets** (CPU): `run_w1.py` (label diagnostics and the control grid),
   `run_w1b.py` (lung niches rebuilt at known construction scales), `run_w2.py leiden` and `run_w2.py seeds` (clustering
   rule and seed spread), `run_w3.py` (component removals and propagation depth), then `prepare_freeze.py`, which fixed
   the scale rule and the margins before any prospective cohort was downloaded.
2. **Prospective cohorts** (CPU): `cohorts/fetch_r3.py` downloads the public records with receipts; `cohorts/prep_spinal.py`,
   `prep_colorectal.py` and `prep_wu.py` write the derived inputs under `data-r3/`; `run_w6_local.py` computes their
   diagnostics and control grid; `run_w7_secondary.py` and `run_w8_selfloop.py` run the secondary analyses.
3. **Encoders** (GPU): `remote/setup_box.sh` prepares a machine; `remote/run_box.sh` fits the stage-1 grids of
   GraphST, STAGATE and Tessera-ST; `remote/resume2_wp4.sh` then scores them, selects each unit's configuration on the
   other units, fits every component removal at five seeds, scores everything under K-means and Leiden, and computes
   the parameter-free counterparts. Each fit is its own process and leaves a record with its input fingerprint;
   `remote/verify_w4.py` checks records, embedding hashes and the stage-2 plan.
4. **Summaries, tables and figures** (CPU, minutes): `analyze_r3.py` (writes `summary/summary.json`),
   `make_macros_r3.py`, `make_tables_r3.py` and `figures/figures_r3.py`; `report_p3.py` prints the encoder comparison.

Tests: `test_r3.py`, `external/test_external_r3.py`, `external/test_tessera_adapter_r3.py`,
`cohorts/test_cohorts_r3.py` and `figures/test_figures_r3.py`. Two practical notes: the score tables are read with
`float_precision="round_trip"`, because pandas' default float parser crashed on some of their values; and scoring and
the counterparts run as independent shard processes (`--shard i --nshards N`) rather than in a Python worker pool.

## Archived versions

The calibration analyses (the controls audit of Tessera-ST) are archived as version 2026.10.11-controls-audit at
[doi:10.5281/zenodo.23308077](https://doi.org/10.5281/zenodo.23308077); their frozen research source is in
[current-release/](current-release/), and the complete archive, with retained arrays and model states, can be
reassembled from that record. The budget, tuning and cluster-number analyses are archived at
[doi:10.5281/zenodo.23999069](https://doi.org/10.5281/zenodo.23999069), and the analyses in `revision-r3/` at
[doi:10.5281/zenodo.24315838](https://doi.org/10.5281/zenodo.24315838); each version keeps every file of the versions
before it.

The root `src/` implementation is byte-identical to the archived implementation. The neural core and the field-aware
graph and data adapters have distinct provenance, documented in
[the comparison record](current-release/provenance/historical-core-versus-adapters.json). A preliminary release of the
software is archived at [doi:10.5281/zenodo.23101191](https://doi.org/10.5281/zenodo.23101191).

Author-owned code is MIT; author-owned research content is CC BY 4.0. [Licensing](LICENSING.md) and
[source notices](NOTICE.md) preserve third-party terms. Breast displays use author-generated coordinate maps; no 10x
breast H&E image or background is redistributed. The separate dataset-licensed CRC images retain their attribution.
