# Tessera-ST — baseline references

The named external spatial-domain methods Tessera positions against. They are **run from their
own repositories behind an adapter** for parity; no upstream source, class name, or figure is
copied into this project's `src/`. Brand names appear here and in `ALLOWED_BASELINE_CONTEXTS.md`
only.

Shared data + metrics with this comparison world is deliberate, so results drop straight onto
the established leaderboard.

## Primary comparison methods (graph spatial-domain encoders)

| Method | Comparison source | Why it is the right comparison | Frozen audit clone |
|--------|-------------------|--------------------------------|--------------------|
| STAGATE | Nat. Commun. 2022 | graph-attention autoencoder; canonical DLPFC ARI baseline | `../spatial-omics-reform/external/STAGATE` |
| GraphST | Nat. Commun. 2023 | contrastive graph self-supervision; closest graph encoder | `../spatial-omics-reform/external/GraphST` |
| SEDR | Nat. Commun. 2024 | masked graph self-supervised spatial embedding | `../spatial-omics-reform/external/SEDR` |
| SpaceFlow | Nat. Commun. 2022 | spatial-domain embedding baseline | `../spatial-omics-reform/external/SpaceFlow` |
| BANKSY | Nat. Genet. 2024 | neighbourhood-augmented feature baseline; comparable ARI tables | `../spatial-omics-reform/external/Banksy` |

The over-smoothing failure these share is the gap Tessera's edge gate targets. A transparent,
brand-neutral over-smoothing reference (`mean_smoothed_kmeans`) lives in `eval/baselines.py` so
the ablation has an internal anchor that needs no external checkout.

## Datasets (identical to the comparison world)

| Dataset | Platform | Ground truth | Use |
|---------|----------|--------------|-----|
| DLPFC / spatialLIBD (12 sections) | 10x Visium | manual cortical layers | primary ARI/NMI/boundary-F1 benchmark |
| Visium mouse brain (ant./post.) | 10x Visium | anatomical regions | cross-tissue generalisation |
| synthetic tessellation | — | exact, by construction | offline CI + ablation isolation |

## Dataset accessions (11-platform panel)

Exact accession identifiers / download locations for the datasets in the 11-platform panel
(`tab:data`), verified live 2026-07-13. These are the *original* publication's data-deposit
locations, not a Tessera-specific mirror; several are additionally redistributed through Squidpy
(`sq.datasets.*`) as noted.

| Dataset (paper key) | Original paper | Accession / location |
|---|---|---|
| DLPFC (sample 151673) | Maynard et al. 2021, *Nat. Neurosci.* | Bioconductor `spatialLIBD` package (`fetch_data()`); raw/processed via LIBD Globus endpoint `jhpce#HumanPilot10x` (research.libd.org/globus/); source github.com/LieberInstitute/HumanPilot; portal spatial.libd.org/spatialLIBD/ |
| seqFISH (mouse embryo) | Lohoff et al. 2022, *Nat. Biotechnol.* | Bioconductor `MouseGastrulationData` package, DOI 10.18129/B9.bioc.MouseGastrulationData; also redistributed via Squidpy (`sq.datasets.seqfish`) |
| MERFISH (hypothalamic preoptic region) | Moffitt et al. 2018, *Science* | Dryad DOI 10.5061/dryad.8t8s248 (datadryad.org/dataset/doi:10.5061/dryad.8t8s248) |
| STARmap (visual cortex) | Wang et al. 2018, *Science* | Wang Lab data portal (wangxiaolab.org/data-portal-1); STARmap resources mirror (starmapresources.org/data) |
| osmFISH (somatosensory cortex) | Codeluppi et al. 2018, *Nat. Methods* | linnarssonlab.org/osmFISH/ |
| MIBI-TOF (TNBC) | Keren et al. 2018, *Cell* | MIBItracker portal, mibi-share.ionpath.com; also redistributed via Squidpy (`sq.datasets.mibitof`) |
| IMC (breast cancer) | Jackson et al. 2020, *Nature* | Zenodo DOI 10.5281/zenodo.3518284 ("The Single-Cell Pathology Landscape of Breast Cancer"); code github.com/BodenmillerGroup/SCPathology_publication |
| Open-ST (HNSCC) | Schott et al. 2024, *Cell* | GEO GSE251926 (confirmed via NCBI GEO record, includes the HNSCC + matched lymph-node series) |
| Slide-seqV2 (hippocampus) | Stickels et al. 2021, *Nat. Biotechnol.* | Broad Single Cell Portal SCP815 (singlecell.broadinstitute.org/single_cell/study/SCP815) |
| CODEX (spleen) | Goltsev et al. 2018, *Cell* | welikesharingdata.blob.core.windows.net/forshare/index.html (authors' shared data repository) |
| BRCA (Visium, tumour) | 10x Genomics Visium demo (no associated paper); per-spot pathology-domain labels from Xu et al. 2024 (SEDR), *Genome Med.* | 10x Genomics "Human Breast Cancer (Block A Section 1)", sample `V1_Breast_Cancer_Block_A_Section_1`, Space Ranger 1.0.0, CC BY 4.0: 10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-0-0 (also `squidpy.datasets.visium("V1_Breast_Cancer_Block_A_Section_1")` / `scanpy.datasets.visium_sge`); 20-region pathology annotation redistributed at github.com/JinmiaoChenLab/SEDR (`data/BRCA1/metadata.tsv`). On-disk processed copy: `spatial-omics-reform/data/processed/brca1_visium_10x/anndata.h5ad` (3,798 spots × 36,601 genes, `Region`/`ground_truth` = 20 domains). Note: 10x reports 3,813 raw spots under tissue; the analyzed n=3,798 is the SEDR-annotated subset (spots carrying a pathology-domain label; `metadata.tsv` has exactly 3,798 rows). |

Accessions above were located by (a) matching the raw/processed file paths hardcoded in
`experiments/*.py` (e.g. `dlpfc_maynard_2021_visium`, `merfish_mouse_hypothalamus`,
`codex_spleen_goltsev2018`) to their originating publication, then (b) confirming each
publication's own data-availability statement. Nothing here is re-derived or newly computed —
this is a provenance lookup only.

## Benchmark/context papers (in the parent program)

`../spatial-omics-reform/references/spatial_omics_pdfs/` — the SRT-clustering and
benchmarking reviews that document over-smoothing, resolution selection, and calibration as
open problems. Cited for motivation only; no result is imported as a Tessera claim.
