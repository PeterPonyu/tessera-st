# Tessera-ST — baseline references

The named external spatial-domain methods Tessera positions against. They are **run from their
own repositories behind an adapter** for parity; no upstream source, class name, or figure is
copied into this project's `src/`. Brand names appear here and in `ALLOWED_BASELINE_CONTEXTS.md`
and `CLAIM_LEDGER.md` only.

Shared data + metrics with this comparison world is deliberate, so results drop straight onto
the established leaderboard.

## Primary comparison methods (graph spatial-domain encoders)

| Method | Venue | Why it is the right comparison | Frozen audit clone |
|--------|-------|--------------------------------|--------------------|
| STAGATE | Nat. Commun. 2022 | graph-attention autoencoder; canonical DLPFC ARI baseline | `../spatial-omics-reform/external/STAGATE` |
| GraphST | Nat. Commun. 2023 | contrastive graph self-supervision; closest graph encoder | `../spatial-omics-reform/external/GraphST` |
| SEDR | Nat. Commun. 2024 | masked graph self-supervised spatial embedding | `../spatial-omics-reform/external/SEDR` |
| SpaceFlow | Nat. Commun. 2022 | spatial-domain embedding baseline | `../spatial-omics-reform/external/SpaceFlow` |
| BANKSY | Nat. Genet. 2024 | neighbourhood-augmented feature baseline; target-venue ARI tables | `../spatial-omics-reform/external/Banksy` |

The over-smoothing failure these share is the gap Tessera's edge gate targets. A transparent,
brand-neutral over-smoothing reference (`mean_smoothed_kmeans`) lives in `eval/baselines.py` so
the ablation has an internal anchor that needs no external checkout.

## Datasets (identical to the comparison world)

| Dataset | Platform | Ground truth | Use |
|---------|----------|--------------|-----|
| DLPFC / spatialLIBD (12 sections) | 10x Visium | manual cortical layers | primary ARI/NMI/boundary-F1 benchmark |
| Visium mouse brain (ant./post.) | 10x Visium | anatomical regions | cross-tissue generalisation |
| synthetic tessellation | — | exact, by construction | offline CI + ablation isolation |

## Benchmark/context papers (in the parent program)

`../spatial-omics-reform/references/spatial_omics_pdfs/` — the SRT-clustering and
benchmarking reviews that document over-smoothing, resolution selection, and calibration as
open problems. Cited for motivation only; no result is imported as a Tessera claim.
