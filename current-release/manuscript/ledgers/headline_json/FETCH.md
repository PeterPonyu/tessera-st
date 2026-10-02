# What a reader must fetch to rerun the headline

No DOI is assigned. None is invented.

## Already in this capsule (enough to recompute printed stats)

`manuscript/ledgers/headline_json/` — copies of the workbench JSON the official numbers are
read from. sha256 (2026-09-18, rematch leftover `working_printed_claims.json` provenance):

```
f97a38e6ac1fc9492f08213844eea433357ec6de8678673d77e82934ed63f23a  expanded_bench.json
20b342a977b6c8c25c97e3e2ba893f8453f09407cf993478c397f8a9deeecb23  stat_rigor.json
41ba1678dc534ecb5a906e4323aea0e76f5025b6e12754418986e1796e7e9af5  robustness_audit.json
960c0b0012354f97048ea5c722213318122d5805d23c9bdd838101b435a28bf9  gtfree_proxy.json
d189a4115ac6dc07b01070ce727fa695c4ca838e257e914e310f6c80efb468e5  proxy_search_corrected.json
7ce2ce8d734f0c1606fecdb868516d7b288aa79adcb1c56404034362e58f4292  mechanism_synth.json
ec7eec7e356e298b1257f196ad519fea0700166aa5d3f85319850fa4e0fa151d  expand_data.json
db67223a364fd81a735603df7480ca47f93f5144bcb584cf3da8154e8eb5d04d  native_baselines.json
```

`manuscript/ledgers/science_core_20260918/` — round 1 (generator scale-up, power, full LOO).
`manuscript/ledgers/science_core_20260918_r2/` — round 2 (disk screen, Bayesian ρ, n=12 Zhuang).
Script: workbench `experiments/science_harden_20260918.py` (capsule `manuscript/experiments/`
is a gitignored symlink into `labs/active/tessera-st/experiments`). Snapshot:
`manuscript/ledgers/science_core_20260918/science_harden_20260918.py`.
Generator source snapshot: `manuscript/experiments/vendored_src/tessera_st/data/synthetic.py`
(sha256 `d061ee6938fea2712b6861f1854775729abf79ed5742fd94fd4e5bf3b194c7e0`).

From the vendored JSON a reader can recompute Spearman / bootstrap / jackknife / Bonferroni
without any h5ad. That does **not** regenerate `expanded_bench.json` itself.

## Must fetch to regenerate the observational bench

1. Code: public git `https://github.com/PeterPonyu/tessera-st` (workbench pin:
   `labs/active/tessera-st`). No Zenodo DOI.
2. Data: local checkout `labs/active/spatial-omics-reform/data` (or set
   `TESSERA_DATA_ROOT` / `SPATIAL_OMICS_ROOT`). The 11 published rows and the 3
   already-scored cancer rows are loaded by `experiments/expanded_bench.py` and
   `experiments/expand_data.py`. Those scripts name the local `.h5ad` paths.
   Accession IDs live in that repo, not here.
3. External method trees under `spatial-omics-reform/external/{STAGATE,SEDR,GraphST,SpaceFlow}`
   for a full method panel. Neighbour-mean generator scale-up does not need them.

## n-raise screen (2026-09-18 round 2)

- `deepstarmap_mouse_brain`: **fails C5**. 198,675 cells, 1017 genes, `spatial` present, but the only labels are method-derived (`Harmony_labels` 22 classes, `FUSEmap_main_level` 19, `FUSEmap_sub_level` 137). BRCA/SEDR is a disclosed exception already on the panel, not a licence for new method-derived rows. sha256 `c0eb7fb3afffc00a52b2455d154062762c386dc34bf7abd1a729cec74db9525f`.
- Raw `Zhuang-ABCA-1` / `Zhuang-ABCA-2`: still fail C1 (no `obsm['spatial']`).
- Processed `niche_abc_zhuang1_brain_slice`: **qualifies**. Allen-CCF-2020 anatomical `domain`, per-cell CCF registration (non-clustering-derived). 22,072 cells, k=17, subsampled to 16,000. Run under the published-panel protocol: contig 0.945, advantage +0.125, n=12 ρ=+0.69 (p=0.013). Ledger: `science_core_20260918_r2/n12_zhuang.json`.
- `niche_abc_zhuang2_brain_slice`: same atlas, second mouse — C4, not a second n.
- Extra Moffitt / IMC / Open-ST slices and processed copies of the published 11: C3/C4.
- `st_CESC` / `st_NSCLC` / `st_PRAD`: technical `segmentation_method` only (C2/C6).
- `wu_breast_atlas`, `sc_mouse_cortex`, lumina/qukun refs: scRNA-seq, no spatial (C1/C7).
- processed Bento MERFISH: n=15, no labels (C2).

Headline remains $n{=}11$. $n{=}12$ is reported alongside. $n{=}14$ is the already-scored ctype expansion. No invented DOI.
