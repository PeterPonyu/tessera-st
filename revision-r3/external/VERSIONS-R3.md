# WP4 external encoders: versions and sources

Checked on the author's laptop on 17 Nov 2026. Every fit record repeats its own package versions, script hashes and
encoder source hashes (`versions` in `fits/<dataset>/<tag>.json`), so these tables describe the environments the
GPU box has to reproduce.

## Environments

| Package | `tessera-native` (GraphST) | `dl` (STAGATE) |
|---|---|---|
| Python | 3.11.18 | 3.13.7 |
| torch (CUDA runtime) | 2.8.0 (12.8) | 2.12.1+cu130 (13.0) |
| torch_geometric | not installed | 2.8.0 |
| GraphST | 1.1.1 | not installed |
| numpy | 2.2.6 | 2.2.6 |
| scipy | 1.16.2 | 1.16.2 |
| pandas | 2.3.3 | 2.3.3 |
| scikit-learn | 1.6.1 | 1.6.1 |
| scanpy | 1.11.4 | 1.11.4 |
| anndata | 0.12.3 | 0.12.3 |
| scikit-misc (seurat_v3 HVG) | 0.5.1 | 0.5.1 |
| leidenalg / igraph | 0.10.2 / 0.11.9 | 0.10.2 / 0.11.9 |
| POT | 0.9.6.post1 | 0.9.6.post1 |
| threadpoolctl | 3.8.0 | 3.6.0 |
| pytest | 9.2.0 | 8.4.2 |

Paths: `/home/zeyufu/miniconda3/envs/tessera-native/bin/python` and `/home/zeyufu/miniconda3/envs/dl/bin/python`
(overridable with `TESSERA_R3_PY_GRAPHST` and `TESSERA_R3_PY_STAGATE`).

## GraphST

- PyPI `GraphST==1.1.1`, wheel `GraphST-1.1.1-py3-none-any.whl`, sha256
  `3d384ac093a20cb928c6e804751b7bf3bc99c545453744098acfa392a36231f3` (as recorded in
  `revision-r1/native/VERSIONS.md`). The installed files match the hashes in the wheel's `RECORD`.
- Installed file sha256: `GraphST.py` 2d1e033e…7f18, `model.py` 01478889…8c95, `preprocess.py` 5959887f…b321,
  `utils.py` 3d2833ce…f388, `__init__.py` c5fe0e3d…88b6 (full hashes in every fit record).
- Not edited. The arms change a constructor argument (`beta = 0`) or, after construction, the instance attribute
  `adj` (the identity). The graph comes from GraphST's own `construct_interaction(adata, n_neighbors=k)`.

## STAGATE_pyG

- `https://github.com/QIFEIDKN/STAGATE_pyG.git`, commit `ae1158ca8cf1eb6bb8ee198298552d44c9ac21db`
  (2023-11-10 10:49:42 +0800). It is imported read-only from
  `/home/zeyufu/Desktop/labs/active/spatial-omics-reform/external/STAGATE` (overridable with `TESSERA_STAGATE_ROOT`).
- **The clone has one uncommitted change, which predates WP4.** In `STAGATE_pyG/gat_conv.py`, the import of
  `SparseTensor, set_diag` from `torch_sparse` is wrapped in `try/except ImportError`, with a sentinel class, because
  torch_sparse has no cu130 wheel. The fallback matters only for a `SparseTensor` edge index. STAGATE always passes
  a `LongTensor` edge index, so it never uses that branch. `gat_conv.py` sha256 is 598666cb…175a at the commit and
  b755a485…09b2 in the working tree.
- Working-tree sha256: `STAGATE.py` 2224eb23…fb92, `Train_STAGATE.py` 997fcaea…efa5, `utils.py` df8295db…9ecd,
  `__init__.py` 8f10eab6…edd4.
- Not edited. The `uniform_attention` arm subclasses `GATConv` and swaps the class name in the
  `STAGATE_pyG.STAGATE` module only while the model is built. `train_STAGATE` runs as released.
- There is no rpy2 in `dl`, so STAGATE's own `mclust_R` cannot run there.

## What each file borrows

| File | Borrowed from (imported unless noted) |
|---|---|
| `common_external.py` | `revision-r1/common_r1.py`: `load_primary`, `celltype_labels`, and `her2st_unit` (frozen `run_her2st_validation.load_joined` + `preprocess`, giving HER2ST PCA50). E4 inputs written by `revision-r1/native/prepare_inputs.py` |
| `graph_ops.py` | Permutation rule of `run_experiments.graph` / `revision-r1/train_core.build_graph`, and the mean convention of `run_experiments.neighbor_embedding` (re-stated in numpy, about 5 lines each) |
| `run_graphst_r3.py` | `revision-r1/native/run_native.py`: `graph_adata` (native input) and the `fit_graphst` pattern. `revision-r1/native/native_common.py`: `offset_frames`, `edge_frame_check` |
| `run_stagate_r3.py` | The graph rule of `labs/capsules/tessera-option-hold/experiments/field_aware_adapters.py` (`stagate_input`, `neighbor_rows`), re-stated in `knn_within_frames`. Importing the capsule would put its own `tessera_st` package on the path and shadow the frozen one |
| `run_external_r3.py` | Record and lock pattern of `run_native.run_one` |
| `score_external_r3.py` | `revision-r1/run_k_sweep.leiden_partition` (round-2 Leiden), and the matched K-means of `native_common.kmeans_matched` |
| `repro_e4.py` | `native_common.kmeans_matched`, `native_common.score`; `results/revision-r1/E4/E4_scores.csv` |
| `test_external_r3.py` | Frame-overlap control of `revision-r1/native/test_frames.py` |
