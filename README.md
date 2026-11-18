# Tessera-ST: matched neighbourhood controls and label diagnostics for spatial encoders

This repository holds Tessera-ST, a gated graph encoder for spatial molecular data, and the code of a study that asks
what a learned spatial encoder adds beyond averaging each observation with its neighbours, and how much the construction
of a reference annotation decides that comparison before any method is run.

## Analysis code and archives

- **Analysis code:** folder `revision-r3/` on the branch
  [`revision-r3`](https://github.com/PeterPonyu/tessera-st/tree/revision-r3), whose README gives the commands for each
  step: label diagnostics of reference annotations, the
  construction-scale experiment, the prospective test, and the comparison of GraphST, STAGATE and Tessera-ST with
  parameter-free propagation on their own graphs.
- **Code, derived inputs and result tables:** [doi:10.5281/zenodo.24315838](https://doi.org/10.5281/zenodo.24315838).
  The same record holds the calibration analyses of Tessera-ST with their complete source and saved results, which are
  also at the tag [`2026.10.11-controls-audit`](https://github.com/PeterPonyu/tessera-st/tree/2026.10.11-controls-audit).
- **Analysis protocol** for the prospective cohorts, deposited before their expression data were downloaded:
  [doi:10.5281/zenodo.24303149](https://doi.org/10.5281/zenodo.24303149).

## This branch

- `src/tessera_st/`: the evaluated Tessera-ST implementation (gated message passing, contrastive objective and data
  adapters), with tests in `tests/`.
- `experiments/`: exploratory benchmark code from the development of Tessera-ST, including an unpublished
  label-contiguity analysis that the study does not rely on.
- `DESIGN.md`, `BASELINE_REFERENCES.md` and `ALLOWED_BASELINE_CONTEXTS.md`: design notes and the clean-room boundary
  for the baseline methods, which are run from their own repositories through adapters.
- `pages-landing/` and `pages-gateway-tests/`: the project page and its checks.

## Licence

Author-owned code is MIT licensed ([LICENSE](LICENSE)); author-written text, tables and result aggregates are
CC BY 4.0. This branch contains no third-party measurements, annotations or images. The archived packages record the
source, licence and attribution of the data they contain ([LICENSING.md](LICENSING.md)).
