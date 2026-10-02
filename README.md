# Tessera-ST: spatial-evaluation protocol audit

**Current scientific release: [`current-release/`](current-release/).**
Author-review version, 2 October 2026; scientific submission remains **HOLD**.
The current contribution is a field-aware evaluation-protocol audit, not a
universal spatial law, a validated selector or a superior new architecture.

- Public code and results: https://github.com/PeterPonyu/tessera-st/tree/author-review-2026-10-02/current-release
- Versioned archive: https://doi.org/10.5281/zenodo.23101191
- Current default **single-column article**: `current-release/manuscript/paper.pdf`.
- Matching readable supplement: `current-release/manuscript/SI.pdf`.
- Scope, source identities and reproduction: `current-release/README.md`,
  `DATA_SOURCES.md`, `ENVIRONMENT.md` and `THIRD_PARTY_NOTICES.md`.

The corrected panel contains seven field-identified objects, seven procedures,
three saved seed identities and 147 completed runs. Saved predictions can be
rescored offline; full refits require separately acquired upstream assays and
external method implementations. This does not certify clinical utility,
independent biological validation, journal acceptance or scientific clearance.

## Verify the current release

```bash
cd current-release
python verify_release.py
python -m pytest -q reproducibility/tests/test_rank_ties.py reproducibility/tests/test_supporting_information.py
bash manuscript/scripts/rebuild_current.sh
```

Use the current-release environment notes. No command installs packages or
downloads raw data automatically. Third-party source/weights/fonts retain
their original terms; author-owned code is MIT and authored text/tables are
CC BY 4.0. See the detailed rights boundaries inside the versioned package.

## Historical development tree

The older root-level `src/`, `experiments/`, `manuscript/`, `DESIGN.md` and
development tests are preserved as **historical workbench material**.
They are not the current scientific package. In particular, the earlier
11/12/14-object association, independent-contiguity mechanism and calibrated
confidence/ECE claims have been withdrawn from current inference. The legacy
soft head has no validated mapping to returned KMeans labels. Historical
benchmark outputs must not be substituted for the corrected panel.

For current source inspection or installation, operate under `current-release/`;
do not assume that an old root-level package or August paper is this release.
The prior IEEE author-review face is retained only as labelled provenance.
