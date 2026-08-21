# Tessera-ST

Spatial-domain detection code for spatial transcriptomics. The encoder learns a per-edge gate that can stop smoothing at a predicted seam.

Graph encoders for spatial domains often smooth node features over the spatial kNN graph. That can mix signal across domain boundaries and shrink small domains. This repository studies that mechanism and the value of a spatial prior.

Status: **11-platform meta-analysis.** Machine-checked by
`experiments/verify_manuscript.py`, exit 0. The object is a
**causal law**: manipulating *only* the ground truth's spatial contiguity on synthetic tissue makes the
value of a spatial prior rise and flip sign (Spearman $\rho=+0.98$ to $+1.00$, $p<0.001$), and the same law
holds observationally across 11 technologically diverse datasets ($\rho=+0.72$, $p=0.012$). No method is a
universal winner (seven distinct winners across the panel; a two-line neighbour-mean floor is competitive).
A label-free `coh_gain` proxy recovers the law's direction ($\rho=+0.55`) but is reported honestly as a
directional, marginal prior — after correcting for its selection among six candidates it is not significant
(search-corrected $p=0.28$). The candidate method (Tessera) itself is retained only as a *worked example* of
how single-dataset evaluation can certify a false ranking and a false mechanism.

## How this differs from a factor-decomposition spatial-domain method

A nonnegative-factor decoder asks *"what gene programs compose each domain?"*. Tessera asks
*"where are the seams, and how confident are we?"* — boundaries are an **explicit output**,
not the by-product of an argmax over a smoothed embedding. Same data, same baselines, same
ARI/NMI table; a different mechanism and two additional boundary-aware metrics.

## The mechanism, in one diagram

```
expression ─▶ edge-gated message passing ─▶ multi-scale fusion ─▶ embedding ─▶ domains
                     │  g_ij→0 = "stop smoothing here"                   │
                     └────────────▶ boundary map  +  calibrated confidence
```

See `DESIGN.md` for the full architecture and rationale.

## Four ablatable components (the study's spine)

| # | Component | OFF degrades to |
|---|-----------|-----------------|
| 1 | Adaptive **edge gating** (anti-over-smoothing core) | plain mean aggregation (classic over-smoothing) |
| 2 | **Multi-scale fusion** (hierarchy) | last layer only |
| 3 | **Boundary-contrastive** loss | reconstruction only |
| 4 | **Calibrated uncertainty** | raw softmax entropy |

`tessera smoke-synth` puts **reference baselines, the component grid, and a data-driven pick**
in one table:

- `ref:nonspatial-kmeans` (non-spatial floor) · `ref:smoothed-kmeans` (naive over-smoothing)
- `full` + 4 leave-one-out + `backbone` (a standard graph-autoencoder stand-in;
  named-method parity is the real-DLPFC gate)
- a printed **data-selected config**: the selected Tessera variant on this data, *not* an assumed
  `full`. Components are hypotheses — anything that does not earn its place is dropped.

## Quick start

```bash
pip install -e ".[test,torch]"    # PyTorch extra; see pyproject.toml

pytest -q                         # 11 tests: metrics, smoke, ablation toggles, clean-room

# Offline synthetic tessellation — train + full ablation grid (no download)
tessera smoke-synth --epochs 150 --out experiments/synth_ablation.json

# Real DLPFC section (needs the real-data extra + a .h5ad with manual layer labels)
pip install -e ".[real-data]"
tessera ablate-real /path/to/dlpfc_151673.h5ad --label-key layer_guess
```

## Metrics — a full panel, every one semi-circular, none enthroned

Spatial-domain detection is unsupervised, so **no metric here is clean external truth** — each is
circular to some degree. We report the full panel, flag each one's circularity, and **triangulate**;
no single metric decides a verdict (see `DESIGN.md` for the full audit).

| Dimension | Metric | Dir. | Circularity |
|-----------|--------|:----:|-------------|
| label agreement | ARI, NMI | ↑ | weak — GT is expression+histology-derived clustering |
| spatial coherence | CHAOS, PAS | ↓ | strong — smoothing is both bias and judge; gameable by one big domain |
| geometric separation | ASW, DBI | ↑ / ↓ | strong — scored in the method's own embedding |
| boundary sharpness | boundary-F1 | ↑ | weak (inherits GT) |
| small-domain recovery | small-IoU | ↑ | weak (inherits GT) |
| calibration | ECE | ↓ | inherits GT |
| marker-prior view | **marker_purity** | ↑ | **lower, not zero — not from this data's clustering, but conditioned on a marker prior that drifts across platform/sample** |

`marker_purity` asks whether each predicted domain is enriched for fixed, prior-knowledge layer
markers. It does not come from re-clustering this dataset, so it is *less* circular — but it is
**not** ground truth: every dataset's conditions differ, so its markers differ, and a fixed list can
be wrong here. It is one more partial view to corroborate against, never a tie-breaker or a metric
to steer training by.

## Layout

```
src/tessera_st/
├── config.py          # AblationConfig — the four toggles + curated grid
├── data/              # synthetic tessellation generator + DLPFC h5ad adapter
├── model/
│   ├── gating.py      # Component 1: learned per-edge gate
│   ├── encoder.py     # edge-gated message passing + Component 2 multi-scale fusion
│   ├── heads.py       # decoder + soft-assignment + temperature
│   └── tessera.py     # assembles components per AblationConfig
├── losses.py          # recon + Component 3 boundary-contrastive
├── train.py           # fit_predict (torch)
├── eval/              # ARI/NMI/ASW/boundary-F1/ECE + brand-neutral reference clusterers
└── ablation.py        # the curated grid runner -> tidy table
```

## Provenance & guardrails

- `DESIGN.md` — thesis, architecture, ablation grid.
- `BASELINE_REFERENCES.md` — the named external methods we compare against (run from their own
  repos; brand names live here only).
- `ALLOWED_BASELINE_CONTEXTS.md` — the clean-room boundary.

Clean-room: baseline brand names never appear in `src/` — enforced by
`tests/test_brand_independence.py` and `scripts/check_independence.sh`.
