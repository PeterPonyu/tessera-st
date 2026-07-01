# Tessera-ST — design

> **Thesis.** A tissue is a *tessellation*: a small number of spatial domains tiled across
> the slide, separated by sharp biological boundaries. Existing graph methods for spatial
> domain detection (graph-autoencoder / graph-attention encoders over a kNN spatial graph)
> smooth node features **indiscriminately** over that graph. Smoothing across a true domain
> boundary bleeds signal between tiles, washes out small domains, and blurs the very edges
> that define the partition. **Tessera learns where to *stop* smoothing.**

This is deliberately a *different* bet from an interpretable nonnegative-factorization decoder.
That line asks "*what programs compose each domain?*". Tessera asks "*where are the seams, and
how sure are we?*" — boundaries are a first-class output, not a by-product of argmax over a
smoothed embedding.

## Why this is a real gap (not a +0.02-ARI reskin)

The spatial-clustering benchmarks in `references/spatial_omics_pdfs/` repeatedly surface three
failure modes that flat, uniformly-smoothing encoders share:

1. **Over-smoothing across boundaries.** Deep message passing on a fixed kNN graph converges
   node features toward a global mean; domain edges and small domains are the first casualties.
2. **No notion of resolution / hierarchy.** Cortex has layers *within* regions. A single-scale
   embedding clustered at one `k` cannot express nested structure.
3. **No calibrated confidence.** Methods emit a hard label per spot with no honest statement of
   *boundary* uncertainty — exactly where biology and reviewers care most.

Tessera targets all three with one coherent mechanism, and each piece is independently ablatable.

## Architecture

```
   spot expression X ──▶ feature MLP ──▶ h⁰
                                          │
          spatial kNN graph G ────────────┤
                                          ▼
   ┌─────────────── edge-gated message passing (L layers) ───────────────┐
   │  for each edge (i,j):  g_ij = σ( φ(|hᵢ−hⱼ|, eucl_dist_ij) )          │   ← COMPONENT 1
   │  message: hᵢ ← hᵢ + Σⱼ g_ij · ψ(hⱼ)          (g→0 ⇒ stop smoothing)   │      edge gating
   │  collect per-layer embeddings  h¹ … h^L      (the "scales")          │
   └──────────────────────────────────────────────────────────────────────┘
                                          │
        multi-scale fusion  z = Fuse(h¹…h^L)   (attention over depths)     ← COMPONENT 2
                                          │                                    multi-scale
                ┌─────────────────────────┼──────────────────────────┐
                ▼                         ▼                          ▼
        feature decoder            boundary score b_i           soft assignment q_i
        (recon loss, always)   b_i = 1 − mean_j g_ij        (cluster head + temperature)
                                          │                          │
                          boundary-contrastive loss          calibrated uncertainty   ← COMPONENTS
                          (InfoNCE on high-gate pos)         (temperature + conformal)    3 & 4
```

Inference: cluster `z` (mini-batch k-means / Leiden) → domain labels; `b` → boundary map;
calibrated entropy of `q` → per-spot confidence.

## The four ablatable components (this is the paper's spine)

| # | Component | ON | OFF (degenerate-to) | Hypothesised effect |
|---|-----------|----|--------------------|---------------------|
| 1 | **Edge gating** | learned per-edge `g_ij` attenuates cross-boundary messages | `g_ij ≡ 1` → plain mean aggregation (classic GCN-style smoothing) | sharper domains, small-domain recall ↑, boundary-F1 ↑ |
| 2 | **Multi-scale fusion** | attention over per-layer embeddings | use last layer only | hierarchy / robustness to depth, ARI ↑ on layered tissue |
| 3 | **Boundary-contrastive loss** | InfoNCE pulling high-gate (intra-domain) neighbours together | reconstruction loss only | cleaner embedding geometry, NMI ↑ |
| 4 | **Calibrated uncertainty** | temperature + split-conformal on boundary confidence | raw softmax entropy | ECE ↓, usable boundary confidence |

The ablation runner sweeps the curated grid (full model + four leave-one-out + backbone) and
emits one tidy table. That table *is* the methods figure.

### Components are hypotheses, not commitments

`full` (all four on) is **not** assumed to be the final model. The four components are
candidate hypotheses; the ablation exists to *falsify* them. The runner reports the
**data-selected** config — the best-scoring Tessera variant on the data at hand — and drops any
component that does not earn its place. If, on honest evaluation, only multi-scale helps, the
selected model is `backbone + multi-scale`, and the paper says so. A component that never wins
on real data gets cut, not shipped "because we built it".

### Baselines and SOTA live in the same table

A component grid in isolation says nothing. Every run also scores:

- **`ref:nonspatial-kmeans`** — clusters raw expression; the floor any spatial method must clear.
- **`ref:smoothed-kmeans`** — averages features over neighbours then clusters; the transparent
  over-smoothing baseline whose failure mode the edge gate targets.
- **`backbone`** — a standard graph-autoencoder + clustering, i.e. a brand-neutral stand-in for
  the *graph spatial-domain encoder family* (the SOTA class).

Parity against the **named** SOTA methods (STAGATE / GraphST / SEDR / BANKSY) is run from their
own repositories behind an adapter and is a **real-DLPFC gate** — synthetic data cannot settle
it, and we do not pretend it can.

## What stays identical to the comparison world

So results drop straight into the established leaderboard:

- **Data:** DLPFC / `spatialLIBD` (12 Visium sections, manual cortical-layer ground truth),
  plus Visium mouse brain and a synthetic *tessellation* generator for offline CI.
- **Reference methods to beat:** graph-autoencoder and graph-attention spatial encoders
  (named only in `BASELINE_REFERENCES.md`; run from their own repos behind an adapter).
### Metrics span orthogonal dimensions (ARI/NMI alone is not enough)

ARI and NMI are **one** axis — chance-corrected label agreement — and they move together. A
method can win ARI yet have spatially shattered, small-domain-eating, mis-calibrated domains.
So the table reports one or two metrics per *independent* dimension:

| Dimension | Metric(s) | Dir. | What it catches that ARI/NMI miss |
|-----------|-----------|:----:|-----------------------------------|
| label agreement | ARI, NMI | ↑ | (the standard axis; kept for leaderboard comparability) |
| spatial coherence | CHAOS | ↓ | domains that are spatially scattered despite high label agreement |
| spatial fragmentation | PAS | ↓ | speckle: spots disagreeing with their spatial neighbourhood |
| geometric separation (INTERNAL ⚠) | ASW, DBI | ↑ / ↓ | embedding clusters that overlap — but see caveat |
| boundary sharpness | boundary-F1 | ↑ | seams smoothed away (the core thesis) |
| small-domain recovery | small-IoU | ↑ | a tiny domain dissolved or shattered — invisible to ARI |
| calibration | ECE | ↓ | over-confident boundary calls |

CHAOS/PAS, boundary-F1 and small-IoU are precisely the spatial axes a uniformly-smoothing
encoder fails on while still scoring respectable ARI — which is why they, not a second
agreement metric, are the ones worth adding.

### Evaluation philosophy: every metric counts, none is enthroned

Spatial-domain detection is unsupervised, so **no metric here is a clean external truth** — they
are all *semi-circular*, just to different degrees. We report the full panel, label each one's
circularity honestly, and **triangulate**; we never delete a metric or let a single one decide a
verdict. When two metrics disagree, that is information about *what each measures*, not proof one
is fake.

| Metric | Real facet it measures | Circularity | Gameable by |
|--------|------------------------|-------------|-------------|
| ARI, NMI | agreement with manual layers | **weak** — but the "ground truth" is itself an expression+histology-derived clustering, not physical truth | — |
| ASW, DBI | cluster separation in the embedding | **strong** — scored in each method's own space; Tessera's contrastive loss directly optimises separation | any pushed-apart embedding |
| CHAOS, PAS | spatial contiguity of domains | **strong** — spatial smoothing is both the inductive bias and the judge | one big domain (perfect CHAOS) |
| boundary_F1, small_IoU | seams / small-domain recovery vs GT | weak (inherits ARI's GT) | — |
| ECE | confidence vs correctness | inherits GT; baselines use post-hoc confidence | — |
| **marker_purity** | domains enriched for **predefined literature** layer markers | **lower, not zero** — not from this data's clustering, BUT conditioned on a marker prior that drifts across platform / individual / disease / batch | — |

The honest reading of e.g. "ASW rises while ARI falls during training": two semi-circular metrics
measuring different things, not one lying. Signals that do **not** come from clustering this dataset
— predefined marker genes (`marker_purity`), histology, cross-sample reproducibility — *lower*
circularity, but they are **not** non-circular ground truth: every dataset's conditions differ, so
its markers differ too, and a fixed marker list can simply be wrong here. So they are extra partial
views to corroborate against, **not** tie-breakers and **not** a metric to steer training by. The
rule stands for them too: enthrone none.

## Non-goals / guardrails

- No performance claim graduates until a real-DLPFC run + full ablation + same-data baseline
  parity + a clean leakage scan land in `CLAIM_LEDGER.md`. The synthetic smoke proves the
  machine runs; it proves nothing biological.
- Clean-room: baseline brand names live only in the three provenance docs, never in `src/`.
