# No universal SOTA in spatial-domain detection: spatial priors help if and only if the ground truth is spatially contiguous — and you can tell in advance without labels

*A confound-controlled benchmark of spatial-domain methods across 11 platforms and 6 methods, with a
label-free rule for when spatial priors are worth using.*

> **Reading guide.** This is the submission-facing synthesis. Every number traces to a machine-checked
> artifact under `experiments/*.json`, asserted by `experiments/verify_manuscript.py` (exit 0). The full,
> append-only evidence trail — including the experiments that *changed* our conclusions — is in
> `DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md` (§1–§8) and `../CLAIM_LEDGER.md`. Figures: `figures/fig1–fig5`.

## Abstract

Spatial-domain detection methods for spatial transcriptomics are typically ranked on one or two Visium
datasets under a single clustering backend, and a steady stream of papers each claim a new state of the
art. We show this practice is unsound and replace it with a predictive rule. Benchmarking six methods —
four graph/feature deep methods (STAGATE, SEDR, GraphST, SpaceFlow), a graph-community method
(SpatialLeiden), and a candidate boundary-aware method (Tessera) — against non-spatial and
neighbour-mean floors across **11 platforms** (Visium ×2, seqFISH, MERFISH, STARmap, osmFISH, MIBI-TOF,
IMC, openST, Slide-seqV2, CODEX), **3 seeds**, and each method at its own best configuration, we find:
**(1)** the clustering backend is a confound as large as the entire between-method spread (max per-method
ΔARI from KMeans→GMM = 0.20 = the full method spread); **(2)** a parameter-free neighbour-mean baseline is
competitive (cross-section ARI 0.473, rank 3/7); **(3)** there is **no universal SOTA** — 7 distinct
methods win across 11 platforms; **(4)** what *is* predictable is the value of **spatial priors in
general**: spatial-prior advantage tracks the ground truth's spatial contiguity at Spearman **+0.72
(p=0.012, 95% bootstrap CI [0.24, 0.92], per-seed ρ ∈ [0.69, 0.77])** — the only significant correlation,
robust to controlling for cluster count, feature dimensionality, and imaging-vs-sequencing modality
(partial ρ=0.67), predictive out-of-sample (leave-one-platform-out), and **causal** in a controlled
synthetic sweep (ρ=+0.98–1.00 when contiguity alone is manipulated). The strength is panel-dependent:
broadening to three additional, noisier cell-type cancer platforms (n=14) **attenuates it to +0.56
(p=0.038)** — still positive and significant, so +0.72 is a diverse-panel upper estimate, not a constant.
**(5)** Crucially, the rule is
**deployable without labels**: naive label-free contiguity proxies *invert*, but a principled proxy —
the spatial-coherence *gain* from smoothing — recovers the trend (ρ=+0.55) and beats base rate
out-of-sample. The actionable rule: *estimate spatial-coherence gain from your unlabelled data; use a
spatial/boundary-aware method only if it is high.* No single method is best; the task–prior match is what
matters, and it is measurable before you commit.

## 1. Introduction

Spatial transcriptomics assays measure gene (or protein) expression while retaining each cell or spot's
location. A core unsupervised task is **spatial-domain detection**: partitioning the tissue into coherent
regions (cortical layers, anatomical domains, tumour vs stroma). Dozens of methods exist — most encode
expression over a spatial neighbour graph so that nearby cells get similar embeddings, then cluster.

Two problems pervade the literature. First, **evaluation is narrow**: methods are usually compared on one
or two human-cortex Visium datasets under a single clustering backend, and "we are SOTA" is declared from
a single ARI table. Second, methods are presented as **universally better**, with little attention to
*when* a spatial prior is appropriate at all. We argue both framings are wrong, and that the right object
of study is not "which method wins" but "**when do spatial priors help, and can you predict it in
advance**".

**Contributions.**
1. A confound-controlled, 11-platform, 6-method, 3-seed benchmark showing the clustering backend is a
   first-order confound and a trivial neighbour-mean floor is competitive (§4).
2. Evidence that there is **no universal SOTA** (7 distinct winners), with a method, SpatialLeiden, whose
   rank profile is *inverted* — best exactly where the deep methods are worst (§5).
3. A **predictive law**: spatial-prior advantage tracks GT spatial contiguity (ρ=+0.72), the only
   significant and seed-robust correlation, validated by bootstrap, leave-one-platform-out, and confound
   control (§6).
4. A **label-free** version of the law that makes it deployable on unlabelled tissue: naive proxies
   invert; a smoothing-coherence-gain proxy recovers it (§7).
5. An honest negative on the candidate method (Tessera is a backend-robust generalist, not a winner),
   used as a worked example of how single-dataset evaluation manufactures false advantage (§8).

## 2. Related work

**Spatial-domain methods.** Graph-autoencoder approaches (STAGATE: graph-attention autoencoder; SEDR:
variational graph autoencoder with self-supervision) and contrastive approaches (GraphST: deep graph
infomax; SpaceFlow: deep graph infomax + spatial regularisation) smooth expression over a spatial kNN
graph. Non-deep pipelines (BANKSY: neighbour-augmented features; SpatialLeiden: spatially-blended features
+ Leiden community detection) and a candidate boundary-aware method (Tessera, edge-gated multi-scale +
boundary contrastive) round out the panel. **Benchmarking.** Prior comparisons typically fix one Visium
benchmark (DLPFC) and one backend; we show both choices silently re-rank methods. **Method selection.**
Unlike work that recommends a single method, we provide a data-measurable criterion for whether *any*
spatial method should be used.

## 3. Methods

**Data.** 11 platforms spanning sequencing- and imaging-based assays, tissues, and ground-truth types
(layer / anatomical domain / region / cell type / niche); large platforms deterministically subsampled to
16k cells (logged). A review-caught byte-duplicate (squidpy `mibitof` == a "colorectal" MIBI-TOF h5ad) was
removed and a duplicate guard added (honest n=11). **Methods.** Tessera, STAGATE, SEDR, GraphST,
SpaceFlow, SpatialLeiden, BANKSY-style, and two floors (non-spatial; neighbour-mean). SEDR/GraphST/
SpaceFlow run under a no-op `numba` stub so they execute on numpy ≥2.3 without changing the environment;
each method's brand name is confined to `experiments/` (clean-room enforced by tests). **Backends.** KMeans
and Gaussian mixture; for backend-fairness (§8c) we add **real R `mclust`** (BIC-based Gaussian mixture
model selection, called via an `Rscript` subprocess since rpy2 does not build on this Python), the clusterer
the deep methods recommend. Each method is scored at its own best configuration (best of backends ×
{refinement off/on}). **Metrics.** A full 11-metric panel (ARI, NMI, CHAOS, PAS, ASW, DBI, CAL,
boundary-F1, small-IoU, ECE, marker-purity), each flagged for circularity; we triangulate rather than
enthrone any single metric. **Contiguity.** GT spatial contiguity = mean fraction of each cell's spatial
kNN sharing its GT label (1 = contiguous domains, ~1/k = scattered). **Spatial-prior advantage** = best
spatial method's ARI − non-spatial floor's ARI.

## 4. The backend is a confound, and a trivial baseline is competitive

Switching KMeans→GMM changes a single method's ARI by 0.036–0.200; the largest shift (STAGATE, 0.20)
equals the entire between-method ARI spread, so a benchmark that fixes one backend silently re-ranks
methods (`rigorous_bench.json`). A two-line neighbour-mean floor reaches cross-section ARI 0.473 (rank
3/7, within 0.02 of the best deep method): new methods should be required to beat neighbour-mean, not just
non-spatial clustering. Methods lie on a structural ARI↔boundary-sharpness Pareto frontier; four
mechanism-transfer experiments slid *along* it, never outward (`pareto_frontier.json`).

## 5. No universal SOTA (figure 3)

Across 11 platforms there are **7 distinct winners** (SpaceFlow, SEDR, STAGATE, Tessera, BANKSY-style,
floor:nonspatial, SpatialLeiden; `expanded_bench.json`) — **8 once SpaGCN, the most-cited method, is added
as a 7th deep SOTA** (it wins only openST, never becoming universal — confirmed both under a unified backend
and with SpaGCN's **genuine native louvain pipeline** in a dedicated real-numba env, which gives the same
inverted profile; `spagcn_panel.json`, `spagcn_native.json`, §8e). SpatialLeiden's profile is **inverted** relative
to the deep methods: #1–#2 on the four lowest-contiguity platforms (CODEX, MIBI-TOF, Slide-seqV2, openST)
yet last (#8/8) on the most-contiguous MERFISH (figure 3) — a graph-community method excels exactly where
the autoencoder/contrastive methods fail. At the top of the contiguous-task leaderboard the winner is a
**seed-level tie** among spatial methods: across three independent runs the MERFISH winner flipped
SpaceFlow→Tessera→SpaceFlow. "Best method" is not a stable concept here.

## 6. The predictive law: contiguity → value of spatial priors (figures 1, 4, 5)

Spatial-prior advantage tracks GT spatial contiguity at Spearman **+0.72 (p=0.012)** — the only
significant correlation among the three we test (the Tessera-rank and SpaceFlow-rank correlations are not
significant), and seed-robust (per-seed ρ ∈ [0.69, 0.77]) (figure 1a). Three stress tests
(`stat_rigor.json`):

- **Uncertainty.** 10k-platform bootstrap: ρ=0.72, **95% CI [0.24, 0.92]**, positive in 98.7% of
  resamples (figure 5a).
- **Out-of-sample.** Leave-one-platform-out: contiguity predicts the *magnitude* of advantage on
  held-out platforms (LOO ρ=0.51), and the decision "is a spatial method worth it" (advantage > 0.05) is
  correct **0.82 vs a 0.55 base rate** (figure 5b).
- **Confounds.** Contiguity is the strongest marginal predictor (vs n_classes −0.57, dimensionality
  −0.35, modality +0.39) and stays predictive after partialling out each (0.55 / 0.86 / 0.80) and **all
  three at once (partial ρ=0.67)** (figure 4). It is not a proxy for cluster count, dimensionality, or
  imaging-vs-sequencing.
- **Causal, not just correlational** (figure 7). In a controlled synthetic experiment that manipulates
  *only* contiguity (expression signal held fixed — verified: non-spatial ARI is 0.345 at every level,
  range 0.000), the spatial-prior advantage rises monotonically with contiguity and flips sign (−0.34 →
  +0.48 for neighbour-mean; −0.33 → +0.54 for STAGATE): Spearman **+0.98 / +1.00 (p<0.001)**. Because
  contiguity is the sole manipulated variable, this is causal evidence that contiguity *drives* the value
  of spatial priors — smoothing helps exactly to the degree a cell's neighbours share its domain.

## 7. Making the rule deployable: a label-free contiguity proxy (figures 1b, 2)

The rule as stated needs the GT — circular at deployment. We test whether contiguity can be estimated
from expression + coordinates alone (`gtfree_proxy.py`). The **obvious proxies invert**: Moran's I of
expression PCs correlates with advantage at −0.34, the unsupervised kNN same-cluster fraction at −0.42
(figure 2). Mechanism: raw-expression spatial autocorrelation reflects platform/cell-type structure,
which runs opposite to where spatial priors help (imaging domains are single-cell-noisy yet contiguous).
The right label-free question is whether spatial smoothing *reveals* structure raw clustering misses. The
proxy **`coh_gain`** — the extra spatial coherence unsupervised clustering gains on spatially-smoothed vs
raw expression — recovers the trend: ρ=+0.56 vs GT contiguity, **ρ=+0.55 vs advantage** (figure 1b),
seed-robust, and out-of-sample it still beats base rate (0.73 vs 0.55). Honest cost: going label-free
recovers the *direction*, not the oracle's significance (+0.72 → +0.55, marginal at n=11).

## 8. The candidate method, and a catalogue of evaluation pitfalls

Tessera, the candidate method, illustrates the thesis. Single-dataset/single-backend/single-metric views
each manufactured an apparent Tessera advantage that dissolved under broader evaluation (a cautionary
table in §4 of the detailed results). Comprehensively, Tessera is a **backend-robust generalist** (most
backend-stable embedding, ΔARI 0.036; biggest beneficiary of label refinement) but **not** a
SOTA-beating domain method. A component ablation on the real domain tasks (MERFISH, osmFISH;
`mechanism_ablation.json`) sharpens this: **only the boundary-contrastive loss earns its place** (dropping
it collapses ARI 0.45→0.17/0.20, back to the plain backbone), while edge-gating and multi-scale are
neutral-to-harmful (−0.01 to −0.03) — even Tessera's named anti-over-smoothing edge gate does not help. We report this as an honest negative — and as the clearest evidence that the
field's "new SOTA" claims are largely single-dataset artifacts.

**§8c — the headline is not a backend artifact.** Giving every method its author-recommended clusterer
(real R `mclust` via an `Rscript` subprocess, the clusterer the graph methods actually recommend) and
re-running all 11 platforms (1-seed sensitivity pass): real mclust genuinely matters (it wins 20 of
81 method×platform best-configs) yet shifts each method's **mean** ARI by ≤0.016 (per-platform cells vary
because this is 1-seed vs the 3-seed published means; conservative for the law). With the genuinely
recommended clusterer the spatial-prior law is **ρ=+0.79 (p=0.004)** — slightly above the published +0.72 —
and there are still **6 distinct winners**. The conclusions are not an artifact of the clustering backend;
the recommended backend makes the law cleaner, not weaker (`native_baselines.json`).

## 9. Discussion

The unit of progress in spatial-domain detection should shift from "a better method" to "a better
understanding of when spatial priors help". Our law gives a cheap, label-free pre-flight check: measure
the spatial-coherence gain on your own data; if it is low, a non-spatial clusterer is both floor and
ceiling and spatial smoothing can hurt; if high, a spatial method is worth it (and which one is within
seed noise). This reframes "no universal SOTA" from a warning into an actionable protocol.

## 10. Limitations (honest)

n=11 after removing a duplicate; local independent platforms are exhausted (remaining candidates are
same-biology slices, deliberately not used to pad n). Only the denominator-free spatial-prior advantage
correlation is significant; the two rank correlations are not. The label-free proxy is marginal at n=11
(the price of having no labels). All metrics are semi-circular or conditioned — no metric here is external
truth. SpaceFlow/GraphST fail on 32–36-dim protein panels (logged). The conclusion has already changed
twice as the panel grew (Tessera-rank law → spatial-prior law); more platforms could revise it again — it
is the current best estimate, not a final word.
