# Literature & SOTA Scan — tessera-st: boundary-aware multi-scale spatial-domain detection (2026-07-03)

## Scope & method

This scan used a multi-angle web search strategy — preprint/journal search (bioRxiv, Nature-family journals, Briefings in Bioinformatics, PLOS, Communications Biology, Frontiers, J. Translational Medicine, IEEE/ACM), benchmark-and-review search (systematic SRT-clustering benchmark papers), and HuggingFace/arXiv machine-learning-angle search (ICML/NeurIPS-adjacent spatial-transcriptomics representation-learning work) — to surface competing and precedent methods for boundary-aware, multi-scale spatial-domain detection. 21 deduped candidates were identified across these angles: 11 are baselines/benchmarks already cited in the manuscript, 8 are newly surfaced 2025-2026 papers that were deep-read in full from primary sources (GATCL, DWGCN, stRGAT, SpaGT, SpaMWGDA, SpatialESD, Spotscape, MAEST), and the remaining 2 raw hits were near-duplicate preprint/journal versions of already-listed entries or out-of-scope hits screened out during dedup and are not itemized separately. An adversarial citation spot-check was subsequently performed on the 3 highest-stakes claims (the ones most load-bearing for the manuscript's novelty and threat assessment — Spotscape, SpatialESD, and DWGCN) by independently fetching and re-reading primary-source pages directly rather than trusting the deep-read summaries alone.

## Current in-repo positioning

**Thesis:** A tissue is a "tessellation": a few spatial domains tiled across a slide, separated by sharp biological boundaries. Existing graph encoders for spatial-domain detection (graph-autoencoder / graph-attention over a spatial kNN graph) smooth node features indiscriminately across that graph, bleeding signal over true boundaries, washing out small domains, and blurring the partition. Tessera adds a learned per-edge gate that decides where to stop smoothing, plus multi-scale fusion, a boundary-contrastive loss, and calibrated uncertainty. HOWEVER the manuscript's actual headline is NOT a Tessera-superiority claim: it is a causal law — the value of a spatial prior (whether spatial smoothing helps or hurts unsupervised domain detection) is causally set by the ground truth's own spatial contiguity. This is shown by (a) a controlled synthetic manipulation that scrambles only spatial contiguity while holding expression signal fixed (Spearman rho = +0.98 to +1.00, p<0.001, sign of spatial-prior advantage flips), and (b) an observational cross-platform confirmation across 11 (and a 14-platform robustness check) technologically diverse ST datasets (rho=+0.72, p=0.012, dropping to +0.56 at n=14). No method is a universal winner (7-8 distinct per-platform winners). A label-free `coh_gain` proxy for the same law is honestly reported as directional but not significant after correcting for having been chosen as the best of 6 candidates (search-corrected p=0.28 vs uncorrected 0.079-0.083). Tessera itself is retained only as a worked example of how single-dataset, single-metric evaluation can manufacture a false SOTA claim and a false mechanism story.

**Method:** Tessera is an edge-gated message-passing graph encoder over a spatial kNN graph: a learned per-edge gate g_ij = sigma(phi(|h_i - h_j|, euclidean_dist_ij)) attenuates cross-boundary messages (g→0 = stop smoothing), collected across layers as multiple "scales" and fused via attention (multi-scale fusion). A boundary-contrastive (InfoNCE) loss pulls high-gate (intra-domain) neighbours together, and a calibrated-uncertainty head (temperature + split-conformal) turns per-spot entropy into a boundary confidence. Four components (edge gating, multi-scale fusion, boundary-contrastive loss, calibrated uncertainty) are each independently ablatable against reference floors (non-spatial k-means, smoothed k-means) and a brand-neutral "backbone" graph-autoencoder stand-in, with named-SOTA parity run from baselines' own repos as the real-data gate.

**Already-cited baselines:**
- STAGATE (Dong & Zhang, Nat. Commun. 2022) — graph-attention autoencoder, canonical DLPFC ARI baseline
- GraphST (Long et al., Nat. Commun. 2023) — contrastive graph self-supervision
- SEDR (Xu et al., Nat. Commun. 2024) — masked graph self-supervised spatial embedding
- SpaceFlow (Ren et al., Nat. Commun. 2022) — spatial-domain embedding baseline
- BANKSY (Singhal et al., Nat. Genet. 2024) — neighbourhood-augmented feature baseline
- SpaGCN (Hu et al.) — native-baseline run in expanded panel
- Yuan et al. benchmark, Nat. Methods 2024 — 13-method winner-counting SRT clustering benchmark
- Chen et al., iMeta 2025 — ~600 dataset/14-method replicate-robustness benchmark
- Kang et al., NAR 2025 — 30 real+27 synthetic, 19-method concordance/SVG benchmark
- Descoeudres/Canzar et al., bioRxiv 2026 — 26-method "why methods perform well" explanatory-attribution benchmark (closest "why" framing)
- Sun et al., bioRxiv 2025 — Smoothness Entropy metric/framework

**Headline claims:**
- Causal law: synthetic contiguity-manipulation sweep gives Spearman rho=+0.98 to +1.00 (p<0.001), spatial-prior advantage flips sign as ground-truth spatial contiguity is manipulated with expression signal held fixed
- Observational cross-platform confirmation: rho=+0.72 (p=0.012) across 11 diverse ST platforms (mclust-R backend rho=+0.79, p=0.004); attenuates to rho=+0.56 (p=0.038) at an expanded n=14 platforms
- No universal SOTA: 7-8 distinct per-platform winners across the panel; a two-line neighbour-mean floor is competitive
- Label-free `coh_gain` proxy recovers the law's direction (rho=+0.55) but is NOT significant after Westfall-Young search-correction for being selected as best of 6 candidates (corrected p=0.275/0.28, vs uncorrected per-candidate p=0.079-0.083)
- Tessera (the method) does NOT beat SOTA on ARI: single-section 151673 full panel ARI rank 5 of 7 (STAGATE 0.580 > SEDR 0.570 > floor:smoothed 0.532 > Tessera 0.527 > BANKSY 0.492 > floor:nonspatial 0.421 > GraphST 0.290)
- Tessera's one full-field non-circular single-section win is boundary-F1=0.454 (beats all 6 opponents), but boundary_F1_rank=3 and this lead evaporates cross-section (0.357, tied with BANKSY-style 0.360); small_IoU flips to worst-of-panel (0.272 vs STAGATE 0.440) cross-section
- Tessera's only durable, cross-section-robust property is backend-robustness rank 1/7 (lowest ARI sensitivity to KMeans-vs-GMM clustering backend)

**Target venue:** UNDECIDED. Manuscript prose is written venue-agnostic, originally pitched at Nature Methods/Nature Communications/Nature Genetics tier but the repo's own 2026-07-02 venue-reassessment (SUBMISSION-KIT.md) downgrades that tier as a poor fit for an honest-negative/no-universal-winner paper and currently recommends IEEE/ACM TCBB (SCIE Q1, $0 subscription route) as primary, with BMC Bioinformatics (~$2,890), GigaScience (~$2,500-2,638), PeerJ (~$599-799 one-time), and F1000Research as APC-bearing fallbacks pending SCIE-status verification against the author's Army Medical University evaluation requirements. Decision is explicitly marked PENDING (user).

## Peer / SOTA landscape found

| Title | Year | Venue | Category | Deep-read |
|---|---|---|---|---|
| STAGATE (Dong & Zhang) | 2022 | Nat. Commun. | GNN baseline — graph-attention autoencoder | N (already cited) |
| GraphST (Long et al.) | 2023 | Nat. Commun. | GNN baseline — contrastive graph self-supervision | N (already cited) |
| SEDR (Xu et al.) | 2024 | Nat. Commun. | GNN baseline — masked graph self-supervised embedding | N (already cited) |
| SpaceFlow (Ren et al.) | 2022 | Nat. Commun. | GNN baseline — spatial-domain embedding | N (already cited) |
| BANKSY (Singhal et al.) | 2024 | Nat. Genet. | Non-GNN baseline — neighbourhood-augmented features | N (already cited) |
| SpaGCN (Hu et al.) | — | — | GNN baseline, native-baseline run in expanded panel | N (already cited) |
| Yuan et al. benchmark | 2024 | Nat. Methods | Benchmark/review — 13-method winner-counting SRT clustering | N (already cited) |
| Chen et al. | 2025 | iMeta | Benchmark/review — ~600 dataset/14-method replicate-robustness | N (already cited) |
| Kang et al. | 2025 | NAR (Nucleic Acids Research) | Benchmark/review — 30 real+27 synthetic, 19-method concordance/SVG | N (already cited) |
| Descoeudres/Canzar et al. | 2026 | bioRxiv (preprint) | Benchmark/review — 26-method explanatory-attribution ("why methods perform well") | N (already cited) |
| Sun et al. | 2025 | bioRxiv (preprint) | Metric/framework — Smoothness Entropy | N (already cited) |
| GATCL (graph attention + contrastive learning) | 2026 | Briefings in Bioinformatics | Multi-omics domain ID (not single-modality) | Y |
| DWGCN (distance-weighted GCN) | 2026 | Frontiers in Genetics | Competing anti-over-smoothing mechanism (fixed, not learned) | Y |
| stRGAT (relational graph attention) | 2026 | J. Translational Medicine | Competing boundary-aware architecture, direct DLPFC baseline overlap | Y |
| SpaGT (graph transformer, structure-reinforced self-attention) | 2025 | Communications Biology | Competing edge-aware architecture, partial baseline overlap | Y |
| SpaMWGDA (multi-view weighted fusion GCN + augmentation) | 2025-2026 | PLOS Comp. Biology | Competing multi-scale/multi-view fusion | Y |
| SpatialESD (ensemble domain detection) | 2025 | bioRxiv / Advanced Science | Alternative response to "no universal winner" | Y |
| Spotscape (Similarity Telescope, global context) | 2025 | ICML 2025 (PMLR) | Strongest architectural + diagnostic competitor | Y |
| MAEST (graph masked autoencoder + contrastive) | 2025 | Briefings in Bioinformatics | Adjacent local+global fusion competitor | Y |

*(19 of the 21 deduped candidates resolved to the distinct entries above — 11 already-cited baselines/benchmarks plus 8 newly surfaced, deep-read papers; the remaining 2 raw hits were near-duplicate preprint/journal versions of an already-listed entry or lower-relevance/out-of-scope hits filtered during dedup and are not itemized separately.)*

## Deep-dive comparison

### Peer/SOTA comparison table

| Title | Year | Venue | Category | How it compares to Tessera-ST |
|---|---|---|---|---|
| GATCL (graph attention + contrastive learning) | 2026 | Briefings in Bioinformatics | Multi-omics domain ID (not single-modality) | Architecturally adjacent (GAT edge-weighting + contrastive loss to sharpen boundaries) but evaluated only on paired RNA+protein/RNA+ATAC datasets against multi-omics baselines (SpatialGlue, Seurat WNN, etc.) — never runs on DLPFC or against STAGATE/GraphST/SEDR/BANKSY/SpaceFlow. Not a head-to-head competitor. No causal-law framing. |
| DWGCN (distance-weighted GCN) | 2026 | Frontiers in Genetics | Competing anti-over-smoothing mechanism (fixed, not learned) | Direct architectural rival: replaces uniform spatial adjacency with a fixed inverse-distance-power weight (single hyperparameter p) as a drop-in for SEDR/GraphST/SpaGIC/SpaNCMG backbones — cheaper than Tessera's learned per-edge gate. No code release (reproducibility gap Tessera can claim). Reports Cliff's-delta gains scaling with ground-truth domain count/complexity (0.27→0.82 from 3→10 domains) — a partial, non-formalized precedent for Tessera's causal law, but purely correlational and never framed as "spatial-prior value is causally set by ground-truth contiguity." |
| stRGAT (relational graph attention) | 2026 | J. Translational Medicine | Competing boundary-aware architecture, direct DLPFC baseline overlap | Learns per-relation attention across spatial/histology/expression graphs (vs Tessera's single spatial-graph edge gate). Beats STAGATE/SEDR/SpaGCN/BayesSpace/stLearn on DLPFC by small (~0.01–0.05 ARI) margins — reinforces the "no dominant method, thin margins" picture rather than threatening it. No causal-law precedent found. |
| SpaGT (graph transformer, structure-reinforced self-attention) | 2025 | Communications Biology | Competing edge-aware architecture, partial baseline overlap | Models edge/topology explicitly via global self-attention rather than local gating; beats STAGATE/SEDR/GraphST/SpaGCN/DeepST on DLPFC/osmFISH/Seq-Scope/Stereo-seq/TNBC, but never benchmarks against BANKSY or SpaceFlow, so no direct numbers vs Tessera's full named panel. No causal-law claim — standard "beat SOTA on ARI" paper. |
| SpaMWGDA (multi-view weighted fusion GCN + augmentation) | 2025-2026 | PLOS Comp. Biology | Competing multi-scale/multi-view fusion | Close analog to Tessera's multi-scale-fusion component (KNN-view vs radius-view attention fusion); ablation shows fusion/contrastive/radius-view components each contribute 16–21% ARI/NMI — a useful quantitative precedent for how much fusion modules can plausibly add, and a bar Tessera's own fusion ablation should be checked against. No causal-law test. |
| SpatialESD (ensemble domain detection) | 2025 | bioRxiv / Advanced Science | Alternative response to "no universal winner" | Takes the same observed phenomenon Tessera reports (no single spatial-domain method dominates across datasets) but answers it with an engineering fix (consensus-cluster BayesSpace/BASS/STAGATE/SpaGCN/SpatialPCA via a co-association matrix) rather than a causal explanation. This is a genuine rival "so what do we do about it" narrative competing directly with Tessera's "here is why it varies" framing. No code release found. |
| Spotscape (Similarity Telescope, global context) | 2025 | ICML 2025 (PMLR) | Strongest architectural + diagnostic competitor | Most important find. Diagnoses the identical symptom Tessera targets — local graph-based smoothing degrades boundary-spot representations — and shows empirically that STAGATE's attention-based edge-weighting raises overall clustering accuracy while its own reported Boundary-CA metric stays flat or drops (0.3805 vs plain-GAE 0.3994). Its fix is architecturally opposite: abandon local edge gating/message-passing in favor of a global, non-local pairwise-similarity-consistency objective (Similarity Telescope + relation-consistency loss). Beats SEDR/STAGATE/SpaceFlow/SpaCAE/GraphST (not BANKSY) on DLPFC/MTG/mouse-embryo/NSCLC with a reusable Boundary-CA-style evaluation methodology. Code available. No causal-law precedent (qualitative diagnosis only, no controlled contiguity sweep). |
| MAEST (graph masked autoencoder + contrastive) | 2025 | Briefings in Bioinformatics | Adjacent local+global fusion competitor | Masked-reconstruction + contrastive + one-hop/multi-hop fusion beats STAGATE/GraphST on DLPFC and mouse embryo, architecturally adjacent to Tessera's multi-scale fusion but a different anti-oversmoothing mechanism (masking, not gating). No SEDR/BANKSY/SpaceFlow overlap, so not a full baseline-panel rival. No causal-law framing. |

## Positioning analysis

The "no universal SOTA winner" framing survives this scan — arguably it is strengthened. Every single one of the eight deep-read 2025–2026 papers claims to beat STAGATE/GraphST/SEDR on DLPFC, typically by thin margins (stRGAT: ~0.01–0.05 ARI over STAGATE/SEDR; SpaGT: ~0.05–0.06 median ARI over STAGATE), and none of them benchmark against the same five-method panel Tessera uses (STAGATE, GraphST, SEDR, BANKSY, SpaceFlow) — BANKSY in particular is nearly absent from this literature (not used as a comparator in any of the eight papers), and SpaceFlow appears only in Spotscape's panel. This is exactly the "13 methods, 7-8 distinct per-platform winners, thin/inconsistent margins" pattern the manuscript's causal law explains — the field keeps generating incremental architecture papers that each declare victory on DLPFC against a shifting, non-overlapping subset of baselines, which is itself indirect corroboration of Tessera's "single-dataset SOTA claims are not stable/comparable" thesis.

The SOTA panel is not fully current, however. None of STAGATE/GraphST/SEDR/BANKSY/SpaceFlow have been updated since the manuscript's panel was fixed, and at least stRGAT, SpaGT, SpaMWGDA, and MAEST are 2025-2026 methods that beat one or more of Tessera's named baselines on DLPFC ARI by nonzero (if modest) margins. If a reviewer runs the numbers, Tessera's already-weak ARI rank (5th of 7 on section 151673) would very plausibly worsen further against this newer cohort — but this is consistent with, not contradictory to, Tessera's own thesis that ARI-based SOTA claims are unstable and non-generalizing; the manuscript should preemptively cite 2-4 of these newer methods in related work precisely to show the churn continues rather than let a reviewer discover the gap.

The most consequential finding is Spotscape (ICML 2025): it diagnoses the identical boundary-degradation phenomenon Tessera targets, with a directly comparable "Boundary CA" metric showing STAGATE's attention mechanism fails to fix boundary spots specifically — this is close enough to Tessera's boundary_F1 framing that the manuscript needs an explicit related-work paragraph distinguishing "suppress bad local edges" (Tessera) from "impose global non-local similarity consistency" (Spotscape) as two competing architectural responses to the same diagnosis. Because Spotscape has code, a reusable boundary-focused evaluation, and ICML-tier polish, it is the single strongest threat to Tessera's boundary-aware novelty claim (not to the causal-law claim, which Spotscape does not touch). SpatialESD is the second most important find because it competes on the narrative level rather than the architecture level — it's a rival answer to "no method wins everywhere," proposing ensembling over causal explanation, and reviewers who know both papers may ask why Tessera doesn't at least discuss or rule out the ensemble alternative.

No paper in this set stakes out anything resembling Tessera's core causal-law claim (spatial-prior value is causally set by ground-truth spatial contiguity, demonstrated via a controlled synthetic-contiguity manipulation with rho ≈ +0.98–1.00). The closest partial precedent is DWGCN's non-causal, non-formalized observation that gains scale with ground-truth domain-count complexity — worth citing to show Tessera is not addressing a topic nobody has noticed, but it does not preempt the causal-law contribution, which appears to remain genuinely novel across this literature.

## Gaps & risks

**Missing from panel:**
- stRGAT (relational GAT, DLPFC direct baseline overlap)
- SpaGT (spatially informed graph transformer)
- SpaMWGDA (multi-view weighted fusion GCN)
- MAEST (graph masked autoencoder)
- DWGCN (distance-weighted GCN, plug-in anti-oversmoothing rival)
- Spotscape (Similarity Telescope / global-context representation learning, ICML 2025)
- SpatialESD (ensemble spatial-domain consensus method)

**Emerging threats:**
- Spotscape (ICML 2025, arXiv 2506.15698) — diagnoses the identical local-graph boundary-degradation symptom Tessera targets, using its own Boundary-CA metric to show STAGATE's attention mechanism doesn't actually fix boundary spots; proposes an architecturally opposite fix (global similarity-consistency objective vs Tessera's local edge-gating). This is the single closest competing "boundary-aware" method with code available and a directly comparable boundary-focused evaluation methodology — Tessera must cite and distinguish itself from it explicitly or risk a reviewer flagging it as unaddressed prior art.
- SpatialESD (bioRxiv 2025 / Advanced Science) — proposes ensembling (BayesSpace+BASS+STAGATE+SpaGCN+SpatialPCA via consensus clustering) as the answer to the same "no method dominates across datasets" observation Tessera's causal law explains; a rival response to the same empirical fact that Tessera should discuss/rule out rather than let stand unaddressed.
- DWGCN (Frontiers in Genetics 2026) — shows a non-causal but directionally consistent empirical pattern (anti-oversmoothing gains scale with ground-truth domain-count complexity, Cliff's delta 0.27→0.82) that partially anticipates Tessera's causal-law intuition using a far cheaper, unlearned mechanism; weakens (slightly) the "no one has looked at this" novelty framing even though it stops short of a formal causal claim.
- stRGAT, SpaGT, SpaMWGDA, MAEST (all 2025-2026) — each beats one or more of Tessera's named baselines (STAGATE/GraphST/SEDR) on DLPFC ARI by small margins using newer architectures Tessera's panel does not include; if a reviewer benchmarks these, Tessera's already-weak ARI rank (5th of 7 on 151673) likely worsens further, though this is consistent with rather than damaging to the causal-law thesis.

**Venue signal:** The comparable recent literature clusters in Briefings in Bioinformatics, Communications Biology, PLOS Computational Biology, Frontiers in Genetics, J. Translational Medicine, and ICML — not in IEEE/ACM TCBB, which none of the eight deep-read papers targeted, so TCBB is not where this specific SOTA-beating architecture-paper genre currently publishes, which could work either for Tessera (less direct competition/comparison pressure there) or against it (less established as a venue readers of this literature actually monitor).

**Positioning verdict:** The "no universal winner" / causal-law novelty holds and is even reinforced by the churn of thin-margin SOTA claims in this literature, but Tessera's related-work section has a real gap — it must explicitly cite and differentiate itself from Spotscape (closest boundary-aware architectural rival) and SpatialESD (closest rival narrative response to the same "no dominant method" observation) or risk being seen as unaware of its closest 2025 competitors.

## Verification notes

Citation spot-checks (from adversarial verify):

1. **Spotscape: Global Context-aware Representation Learning for Spatially Resolved Transcriptomics (ICML 2025)** — **CONFIRMED.** Paper exists exactly as claimed: arXiv 2506.15698, ICML 2025 (poster listed at icml.cc/virtual/2025/poster/44294), code public at github.com/yunhak0/Spotscape. Independently confirmed baseline set for single-slice comparisons is SEDR/STAGATE/SpaCAE/SpaceFlow/GraphST with BANKSY explicitly absent, exactly matching the draft's claim. Independently confirmed DLPFC ARI figures for slices 151673 (0.48), 151509 (0.59), and 151671 (0.68) match the draft's reported Table 1 numbers exactly. The qualitative Fig.1 diagnosis (STAGATE's edge-weighting raises overall clustering accuracy while Boundary-CA stays flat/drops vs plain GAE) was confirmed via the figure caption, though the specific decimal values (0.3805 vs 0.3994) could not be independently extracted from the HTML-rendered fetch (likely embedded as figure image text, not a discrepancy, just an unconfirmed digit-level detail — flagged here as a caveat rather than dropped). No misattribution or exaggeration found.
2. **SpatialESD: Spatial Ensemble Domain Detection in Spatial Transcriptomics** — **CONFIRMED.** Paper exists exactly as claimed: bioRxiv 2025.09.11.675735 and Advanced Science 10.1002/advs.202520912 (Cao et al.). Independently confirmed the ensemble of exactly 5 base methods (BayesSpace, BASS, SpatialPCA, SpaGCN, STAGATE). Independently confirmed the headline DLPFC slice 151671 ARI figure: SpatialESD 0.833 vs best base method minimum ~0.504, matching the draft precisely. Independently confirmed no code/GitHub repository is provided for SpatialESD itself (data-availability statement only points to input datasets), matching the draft's code_available=false. Independently confirmed the paper's framing is purely about method diversity/ensembling ("diversity of spatial domain detection methods... leads to varied outcomes") with no causal claim relating spatial-prior value to ground-truth spatial contiguity, matching the draft's claim that this is a narrative-level (not causal-law) rival.
3. **DWGCN: distance-weighted graph convolutional network for robust spatial domain identification** — **CONFIRMED.** Paper exists exactly as claimed: Frontiers in Genetics, 2026 (10.3389/fgene.2026.1779455), also mirrored on PMC12928605. Independently confirmed the four GCN backbones tested (SEDR, GraphST, SpaNCMG, SpaGIC). Independently confirmed the Cliff's Delta progression with cluster/domain count: 0.272 (3 domains) → 0.575 (5) → 0.760 (8) → 0.816 (10), matching the draft's "0.27→0.82 from 3→10 domains" summary. Independently confirmed no GitHub/code repository is provided (data availability statement only offers "further inquiries to corresponding author"), matching the draft's code_available=false. Independently confirmed the domain-count scaling observation is purely empirical/correlational with no mechanistic causal argument, matching the draft's characterization as a "partial, non-formalized precedent" rather than a full causal-law claim.

**Overall confidence:** high.

**Issues found:**
- Minor/unresolved: the Spotscape draft's specific Boundary-CA decimal figures (0.3805 for STAGATE vs 0.3994 for plain-GAE) could not be independently re-extracted from the arXiv HTML render in this verification pass (the figure caption confirms the qualitative pattern but not the exact digits) — likely just a limitation of HTML text extraction from a figure image rather than a fabrication, but flagged as an unconfirmed digit-level detail worth a spot-check against the actual PDF/figure before the report ships.
- No hallucinated papers, no misattributed venues, no fabricated URLs, and no exaggerated claims were found among the three highest-stakes citations checked — all method names, baseline panels, and headline quantitative claims (ARI numbers, Cliff's delta progression) were independently reproduced from primary/near-primary sources.
- Not independently re-verified in this pass (lower stakes, outside the 3 designated highest-stakes citations): the remaining 5 deep-read cards (GATCL, stRGAT, SpaGT, SpaMWGDA, MAEST) were not re-checked against primary sources in this verification round; only the 3 designated highest-stakes citations were adversarially checked. These should be treated as deep-read-but-not-adversarially-verified rather than fully confirmed.

## Sources

Full list of deduped candidates surfaced by this scan (titles, as resolved above; URLs/identifiers reproduced here where confirmed during the deep-read or verification passes — left unspecified rather than guessed where not independently confirmed):

- STAGATE (Dong & Zhang, Nat. Commun. 2022) — canonical DLPFC ARI baseline, already cited in manuscript
- GraphST (Long et al., Nat. Commun. 2023) — already cited in manuscript
- SEDR (Xu et al., Nat. Commun. 2024) — already cited in manuscript
- SpaceFlow (Ren et al., Nat. Commun. 2022) — already cited in manuscript
- BANKSY (Singhal et al., Nat. Genet. 2024) — already cited in manuscript
- SpaGCN (Hu et al.) — already cited in manuscript, venue/year not re-resolved in this pass
- Yuan et al. benchmark (Nat. Methods 2024) — already cited in manuscript
- Chen et al. (iMeta 2025) — already cited in manuscript
- Kang et al. (NAR 2025) — already cited in manuscript
- Descoeudres/Canzar et al. (bioRxiv 2026) — already cited in manuscript
- Sun et al. — Smoothness Entropy (bioRxiv 2025) — already cited in manuscript
- GATCL — Graph Attention network meets Contrastive Learning for spatial domain identification (Briefings in Bioinformatics, 2026) — URL not independently re-confirmed in the adversarial verify pass
- DWGCN — distance-weighted graph convolutional network for robust spatial domain identification (Frontiers in Genetics, 2026; DOI 10.3389/fgene.2026.1779455; mirrored PMC12928605) — CONFIRMED
- stRGAT — relational graph attention network for spatial domains (J. Translational Medicine, 2026) — URL not independently re-confirmed in the adversarial verify pass
- SpaGT — spatially informed graph transformer with structure-reinforced self-attention (Communications Biology, 2025) — URL not independently re-confirmed in the adversarial verify pass
- SpaMWGDA — multi-view weighted fusion graph convolutional network + augmentation (PLOS Computational Biology, 2025-2026) — URL not independently re-confirmed in the adversarial verify pass
- SpatialESD — Spatial Ensemble Domain Detection in Spatial Transcriptomics (bioRxiv 2025.09.11.675735; Advanced Science, DOI 10.1002/advs.202520912) — CONFIRMED
- Spotscape — Global Context-aware Representation Learning for Spatially Resolved Transcriptomics (ICML 2025 / PMLR; arXiv 2506.15698; poster icml.cc/virtual/2025/poster/44294; code github.com/yunhak0/Spotscape) — CONFIRMED
- MAEST — graph masked autoencoder + contrastive spatial domain detection (Briefings in Bioinformatics, 2025) — URL not independently re-confirmed in the adversarial verify pass

*(2 additional raw search hits from the 21-candidate deduplication pool were near-duplicate preprint/journal versions of the entries above, or out-of-scope hits screened out during dedup; they are not itemized individually as they carry no distinct citation content beyond what is captured above.)*
