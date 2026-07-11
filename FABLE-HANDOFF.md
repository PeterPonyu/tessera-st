# FABLE-HANDOFF.md — tessera-st

Prepared 2026-07-02. Self-contained handoff for a future model ("Fable") to pick up cold.
Honest, evidence-backed, uncertainty marked. Reconciles with `CLAIM_LEDGER.md` (574 lines,
chronological, every entry LOCKED until it clears the 6 graduation gates) and `DESIGN.md`.

Hardware on this machine: RTX 5090 Laptop GPU (24 GB VRAM), conda env `dl` (torch 2.x + CUDA).
**No new GPU/heavy recompute is authorized for handoff work** — see `CLAUDE.md` §3. Everything
below is grounded in artifacts already on disk.

## Purpose

`tessera-st` is a boundary-aware, multi-scale spatial-domain-detection method for spatial
transcriptomics (ST), built on the thesis "a tissue is a *tessellation*: a small number of spatial
domains tiled across the slide, separated by sharp biological seams; existing graph encoders smooth
indiscriminately across those seams — Tessera learns where to *stop* smoothing" (`DESIGN.md`). The
project has grown from a method scaffold into an **11-platform meta-analysis and benchmark study**
(`manuscript/paper.tex`), and the manuscript's actual headline finding is now broader than the
method itself: a **causal law** for when spatial priors help at all.

## §1 Methodology classification

**Two things are true at once, and both must be represented honestly:**

1. **Tessera is a boundary-aware spatial-domain-detection *method*** — an edge-gated message-
   passing graph encoder (learned per-edge gate `g_ij → 0` = "stop smoothing here") + multi-scale
   fusion + a boundary-contrastive loss + a calibrated uncertainty head, four independently
   ablatable components (`src/tessera_st/model/`, `DESIGN.md`).
2. **The manuscript is now primarily an honest-negative benchmark + causal-mechanism study**, not
   a Tessera-superiority paper. Its title is *"A causal law for spatial-domain detection: the
   value of a spatial prior is set by the ground truth's spatial contiguity"* (retitled from an
   earlier "No universal SOTA in spatial-domain detection" — CLAIM_LEDGER §R5, 2026-07-01). Tessera
   is retained in that paper as a **worked example** of how single-dataset, single-metric
   evaluation can manufacture a false SOTA claim and a false mechanism story — not as the method
   being pitched for adoption.

This is a research-scaffold-turned-meta-analysis, not a from-scratch new-estimator paper, though
the edge-gating/boundary-contrastive mechanism itself is original (clean-room implemented — no
baseline source copied; see `ALLOWED_BASELINE_CONTEXTS.md`).

## §2 SOTA-to-beat / yardstick — be honest, this is NOT a beat-SOTA-on-ARI story

**Tessera does not beat SOTA on the standard ARI axis, and this is recorded, not hidden.**

- Single-section 151673, full comparison panel (STAGATE / SEDR / GraphST / BANKSY-style / 2 floors,
  fair GMM clustering backend, 3 seeds — `experiments/full_panel_v2.json`, pinned by
  `experiments/verify_manuscript.py`): **Tessera ARI rank 5 of 7**; STAGATE 0.580 > SEDR 0.570 >
  floor:smoothed 0.532 > **Tessera 0.527** > BANKSY 0.492 > floor:nonspatial 0.421 > GraphST 0.290
  (GraphST under-fit by the adapter, not its true level — disclosed).
- Tessera's **one full-field, non-semi-circular win** in that single-section panel is
  **boundary_F1 = 0.454** (beats all 6 opponents) — its designed-for thesis axis.
  **boundary_F1_rank = 3** and **backend_robustness_rank = 1** are the two numbers pinned by the
  verifier (`experiments/verify_manuscript.py`), i.e. the boundary lead is *not* universal even in
  the single-section view; the backend-robustness lead is.
- Cross-section (3 donors, `experiments/run_multisection_panel.py`): the single-section boundary
  story **does not generalize**. Boundary_F1 advantage evaporates (0.357, tied with BANKSY-style
  0.360); small_IoU flips to **worst** of panel (0.272 vs STAGATE 0.440); ARI/NMI trail. BANKSY-
  style (the simplest, non-DL baseline) has the **highest** cross-section ARI (0.512). Tessera's
  only durable cross-section-robust, non-circular property is **backend-robust embeddings**
  (lowest ARI sensitivity to KMeans-vs-GMM backend) — the calibration-ECE edge also evaporates once
  SEDR is included in the panel (SEDR ECE 0.189 < Tessera 0.247).
- An exhaustive sweep of borrowed SOTA mechanisms (DEC self-training, contrastive annealing, pure
  reconstruction, GAT attention gating) plateaus at ARI 0.52–0.56 and never clears STAGATE's 0.577
  — diagnosed as (a) DLPFC domain detection being a saturated task for graph-autoencoder methods,
  and (b) Tessera's contrastive term causing intrinsic long-training decay (`diag_collapse.py`
  probe: ARI peaks ~0.44 at epoch ~175 then falls to ~0.34 while loss keeps falling — an
  objective/task misalignment, not under-fitting or embedding collapse).

**The real yardsticks, in order of how defensible they are:**

1. **The causal law itself** (the actual paper headline, not Tessera-specific): synthetic
   contiguity-manipulation sweep, Spearman ρ = **+0.98 to +1.00, p<0.001** (spatial-prior advantage
   flips sign as GT contiguity is manipulated, expression signal held fixed); observational
   cross-platform confirmation on 11 platforms, ρ = **+0.72, p=0.012** (mclust-R backend:
   **+0.79, p=0.004**). **This is the number a future method should be measured against if the
   contribution is "predicting when a spatial prior helps," not "detecting domains better."**
2. **Boundary-F1 + backend-robustness**, not ARI, if the contribution is specifically
   Tessera-style boundary-aware detection. Current Tessera numbers: boundary_F1 0.454 (single-
   section, does not fully generalize), backend-robustness rank 1/7 (does generalize).
3. **No-universal-SOTA breadth**: 7–8 distinct per-platform winners across the 11-platform panel
   (`expanded_bench.json`; SpaGCN native run pushes it to 8). A new method claiming universality
   would need to contradict this pattern, not just win one platform.
4. Do **not** use single-section 151673 ARI as the yardstick — it is explicitly shown to flip
   across platforms (seqFISH reverses which methods win; DLPFC's "SOTA" STAGATE drops to last on
   seqFISH) and is the exact failure mode this project's own manuscript is built to expose.

## §3 Evaluation metrics

Full panel from `CLAIM_LEDGER.md` / `DESIGN.md`; **every metric is semi-circular, none is
enthroned, no verdict graduates from a single metric**:

| Dimension | Metric | Dir. | Circularity |
|---|---|:--:|---|
| label agreement | ARI, NMI | ↑ | weak — GT is itself an expression+histology-derived clustering |
| spatial coherence | CHAOS, PAS | ↓ | strong — smoothing is both the inductive bias and the judge; gameable by one big domain |
| geometric separation | ASW, DBI | ↑/↓ | strong — scored in the method's own embedding |
| boundary sharpness | boundary-F1 | ↑ | weak (inherits GT) |
| small-domain recovery | small-IoU | ↑ | weak (inherits GT) |
| calibration | ECE | ↓ | inherits GT |
| marker-prior view | marker_purity | ↑ | lower, not zero — not derived from this data's clustering, but conditioned on a marker prior that drifts across platform/sample |

Plus the meta-analysis-level metrics that carry the actual paper headline:
- **Spearman ρ(GT spatial contiguity, spatial-prior advantage)** — the causal-law correlation,
  both manipulated (synthetic, ρ=+0.98–1.00) and observational (11 platforms, ρ=+0.72–0.79).
- **`coh_gain`** — a label-free proxy for the same law (extra spatial coherence an unsupervised
  clustering gains from spatial smoothing). Honestly reported as **directional but not
  significant** once corrected for having been selected as the best of 6 candidates:
  search-corrected p = **0.28** (Westfall–Young max-statistic permutation, B=10,000), vs an
  uncorrected-per-candidate p = 0.079/0.083. Do not cite the uncorrected number as if it were the
  headline — the ledger and verifier both require the corrected one to be reported alongside it.

`experiments/verify_manuscript.py` is the machine gate: it loads the JSON artifacts and asserts
every headline number the manuscript states — 18+ checks as of R4/R5, including the ARI/boundary/
backend ranks above, the causal-sweep ρ's, the search-corrected proxy p, bootstrap/LOPO robustness
checks, and duplicate/vacuous-pass guards. Exit 0 is required before treating any manuscript number
as trustworthy; run it (`python experiments/verify_manuscript.py` from repo root) rather than
trusting prose.

## §4 Assets on disk

- **`experiments/*.json`** (35 files, **git-ignored / local-only** per `.gitignore` — regenerate
  from the corresponding `experiments/*.py` script if missing, but that is new compute and is
  out of scope by default; see `CLAUDE.md` §3). Key ones: `full_panel_v2.json`,
  `expanded_bench.json`, `hardened_bench.json`, `multisection_panel.json`,
  `mechanism_synth.json`, `proxy_search_corrected.json`, `native_baselines.json`,
  `dlpfc_151673_full.json`, `pareto_frontier.json`, `tessera_verdict.json`, `rigorous_bench.json`.
- **`experiments/*.py`** (~30 scripts) — benchmark/diagnostic runners, ~30 of which hardcode paths
  into `../spatial-omics-reform/{data,external,...}` for real DLPFC/multi-platform data and the
  named baseline repos (STAGATE/GraphST/SEDR/SpaceFlow/BANKSY). This is a **kept, legitimate
  reproduction dependency** (`ALLOWED_BASELINE_CONTEXTS.md`), not contamination.
- **`manuscript/`** — `paper.tex` (flagship, lualatex + pgfplots/TikZ figure stack), `refs.bib`
  (31 entries), `make_figs.R` (reads `experiments/*.json` → `manuscript/figs/*.tex`, ~10 fig
  files + 4 PNGs), `Makefile`, `paper.pdf` (built), `DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md`,
  `PAPER.md`.
- **`experiments/verify_manuscript.py`** — the machine-checked gate described in §3.
- **`src/tessera_st/`** — the actual method implementation (config/ablation toggles, data
  adapters, model, losses, eval metrics, CLI). MIT-licensed, clean-room.
- **`tests/`** — pytest suite (23 tests as of R4/R5) including `test_brand_independence.py`
  (clean-room enforcement) and `test_refine.py` (spatial label-refinement post-step).
- **`CLAIM_LEDGER.md`, `DESIGN.md`, `README.md`, `BASELINE_REFERENCES.md`,
  `ALLOWED_BASELINE_CONTEXTS.md`** — the provenance/claim/guardrail docs; read before changing any
  scientific statement.
- **`.zenodo.json` + `scripts/deposit_zenodo.py`** — submission-readiness artifacts (draft-only
  deposit tooling; not executed, no `ZENODO_TOKEN` on this machine — human input required).

## §5 Path-to-our-own-SOTA / current status

**Current status: 19pp manuscript draft, all scientific claims LOCKED in `CLAIM_LEDGER.md`,
machine-verified (`verify_manuscript.py` exit 0 as of the last recorded run), 6 graduation gates
defined but the paper does not depend on any of them passing — the headline claim is the causal
law, which is already evidenced, not the Tessera-superiority claims, which remain honestly
negative/LOCKED.**

The 6 graduation gates (from `CLAIM_LEDGER.md`, apply to claims **C1–C5**, i.e. Tessera-component
claims, not the causal-law headline):
1. Real-data run — ≥3 DLPFC sections through `tessera ablate-real` (done, exceeded: 3 donors +
   11-platform panel).
2. Full ablation reproduced on real data with seed variance ≥3 seeds (done for 151673).
3. Same-data baseline parity — named baselines on identical inputs (done: STAGATE/SEDR/GraphST/
   BANKSY-style all run under a shared eval panel and a shared, fairness-audited clustering
   backend).
4. Boundary + calibration evidence beyond ARI/NMI (done — boundary_F1/ECE reported throughout,
   and shown to be backend-robust for Tessera but not the standout claimed earlier once SEDR is
   included).
5. Clean leakage scan — `scripts/check_independence.sh` green, no baseline brand in `src/` (green,
   enforced by `tests/test_brand_independence.py`).
6. Failure-mode analysis + honest documentation of where Tessera loses (extensively done — this is
   the majority of `CLAIM_LEDGER.md`'s content: cross-section ARI/small-IoU losses, training-
   objective misalignment, seed instability, backend-confound analysis).

**What is NOT done / open, if a future agent wants to push this further (not authorized without
new compute — see `CLAUDE.md` §3):**
- Claims C1–C5 (component-level superiority claims) remain LOCKED — none has graduated to a
  paper-ready row, and per the ledger's honest-negative policy they may never graduate as
  "Tessera wins" claims; the more likely honest outcome is documenting which components are
  load-bearing (boundary_contrastive, robustly) vs liabilities/marginal (multi_scale, edge_gating).
- SEDR/GraphST were only runnable via a numba stub (`experiments/_numba_stub.py`) in this
  environment; their *true* native-pipeline numbers may differ.
- Only 3 of 6 available DLPFC sections and 11 of the platforms considered were used in the
  headline cross-platform numbers; more platforms would sharpen (or further attenuate, as the
  n=11→n=14 expansion already did: ρ dropped from +0.72 to +0.56) the causal-law correlation.
- Zenodo deposit and author affiliation/ORCID/funding fields are marked `[to be completed]` in
  `paper.tex` — human-only inputs, not something to fabricate.
- Venue is undecided (Nat. Methods/Commun./Genet. tier cited, none locked) — see `CLAUDE.md` §4.

## §6 Feasibility

**GREEN for polishing work; RED/deferred for new compute.** All 35 result JSONs, the trained-model
artifacts they summarize, and the manuscript figure pipeline already exist locally. Re-rendering
figures (`make_figs.R`), re-running `verify_manuscript.py`, recompiling LaTeX, and fixing
`refs.bib`/prose all require zero GPU time and zero new data pulls. The `../spatial-omics-reform`
dependency (real DLPFC h5ad files + baseline repo checkouts) is present on this machine per
`BASELINE_REFERENCES.md`, so if new compute is ever explicitly authorized, the datasets and
baseline adapters are already wired — but that authorization has not been given by default.
Hardware available if/when authorized: RTX 5090 Laptop GPU (24 GB VRAM).

## §7 Sources

- `CLAIM_LEDGER.md` (this repo, 574 lines) — the full chronological claim/evidence log; the
  primary source for every number in this handoff.
- `DESIGN.md` (this repo) — architecture thesis, ablatable components, metric-circularity audit.
- `BASELINE_REFERENCES.md`, `ALLOWED_BASELINE_CONTEXTS.md` (this repo) — named baselines
  (STAGATE Nat. Commun. 2022, GraphST Nat. Commun. 2023, SEDR Nat. Commun. 2024, SpaceFlow Nat.
  Commun. 2022, BANKSY Nat. Genet. 2024), the `../spatial-omics-reform/external/*` frozen audit
  clones, and the clean-room policy.
- `manuscript/paper.tex`, `manuscript/refs.bib` (31 entries), `manuscript/DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md`,
  `manuscript/PAPER.md` — the manuscript itself and its related-work positioning (5 directly-
  overlapping 2024–2026 benchmarks added and DOI-verified per CLAIM_LEDGER §R5).
- `experiments/verify_manuscript.py` — machine-readable ground truth for every headline number;
  re-run it rather than trusting any prose summary, including this one.

---
STATUS: PARTIAL-HONEST-NEGATIVE (method) / SUPPORTED (causal-law headline) — tessera-st is a
boundary-aware ST spatial-domain-detection method that does NOT beat named SOTA on ARI
(single-section rank 5/7; cross-section boundary-F1 lead does not generalize; durable edge is
backend-robust embeddings only) — manuscript.tex retitled to lead with a machine-verified CAUSAL
LAW (spatial-prior value tracks GT spatial contiguity: synthetic ρ=+0.98–1.00 p<0.001,
cross-platform ρ=+0.72 p=0.012, n=11) with Tessera retained only as a worked example; no method is
universal winner (7-8 distinct winners/11 platforms); label-free coh_gain proxy is directional but
NOT significant after search-correction (p=0.28); all claims C1-C5 remain LOCKED pending 6
graduation gates; venue undecided (Nat Methods/Commun/Genet tier); no new GPU/compute authorized
by default — next work is manuscript polishing (resync from existing experiments/*.json,
verify_manuscript.py, LaTeX compile) unless new compute is explicitly authorized.
