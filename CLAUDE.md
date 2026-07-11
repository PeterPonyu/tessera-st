# CLAUDE.md — tessera-st isolation charter

## 1. Identity & scope

`tessera-st` is an **independent, self-contained workspace**: a boundary-aware spatial-domain-
detection research line for spatial transcriptomics (ST), built around the thesis "a tissue is a
tessellation; learn where to stop smoothing" (see `DESIGN.md`). It is a sibling of, but not
subordinate to, other projects under `~/Desktop/labs/active/` (e.g. `spatial-omics-reform`,
`deconv-lab`). Operate on **this project only**. Do not read, infer conventions from, or apply
decisions/state from sibling projects unless a file inside `tessera-st` explicitly points there
(see §6).

## 2. Isolation boundary

This workspace is **decoupled from legacy OMC global memory and sibling-project state**. Any
`.omc/`, `.omx`, session cache, or notepad content that references other projects (dl-research,
spatial-omics-reform, deconv-lab, ml-reliability-research, etc.) is **legacy noise carried over
from the shared home-directory environment** — it is not a dependency of tessera-st and must not
be treated as ground truth for this repo. When in doubt, ground decisions in this repo's own
files: `CLAIM_LEDGER.md`, `DESIGN.md`, `README.md`, `manuscript/paper.tex`,
`experiments/verify_manuscript.py`. This repo has its own `.omc/` (project-local state) and its
own `.claude/`; do not merge or reconcile them with another project's.

## 3. Compute constraint — no new heavy recompute

**No new experiments, no new GPU training runs, no heavy recompute is authorized by default.**
"Manuscript polishing" in this repo means, in order of preference:

1. **Resync** already-computed values: read existing `experiments/*.json` artifacts (35 on disk,
   git-ignored/local-only — see `.gitignore`) and make sure `manuscript/paper.tex` / figures /
   `CLAIM_LEDGER.md` state exactly what those JSONs say. Never re-derive a number from raw data
   when a computed value already exists on disk.
2. **Re-render** figures from existing data: `manuscript/make_figs.R` reads `experiments/*.json`
   and emits `manuscript/figs/*.tex`; re-running it is a formatting/regen step, not new compute.
3. **Verify** refs and headline numbers: `experiments/verify_manuscript.py` machine-checks 18
   headline numbers in the manuscript against the JSON artifacts (extended by later rounds — see
   `CLAIM_LEDGER.md`); run it to check drift, not to generate new results.
4. **Compile** the LaTeX (`manuscript/paper.tex`, via `manuscript/Makefile`, lualatex for the
   pgfplots/TikZ figure stack) and fix bib/reference issues in `refs.bib`.

Any task that would require running a new benchmark, training a new model, adding a new dataset,
or launching a new ablation is **out of scope** unless the user explicitly authorizes new compute
in the current request. When a gap is found that would need new computation to close, surface it
and defer to the user — do not just go run it.

## 4. Flagship manuscript & venue

- Flagship: `manuscript/paper.tex` — currently titled *"A causal law for spatial-domain
  detection: the value of a spatial prior is set by the ground truth's spatial contiguity"*
  (retitled from an earlier "No universal SOTA" framing — see CLAIM_LEDGER §R5).
- **Venue is UNDECIDED.** The paper cites and is pitched at the Nat. Methods / Nat. Commun. /
  Nat. Genet. tier, but no venue is locked in. Keep prose, framing, and formatting
  **venue-agnostic** — do not hard-commit to one journal's template, word limits, or house style
  without being told which venue was chosen.
- Supporting docs: `manuscript/DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md`, `manuscript/PAPER.md`,
  `manuscript/README.md`.

## 5. Honest-claims guardrail (binding on all future work in this repo)

Tessera (the method) is **not positioned as SOTA-beating**. Preserve the honest-negative framing
recorded in `CLAIM_LEDGER.md`:

- On the single-section 151673 full comparison panel (STAGATE/SEDR/GraphST/BANKSY-style + floors,
  machine-checked in `experiments/verify_manuscript.py`): Tessera's ARI rank is **5** (of 7),
  boundary-F1 rank **3**, backend-robustness rank **1** — it is not the best label-agreement
  method, and its boundary-sharpness edge does **not** reliably generalize cross-section (the
  multisection panel shows the boundary_F1 lead evaporating and small-domain recovery becoming
  worst-of-panel). Its most durable, cross-section-robust property is **backend-robust
  embeddings** (lowest ARI sensitivity to KMeans-vs-GMM clustering backend).
- The manuscript's current headline is **not** a Tessera-superiority claim at all: it is a
  **causal law** (spatial-prior value tracks GT spatial contiguity, demonstrated both by a
  controlled synthetic manipulation and observationally across 11 platforms) with **no universal
  winner** across methods (7–8 distinct per-platform winners recorded). Tessera is retained as one
  *worked example* in that story, not the story's hero.
- **Never enthrone or one-stroke-kill a method on a single metric.** All spatial-domain-detection
  metrics here are semi-circular (see `DESIGN.md` and `CLAIM_LEDGER.md`'s "Evaluation caveat"
  section) — ARI/NMI are GT-derived-clustering-circular, ASW/DBI are scored in the method's own
  embedding, CHAOS/PAS reward one big domain, and even `marker_purity` is conditioned on a marker
  prior that drifts across platform/sample. Triangulate across the full panel; this mirrors the
  user's standing cross-project evaluation policy (no one-metric kills, no circular ground-truth
  overclaims).
- If asked to "fix" or "improve" the manuscript's framing of Tessera, do not quietly re-inflate
  the claims back toward a SOTA narrative. Any claim-status change must go through
  `CLAIM_LEDGER.md`'s LOCKED → graduated gate process, not a rewording.

## 6. Legitimate external dependency: spatial-omics-reform

About 30 scripts under `experiments/*.py` hardcode paths into
`/home/zeyufu/Desktop/labs/active/spatial-omics-reform/{data,external,...}` for real datasets
(DLPFC/spatialLIBD, multi-platform ST data) and baseline method repos (STAGATE, GraphST, SEDR,
SpaceFlow, BANKSY). **This is a KEPT, intentional reproduction dependency, not contamination** —
it lets Tessera's benchmark drop onto the same data/baselines as the established leaderboard (see
`BASELINE_REFERENCES.md`). Do not "clean up" or redirect these paths without understanding this.
The clean-room boundary that *does* apply: baseline brand names (STAGATE/GraphST/SEDR/SpaceFlow/
BANKSY) are banned from `src/`, `tests/`, and `scripts/` — allowed only in `README.md`,
`DESIGN.md`, `BASELINE_REFERENCES.md`, `CLAIM_LEDGER.md`, and `experiments/**` result notes. This
is enforced by `tests/test_brand_independence.py` and `scripts/check_independence.sh`. Full policy
in `ALLOWED_BASELINE_CONTEXTS.md`.

## 7. Pointers

- `FABLE-HANDOFF.md` — self-contained handoff summary (purpose, SOTA-to-beat, metrics, assets,
  status) for a future agent picking this project up cold.
- `CLAIM_LEDGER.md` — every scientific claim, its LOCKED status, required evidence, and the full
  chronological result log (read this before touching any claim in the manuscript).
- `DESIGN.md` — architecture thesis, the four ablatable components, and the full metric-circularity
  audit.
