# Current working face — original-manuscript revision, 2026-09-23

Use **`paper.tex` → `paper.pdf`** for the integrated field-aware revision.
This is the continued original manuscript, not a candidate core summary.
Status remains **SCIENTIFIC_SUBMISSION_HOLD**. The new seven-object, seven-procedure,
three-seed panel is complete (147 runs); SpaceFlow/GraphST/SpaGCN and the larger
historical panel are not repaired by that completion.

Build from the capsule root with `bash manuscript/scripts/rebuild_current.sh`.
It verifies bound results, redraws only revision figures from bundled ledgers,
and uses native `latexmk`; no model fitting or download. Numerical provenance is
in `ledgers/revision_20260923/`; current identity is in `release-current.json`.
Detailed limits and verification commands are in `../REPRODUCIBILITY-CURRENT.md`.

The 1 October 2026 continuation also builds **`SI.tex` -> `SI.pdf`**.
Its five readable tables export the committed field identities, named ARI
contrasts, row-order counterexample and paired generator endpoints. Every
table has a scientific entry in the main Results; none is a new model run
or an independent validation. The SI exporter saves input/table hashes in
`ledgers/revision_20260923/supporting_tables-provenance.json`, and current
build and release checks bind both PDFs. Historical supplements are unchanged.

`submission_flat/` and `submission_flat_candidate/` are **historical snapshots**,
untouched by this integration. Their old hashes, readiness text and near-ready
designations cannot certify the revised working face. A prior worker's inability
to view images is recorded; rendered pages still need visual review.

## Archived manuscript index (not current authority)

The following description records the older submission face. Its old headline,
working-leftover designation and build comments are not current instructions.

# Previous manuscript index — Tessera frozen face

**Official face is the 14-page observational / synthetic-intervention kit**, not the
working leftover and not the retired causal-law freeze.

- Official: `submission_flat/paper.pdf` sha256 `c407880a…099b` (14 pp; pin: `official-face.sha256`).
  Prior official `06c083df` / 14 pp is the previous rebind of this observational face.
  Prior official `dcd52090` / 15 pp is archived at `historical/official-dcd52090-pre-deep-audit/`.
  Prior official `a2804530` / 15 pp is archived at `historical/official-a2804530-science-20260918/`.
  Prior official `28328e15` / 14 pp is archived at `historical/official-28328e15-observational-20260918/`.
  Prior official `0549af0e` / 14 pp is archived at `historical/official-0549af0e-observational-20260911/`.
  Prior official `715a635c` / 13 pp is archived at `historical/official-715a635c-observational-20260911/`.
- Retired freeze `247b0657` / 13 pp: `historical/freeze-247b0657-causal-law-20260907/`.
  **Not official.**
- Working leftover compositor `961d6662` / 22 pp is archived at
  `historical/leftover-22pp-compositor-20260910/`. `manuscript/paper.tex` now
  compiles the **default IEEEtran journal face** (same 14 pp source as
  `submission_flat/`, `\includegraphics` of PDF panels, not `\resizebox+\input`
  TikZ). **Not a second official face.** Official remains `submission_flat/paper.pdf`.
- 18 pp contrastive paper: retired (`historical/retired-18pp-contrastive-33a5f9b/`).

See `../FACE-HONESTY.md` and `../SOURCE_PIN.md`.

The durable scientific result of the Tessera-ST program. Produced and quality-gated via ultragoal
(plan/ledger under `../.omc/ultragoal/`).

## Read this

**[`DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md`](DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md)** — historical
lab note of the confound-controlled benchmark that scaled to **11 platforms, 6 real methods,
3 seeds**. It is **not** a deployable recipe. Do not read it as “use a spatial method iff
contiguity is high.” Official claims live on `submission_flat/paper.tex`.

## Headline (official 14 pp; *n*=11, 6 methods incl. SpaceFlow + SpatialLeiden, 3 seeds)

1. **Observational association, not a deployable law.** GT spatial contiguity tracks the value of
   spatial priors at Spearman **+0.72 (uncorrected p = 0.012, 95% CI [0.24, 0.92])** — a
   diverse-panel upper estimate and a definitional calibration. Dropping MERFISH lowers ρ to
   **0.63**; *n*=14 attenuates to **+0.56 (p = 0.038)**; family-of-7 Bonferroni is **p = 0.082**
   (not significant at α = 0.05). Tessera-specific rank correlation is **−0.36 (p = 0.28)**;
   SpaceFlow rank is **−0.59 (n = 9, p = 0.10)**. There is **no** “use a spatial method iff it
   is high” rule. The label-free proxy has the same sign (ρ = +0.55) but search-corrected
   **p = 0.28**.
2. **No universal SOTA** — **7 distinct winners across 11 platforms** (SpatialLeiden, a graph-community
   method with an *inverted* profile, wins the least-contiguous platform CODEX and ranks #2 on the scattered
   ones yet last on contiguous MERFISH — method–task fit is real and method-specific).
3. **Seed-level tie at the top, not a Tessera win** — on the contiguous spatial-domain tasks a spatial
   method wins, but which one is within seed noise: across three independent runs the MERFISH winner flipped
   SpaceFlow→Tessera→SpaceFlow. Tessera is a backend-robust **generalist** (ΔARI 0.036), not uniquely best.

> An internal code review (G008 gate) caught a duplicate dataset (n 12→11), a Spearman tie-handling artifact,
> and single-seed "wins"; all corrected. The hardening round (G009–G010) then added a 6th method, a 3rd seed,
> and ρ seed-robustness — the signed association (+0.72, per-seed ρ ∈ [0.69, 0.77]) survived all of it
> as an upper estimate, not as a significant family-7 result.

Earlier findings still hold within their scope: backend confound on a **DLPFC case** (KMeans→GMM moves ARI
up to 0.20), a neighbour-mean floor is competitive (cross-section ARI 0.473), and the
ARI↔boundary_F1 Pareto trade-off is structural (4 mechanism transfers slid along it).

## Evidence chain (every number is machine-checked)

| claim region | artifact | producer |
|--------------|----------|----------|
| backend confound, cross-section table | workbench `experiments/rigorous_bench.json` (external) | `rigorous_bench.py` |
| Pareto frontier | workbench `experiments/pareto_frontier.json` (external) | `pareto_frontier.py` |
| Tessera per-metric ranks | workbench `experiments/tessera_verdict.json` (external) | `tessera_verdict.py` |
| **§7 — 11 platforms, 6 methods, 3 seeds** | workbench `experiments/expanded_bench.json` (external) | `expanded_bench.py` |
| **official Table II *n* cells / *K* / contiguity** | `ledgers/table_i_printed.json` (filename historical) | rematch only |
| **leftover-only printed-claim check** | `ledgers/working_printed_claims.json` | leftover `paper.tex` only |
| **full number↔manuscript check** | workbench-only; not a capsule property | `verify_manuscript.py` |

The capsule is **not** self-contained. `experiments/` is **not** in the official kit.
No Zenodo DOI. No SI.

## Reproduce

```bash
conda activate dl
# full-factorial benchmark lives in the workbench, not in this capsule
# python experiments/...   # requires labs/active/tessera-st
make -C manuscript verify-committed   # leftover + candidate token check
make -C manuscript                    # leftover 22 pp compositor; not official
# official 14 pp: pdflatex in manuscript/submission_flat with SOURCE_DATE_EPOCH=0
```

## Honest scope

The official *n*=11 Spearman is an association and a calibration, not a discovery and not a
deployable law. Family-7 *p*=0.082 stays on the page. BRCA labels are SEDR pathology domains;
SEDR is also a scored competitor. Working leftover `961d6662` is not official.
