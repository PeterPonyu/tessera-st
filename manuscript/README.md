# Manuscript index — DLPFC spatial-domain benchmark

The durable scientific result of the Tessera-ST program. Produced and quality-gated via ultragoal
(plan/ledger under `../.omc/ultragoal/`).

## Read this

**[`DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md`](DLPFC_BENCHMARK_SCIENTIFIC_RESULT.md)** — confound-controlled
benchmark of spatial-domain methods that scaled to **11 platforms, 6 real methods, 3 seeds**. Headline in
**§7** (supersedes the DLPFC/6-platform sections §1–§6).

## Headline (§7, n=11, 6 methods incl. SpaceFlow + SpatialLeiden, 3 seeds)

1. **The law is about spatial priors, not any one method.** GT spatial contiguity predicts the value of
   spatial priors at Spearman **+0.72 (p = 0.012, seed-robust per-seed ρ ∈ [0.69, 0.77])** — the only
   significant correlation. The Tessera-specific rank correlation is **−0.36 (p = 0.28, ns)** and SpaceFlow's
   own rank trends the same way but is marginal (−0.59, n = 9, p = 0.10). Rule: *measure GT contiguity; use a
   spatial method iff it is high.*
2. **No universal SOTA, harder** — **7 distinct winners across 11 platforms** (SpatialLeiden, a graph-community
   method with an *inverted* profile, wins the least-contiguous platform CODEX and ranks #2 on the scattered
   ones yet #9 on contiguous MERFISH — method–task fit is real and method-specific).
3. **Seed-level tie at the top, not a Tessera win** — on the contiguous spatial-domain tasks a spatial
   method wins, but which one is within seed noise: across three independent runs the MERFISH winner flipped
   SpaceFlow→Tessera→SpaceFlow. Tessera is a backend-robust **generalist** (ΔARI 0.036), not uniquely best.

> An internal code review (G008 gate) caught a duplicate dataset (n 12→11), a Spearman tie-handling artifact,
> and single-seed "wins"; all corrected. The hardening round (G009–G010) then added a 6th method, a 3rd seed,
> and ρ seed-robustness — the significant lead result (+0.72, per-seed ρ ∈ [0.69, 0.77]) survived all of it.

Earlier findings still hold within their scope: backend confound (KMeans→GMM moves ARI up to 0.20 = the
whole between-method spread), a neighbour-mean floor is competitive (cross-section ARI 0.473), and the
ARI↔boundary_F1 Pareto trade-off is structural (4 mechanism transfers slid along it).

## Evidence chain (every number is machine-checked)

| claim region | artifact | producer |
|--------------|----------|----------|
| backend confound, cross-section table | `../experiments/rigorous_bench.json` | `rigorous_bench.py` |
| Pareto frontier | `../experiments/pareto_frontier.json` | `pareto_frontier.py` |
| Tessera per-metric ranks | `../experiments/tessera_verdict.json` | `tessera_verdict.py` |
| single-section 7-method panel (bF1 0.453) | `../experiments/full_panel_v2.json` | `run_full_panel_v2.py` |
| STAGATE KMeans 0.246 vs GMM 0.577 | `../experiments/sota_stagate_151673.json` | `run_sota.py` |
| 4 mechanism-transfer negatives | `CLAIM_LEDGER` + `../experiments/diag_{aug,vae,attn_full,decouple}.py` | — |
| Tessera component ablation (only boundary-contrastive load-bearing) | `../experiments/mechanism_ablation.json` | `mechanism_ablation.py` |
| **§7 — 11 platforms (dedup+guard), 6 methods (+SpaceFlow +SpatialLeiden), 3 seeds, scipy ρ+p+seed-robustness, 7 winners** | `../experiments/expanded_bench.json` | `expanded_bench.py` |
| **§8a — label-free proxy: naive proxies invert, coh_gain recovers (+0.55)** | `../experiments/gtfree_proxy.json` (+ `gtfree_proxy_search.json`) | `gtfree_proxy.py` |
| **§8b — bootstrap CI, LOPO out-of-sample, confound partial-correlations** | `../experiments/stat_rigor.json` | `stat_rigor.py` |
| **§8c — backend fairness (real R mclust via Rscript); law strengthens to +0.79** | `../experiments/native_baselines.json` | `native_baselines.py` |
| **figures (vector .tex: law, proxy, profile, confounds, robustness, backend, mechanism, heatmap, spatial map) + per-dataset table** | `figs/fig_*.tex`, `figs/tab_perdataset.tex` | `make_figs.R` (reads `experiments/*.json` directly) |
| **number↔manuscript check** (incl. R2) | exit 0 | `verify_manuscript.py` |

## Reproduce

```bash
conda activate dl
# full-factorial benchmark (SEDR/GraphST run via the numba stub — env unchanged)
python experiments/rigorous_bench.py        # -> rigorous_bench.json
python experiments/pareto_frontier.py       # -> pareto_frontier.json
python experiments/tessera_verdict.py       # -> tessera_verdict.json
python experiments/expanded_bench.py        # -> expanded_bench.json (§7: 11 platforms, 6 methods)
# R2 (§8): make the rule deployable + stress-test it
python experiments/gtfree_proxy.py          # -> gtfree_proxy.json  (label-free proxy; naive ones invert)
python experiments/stat_rigor.py            # -> stat_rigor.json    (bootstrap CI, LOPO, confounds)
python experiments/native_baselines.py      # -> native_baselines.json (real R mclust backend fairness)
python experiments/verify_manuscript.py     # asserts manuscript numbers == artifacts incl. R2 (exit 0)

# vector figures + PDF (R + pgfplots/TikZ; compile with pdflatex)
make -C manuscript                          # make_figs.R reads experiments/*.json -> .tex -> paper.pdf
```

## Honest scope

§7 rests on **11 platforms** (Visium ×2, seqFISH, MERFISH, STARmap, osmFISH, MIBI-TOF, IMC, openST,
Slide-seqV2, CODEX) and **6 real methods** (STAGATE/SEDR/GraphST/SpaceFlow/SpatialLeiden/BANKSY-style),
3 seeds, best-config. Honest gaps (review-caught): a **duplicate dataset** was removed (squidpy mibitof ==
colorectal MIBI-TOF; n 12→11, guard added); **local independent platforms are exhausted** (remaining
candidates are same-biology slices/sections — not used to pad n); only the denominator-free spatial-prior
advantage (+0.72, p=0.012, seed-robust ρ ∈ [0.69, 0.77]) is significant — the two rank correlations are ns;
the contiguous-task winner flips across runs (reported as a tie); SpaceFlow/GraphST fail on the 32–36-dim
protein panels (logged); large platforms subsampled to 16k; all metrics semi-circular/conditioned. The
conclusion already *changed twice* (panel growth, then review) — reported as the current best estimate, not final.
