# Differentiation audit — Spotscape / SpatialESD / DWGCN

**Date:** 2026-08-10  
**Against:** `docs/LITERATURE_SOTA_SCAN_2026-07-03.md`  
**Scope:** text audit/harden only — no new ARI tables, no Boundary-CA runs.

## Checklist (plan Step 2 accept criteria 1–4)

| # | Requirement | Pass? | Evidence |
|---|---|---|---|
| 1 | Same symptom (local-graph boundary degradation) | **PASS** | `manuscript/paper.tex` Related Work ¶Boundary-aware…: “targets exactly the boundary-degradation symptom… local graph smoothing blurs the spots near domain seams”; Spotscape “diagnoses the symptom most sharply… Boundary-CA… leaving boundary spots no better than a plain graph autoencoder.” Lit scan §Spotscape confirms identical diagnosis. |
| 2 | Opposite fix (Tessera local edge gate vs Spotscape global Similarity Telescope) | **PASS** (+ light harden 2026-08-10) | Pre-audit: “remedy is architecturally opposite to a local edge gate---a global, non-local similarity-consistency objective.” Harden: name Spotscape’s **Similarity Telescope** explicitly so reviewers cannot miss the architectural polarity. Lit scan: “abandon local edge gating… in favor of… Similarity Telescope.” |
| 3 | SpatialESD ensemble as alternative “so what”; causal law complementary | **PASS** | Related Work: SpatialESD “responds to the same ‘no single method dominates…’ fact… with an engineering fix---consensus-ensembling… rather than an account of *when* a spatial prior is warranted”; closing sentence: “causal-contiguity law… complementary… rather than a rival.” Lit scan §SpatialESD: rival narrative, not causal claim. |
| 4 | DWGCN = correlational domain-complexity precedent only | **PASS** | Related Work: DWGCN “reports, as a purely correlational observation, that its gains grow with… domain-count complexity: a partial, non-formalised precedent… never framed as a causal claim.” Lit scan: Cliff’s δ 0.27→0.82; no causal framing. |

## Optional echoes (plan item 5)

| Location | Status |
|---|---|
| Discussion | **Added** one sentence (2026-08-10): Spotscape architecture vs SpatialESD ensemble are complementary foils, not ARI rivals this paper claims to beat. |
| Limitations | Not required — Related Work + Discussion cover reviewer risk; no SOTA language added. |

## Claim safety

- No new SOTA / “universal winner” language introduced.
- CLAIM_LEDGER R5/R6 honest-negative framing unchanged (Tessera = worked example; ARI rank 5/7).
- Cover letter echoes checklist items 1–3 (see `COVER-LETTER.md`).

## Verifier

Run after text edits: `experiments/verify_manuscript.py` (Step 4 gate).
