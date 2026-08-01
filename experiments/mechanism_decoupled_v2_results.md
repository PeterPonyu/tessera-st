# Decoupling experiment, full-fidelity re-run (v2) — results

**ADOPTED as R6 robustness evidence.** The result is integrated in `manuscript/paper.tex`,
recorded in `CLAIM_LEDGER.md` R6, and checked directly from the local JSON artifact by
`experiments/verify_manuscript.py`. The causal claim remains `LOCKED` under the ledger's global
honest-claims policy. v1's `mechanism_decoupled.{py,json}` remains as a superseded audit artifact;
v2 is `mechanism_decoupled_v2.{py,json}`.

## What was run

Re-run of the 2-arm decoupling (`mechanism_decoupled.py`, arm definitions copied **verbatim**) at
the compute fidelity of the paper's primary headline result `mechanism_synth.py`:

| knob | v1 (staged) | v2 (this run) | primary `mechanism_synth.py` |
|---|---|---|---|
| STAGATE epochs | 120 | **400** | 400 |
| device / threads | CPU / 2 | **GPU (cuda:0) / 24** | GPU / default |
| sweep levels | 6 (0.2 step) | **11 (0.1 step)** | 8 |
| seeds | 3 | **5** | 3 |
| substrates | 1 | **2** | 1 |

- **Arm A (alignment-only):** coords + labels never touched (contiguity fixed); a fraction of
  cells has its expression vector swapped → erodes expression↔position alignment.
- **Arm B (contiguity-only):** coords never touched; expression always regenerated from the
  current label (alignment perfect by construction); a fraction of positions has its label
  overwritten with a random domain id → breaks label spatial contiguity.
- **S1** = exact primary params (n_side=34, n_genes=40, n_domains=5, signal=0.3, noise=1.0).
- **S2** = independent substrate (n_side=40, n_genes=60, n_domains=6, signal=0.35, noise=1.0).

Runtime 2371 s (~40 min) on the RTX 5090. Deterministic (fixed seeds). Note: interrupted once
mid-run by an unrelated process kill; restarted from scratch with per-arm checkpointing added —
the restart reproduced the pre-kill per-level numbers exactly.

## Headline numbers (ρ and normalized_slope, both arms, both substrates, both priors)

`normalized_slope` = OLS slope × observed x-range = total change in advantage across the arm
(the effect-size metric). ρ = Spearman of advantage vs the arm's own driver (alignment for A,
contiguity for B).

| substrate | arm (driver) | ρ smooth | ρ STAGATE | nslope smooth | nslope STAGATE | adv range (smooth) | sign-flip? | floor range |
|---|---|---|---|---|---|---|---|---|
| S1 | A (alignment) | 0.991 | 1.000 | 0.549 | 0.623 | [+0.017, +0.485] | **no** | 0.364 |
| S1 | B (contiguity) | 0.973 | 0.973 | **0.851** | **0.906** | [**−0.424**, +0.459] | **yes** | **0.050** |
| S2 | A (alignment) | 0.745 | 0.873 | 0.339 | 0.429 | [+0.014, +0.402] | **no** | 0.682 |
| S2 | B (contiguity) | 0.991 | 0.991 | **0.918** | **0.953** | [**−0.715**, +0.196] | **yes** | **0.027** |

`normalized_slope(B) − normalized_slope(A)`: S1 = +0.302 (smooth) / +0.283 (STAGATE);
S2 = +0.579 / +0.524. Positive everywhere → contiguity has the larger effect on advantage on
every substrate and both priors.

## Old (v1) vs new (v2) — did restoring fidelity change anything?

No. v2's S1 (same substrate as v1) reproduces v1 almost exactly:

| metric | v1 (120 ep) | v2 S1 (400 ep) |
|---|---|---|
| Arm A nslope smooth / STAGATE | 0.512 / 0.605 | 0.549 / 0.623 |
| Arm B nslope smooth / STAGATE | 0.818 / 0.890 | 0.851 / 0.906 |
| contrast nslope (B−A) smooth / STAGATE | +0.305 / +0.286 | +0.302 / +0.283 |

The 120→400-epoch cut did **not** bias v1's S1 result. What the extra fidelity/coverage adds is
(1) a denser sweep that resolves Arm B's sign-flip crossing, (2) a second substrate that makes the
separation starker, and (3) quantified sign-flip and floor-flatness discriminators.

## The three discriminators that matter (ρ is not one of them)

Both arms are monotone by construction, so ρ saturates near 1 and cannot separate them — this is
why v1's auto-verdict ("tracks alignment more tightly", off ρ 1.0 vs 0.943) is an **artifact of a
bad metric**, not a real finding. The load-bearing comparisons:

1. **Effect size (normalized_slope):** contiguity > alignment on every substrate/prior
   (B−A = +0.28 … +0.58).
2. **Sign-flip (the paper's actual headline phenomenon):** the advantage goes from positive to
   **negative** only in the contiguity arm — S1 crosses 0 near contiguity ≈ 0.58, S2 near ≈ 0.76.
   The alignment arm's advantage only decays toward 0, never goes negative. The paper's claim that
   a spatial prior *hurts* at low contiguity (negative advantage) is reproduced **only** by
   breaking contiguity.
3. **Clean vs confounded manipulation:** the contiguity arm holds the non-spatial floor essentially
   flat (range 0.03–0.05), isolating spatial arrangement. The alignment arm's expression-swap
   destroys the recoverable signal for everyone (floor collapses by 0.36 on S1, 0.68 on S2), so its
   shrinking advantage is substantially *generic signal loss* — precisely the confound a referee
   worries about, and it lands on the **alignment** arm, not the contiguity arm.

Cross-substrate rollup: `norm_slope_both_favor_B` = true on both; `only_B_flips_sign` = true on both
(both priors); `B_floor_flatter_than_A` = true on both.

## Honest read

**This strengthens the causal claim that contiguity — not generic expression↔position alignment —
sets the spatial prior's value, and it overturns v1's misleading auto-verdict.** On effect size,
on the sign-flip that is the paper's actual headline, and on manipulation-cleanliness, contiguity
wins on both substrates and both priors.

Caveats (so the claim is stated at the right altitude, not over-inflated):
- **Still synthetic.** This strengthens the *controlled-manipulation* leg only; the observational
  11-platform leg is separate.
- **The arms are not perfectly symmetric — and that asymmetry is itself the point.** You cannot
  erode expression↔position alignment without also moving expression, which unavoidably reduces
  recoverable signal (Arm A's floor must collapse). You *can* break contiguity while holding both
  alignment and signal fixed (Arm B). So the honest statement is not "alignment doesn't matter"
  (Arm A's advantage does track alignment) but "contiguity is the lever that can be moved **in
  isolation**, and only moving it reproduces the full positive→negative sign-flip."
- **Don't read ρ as a tie-break.** It is uninformative here by construction.
