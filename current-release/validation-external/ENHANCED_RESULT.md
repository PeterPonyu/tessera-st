# Enhanced pathologist targets — 11 October 2026

**Superseded in part.** The same-graph controls added afterwards
(`../validation-extension/RESULT_EXTENSION.md`) show that expression weighting is
not what carries the benefit: the weighted mean was not above the unweighted mean
on the same graph (4 patients above, 4 below), and the unweighted mean itself is
above the tuned control in 8 of 8 patients on the tumor boundary. The weighting
interpretation below is withdrawn; the per-patient scores stand.

The first HER2ST comparison kept every label string, including `undetermined`.
This follow-up, locked in `ENHANCED_PROTOCOL.md` before the refit, asks two
sharper questions and adds one constructive arm. The released encoder is
unchanged. The new arm weights Tessera's frozen six-neighbor graph by
`exp(-d / median(d))`, where `d` is the PCA-feature difference. That formula
comes from the development result that an expression difference already beat
the learned gate. It was not tuned on these patients.

Values are three-seed mean adjusted Rand index.

## Tumor versus other named tissue

`invasive cancer` and `cancer in situ` versus adipose tissue, connective
tissue, immune infiltrate, and breast glands. `undetermined` is excluded from
the score and kept in the graph.

| Patient | Tessera | Tuned control | Expression-weighted | Weight minus control |
|---|---:|---:|---:|---:|
| A | 0.117 | 0.307 | 0.417 | +0.111 |
| B | 0.734 | 0.672 | 0.673 | +0.002 |
| C | 0.121 | −0.007 | 0.046 | +0.053 |
| D | 0.217 | 0.377 | 0.339 | −0.038 |
| E | 0.010 | 0.034 | 0.048 | +0.014 |
| F | 0.009 | 0.087 | 0.118 | +0.031 |
| G | 0.417 | 0.349 | 0.398 | +0.049 |
| H | 0.579 | 0.501 | 0.611 | +0.110 |

The expression-weighted neighbor mean is ahead in 7 of 8 patients (mean
+0.042, descriptive sign-flip 0.055). Patients C, E, and F remain near zero,
so the count is not eight large gains. The clear gains are A, G, and H.
Released Tessera splits 4/4 with the tuned control (mean −0.014). Its high
scores are B (0.734), H (0.579), and G (0.417).

## Named compartments

`undetermined` removed, every other pathologist string kept. The tuned control
is ahead of Tessera in 5 of 8 patients (mean −0.020). The expression-weighted
mean is ahead of that control in 6 of 8 (mean +0.012). Patient E's Tessera
advantage is 0.172 against a control score of 0.005.

## What this changes

The improved rule is to give the simple spatial arm the scale the annotation
implies, then weight those neighbors by expression difference. Lung region
labels, built from 200-neighbor compositions, selected k=100 or 200. These
pathologist boundaries selected k=6 or 10. At that local scale, the learned
gate is not the part that matches the boundary. The expression weight is.

Limits: one pathologist, no rechecked registration, the
G1/G2 filename mismatch, overlapping folds, and no calibration analysis remain.
The figure is `figures/claim-boundary.pdf`.
