# experiments/

Generated ablation outputs. **None of these is a scientific claim.**

## `synth_ablation.json`

The curated component grid on the synthetic *tessellation* fixture. This is a **software
correctness + harness-discrimination check**, not biology:

- It confirms the deep model trains, the four component toggles change behaviour, and the
  metrics (ARI / NMI / ASW / boundary-F1 / ECE) compute on real outputs.
- The synthetic fixture noise is deliberately set so a non-spatial clusterer does **not**
  trivially solve it — i.e. the over-smoothing regime the components are designed for. With
  near-zero noise everything (including the bare backbone) hits ARI≈1.0 and the table shows
  no headroom; that easy regime is uninformative by construction.
- Whether each component *helps* is an empirical question reserved for the real DLPFC ablation
  (`tessera ablate-real`); no such claim is made without that evidence, seed variance and
  same-data baseline parity.

Reproduce:

```bash
tessera smoke-synth --epochs 200 --out experiments/synth_ablation.json
```

## `gate2_baseline_parity/` (planned)

Same-data runs of the named graph encoders (from their own repos, behind an adapter) on the
identical DLPFC inputs, for the parity table. Empty until the real-data gate opens.
