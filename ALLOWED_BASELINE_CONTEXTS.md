# Tessera-ST — allowed baseline contexts (clean-room boundary)

Tessera is implemented clean-room. Baseline methods inform *positioning, metrics, and data
conventions* — never code.

## Allowed

- Cite baseline papers and name baseline methods in: `README.md`, `DESIGN.md`,
  `BASELINE_REFERENCES.md`, `CLAIM_LEDGER.md`, and `experiments/**` result notes.
- Reproduce baseline *behaviour* by running their upstream repos behind an adapter to produce
  comparison numbers on the same DLPFC inputs.
- Reuse field-standard conventions that are not anyone's IP: kNN spatial graph, HVG→PCA
  preprocessing, ARI/NMI/ASW metrics, the DLPFC manual-layer ground truth.

## Not allowed

- Copying upstream source, class names, parameter names, comments, figures, or narrative into
  `src/`, `tests/`, or `scripts/`.
- Naming any baseline brand in executable source or in package/module/API names.
- Importing a baseline paper's reported result as a Tessera claim.

## Enforcement

- `tests/test_brand_independence.py` fails if a banned brand string appears anywhere in `src/`.
- `scripts/check_independence.sh` is the CI gate (scoped to `src/`, `tests/`, `scripts/`).
- Mirrors the parent program's `leakage_scan.py` discipline so Tessera can graduate cleanly.
