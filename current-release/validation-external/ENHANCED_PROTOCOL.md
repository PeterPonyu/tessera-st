# Enhanced pathologist-target protocol

Locked 11 October 2026 after the first HER2ST comparison and before these
refits. The first comparison remains the recorded primary test. This file does
not replace it.

## Why the question changes

The released encoder's gates are trained to down-weight edges that cross a
tissue boundary. The first HER2ST adjusted Rand index treated every label
string, including `undetermined`, as a compartment. `undetermined` is an
unassigned spot, not a region the pathologist drew. The lung follow-up had
already shown that a neighbor mean wins when it is given the scale at which
the label was constructed (k=100 or 200 for those regions). The HER2ST
selector, on the first target, returned local configurations (k=6 or 10).
The enhanced question asks whether that conclusion holds for the compartments
and the tumor boundary actually drawn.

## Targets

Both targets use the same joined spots, pixel geometry, preprocessing, frozen
Tessera hyperparameters, seeds, and leave-one-patient-out control grid as
`PROTOCOL.md`. Unassigned spots stay in the graph and the principal components.
They are excluded from adjusted Rand index and from seam labels.

- Named compartments. `undetermined` is invalid. Every other label string
  remains its own class. K is the number of remaining classes.
- Tumor boundary. `invasive cancer` and `cancer in situ` are tumor.
  `adipose tissue`, `connective tissue`, `immune infiltrate`, and
  `breast glands` are other tissue. `undetermined` is invalid. Any other
  string stops the run. K is 2. A patient with fewer than 20 spots on either
  side is listed and left out of that sign-flip.

## Arms

- Released Tessera, refit at the target's K because the unused cluster-head
  initialization consumes random state.
- The same nested spatial-control grid. PCA50 is reported and is not eligible
  for selection.
- One constructive arm, not eligible for selection: an expression-gated mean on
  Tessera's frozen k=6 graph. The edge weight is `exp(-d / median(d))`, where
  `d` is the Euclidean distance in the PCA input. Similar neighbors keep more
  weight. This uses the development result that an expression difference
  outranked the learned gate. It is not tuned on HER2ST.

## Reading

The primary contrast is still Tessera minus the nested control, on each target
separately. The expression-gated mean is a second contrast. Either sign is
recorded. A positive scientific result is a stable rule about which use of
position matches the annotation, including a result in which the simple
control remains ahead.
