# Tessera panel style and remaining validation plan

Updated 11 October 2026. The current panel contract covers 10 main figures,
eight supplementary figures and 61 semantic panel/row markers. The saved
10-11 October follow-up analyses are reported in the current manuscript and
`FOLLOWUP_PROVENANCE.md`. The HER2ST pathologist-label comparison in
`validation-external/VALIDATION_RESULT.md` has now been run. The final document
build has 52 passing tests and 18/10/18 pages (manuscript/supporting
information/atlas). Original receipts, results, and every figure remain
unchanged. The proposed multi-panel validation figures below were not added.

## Historical snapshot: 2026-10-10 visual polish (superseded layout)

Figure 4b was re-laid out (no overlapping "Reference" label); titles and markers use baseline
alignment so descenders no longer shift them; Figure 6a shows unsigned zeros; category legends use
readable names and add a row instead of widening the page; colorectal bars read 100 µm on a white
backing; Figure 12 uses "Self-only" consistently; Figure 14c uses integer degrees; Figure S2 is drawn
near its placed width so its type is no longer enlarged. Values, captions' claims and page counts
(26/6/18) are unchanged; 44 tests pass; minimum statistical marker 12.93 pt. The editable-source archive
was not regenerated. Backup: `backups/before-visual-polish-20261010/`.

## Panel labels, titles and typography

- Use bare lowercase panel labels `a`, `b`, etc., in reading order. Dense repeated
  map/experiment grids use one marker per semantic row rather than one marker per
  plotted cell; the caption states the column mapping explicitly.
  Only the label is bold. Titles, module names, numbers, legends, axis labels,
  scale bars, explanatory text and caption prefixes use regular weight.
- Keep label and title as separate text elements. Statistical cell headers
  reserve an 18-point title offset. Statistical markers are 13.5 points, larger
  than regular text (at most 9 points); native SVG markers are 38 units, larger
  than every regular SVG text element. Schematic titles start 34 units to the
  right of the header origin; markers start 12 units to its left. Cell headers
  share a lower baseline. Repeated grids use a common left-margin lane, six
  points clear of the widest first-axis decoration, with regular rotated row
  identifiers below the bare letters. Row identifiers do not compete with
  per-cell titles or y-axis labels.
- Measure marker size after LaTeX placement, not only in the standalone figure.
  All 45 text-based statistical markers must remain at least 12 points in the
  final manuscript/supplement; the verified minimum is 13.45 points. The 16
  schematic letters are outlined vector text and are approximately 13.92 points
  at manuscript width. Inspect every figure individually at its final size.
- Use the manuscript's Times-compatible serif family throughout: newtxtext
  for PGF statistical PDFs and TeX Gyre Termes for SVG exports. Math uses the
  existing newtxmath contract. Do not introduce decorative sans-serif labels,
  italic titles or bold module names.
- Shorten titles before shrinking type. Move numeric ARI annotations into a
  small regular-weight map annotation; put full specimen identities and
  interpretation in the caption. Do not repeat long figure-level banners.
- Caption panel identifiers are explanatory mappings, not cross-figure calls.
  Fine-grained cross-references belong only to Results prose. Introduction,
  Methods and Discussion may cite a whole figure but not its lettered panels.

## Density and color

- Statistical exports use explicit point geometry on a 496.8-point-wide canvas
  and are placed at 100% in the manuscript and supplement. Map columns and
  rows reserve measured title, marker, annotation and legend space before
  aspect adjustment. Measured coordinates are never stretched to fill a slot.
- Reduce figure-specific row and column spacing without obscuring ticks,
  colorbars, metadata, titles or legends. Preserve enough room for two-line
  headers and rotated heatmap ticks. Data-associated white space inside an
  actual tissue image is not removed by a score-selected crop.
- Use pure black `#000000` for all text, including markers and heatmap numbers.
  Lighten heatmap fills while retaining the same normalization and numerical
  values so black numbers remain readable. Preserve blue `#176b93`, orange `#bd5b30`,
  green `#527762` and purple `#82618e` for established comparisons/modules.
  In method legends, PCA50 is blue, Tessera orange, self-only green and
  no-contrast purple; other controls retain declared, consistent colors.
  Color meanings are local and stated in legends: method colors are not
  interchangeable with patient-cohort colors. Supplied cell categories retain
  consistent mappings across matched maps; unknown categories remain gray.
- Use sequential colormaps for nonnegative quantities and a zero-centered
  diverging colormap for signed contrasts. Retain symbols, lines, labels and
  numeric values so color is not the only information channel. Do not add
  colorful panel-label badges that compete with the measured data.
- Native SVG arrows use explicit filled polygon heads, not export-dependent
  SVG markers. Keep head length 12 units, width 9 units and shaft stroke 2.8
  units across colors: approximately 4.40-by-3.30-point heads at manuscript
  width. Shorten the last orthogonal shaft segment so it terminates exactly
  at the head base without moving the original tip. Retain a visible terminal
  shaft leg, a butt cap and a head matched to the shaft color, including neutral
  gray. Route around text, endpoint circles and lock icons; expose connectors
  on both sides of a lock rather than hiding them beneath the glyph.
- Source `figure08-breast-mixtures` is manuscript Figure 8. Its C/D row uses
  explicit axes positions with a compact gap and a separately reserved
  colorbar lane. Colorbar placement cannot automatically shrink or recenter
  the map.
- Categorical map legends use one or two rows (more horizontal columns) and
  sit six points below the lowest actual map annotation. Colorectal map
  columns include measured colorbar-label width in their packing and align
  assignment and protein maps on the same left edges.
- Arrange BCC as four patient rows with Reference/PCA50/Neighbor mean/Tessera
  columns and one bare marker per row. Lung and MIBI use the same column
  mapping. Histology, colorectal and efficiency grids use one marker per
  semantic row where the cells are repeated measurements of one unit;
  transfer uses four separate task/metric panels. State the patient identities
  and column mapping in each caption.
- Use existing prose to fill space beneath late manuscript floats. Allow the
  references to follow the final figure and declarations on the same page;
  the final float barrier was removed after checking the resulting order. Short supplementary
  figures may share a page with a fixed 12-point gap. Do not shrink figures,
  stretch coordinates or add unsupported panels to force page density. The
  review atlas deliberately retains one figure per page.

## Specific revisions to the existing evidence

| Issue | Implemented revision | Verification |
|---|---|---|
| Label/title styling mixed together | Separate label and title objects; only the bare `letter` is bold | Artist audit, native SVG audit and exported-PDF font audit |
| Large outer margins and loose map columns | Tight export plus aspect-preserving map packing | Figure-by-figure render review; PDF dimensions compared with backup |
| Long or redundant headings | Short task titles; full identity in caption; ARI inside map | Header/legend overlap checks and visual inspection |
| Claims disconnected from supporting panels | Results references specify the relevant row labels; ranges checked against actual semantic label counts | Cross-reference regression test and compiled-reference check |
| Arrowheads disappear or shafts are hidden at page size | Explicit filled heads, continuous orthogonal shaft joins and unobscured terminal legs | SVG geometry tests and individual final-page figure review |
| Later panel letters become too small after placement | Larger source markers and a final-document 12-point minimum for statistical letters | Final PDF measurement and visual review of every placement |
| Displayed calculations lack corresponding Methods | All 18 figures and 61 semantic sections map to implementation-matched definitions and 31 equation identities | Hash-bound Methods audit and targeted denominator/tie regression tests |
| Repetitive, revision-centered prose | Introduction and Discussion lead with the task or finding; retain mechanism, sources and limitations | Comparable-length check and final-page review |
| Presentation mistaken for new validation | Separate the completed style changes from the experiments below | Unchanged results/code hash tests and explicit statement of what was not run |

## Implemented Methods coverage

`qa/figure_methods_audit.json` records definitions and equation identities for
every semantic section, with final figure, manuscript and computation-source
hashes. The Methods distinguish valid-reference-edge homogeneity from
all-predicted-edge homogeneity, handle tied/absent-class gate AUROC, and document
seed/stability summaries, density histograms, fixed registered ROIs,
display-only cluster matching, composition/entropy/purity, evaluation
restrictions, ranked cluster sizes, tied-rank association, protein-versus-RNA
normalization, multiset graph degrees and recorded cost summaries. Conceptual
glyphs and descriptive source cardinalities are identified as such rather than
assigned artificial equations. Fine-grained panel references remain confined
to Results; this mapping lives in the QA record, not lettered Methods citations.

This is an implementation comparison plus automated coverage/hash checking,
not independent validation of the scientific claim. All original estimator,
protocol, input, prediction, transform, weight and numerical-ledger identities
remain unchanged. The current Methods contain 27 main and four supplementary
equation identities; statistical figures retain their designed print size.

## Scientific follow-up

Executed (see `validation-external/`, `validation-extension/` and the paper): an
independent eight-patient pathologist-labelled cohort with frozen hyperparameters
and leave-one-patient-out control selection; same-graph, weighted, shuffled-weight
and shuffled-coordinate neighbor means; edge-level seam AUROC with patient
bootstrap; three replicate fits; three further tissues. Main figures 11 and 12 and
Supplementary Fig. S9 show these results. Not run: calibration and abstention, a
second annotator, registration review.

### Independent tumor-boundary reference

Use registered tissue sections with independently prepared, blinded pathology
annotations. Lock patient identity, registration checks, the definition of a
boundary and the matching tolerance before accessing model outcomes. Record
whether annotators used any RNA/protein categories that also entered the model.
Use independently annotated specimens as the evaluation unit; adjudicate
annotation disagreements without model overlays.

A future four-panel figure would show `(a)` actual H&E and annotation provenance,
`(b)` independent annotations and disagreement, `(c)` frozen-method boundaries
on the same geometry, and `(d)` patient-level precision/recall and boundary
distance with uncertainty. Do not create these panels until those measurements
exist. Success requires a valid independent target and prespecified patient-level
comparison; molecular label agreement alone cannot satisfy this gate.

### External cohort and adaptive-policy test

The HER2ST run is the smallest version of this test: eight published patients,
frozen hyperparameters, and scale selection that excludes the held-out patient.
It has no author-locked effect margin, no calibration split, and no second
annotator. The six-panel figure was not made.

Obtain an independently sourced cohort after checking access and source rights.
Keep development, calibration and final test patients disjoint. Select any
smoothing/component policy only on development patients, then freeze the
feature schema, transformations, weights, centroids, policy and metric set.
Report unsupported categories and acquisition differences rather than silently
discarding them. Do not reuse a current favorable ablation as an untouched test.

A future six-panel figure would show `(a)` patient/site lineage, `(b)` frozen
processing, `(c)` regional scores, `(d)` cell-type scores, `(e)` unsupported/OOD
categories and `(f)` paired patient contrasts against both PCA and neighbor mean.
Assess the claimed target separately; do not require or imply universal gains.
Sample-size planning and a practically meaningful effect threshold must be
locked with the author before collecting/scoring the final cohort.

### Predictor-specific calibration and abstention

Define the evaluated source-centroid predictor and its error event first.
Prespecify a confidence statistic based on source-centroid distances rather
than treating the untrained soft head or gate seam score as probabilities.
Fit any temperature/isotonic/conformal step on separate calibration patients;
do not use target labels to select the mapping or abstention threshold.

A future six-panel figure would show `(a)` train/calibrate/test patient split,
`(b)` confidence construction, `(c)` held-out reliability, `(d)` coverage-risk
against uncalibrated and random-abstention controls, `(e)` patient-level error
discrimination with bootstrap intervals and `(f)` coverage/unsupported-class
accounting. The workspace's discrimination gate is a patient-resampled AUROC
95% interval lower bound above 0.5, subject to an adequate cohort and locked
resampling design. This is not by itself a calibration or clinical-utility gate;
those need prespecified predictor-level metrics and coverage assumptions.

No proposed panel will be populated with synthetic biological evidence. The
HER2ST comparison is a scored external test, not one of these unpublished
panels. Calibration, a second blinded annotation, and registration review
remain undone and are stated as limits in the paper.
