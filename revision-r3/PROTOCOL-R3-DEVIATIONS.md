# Deviations from the frozen protocol

PROTOCOL-R3.md stays byte-identical to its public deposit (doi:10.5281/zenodo.24303149). Every departure from it, and every implementation decision it left open, is listed here with its date and reason. None was made after looking at any score from the new cohorts.

## 17 Nov 2026, before any new-cohort score

1. **Reproduction check of section 2.5 / WP2.** The check that CPU refits at seeds 1–3 equal the saved fits of the 13 bounded objects was wrong as written: those saved fits were made on a GPU, and a CPU refit cannot match a GPU fit bit for bit. The earlier exact reproduction compared the CPU training loop with the frozen CPU loop.
   - **Replaced by** agreement in ARI. The per-seed median absolute difference between one-thread CPU refits and the saved GPU fits is 0.0013 (maximum 0.035), and every dataset mean is within 0.011.
   - **Seed noise.** Spread over seeds 1–10 on one CPU thread has a median SD of 0.011 ARI.
2. **STAGATE grid (section 5).** STAGATE ships no default radius. Its released graph in the field-aware adapter is the six-nearest-neighbour graph, so the tuned grid is {6, 15, 30} neighbours × {500, 1000} epochs, six configurations.
3. **Message-passing depth L (section 5).**
   - **STAGATE: L = 1.** Its second graph-attention layer is called with attention switched off and applies only a linear map.
   - **GraphST: L = 2.** Its clustering embedding is propagated once in the encoder and once in the decoder, with no nonlinearity between.
   - Both values come from receptive-field tests of the adapters.
4. **Native input (section 5).** Native preprocessing is available only for NSCLC and HER2ST, as in round 2. The other bounded objects carry only gene-selected matrices, and the new cohorts store no counts. The native-input arm is therefore limited to those two cohorts.
5. **Clustering (section 5).** Each method's own mclust clustering is not run; the analysis environments have no R bridge. K-means (primary) and Leiden (secondary) are scored for every arm, as specified.
6. **Spinal cord record (section 3.2).** Zenodo record 18039571, named in the protocol, is used. A later version of the record (20533289) re-exports the files with different checksums; it is not used.
7. **Colorectal selection (section 3.2).** The smallest-archive rule selects, for patient A938797, an archive whose name carries an `_X` suffix that the source does not explain. The rule is followed as written.
8. **Bounding (section 3.3).**
   - Frames are visited in sha256 order of the frame identifier, and every frame is considered: a frame is admitted if it fits the remaining budget, and later frames are still visited.
   - Spinal frame identifiers are `<mouse>|<section>`.
9. **Labels.** Released label strings "unknown", "na" and "none" (any case) are treated as missing, as by the round-1 encoder.
10. **Wu metadata inspection (section 3.2).** The schema-level inspection of the Wu class vocabulary on 17 Nov was made by the source check in a temporary location. No Wu file was kept in the project before the download receipts.
11. **Spinal cord branch (section 3.2).** The deposited file has per-cell columns `Laminae` and `Region`, but they are attributes of neuronal cell types, not spatial annotations.
    - **Evidence from the schema inspection:**
      - each named neuronal type maps to one laminar range;
      - the class counts are identical (for example 4,169 cells of one type and 4,169 cells in range L4-10);
      - the columns are missing for all non-neuronal cells, consistent with the source's statement that subtypes were assigned to laminae from prior knowledge.
    - **No depth is deposited:** no per-cell relative depth or traced surface.
    - **Consequence:** branch C applies. The targets are the deposited cell-type levels: `1st round cluster` (14 classes) and `MERFISH cell type annotation` (60 classes). No lamina target is built.
    - **Effect on the tests:** the spinal cord enters P1 and P3 as a cohort of expression-assigned targets. It does not provide a thin-structure test.
12. **Colorectal pathology headers (section 3.2).** Two of the four new patients failed the frozen pilot parser. That parser accepts the label column only as "Pathologist Annotation(s)". The cause was the header name, not ambiguous data: each table holds exactly one label column, headed "Pathologist annotation" (SN123_A798015_Rep1) or "Pathologist_KH" (SN123_A938797_Rep1_X).
    - **Fix.** The header was renamed in a temporary copy of the annotation archive. The unchanged preparation then ran, and its barcode-uniqueness check passed.
    - **Records.** The original header and both archive hashes are in each input.json.
    - **When decided.** On 17 Nov, after reading only the header lines and before any score, so that two of four patients are not lost to a header spelling.
13. **Provenance label of Open-ST (calibration only).** The calibration code labelled the Open-ST target "pathologist", but its source annotation is molecular, so it is expression-assigned. Section 3.2's vocabulary never listed it as pathologist. The label was corrected and WP1 re-run on 17 Nov. This changes only the provenance-class summaries of the calibration datasets: no number in P1, P2 or P3, and no frozen threshold, uses it.
