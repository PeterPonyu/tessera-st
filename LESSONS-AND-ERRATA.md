# LESSONS-AND-ERRATA — tessera-st manuscript polish pass

Date: 2026-07-02
Agent role: APPLY + RECORD (mechanical-fixes-only pass)
Scope: `manuscript/paper.tex`, `manuscript/refs.bib`. No new experiments, no GPU, no
heavy recompute. On-disk JSON artifacts were read to adjudicate one numeric finding.

Backups written before any edit:
- `manuscript/paper.tex.prepolish.bak`
- `manuscript/refs.bib.prepolish.bak`

Toolchain: TeX Live 2023 (LuaHBTeX 1.17.0), bibtex 0.99d, Rscript, python3.
Build sequence: `lualatex` -> `bibtex` -> `lualatex` -> `lualatex` (mirrors `Makefile` `pdf`
target; figures were NOT re-rendered because every applied fix lives in `paper.tex` captions
or `refs.bib`, not in the generated `figs/*.tex`).

Post-fix gates:
- LaTeX compile rc=0 on all three passes.
- `experiments/verify_manuscript.py`: ALL VERIFICATION CHECKS PASS.
- Undefined references: 0. Multiply-defined labels: 0.
- Pages: 13. Overfull \hbox: 0. Undefined references: 0. Multiply-defined labels: 0
  (current 2026-07-14 rebuild, synchronized 2026-08-01).

---

## Defects APPLIED (verdict CONFIRMED AND mechanical==true)

### E1 — `sun2025smoothness` author list corrupted (major)
- What caught it: reviewer cross-check against Crossref DOI 10.1101/2025.06.23.660861.
- Defect: 7 wrong given-name initials, a swapped Reinders/Eils order, and an omitted
  `SpaceHack 2.0 participants` consortium author. Entry was falsely comment-marked VERIFIED.
- Fix: replaced the `author=` field with the full corrected list (Biharie, Kirti; Cai,
  Peiying; Turner, Meghan A.; Emons, Martin; Gunz, Samuel; Zacharias, Martin; Long, Brian;
  `{SpaceHack 2.0 participants}` inserted; Reinders before Eils).
- Verify: `Biharie, Kirti` present in `refs.bib`; `Biharie` renders in `paper.bbl`.
- Prevention: never trust an in-file `% VERIFIED` comment; re-resolve every author string
  against Crossref/DOI at submission time. A "VERIFIED" tag is a claim, not evidence.

### E2 — `yuan2024benchmark` second author + missing issue (major)
- What caught it: Crossref DOI 10.1038/s41592-024-02215-8.
- Defect: `Zhao, Fusheng` should be `Zhao, Fangyuan`; issue number 4 missing.
- Fix: `Zhao, Fusheng` -> `Zhao, Fangyuan`; added `number={4}` (vol 21, no 4, pp 712–722).
- Verify: `Zhao, Fangyuan` present; `Fangyuan` renders in `paper.bbl`.
- Prevention: same as E1 — DOI-resolve author given-names, not just surnames.

### E3 — orphan bib entry `mcinnes2018umap` (minor)
- What caught it: grep of `\cite`/usage across `manuscript/`; UMAP is never discussed.
- Fix: deleted the 5-line `@article{mcinnes2018umap ...}` block.
- Verify: `mcinnes` count 0 in both `refs.bib` and `paper.bbl`; refs count 31 -> 30.
- Prevention: run an orphan-entry check (cited-keys vs defined-keys) before every build.

### E4 — three unused `\label` anchors (taste)
- What caught it: full `\label` vs `\ref/\cref/\autoref` cross-check; every `\ref` resolves,
  only these three labels were never referenced.
- Fix: removed `\label{sec:methods}` (§Methods heading), `\label{fig:spatialmap}`,
  `\label{sec:backend}`.
- Verify: 0 remaining occurrences; 0 multiply-defined-label warnings; compile clean.
- Prevention: label hygiene lint (unused-label report) in the build.

### E5 — fig:spatialmap caption overstated ARI coverage (taste)
- Defect: "each panel title gives its ARI" — only the 2 clustering panels carry an ARI.
- Fix: "each **clustering** panel title gives its ARI".

### E6 — fig:profile caption overstated SpatialLeiden wins (taste)
- Defect: "wins low-contiguity datasets" (plural) but it wins only one (CODEX); rank 2 on
  the others. The in-body §nosota text is already accurate.
- Fix: "ranks first or second on the low-contiguity datasets, last on high-contiguity MERFISH".
- Prevention: caption claims must match the table they summarise; a "wins" verb needs a
  per-dataset winner-count check.

### E7 — fig:heatmap caption mislabelled a whole column a near-tie (minor)
- Defect: "the high-contiguity column (right) is a near-tie" — only the top two cells
  (SpaceFlow 0.43, Tessera 0.41) are close; the rest are far below.
- Fix: "the **winner in the** high-contiguity column (right, MERFISH) is a near-tie
  (SpaceFlow vs. Tessera)".
- Prevention: distinguish "column is a near-tie" from "the top of the column is a near-tie".

---

## Defects DEFERRED (not applied — see reason)

### D1 — coh_gain uncorrected proxy p: 0.079 vs 0.083 (CONFIRMED, but reclassified NOT mechanical)
- Finding proposed replacing `0.079` with `0.083` in 4 places, asserting it was verifier-safe.
- REALITY (this pass): `experiments/verify_manuscript.py` line 275 ASSERTS that `paper.tex`
  contains `0.079` (the asymptotic Spearman p companion, `gtfree_proxy.json`
  `principled_proxy_coh_gain.vs_advantage[1]` = 0.0788). Applying the change made the machine
  gate FAIL ("coh_gain uncorrected proxy p-value '0.079' ... 0 ... FAIL"). The finding's
  premise that the verifier does not assert 0.079 was factually wrong.
- The scientific point is real: the manuscript pairs a permutation-null search-corrected
  p=0.28 (`proxy_search_corrected.json` p_search_corrected=0.275) with an ASYMPTOTIC
  uncorrected p=0.079, whereas the permutation null's own uncorrected companion is
  p_uncorrected_selected=0.083. Fixing it consistently requires EITHER relabelling 0.079 as
  "asymptotic Spearman p" (prose judgment) OR switching to 0.083 AND updating the verifier's
  ground-truth assertion + provenance note (touches the machine gate, which is governed by
  CLAIM_LEDGER's LOCKED->graduated process). Both are out of scope for a mechanical apply.
- RESOLVED 2026-08-01: retained `0.079` and labelled it explicitly as the asymptotic
  uncorrected Spearman p-value; added the permutation-null uncorrected companion `0.083` in
  the causal section. `verify_manuscript.py` now derives and asserts both values and both labels.
- Prevention: before trusting a finding's "verifier-safe" tag, grep the verifier for the exact
  literal; the machine gate is authoritative over a finding's self-assessment.

### D2 — Methods §Data enumerates only 10 of 11 datasets; BRCA omitted (CONFIRMED, mechanical==false)
- `paper.tex:192-197` and Data-availability `:595-599` list 10 datasets but say "11"; BRCA
  (row 4 of tab:data, Visium tumour) is missing from both prose lists.
- Not applied: the natural fix inserts "BRCA breast tumour (Visium)" but BRCA has NO
  provenance `\citep` anywhere (see D3); adding a citation-less list item would break the
  pattern every sibling dataset follows. Must be resolved together with a real BRCA source.
- Deferred to user (needs the accession, not a fabricated citation).
- RESOLVED 2026-07-13: accession located (see D3). Inserted "a breast-tumour Visium section
  \citep{tenx2020brca} with per-spot pathology-domain labels from SEDR \citep{xu2024sedr}" into
  the Methods §Data enumeration and Data-availability; both prose lists now enumerate all 11.

### D3 — BRCA 11th dataset has no provenance citation (CONFIRMED, mechanical==false)
- BRCA appears only in `tab:data` and `tab_perdataset.tex` with no `\citep`, is absent from
  Data-availability, and is undocumented in `BASELINE_REFERENCES.md`.
- Not applied: requires the actual accession (e.g. the 10x Genomics breast-cancer Visium
  sample). Do NOT fabricate a citation. Deferred to user to supply the accession, then add a
  `refs.bib` entry + `\citep` + `BASELINE_REFERENCES.md` line, and resolve D2 in the same edit.
- RESOLVED 2026-07-13: the "e.g." guess was correct. Source is 10x Genomics' public
  "Human Breast Cancer (Block A Section 1)" Visium demo (sample
  `V1_Breast_Cancer_Block_A_Section_1`, Space Ranger 1.0.0, CC BY 4.0), with the 20-region
  per-spot pathology-domain annotation from Xu et al. 2024 (SEDR). The table numbers were NOT
  placeholders: `experiments/expanded_bench.json` records `n_cells 3798, k 20,
  GT_contiguity 0.865, subsampled false`, matching the on-disk h5ad
  (`spatial-omics-reform/data/processed/brca1_visium_10x/anndata.h5ad`, 3,798 × 36,601,
  `Region`=20 domains). Added `@misc{tenx2020brca}` to `refs.bib`, `\citep` in Methods §Data +
  Data-availability, and the accession row in `BASELINE_REFERENCES.md`. No fabrication.

### D4 — fig:ablation caption lacks a non-comparability note (CONFIRMED, mechanical==false)
- `mechanism_ablation.json` "full" Tessera ARI on MERFISH = 0.4525 exceeds the main-panel
  MERFISH winner (SpaceFlow 0.429); legitimate because the ablation is a separate
  2-dataset/3-seed case-study run, but the caption does not flag that its absolute ARIs are
  not comparable to the cross-dataset panel, so a reader could infer Tessera beats the stated
  winner — an inflation risk that the honest-negative posture forbids.
- RESOLVED 2026-08-01: caption now states that the absolute ARIs come from a separate
  two-dataset case-study run and are not directly comparable with the cross-dataset panels.

### D5 — CLAIM_LEDGER R4 page/overfull line is stale (CONFIRMED, mechanical==false)
- Initial finding in the polish pass: `CLAIM_LEDGER.md:315` said "paper.pdf 12pp / 0 overfull /
  0 undefined", while that pass produced 19pp / 2 overfull / 0 undefined.
- RESOLVED 2026-08-01 from a fresh `make clean pdf`: current build is 13pp / 0 overfull /
  0 undefined / 0 multiply-defined. `CLAIM_LEDGER.md` and `SUBMISSION-KIT.md` now match the log.

---

## Prevention rules distilled this pass
1. `% VERIFIED` bib comments are claims, not proof — re-resolve author given-names via
   Crossref/DOI at submission; surnames alone are insufficient (E1, E2).
2. Run cited-vs-defined key diff (orphans) and unused-`\label` diff every build (E3, E4).
3. Every figure/table caption claim must match the artifact it summarises; verbs like "wins"
   and "near-tie" need a per-cell/per-dataset count check (E5, E6, E7).
4. A finding's "verifier-safe/mechanical" self-tag is not authoritative — grep
   `verify_manuscript.py` for the exact literal before editing a headline number (D1).
5. Never fabricate a citation to satisfy an enumeration count; surface the missing accession
   and defer (D2, D3).
6. Ground-truth-gate and CLAIM_LEDGER edits go through the ledger owner, not a mechanical pass
   (D1, D5).
