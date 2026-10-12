# Scientific Reports document candidate

Neighborhood averaging accounts for spatial gains of a graph encoder in tumor spatial omics

This is an unpublished local preparation. It has no new manuscript or software DOI and has not been submitted or deposited. Historical software identifiers refer to the earlier immutable software releases and do not archive this revision.

## Reconstruct the documents

From the extracted source directory run `python3 source/scripts/build_scientific_reports.py`. The command requires Python 3 (standard library only), TeX Live with latexmk, pdfLaTeX, BibTeX and TeXcount, and Poppler's pdftotext and pdfinfo. Font requirements: newtx text/math supplied by TeX Live; figures are supplied as vector PDFs. The supplied sources, vector figures, tables and bibliography suffice. This document command fits no models and downloads no data.

The current journal manuscript has a 186-word abstract, 3499 prose words across Introduction, Results and Discussion (Methods excluded), 7 main figures and 1 main tables. The separate supplement contains 14 figures and 18 tables. TeXcount excludes headings and mathematical tokens; the count method and source hashes are recorded in qa/scientific-reports/format-check.json. Author contributions follow the references. All supplementary display items are cited from the main text. Actual draft page counts are not journal typeset-page counts.

## Evidence and remaining concerns

The eight-patient breast comparison and the separately specified three-patient colorectal pilot are distinct evidence sets. The colorectal gate has weak seam information without demonstrated added value over raw features. The QC-target amendment followed the first PCA/mean scores. Blinded reader agreement, independent registration assessment, a larger effect-margin design and fresh scientific review remain open.

The review is operator editorial, numerical and document verification. It is not author approval or independent scientific peer review. The active leader runtime reports gpt-6.1-sol with max effort. No child-agent conclusion is credited. Reproduction in an extracted document source archive does not certify a fresh model fit, a clean-machine dependency installation or a published software archive.

The manuscript/figure font contract is preserved; the sans-serif figure-lettering recommendation remains a disclosed style exception.

## Author completion before submission

Confirm affiliation and correspondence details, funding and interests, data and image reuse permissions, submission exclusivity and related manuscripts, prior editor discussions, and reviewer suggestions or exclusions. The cover letter is a review draft until those statements are completed. Ethics and consent for original measurements remain attributed to the source studies; no new institutional waiver is claimed.

Guidance: https://www.nature.com/srep/author-instructions/submission-guidelines and https://www.nature.com/documents/srep-checklist-for-initial-submissions.pdf.

## Reading layout with embedded figures

`build_scientific_reports.py` also writes `output/scientific-reports/scientific-reports-manuscript-inline.pdf` (source `source/manuscript/paper_srep_inline.tex`): the same prose and numbers with each figure and the table floated after the paragraph that first cites it (never before it, within two pages, in numerical order; 20 draft pages, 2.5 MB). The journal accepts either arrangement at first submission and limits a single-file submission to 3 MB. The checklist layout above remains the default upload form. Supplementary items are cited as whole items in both layouts because the journal does not allow panel-level supplementary citation.
