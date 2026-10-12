# Editable Tessera controls audit release

Software/source version `2026.10.11-controls-audit`; software DOI
`10.5281/zenodo.23308077` is reserved until the record owner publishes.
The evaluated original implementation DOI `10.5281/zenodo.23101191` is distinct.
No manuscript DOI is assigned by this package.

The source includes the full canonical manuscript, Scientific Reports manuscript
and inline reading layout, and supplementary-information source. The three final
journal PDFs are in `output/scientific-reports/`. The current journal-format audit
records a 192-word abstract, seven main figures, one main table, fourteen
supplementary figures and eighteen supplementary tables.

Run the four portable checks in README.md to verify the sealed package, import
closure, retained scientific evidence and recovered producer sources. Their
default execution uses no network, GPU, author-source checkout or source symlinks.
The old author-workspace CRC verifier additionally checks omitted raw ZIPs and
historical integration backups; its wider scope is not claimed by the portable
command. Full refitting remains a distinct operation with documented missing
historical embeddings/partitions.

The current 82-test suite covers saved numerical bindings, source contracts and
current journal presentation. The test corrections and initial stale-presentation
failures are documented in `qa/release-closure/`. The fixed numerical reference
baselines and all scientific protocols/locks remain unchanged. A source suite
run is distinct from the four fresh-extraction checks.

For document reconstruction, `bash source/scripts/build_documents.sh` uses the
supplied native/journal graphics and table sources plus latexmk, pdfLaTeX,
pdftotext, pdfinfo and TeXcount. Optional figure-style/PDF QA tooling has separate
runtime requirements recorded in `requirements-document-qa.txt`; it is not
needed for saved-result verification or use of the already generated graphics.
Arial font binaries are not distributed as standalone files.

Breast Figure 7 uses author-generated measured-coordinate maps without the 10x
breast H&E image/background. Its historic cross-reference filename is retained.
The separate CRC image/registration evidence retains source-specific Dataset
CC BY 4.0 attribution. See LICENSING.md, NOTICE.md and the source-rights records.
