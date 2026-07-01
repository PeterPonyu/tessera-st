# Manuscript build: machine-checked artifacts -> vector .tex -> paper.pdf.
#
#   make figs    R reads experiments/*.json directly and emits pgfplots/TikZ + the spatial map
#                + tables into figs/ (no Python, no intermediate CSV)
#   make pdf     compile paper.pdf with lualatex x2 + bibtex                  (default)
#   make verify  machine-check the manuscript numbers against the JSON artifacts
#   make clean   remove LaTeX build artifacts and scratch test files
#
# Requires: Rscript (jsonlite, ggplot2, tikzDevice, ggrastr, mclust), lualatex, python3 (verify only).

PY   ?= python3
R    ?= Rscript
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(HERE)/..)
LATEX := lualatex -interaction=nonstopmode -halt-on-error

.PHONY: all pdf figs verify clean
all: pdf

figs:
	cd $(ROOT) && $(R) manuscript/make_figs.R

pdf: figs
	cd $(HERE) && $(LATEX) paper.tex
	cd $(HERE) && bibtex paper
	cd $(HERE) && $(LATEX) paper.tex
	cd $(HERE) && $(LATEX) paper.tex
	@echo "built $(HERE)paper.pdf"

verify:
	cd $(ROOT) && $(PY) experiments/verify_manuscript.py

clean:
	cd $(HERE) && rm -f paper.aux paper.log paper.out paper.bbl paper.blg \
	    paper.fls paper.fdb_latexmk _figtest.* _b.* _gallery.*
