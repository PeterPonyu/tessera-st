# Existing numerical and document environment

Saved field-run package versions: NumPy 2.2.6, SciPy 1.16.3,
scikit-learn 1.8.0, PyTorch 2.12.0+cu130, torch-geometric 2.7.0,
AnnData 0.12.10. These are recorded run identities, not a declaration that
all versions are required for byte-identical PDF compilation.

Offline verification uses Python, NumPy, SciPy, scikit-learn and PyMuPDF.
Adapter tests additionally use PyTorch, torch-geometric and AnnData. Document
build uses existing TeX Live/latexmk (newtx, natbib, plainnat, pgfplots, etc.),
Poppler and PyMuPDF. Figure redraw additionally uses Matplotlib and a locally
licensed Arial font. Full source fits additionally require external methods
and assay-specific data inputs. No package or font is installed by these
verification/build commands.
