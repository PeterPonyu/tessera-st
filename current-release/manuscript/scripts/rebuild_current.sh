#!/usr/bin/env bash
# Current working-face build. No training, downloads, or historical reruns.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOURCE_DATE_EPOCH=0 FORCE_SOURCE_DATE=1 TZ=UTC
revision_python="${PY:-python}"
"$revision_python" ../verify_release.py
"$revision_python" scripts/build_supporting_information.py
"$revision_python" verify_committed_claims.py --source-only
latexmk -g -pdf -interaction=nonstopmode -halt-on-error paper.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error SI.tex
"$revision_python" verify_committed_claims.py
# This portable public build consumes the committed vector figures and tables.
# It does not silently refresh original capsule attestations or redraw assets.
"$revision_python" scripts/check_revision_pdf.py --render
