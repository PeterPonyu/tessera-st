#!/usr/bin/env bash
# Portable reconstruction of supplied documents; no fitting or downloads.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/../.." && pwd)"
task_build="$task_root/build-portable"
mkdir -p "$task_build" "$task_root/output/pdf"
cd "$task_root/source/manuscript"
for task_stem in paper supporting atlas; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$task_build" "$task_stem.tex"
done
cp "$task_build/paper.pdf" "$task_root/output/pdf/tessera-revised-manuscript.pdf"
cp "$task_build/supporting.pdf" "$task_root/output/pdf/tessera-supporting-information.pdf"
cp "$task_build/atlas.pdf" "$task_root/output/pdf/tessera-figure-atlas.pdf"
cd "$task_root"
python3 source/scripts/build_scientific_reports.py
