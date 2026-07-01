#!/usr/bin/env bash
# Clean-room gate: no baseline brand may appear in the executable zones (src/ tests/ scripts/).
# Brand names are permitted only in the provenance docs. Mirrors the parent program's
# leakage_scan.py. Exits non-zero on any leak.
set -euo pipefail

cd "$(dirname "$0")/.."

BANNED="stagate|graphst|spaceflow|sedr|banksy|spagcn|inspire|harvest|staligner|deepst"
ZONES="src tests scripts"

# this script and the clean-room test both name brands to *define* the banned list;
# exclude those two definers from the scan.
hits=$(grep -rniE "$BANNED" $ZONES --include='*.py' --include='*.sh' \
        --exclude='check_independence.sh' --exclude='test_brand_independence.py' || true)

if [[ -n "$hits" ]]; then
  echo "FAIL: baseline brand leaked into executable zone:"
  echo "$hits"
  exit 1
fi
echo "OK: clean-room boundary intact (no baseline brand in $ZONES)"
