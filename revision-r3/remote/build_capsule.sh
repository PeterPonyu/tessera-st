#!/usr/bin/env bash
# Laptop: build the WP4 capsule from the committed repository and the small derived inputs.
set -euo pipefail
P=${TESSERA_PROJECT:?set TESSERA_PROJECT to the project folder that holds repository/, data/ and results/}
S=${CAPSULE_DIR:-$P/capsule}
test -z "$(git -C "$P/repository" status --porcelain -- revision-r3)" || { echo "commit revision-r3 first"; exit 1; }
rm -rf "$S"; mkdir -p "$S/stage/tessera-st/results/revision-r1/E4"
git -C "$P/repository" archive --prefix=tessera-st/repository/ HEAD | tar -x -C "$S/stage"
cp -r "$P/data" "$S/stage/tessera-st/data"
cp -r "$P/validation" "$S/stage/tessera-st/validation"
cp -r "$P/results/revision-r1/E4/inputs" "$S/stage/tessera-st/results/revision-r1/E4/inputs"
cp "$P/results/revision-r1/E4/E4_scores.csv" "$S/stage/tessera-st/results/revision-r1/E4/"
rsync -a --exclude raw "$P/data-r3/" "$S/stage/tessera-st/data-r3/"
cp -r /home/zeyufu/Desktop/labs/active/spatial-omics-reform/external/STAGATE "$S/stage/STAGATE"
tar -C "$S/stage" -czf "$S/capsule.tar.gz" tessera-st STAGATE
(cd "$S" && sha256sum capsule.tar.gz > CAPSULE-SHA256.txt && cat CAPSULE-SHA256.txt)
git -C "$P/repository" rev-parse HEAD > "$S/CAPSULE-COMMIT.txt"
du -sh "$S/capsule.tar.gz"
