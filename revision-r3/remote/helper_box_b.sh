#!/usr/bin/env bash
# Helper box for WP4 (17 Nov; second RTX 3090, same image and environment as the main box).
# It works through the same job lists as the main box but from the other end, so the two meet in the middle.
# Finished fits are exchanged by sync_ab.sh, which runs on the main box; the runner skips any job whose
# record already exists, so a job done on one box is not repeated on the other once synced.
#   Stage 1: GraphST grids on every core dataset and Tessera-ST grids on the new cohorts, in reverse order,
#            one runner process per dataset (GraphST memory grows across fits within a process).
#   Stage 2: after the main box's patient-excluded selection has been synced here, the stage-2 commands
#            in reverse order (stage2_r3.py --reverse).
set -u
source /root/env.sh
EXT=$PROJ/repository/revision-r3/external
REM=$PROJ/repository/revision-r3/remote
OUT=$TESSERA_R3_OUT
mkdir -p "$OUT/logs"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$OUT/logs/helper.log"; }
CORE=$($PY -c "import sys; sys.path.insert(0, '$EXT'); import common_external as ce; print(' '.join(reversed(ce.dataset_ids('core'))))")
R3=$($PY -c "import sys; sys.path.insert(0, '$EXT'); import common_external as ce; print(' '.join(reversed(ce.r3_ids())))")
log "helper start: stage 1 in reverse order"
graphst() {
  for ds in $CORE; do
    $PY "$EXT/run_external_r3.py" --encoders graphst --datasets "$ds" --arms released --configs all --seeds 1 2 3 \
      --input-modes common --device cuda >> "$OUT/logs/b-s1-graphst-$1.log" 2>&1
  done
}
tessera() {
  for ds in $R3; do
    $PY "$EXT/run_external_r3.py" --encoders tessera --datasets "$ds" --arms released --configs all --seeds 1 2 3 \
      --input-modes common --device cuda >> "$OUT/logs/b-s1-tessera-$1.log" 2>&1
  done
}
for i in 1 2 3; do graphst "$i" & done
for i in 1 2 3; do tessera "$i" & done
wait
log "stage 1 share done; waiting for the main box's selection"
while [ ! -s "$OUT/selected_configs.json" ]; do sleep 30; done
sleep 60   # let the sync finish copying selection.csv as well
log "selection received; stage 2 in reverse order"
$PY "$REM/stage2_r3.py" --out "$OUT" --workers 6 --reverse > "$OUT/logs/b-stage2.log" 2>&1
log "stage 2 share done"
