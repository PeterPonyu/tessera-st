#!/usr/bin/env bash
# WP4 on the box (run inside tmux after setup_box.sh). Stage 1 grids, K-means scoring, patient-excluded
# selection, stage 2 arms at seeds 1-5, GPU repeats, full scoring, counterparts, the E4 reproduction check.
set -uo pipefail
source /root/env.sh
EXT=$PROJ/repository/revision-r3/external
REM=$PROJ/repository/revision-r3/remote
OUT=$TESSERA_R3_OUT
NPROC=${NPROC:-14}   # the container reports the host's cores; this box rents 16
mkdir -p "$OUT/logs"
stamp() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/logs/run_box.log"; }
R3_IDS=$($PY -c "import sys; sys.path.insert(0, '$EXT'); import common_external as ce; print(' '.join(ce.r3_ids()))")
stamp "stage 1 start"
for i in 1 2; do
  $PY "$EXT/run_external_r3.py" --encoders graphst --datasets core --arms released --configs all --seeds 1 2 3 --input-modes common --device cuda > "$OUT/logs/s1-graphst-$i.log" 2>&1 &
  $PY "$EXT/run_external_r3.py" --encoders stagate --datasets core --arms released --configs all --seeds 1 2 3 --input-modes common --device cuda > "$OUT/logs/s1-stagate-$i.log" 2>&1 &
  $PY "$EXT/run_external_r3.py" --encoders tessera --datasets $R3_IDS --arms released --configs all --seeds 1 2 3 --input-modes common --device cuda > "$OUT/logs/s1-tessera-$i.log" 2>&1 &
done
wait
stamp "stage 1 done"
$PY "$EXT/score_external_r3.py" --out "$OUT" --processes "$NPROC" --rules kmeans > "$OUT/logs/score1.log" 2>&1
$PY "$REM/select_external_r3.py" --scores "$OUT/scores.csv" --out "$OUT" > "$OUT/logs/select.log" 2>&1
stamp "selection done"
$PY "$REM/stage2_r3.py" --out "$OUT" --workers 6 > "$OUT/logs/stage2.log" 2>&1
stamp "stage 2 done"
$PY "$REM/gpu_repeats_r3.py" --out "$OUT/gpu_repeats" > "$OUT/logs/gpu_repeats.log" 2>&1 &
$PY "$EXT/score_external_r3.py" --out "$OUT" --processes "$(( NPROC - 2 ))" --rules kmeans leiden > "$OUT/logs/score2.log" 2>&1
$PY "$REM/counterparts_r3.py" --out "$OUT" --processes "$NPROC" > "$OUT/logs/counterparts.log" 2>&1
wait
$PY "$EXT/repro_e4.py" --device cuda --out "$OUT/repro" > "$OUT/logs/repro.log" 2>&1
stamp "all done"
tar -C "$PROJ/results/revision-r3" --exclude='W4/embeddings' -czf /root/work/W4-results.tgz W4
sha256sum /root/work/W4-results.tgz > /root/work/W4-results.tgz.sha256
stamp "packed"
