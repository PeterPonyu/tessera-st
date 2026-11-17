#!/usr/bin/env bash
# Resume WP4 on the main box after stage 1 (17 Nov). The first pass of run_box.sh stopped at K-means scoring: a
# HER2ST dataset reloaded inside a busy worker process did not match its recorded fingerprint, so selection, stage 2,
# the GPU repeats, full scoring and the counterparts did not run (the E4 reproduction did). score_external_r3.py and
# counterparts_r3.py now load each dataset once in the parent process. The steps are those of run_box.sh, but every
# step must succeed before the next starts.
set -uo pipefail
source /root/env.sh
EXT=$PROJ/repository/revision-r3/external
REM=$PROJ/repository/revision-r3/remote
OUT=$TESSERA_R3_OUT
NPROC=${NPROC:-14}
stamp() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/logs/run_box.log"; }
fail() { stamp "$1 FAILED; see logs"; exit 1; }
stamp "resume after stage 1"
$PY "$EXT/score_external_r3.py" --out "$OUT" --processes "$NPROC" --rules kmeans > "$OUT/logs/score1.log" 2>&1 || fail "score1"
$PY "$REM/select_external_r3.py" --scores "$OUT/scores.csv" --out "$OUT" > "$OUT/logs/select.log" 2>&1 || fail "selection"
stamp "selection done"
$PY "$REM/stage2_r3.py" --out "$OUT" --workers 6 > "$OUT/logs/stage2.log" 2>&1 || fail "stage 2"
stamp "stage 2 done"
$PY "$REM/gpu_repeats_r3.py" --out "$OUT/gpu_repeats" > "$OUT/logs/gpu_repeats.log" 2>&1 &
REPEATS=$!
$PY "$EXT/score_external_r3.py" --out "$OUT" --processes "$(( NPROC - 2 ))" --rules kmeans leiden > "$OUT/logs/score2.log" 2>&1 || fail "score2"
$PY "$REM/counterparts_r3.py" --out "$OUT" --processes "$NPROC" > "$OUT/logs/counterparts.log" 2>&1 || fail "counterparts"
wait "$REPEATS" || fail "gpu repeats"
if [ ! -s "$OUT/repro/summary.json" ] && [ -z "$(ls -A "$OUT/repro" 2>/dev/null)" ]; then
  $PY "$EXT/repro_e4.py" --device cuda --out "$OUT/repro" > "$OUT/logs/repro.log" 2>&1 || fail "repro"
fi
stamp "all done"
tar -C "$PROJ/results/revision-r3" --exclude='W4/embeddings' -czf /root/work/W4-results.tgz W4
sha256sum /root/work/W4-results.tgz > /root/work/W4-results.tgz.sha256
stamp "packed"
