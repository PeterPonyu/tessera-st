#!/usr/bin/env bash
# Restart of the WP4 stage-1 GraphST grid on the box (17 Nov). Both original GraphST runners were killed by
# the container's OOM killer (cgroup limit about 85 GiB) after about 100 fits each, consistent with memory
# accumulating across GraphST fits within one process. Here each dataset gets its own runner process, so that
# memory is released after at most 24 fits. The script also reruns four jobs that stale locks left by the
# first attempt had blocked, and, once no stage-1 runner remains, resumes the paused run_box.sh.
# Usage (inside tmux): bash restart_graphst_s1.sh <pid of the paused run_box.sh>
set -u
RUNBOX_PID=$1
source /root/env.sh
EXT=$PROJ/repository/revision-r3/external
OUT=$TESSERA_R3_OUT
DATASETS=$($PY -c "import sys; sys.path.insert(0, '$EXT'); import common_external as ce; print(' '.join(ce.dataset_ids('core')))")
echo "[$(date -u +%FT%TZ)] graphst stage-1 restart (per-dataset processes)" >> "$OUT/logs/restart.log"
worker() {
  for ds in $DATASETS; do
    $PY "$EXT/run_external_r3.py" --encoders graphst --datasets "$ds" --arms released --configs all --seeds 1 2 3 \
      --input-modes common --device cuda >> "$OUT/logs/s1-graphst-r$1.log" 2>&1
  done
}
for i in 1 2 3; do worker "$i" & done
wait
$PY "$EXT/run_external_r3.py" --encoders stagate --datasets NSCLC_Lung5_Rep2 --arms released --configs all \
  --seeds 1 2 3 --input-modes common --device cuda >> "$OUT/logs/s1-stagate-r1.log" 2>&1
echo "[$(date -u +%FT%TZ)] graphst restart and blocked jobs done; waiting for the other stage-1 runners" >> "$OUT/logs/restart.log"
while pgrep -f "run_external_r3.py --encoders (stagate|tessera|graphst) --datasets" > /dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] resuming run_box.sh (pid $RUNBOX_PID)" >> "$OUT/logs/restart.log"
kill -CONT "$RUNBOX_PID"
