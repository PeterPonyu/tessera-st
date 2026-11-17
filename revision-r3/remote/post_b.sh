#!/usr/bin/env bash
# Post-processing on the helper box (17 Nov). The helper has a 46-core quota against the main box's 15 and holds every
# fit record and embedding through the two-way sync, so once every planned stage-2 job has a record it runs full
# scoring and the counterparts as SH independent shards (no Python pools), assembles them, verifies the tree and packs.
# The main box runs the same steps with fewer shards; the two outputs serve as a cross-check.
set -uo pipefail
source /root/env.sh
EXT=$PROJ/repository/revision-r3/external
REM=$PROJ/repository/revision-r3/remote
OUT=$TESSERA_R3_OUT
SH=${SH:-40}
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/logs/post_b.log"; }
fail() { log "$1 FAILED; see logs"; exit 1; }
shards() {
  local label=$1; shift
  local pids=() bad=0
  for i in $(seq 0 $((SH - 1))); do
    "$@" --shard "$i" --nshards "$SH" > "$OUT/logs/b-$label-shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || bad=$((bad + 1)); done
  [ "$bad" -eq 0 ] || fail "$label ($bad of $SH shards)"
}
log "waiting for the helper's stage 2 and for every planned stage-2 job to have a record"
# wait for the helper's own completion line (pgrep on the script name also matches the tmux server that launched it)
until grep -q "stage 2 share done" "$OUT/logs/helper.log" 2>/dev/null; do sleep 60; done
until $PY "$REM/verify_w4.py" "$OUT" --stage2 > "$OUT/logs/b-verify-wait.log" 2>&1; do sleep 120; done
log "all planned stage-2 jobs present and verified"
shards score2 "$PY" "$EXT/score_external_r3.py" --out "$OUT" --rules kmeans leiden
$PY "$EXT/score_external_r3.py" --out "$OUT" --rules kmeans leiden > "$OUT/logs/b-score2.log" 2>&1 || fail "score2 assembly"
log "scoring done"
shards counterparts "$PY" "$REM/counterparts_r3.py" --out "$OUT"
$PY "$REM/counterparts_r3.py" --out "$OUT" --merge > "$OUT/logs/b-counterparts.log" 2>&1 || fail "counterparts merge"
log "counterparts done"
$PY "$REM/verify_w4.py" "$OUT" --stage2 > "$OUT/logs/b-verify-final.log" 2>&1 || fail "final verification"
tar -C "$PROJ/results/revision-r3" --exclude='W4/embeddings' -czf /root/work/W4-results-b.tgz W4
sha256sum /root/work/W4-results-b.tgz > /root/work/W4-results-b.tgz.sha256
log "packed"
