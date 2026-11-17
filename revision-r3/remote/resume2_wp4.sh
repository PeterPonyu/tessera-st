#!/usr/bin/env bash
# Resume WP4 after stage 1 without any Python worker pool (17 Nov, second design). Every failure so far came from
# pools inside one Python process (a forked worker crashing, a spawned pool deadlocking), while the fits, each its
# own process, never failed. Scoring and the counterparts therefore run as independent shard processes that write
# whole per-fit or per-shard files; a plain single-process run then assembles and checks. Every step must succeed.
set -uo pipefail
source /root/env.sh
EXT=$PROJ/repository/revision-r3/external
REM=$PROJ/repository/revision-r3/remote
OUT=$TESSERA_R3_OUT
SH=${SH:-14}
stamp() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/logs/run_box.log"; }
fail() { stamp "$1 FAILED; see logs"; exit 1; }
shards() {   # shards <label> <command...>: SH independent processes with --shard i --nshards SH
  local label=$1; shift
  local pids=() bad=0
  for i in $(seq 0 $((SH - 1))); do
    "$@" --shard "$i" --nshards "$SH" > "$OUT/logs/$label-shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || bad=$((bad + 1)); done
  [ "$bad" -eq 0 ] || fail "$label ($bad of $SH shards)"
}
stamp "resume after stage 1 (shards, no pools)"
shards score1 "$PY" "$EXT/score_external_r3.py" --out "$OUT" --rules kmeans
$PY "$EXT/score_external_r3.py" --out "$OUT" --rules kmeans > "$OUT/logs/score1.log" 2>&1 || fail "score1 assembly"
$PY "$REM/select_external_r3.py" --scores "$OUT/scores.csv" --out "$OUT" > "$OUT/logs/select.log" 2>&1 || fail "selection"
stamp "selection done"
$PY "$REM/stage2_r3.py" --out "$OUT" --workers 6 > "$OUT/logs/stage2.log" 2>&1 || fail "stage 2"
stamp "stage 2 done"
shards score2 "$PY" "$EXT/score_external_r3.py" --out "$OUT" --rules kmeans leiden
$PY "$EXT/score_external_r3.py" --out "$OUT" --rules kmeans leiden > "$OUT/logs/score2.log" 2>&1 || fail "score2 assembly"
stamp "scoring done"
shards counterparts "$PY" "$REM/counterparts_r3.py" --out "$OUT"
$PY "$REM/counterparts_r3.py" --out "$OUT" --merge > "$OUT/logs/counterparts.log" 2>&1 || fail "counterparts merge"
stamp "counterparts done"
stamp "all done"
tar -C "$PROJ/results/revision-r3" --exclude='W4/embeddings' -czf /root/work/W4-results.tgz W4
sha256sum /root/work/W4-results.tgz > /root/work/W4-results.tgz.sha256
stamp "packed"
