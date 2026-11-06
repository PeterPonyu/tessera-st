#!/usr/bin/env bash
# Amendment A2 job on the rented box: hardware check, k = 200 lung grid, selection and mechanism arms.
# Usage: nohup setsid bash run_a2.sh ROOT > /dev/null 2>&1 &
# Writes ROOT/r1out/E2/remote-a2/{started_utc,finished_utc,EXIT,outputs.sha256,DONE} and one log per stage.
set -uo pipefail
ROOT=${1:?root}
PY=${PY:-python}
cd "$ROOT"
export TESSERA_DATA="$ROOT/inputs/data" TESSERA_R1_OUT="$ROOT/r1out" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REC=r1out/E2/remote-a2
date -u +%FT%TZ > "$REC/started_utc"
stage() { "$PY" repo/revision-r1/run_tuning.py "$1" --processes 2 > "$REC/$1.log" 2>&1; echo "$1 $?" >> "$REC/stages.txt"; }
stage hwcheck-a2
stage grid-gpu-a2
if grep -q '^grid-gpu-a2 0$' "$REC/stages.txt"; then stage mechanism-a2; fi
date -u +%FT%TZ > "$REC/finished_utc"
(cd r1out && find E2 -type f ! -name outputs.sha256 ! -name DONE -print0 | sort -z | xargs -0 sha256sum) > "$REC/outputs.sha256"
awk '{s += $2} END {print (s == 0 ? 0 : 1)}' "$REC/stages.txt" > "$REC/EXIT"
touch "$REC/DONE"
