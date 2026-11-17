#!/usr/bin/env bash
# Runs on the main WP4 box: every 2 minutes, an add-only, two-way exchange of finished fits with the helper box,
# embeddings before records so that a record never arrives without its embedding; locks and temporary files are
# never copied and nothing is deleted or overwritten. Also sends the patient-excluded selection to the helper.
# Usage: bash sync_ab.sh <helper host> <helper ssh port>
set -u
HOST=$1
PORT=$2
W=/root/work/tessera-st/results/revision-r3/W4
RSH="ssh -i /root/.ssh/box_b -o IdentitiesOnly=yes -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
OPTS=(-a --ignore-existing "--exclude=*.lock" "--exclude=*.tmp" -e "$RSH")
while true; do
  for d in embeddings fits; do
    rsync "${OPTS[@]}" "root@$HOST:$W/$d/" "$W/$d/" || echo "[$(date -u +%FT%TZ)] pull $d failed" >> "$W/logs/sync.log"
    rsync "${OPTS[@]}" "$W/$d/" "root@$HOST:$W/$d/" || echo "[$(date -u +%FT%TZ)] push $d failed" >> "$W/logs/sync.log"
  done
  if [ -s "$W/selected_configs.json" ]; then
    rsync -a -e "$RSH" "$W/selection.csv" "$W/selected_configs.json" "root@$HOST:$W/" 2>> "$W/logs/sync.log"
  fi
  echo "[$(date -u +%FT%TZ)] sync: $(ls $W/fits/*/*.json 2>/dev/null | wc -l) records here" >> "$W/logs/sync.log"
  sleep 120
done
