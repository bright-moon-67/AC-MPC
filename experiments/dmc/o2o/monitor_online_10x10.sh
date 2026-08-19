#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$1"
DATASET="$2"
KOOPMAN="$3"
DEVICE="${4:-cpu}"
ONLINE_STEPS="${5:-20000}"
OFFLINE_UPDATES="${6:-100000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PARALLEL_WORKERS="${O2O_EVAL_WORKERS:-10}"

METHODS=(
  Cal-QL-Raw
  Cal-RLPD-Raw
  Cal-QL-AC-KMPC
  Cal-QL-MPVE
  Cal-RLPD-KMPC
  Cal-RLPD-MPVE
)

while :; do
  all_done=1
  for method in "${METHODS[@]}"; do
    dir="$RUN_ROOT/$method"
    result="$dir/evaluation_10x10.json"
    if [[ -f "$result" ]]; then
      continue
    fi
    all_done=0
    if [[ ! -f "$dir/run.json" || ! -f "$dir/latest.pt" ]]; then
      continue
    fi
    if ! "$PYTHON_BIN" - "$dir/run.json" "$OFFLINE_UPDATES" "$ONLINE_STEPS" <<'PY'
import json
import sys

path, expected_offline, expected_online = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    value = json.load(open(path, encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(
    0
    if value.get("completed") is True
    and value.get("execution_scope") == "offline_to_online"
    and value.get("offline_updates_completed") == expected_offline
    and value.get("online_steps_completed") == expected_online
    else 1
)
PY
    then
      continue
    fi
    extra=(
      --dataset "$DATASET"
      --device "$DEVICE"
      --checkpoint latest
      --seed-base 9100000
      --num-seeds 10
      --episodes-per-seed 10
      --parallel-workers "$PARALLEL_WORKERS"
      --output "$result"
    )
    if [[ "$method" != *Raw ]]; then
      extra+=(--koopman "$KOOPMAN")
    fi
    echo "[$(date -u +%FT%TZ)] evaluating $method latest at online=$ONLINE_STEPS" >&2
    PYTHONPATH=. "$PYTHON_BIN" -m experiments.dmc.o2o.evaluate_10x10 \
      --run-dir "$dir" "${extra[@]}" >"$dir/evaluation_10x10.log" 2>&1
  done
  if (( all_done )); then
    echo "[$(date -u +%FT%TZ)] all online 10x10 evaluations complete" >&2
    exit 0
  fi
  sleep 10
done
