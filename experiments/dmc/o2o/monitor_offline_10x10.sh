#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$1"
DATASET="$2"
KOOPMAN="$3"
DEVICE="${4:-cuda}"
OFFLINE_UPDATES="${5:-50000}"
ONLY_METHOD="${6:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

METHODS=(Cal-QL-Raw Cal-RLPD-Raw Cal-QL-KMPC Cal-QL-MPVE Cal-RLPD-KMPC Cal-RLPD-MPVE)
declare -A TRAIN_METHODS=(
  [Cal-QL-Raw]=Cal-QL-Raw
  [Cal-RLPD-Raw]=Cal-RLPD-Raw
  [Cal-QL-KMPC]=Cal-QL-AC-KMPC
  [Cal-QL-MPVE]=Cal-QL-MPVE
  [Cal-RLPD-KMPC]=Cal-RLPD-KMPC
  [Cal-RLPD-MPVE]=Cal-RLPD-MPVE
)
if [[ -n "$ONLY_METHOD" ]]; then
  if [[ -z "${TRAIN_METHODS[$ONLY_METHOD]+present}" ]]; then
    echo "unknown method filter: $ONLY_METHOD" >&2
    exit 2
  fi
  METHODS=("$ONLY_METHOD")
fi

while :; do
  all_done=1
  for name in "${METHODS[@]}"; do
    dir="$RUN_ROOT/$name"
    result="$dir/evaluation_10x10.json"
    if [[ -f "$result" ]]; then
      continue
    fi
    all_done=0
    if [[ ! -f "$dir/run.json" || ! -f "$dir/latest.pt" ]]; then
      continue
    fi
    if ! "$PYTHON_BIN" - "$dir/run.json" "$OFFLINE_UPDATES" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
try:
    value = json.load(open(path, encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if value.get("completed") is True
                 and value.get("execution_scope") == "offline_only"
                 and value.get("offline_updates_completed") == expected
                 and value.get("online_steps_completed") == 0 else 1)
PY
    then
      continue
    fi
    method="${TRAIN_METHODS[$name]}"
    extra=(--dataset "$DATASET" --device "$DEVICE" --checkpoint latest
           --seed-base 9100000 --num-seeds 10 --episodes-per-seed 10
           --output "$result")
    if [[ "$method" != *Raw ]]; then
      extra+=(--koopman "$KOOPMAN")
    fi
    echo "[$(date -u +%FT%TZ)] evaluating $name (10 seeds x 10 episodes)" >&2
    PYTHONPATH=. "$PYTHON_BIN" -m experiments.dmc.o2o.evaluate_10x10 \
      --run-dir "$dir" "${extra[@]}" >"$dir/evaluation_10x10.log" 2>&1
  done
  if (( all_done )); then
    echo "[$(date -u +%FT%TZ)] all offline 10x10 evaluations complete" >&2
    exit 0
  fi
  sleep 10
done
