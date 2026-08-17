#!/usr/bin/env bash
# Complete the Cartpole 3M-data pipeline without an SSH session:
#   wait for the approved data-source PPO -> build dataset -> train Koopman
#   -> validate four actor runs -> launch the four runs in independent sessions.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${DMC_CONFIG:-experiments/dmc/configs/data/cartpole_swingup_koopman_data_v2.yaml}"
PROFILE="${DMC_PROFILE:-development}"
DEVICE="${DMC_DEVICE:-cuda}"
ENV_WORKERS="${DMC_ENV_WORKERS:-16}"
PREFLIGHT="${DMC_PREFLIGHT:-runs/dmc/preflight/cartpole_swingup_koopman_data_v2_3m.json}"
APPROVAL="${DMC_APPROVAL:-runs/dmc/approvals/cartpole_swingup_koopman_data_v2_3m.json}"

DATA_PPO_OUT="${DMC_DATA_PPO_OUT:-runs/dmc/ppo/cartpole_swingup/koopman_data_v2_3m/seed_20260812/PPO}"
COLLECT_DIR="${DMC_COLLECT_DIR:-runs/dmc/data/cartpole_swingup/development/seed_20260812}"
DATASET="${DMC_DATASET:-runs/dmc/data/cartpole_swingup/development/cartpole_swingup_koopman_v2_3m.npz}"
KOOPMAN_OUT="${DMC_KOOPMAN_OUT:-runs/dmc/koopman/cartpole_swingup/development_v2_3m}"
KOOPMAN_DRY_OUT="${DMC_KOOPMAN_DRY_OUT:-runs/dmc/dry_runs/cartpole_swingup/development/koopman_v2_3m}"
ACTOR_DRY_BASE="${DMC_ACTOR_DRY_BASE:-runs/dmc/dry_runs/cartpole_swingup/development/koopman_v2_3m_actors}"
ACTOR_BASE="${DMC_ACTOR_BASE:-runs/dmc/ppo/cartpole_swingup/koopman_v2_3m_actors/seed_20260812}"
PIPELINE_DIR="${DMC_PIPELINE_DIR:-runs/dmc/pipelines/cartpole_swingup_koopman_v2_3m}"
PIPELINE_LOG="$PIPELINE_DIR/pipeline.log"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "required file is missing: $1"
}

process_uses_output_dir() {
  local output_dir="$1"
  pgrep -af 'experiments\.dmc\.ppo\.train_dmc_ppo' 2>/dev/null \
    | grep -F -- "--output-dir $output_dir" >/dev/null
}

if [[ "${1:-}" != "--foreground" ]]; then
  mkdir -p "$PIPELINE_DIR"
  if [[ -f "$PIPELINE_DIR/pipeline.pid" ]]; then
    EXISTING_PID="$(tr -dc '0-9' < "$PIPELINE_DIR/pipeline.pid")"
    if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
      fail "pipeline is already running as PID $EXISTING_PID"
    fi
  fi
  nohup setsid "$0" --foreground >> "$PIPELINE_LOG" 2>&1 < /dev/null &
  PIPELINE_PID=$!
  printf '%s\n' "$PIPELINE_PID" > "$PIPELINE_DIR/pipeline.pid"
  log "detached pipeline PID $PIPELINE_PID; log: $PIPELINE_LOG"
  exit 0
fi

mkdir -p "$PIPELINE_DIR"
printf '%s\n' "$$" > "$PIPELINE_DIR/pipeline.pid"
trap 'PIPELINE_RC=$?; log "pipeline exit code $PIPELINE_RC"; exit "$PIPELINE_RC"' EXIT

require_file "$CONFIG"
require_file "$PREFLIGHT"
require_file "$APPROVAL"

log "waiting only for the 3M durable collection (the source PPO may stop later)"
while [[ ! -f "$COLLECT_DIR/collection_status.json" ]]; do
  if ! process_uses_output_dir "$DATA_PPO_OUT"; then
    fail "data-source PPO stopped before creating collection_status.json"
  fi
  sleep 30
done

python - "$COLLECT_DIR/collection_status.json" <<'PY'
import json
import sys

status_path = sys.argv[1]
with open(status_path, encoding="utf-8") as stream:
    status = json.load(stream)
if status.get("total_transitions") != 3_000_000:
    raise SystemExit("collector did not durably finish 3M transitions")
budget = status.get("budget", {})
stages = budget.get("stages", [])
if [stage.get("name") for stage in stages] != ["early", "mid", "late"]:
    raise SystemExit("collector stage labels are incomplete")
if [stage.get("selected_transitions") for stage in stages] != [1_000_000] * 3:
    raise SystemExit("collector early/mid/late quotas are incomplete")
if budget.get("pending_complete_episode_transitions") != 0:
    raise SystemExit("collector still has non-durable complete episodes")
PY
log "3M complete-episode collection verified; PPO completion is not required"

if [[ ! -f "$DATASET" ]]; then
  log "building strict episode-split dataset: $DATASET"
  MUJOCO_GL=egl PYTHONPATH=. python -m experiments.dmc.collect.build_dmc_datasets \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --collect-root runs/dmc/data \
    --output "$DATASET"
else
  log "reusing existing dataset; Koopman dry-run will validate its identity"
fi
require_file "$DATASET"

log "running zero-step Koopman validation"
MUJOCO_GL=egl PYTHONPATH=. python -m experiments.dmc.koopman.train_dmc_koopman \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --preflight-file "$PREFLIGHT" \
  --dataset "$DATASET" \
  --output-dir "$KOOPMAN_DRY_OUT" \
  --device "$DEVICE" \
  --dry-run

log "starting Koopman training (safe resume is enabled)"
MUJOCO_GL=egl PYTHONPATH=. python -m experiments.dmc.koopman.train_dmc_koopman \
  --config "$CONFIG" \
  --profile "$PROFILE" \
  --preflight-file "$PREFLIGHT" \
  --approval-file "$APPROVAL" \
  --dataset "$DATASET" \
  --output-dir "$KOOPMAN_OUT" \
  --device "$DEVICE"

KOOPMAN_BEST="$KOOPMAN_OUT/best.pt"
require_file "$KOOPMAN_BEST"
require_file "$KOOPMAN_OUT/report.json"
log "Koopman completed successfully: $KOOPMAN_BEST"

log "validating all four actor commands without environment/optimizer steps"
for ACTOR in PPO KMPC AB-PQ AC-MPC-MPVE; do
  EXTRA_ARGS=()
  if [[ "$ACTOR" == PPO ]]; then
    EXTRA_ARGS+=(--no-collect)
  else
    EXTRA_ARGS+=(--koopman "$KOOPMAN_BEST")
  fi
  MUJOCO_GL=egl PYTHONPATH=. python -m experiments.dmc.ppo.train_dmc_ppo \
    --config "$CONFIG" \
    --profile "$PROFILE" \
    --train-seed-index 0 \
    --preflight-file "$PREFLIGHT" \
    --actor "$ACTOR" \
    --output-dir "$ACTOR_DRY_BASE/$ACTOR" \
    --device "$DEVICE" \
    --env-workers "$ENV_WORKERS" \
    --dry-run \
    "${EXTRA_ARGS[@]}"
done

log "launching PPO, KMPC, AB-PQ and AC-MPC-MPVE in independent sessions"
PID_TMP="$PIPELINE_DIR/.actor_pids.$$.tmp"
: > "$PID_TMP"
for ACTOR in PPO KMPC AB-PQ AC-MPC-MPVE; do
  ACTOR_OUT="$ACTOR_BASE/$ACTOR"
  mkdir -p "$ACTOR_OUT"
  if [[ -f "$ACTOR_OUT/final.json" ]]; then
    log "$ACTOR is already complete; skipping"
    continue
  fi
  if process_uses_output_dir "$ACTOR_OUT"; then
    log "$ACTOR is already active; skipping duplicate launch"
    continue
  fi
  EXTRA_ARGS=()
  if [[ "$ACTOR" == PPO ]]; then
    EXTRA_ARGS+=(--no-collect)
  else
    EXTRA_ARGS+=(--koopman "$KOOPMAN_BEST")
  fi
  nohup setsid env MUJOCO_GL=egl PYTHONPATH=. \
    python -m experiments.dmc.ppo.train_dmc_ppo \
      --config "$CONFIG" \
      --profile "$PROFILE" \
      --train-seed-index 0 \
      --preflight-file "$PREFLIGHT" \
      --approval-file "$APPROVAL" \
      --actor "$ACTOR" \
      --output-dir "$ACTOR_OUT" \
      --device "$DEVICE" \
      --env-workers "$ENV_WORKERS" \
      "${EXTRA_ARGS[@]}" \
      >> "$ACTOR_OUT/persistent.log" 2>&1 < /dev/null &
  ACTOR_PID=$!
  printf '%s\t%s\t%s\n' "$ACTOR" "$ACTOR_PID" "$ACTOR_OUT" >> "$PID_TMP"
  log "launched $ACTOR as PID $ACTOR_PID"
done
mv "$PID_TMP" "$PIPELINE_DIR/actor_pids.tsv"
log "pipeline complete; actor jobs are detached and SSH-safe"
