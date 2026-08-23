#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_turn_pilot_20260823/balanced45_a20"
run_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823"
source_model="$source_root/best/best_model.zip"
source_vecnorm="$source_root/vecnormalize.pkl"

test -f "$source_model"
test -f "$source_vecnorm"
test ! -e "$run_root"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2" minimum="$3"
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -120 "$run_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= minimum )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch_train() {
  local name="$1" seed="$2" turn="$3" anchor="$4" lr="$5" weights="$6"
  local output="$run_root/$name" log="$run_root/$name.log"
  local weighted_args=()
  if [[ -n "$weights" ]]; then
    weighted_args=(--entry-sampling-weights "$weights")
  fi
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,3 \
      --waypoint-segment-count-probabilities 0.25,0.75 \
      --waypoint-maximum-turn-degrees "$turn" \
      --waypoint-maximum-extent 0.035 \
      --waypoint-single-line-probability 0.20 \
      --internal-waypoint-capture-radius 0.010 \
      "${weighted_args[@]}" \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-panel-seed 863000 --eval-num-envs 4 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 120000 --num-envs 20 \
      --checkpoint-freq 30000 --eval-freq 30000 --eval-episodes 48 \
      --learning-rate "$lr" --batch-size 2048 \
      --gradient-steps 8 --learning-starts 10000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid turn=$turn anchor=$anchor lr=$lr weights=${weights:-uniform}"
  wait_for_workers "$pid" "$name" 24
}

launch_source_panel() {
  local name="source_panel" output="$run_root/source_panel"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-maximum-turn-degrees 60 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-panel-seed 863000 --eval-num-envs 8 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 1 --num-envs 1 \
      --checkpoint-freq 1 --eval-freq 1 --eval-episodes 48 \
      --learning-starts 100 --actor-anchor-coef 100 \
      --device cuda --seed 2308270 \
      >"$run_root/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid"
  wait_for_workers "$pid" "$name" 9
}

cd "$project_root"
launch_train turn30_uniform_a100 2308271 30 100 0.000002 ""
launch_train turn45_uniform_a100 2308272 45 100 0.000002 ""
launch_train turn45_weak_a100 2308273 45 100 0.000002 "0.08,0.14,0.08,0.20,0.22,0.28"
launch_train turn60_weak_a300 2308274 60 300 0.000001 "0.08,0.14,0.08,0.20,0.22,0.28"
launch_source_panel

echo "launched four gate-aware pilots and their source panel under $run_root"
