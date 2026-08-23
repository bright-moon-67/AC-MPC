#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_rehearsal_anchor_pilot_20260822/anchor10_r70"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_4199960_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4199960_steps.pkl"
run_root="$project_root/runs/manisoft_waypoint_sac_range_pilot_20260822"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 120); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -80 "$run_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 28 )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch() {
  local name="$1" leak="$2" extent="$3" anchor="$4" seed="$5"
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_waypoint_rehearsal.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,2 \
      --waypoint-maximum-turn-degrees 90 \
      --waypoint-maximum-extent "$extent" \
      --waypoint-single-line-probability 0.60 \
      --eval-waypoint-single-line-probability 0 \
      --cartesian-action-leak "$leak" \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 100000 --num-envs 28 \
      --checkpoint-freq 25000 --eval-freq 25000 --eval-episodes 36 \
      --learning-rate 0.00001 --batch-size 2048 \
      --gradient-steps 8 --learning-starts 20000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid leak=$leak extent=$extent anchor=$anchor"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
test -f "$source_model"
test -f "$source_vecnorm"
launch control_l008_e045 0.008 0.045 10 2208231
launch range_l006_e045 0.006 0.045 5 2208232
launch range_l005_e045 0.005 0.045 5 2208233
launch feasible_l008_e035 0.008 0.035 10 2208234
