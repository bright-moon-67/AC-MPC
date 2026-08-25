#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_stage35_potential_pilot_20260823"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_4879664_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4879664_steps.pkl"

test -f "$source_model"
test -f "$source_vecnorm"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -120 "$run_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 5 )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch_branch() {
  local name="$1" seed="$2" progress_scale="$3" waypoint_stall_steps="$4"
  local output="$run_root/$name"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,3 \
      --waypoint-segment-count-probabilities 0.10,0.90 \
      --waypoint-maximum-turn-degrees 60 \
      --waypoint-maximum-extent 0.035 \
      --waypoint-single-line-probability 0.10 \
      --internal-waypoint-capture-radius 0.010 \
      --internal-waypoint-progress-scale "$progress_scale" \
      --waypoint-stall-steps "$waypoint_stall_steps" \
      --waypoint-stall-distance-epsilon 0.0003 \
      --entry-sampling-weights 0.08,0.14,0.08,0.20,0.22,0.28 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-waypoint-stall-steps 0 \
      --eval-panel-seed 871000 --eval-num-envs 1 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 20000 --num-envs 4 \
      --checkpoint-freq 10000 --eval-freq 10000 --eval-episodes 48 \
      --learning-rate 0.000005 --batch-size 2048 \
      --gradient-steps 2 --learning-starts 2000 \
      --actor-anchor-coef 20 \
      --device cuda --seed "$seed" \
      >"$run_root/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid progress_scale=$progress_scale waypoint_stall_steps=$waypoint_stall_steps"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch_branch control_p0 2308321 0.00 0
launch_branch potential_p15 2308322 0.15 0
launch_branch potential_p25 2308323 0.25 0
launch_branch potential_p25_stall250 2308324 0.25 250

echo "launched four Stage 3.5 causal pilots under $run_root"
