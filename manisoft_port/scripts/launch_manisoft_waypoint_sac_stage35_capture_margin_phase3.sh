#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_stage35_potential_pilot_20260823"
phase_root="$run_root/phase3_capture_margin"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_4879664_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4879664_steps.pkl"

test -f "$source_model"
test -f "$source_vecnorm"
mkdir -p "$phase_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -120 "$phase_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 8 )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch_branch() {
  local name="$1" seed="$2" progress_scale="$3" margin_scale="$4" train_radius="$5"
  local output="$phase_root/$name"
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
      --internal-waypoint-capture-radius "$train_radius" \
      --internal-waypoint-progress-scale "$progress_scale" \
      --internal-waypoint-distance-penalty-scale "$margin_scale" \
      --waypoint-stall-steps 0 \
      --entry-sampling-weights 0.08,0.14,0.08,0.20,0.22,0.28 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-waypoint-stall-steps 0 \
      --eval-panel-seed 871000 --eval-num-envs 4 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 25000 --num-envs 4 \
      --checkpoint-freq 25000 --eval-freq 25000 --eval-episodes 48 \
      --learning-rate 0.000010 --batch-size 2048 \
      --gradient-steps 2 --learning-starts 2000 \
      --actor-anchor-coef 5 \
      --device cuda --seed "$seed" \
      >"$phase_root/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$phase_root/$name.pid"
  echo "$name PID=$pid progress=$progress_scale margin=$margin_scale radius=$train_radius"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch_branch margin05_p0_r10 2308351 0.00 0.05 0.010
launch_branch margin10_p0_r10 2308352 0.00 0.10 0.010
launch_branch margin10_p25_r10 2308353 0.25 0.10 0.010
launch_branch margin10_p25_r12 2308354 0.25 0.10 0.012

echo "launched four capture-margin phase-3 branches under $phase_root"
