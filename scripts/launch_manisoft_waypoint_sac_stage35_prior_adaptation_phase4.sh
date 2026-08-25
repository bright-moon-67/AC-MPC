#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_stage35_potential_pilot_20260823"
phase_root="$run_root/phase4_prior_adaptation"
source_root="$run_root/phase3_capture_margin/margin10_p25_r10"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_4904664_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4904664_steps.pkl"

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
  local name="$1" seed="$2" prior_weight="$3" anchor="$4" lr="$5"
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
      --internal-waypoint-capture-radius 0.010 \
      --internal-waypoint-progress-scale 0.25 \
      --internal-waypoint-distance-penalty-scale 0.10 \
      --waypoint-stall-steps 0 \
      --cartesian-prior-weight "$prior_weight" \
      --cartesian-prior-proportional-gain 20 \
      --cartesian-prior-feedforward-scale 1 \
      --entry-sampling-weights 0.08,0.14,0.08,0.20,0.22,0.28 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-waypoint-stall-steps 0 \
      --eval-panel-seed 871000 --eval-num-envs 4 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 40000 --num-envs 4 \
      --checkpoint-freq 40000 --eval-freq 40000 --eval-episodes 48 \
      --learning-rate "$lr" --batch-size 2048 \
      --gradient-steps 2 --learning-starts 3000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$phase_root/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$phase_root/$name.pid"
  echo "$name PID=$pid prior=$prior_weight anchor=$anchor lr=$lr"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch_branch w30_a10 2308361 0.30 10 0.000005
launch_branch w40_a20 2308362 0.40 20 0.000005
launch_branch w40_a5 2308363 0.40 5 0.000010
launch_branch w50_a10 2308364 0.50 10 0.000005

echo "launched four prior-adaptation phase-4 branches under $phase_root"
