#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_stage35_potential_pilot_20260823"
phase_root="$run_root/phase2"
checkpoint_steps=4889664

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
  local name="$1" source_name="$2" seed="$3" progress_scale="$4" anchor="$5" lr="$6"
  local source_root="$run_root/$source_name"
  local model="$source_root/checkpoints/sac_table_waypoint_polyline_${checkpoint_steps}_steps.zip"
  local vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_${checkpoint_steps}_steps.pkl"
  local output="$phase_root/$name"
  test -f "$model"
  test -f "$vecnorm"
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
      --waypoint-stall-steps 0 \
      --entry-sampling-weights 0.08,0.14,0.08,0.20,0.22,0.28 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-waypoint-stall-steps 0 \
      --eval-panel-seed 871000 --eval-num-envs 4 \
      --resume "$model" --vec-normalize "$vecnorm" \
      --total-timesteps 20000 --num-envs 4 \
      --checkpoint-freq 20000 --eval-freq 20000 --eval-episodes 48 \
      --learning-rate "$lr" --batch-size 2048 \
      --gradient-steps 2 --learning-starts 2000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$phase_root/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$phase_root/$name.pid"
  echo "$name PID=$pid source=$source_name scale=$progress_scale anchor=$anchor lr=$lr"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch_branch control_a20 control_p0 2308331 0.00 20 0.000005
launch_branch p25_a20 potential_p25 2308332 0.25 20 0.000005
launch_branch p25_a5 potential_p25 2308333 0.25 5 0.000010
launch_branch p50_a5 potential_p25 2308334 0.50 5 0.000010

echo "launched four Stage 3.5 phase-2 branches under $phase_root"
