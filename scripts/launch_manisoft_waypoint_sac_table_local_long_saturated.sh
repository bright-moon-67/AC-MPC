#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
pilot_root="$project_root/runs/manisoft_waypoint_sac_table_local_pilot_20260822"
run_root="$project_root/runs/manisoft_waypoint_sac_table_local_long_saturated_20260822"

mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 90); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 40 )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch() {
  local name="$1" learning_rate="$2" batch_size="$3" seed="$4"
  local source="$pilot_root/$name"
  local output="$run_root/$name"
  local log="$run_root/$name.log"
  local pid_file="$run_root/$name.pid"
  test -f "$source/training_complete.json"
  test -f "$source/final_model.zip"
  test -f "$source/vecnormalize.pkl"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_physical.yaml \
      --output "$output" --curriculum table_local \
      --total-timesteps 980000 --num-envs 40 \
      --checkpoint-freq 50000 --eval-freq 25000 --eval-episodes 24 \
      --learning-rate "$learning_rate" --batch-size "$batch_size" \
      --gradient-steps 8 --device cuda --seed "$seed" \
      --resume "$source/final_model.zip" --vec-normalize "$source/vecnormalize.pkl" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size envs=40 gradients=8"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
# Start the slower large-batch branches first so all four overlap for most of
# the wall clock and jointly exercise both the simulator and GPU updater.
launch c_fast_large 0.000050 16384 860442
launch d_slow_large 0.000025 16384 870442
launch main_balanced 0.000025 2048 880442
launch b_conservative 0.000010 2048 890442
