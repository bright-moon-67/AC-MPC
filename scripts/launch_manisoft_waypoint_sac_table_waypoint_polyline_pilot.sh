#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_table_cartesian_long_20260822/c_fast_auto"
run_root="$project_root/runs/manisoft_waypoint_sac_table_waypoint_polyline_pilot_20260822"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 120); do
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
  local name="$1" learning_rate="$2" batch_size="$3" gradient_steps="$4" seed="$5"
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_waypoint_polyline.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,3 \
      --resume "$source_root/final_model.zip" \
      --vec-normalize "$source_root/vecnormalize.pkl" \
      --total-timesteps 300000 --num-envs 40 \
      --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 48 \
      --learning-rate "$learning_rate" --batch-size "$batch_size" \
      --gradient-steps "$gradient_steps" --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size gradients=$gradient_steps"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
test -f "$source_root/final_model.zip"
test -f "$source_root/vecnormalize.pkl"
launch main_auto 0.00010 2048 8 1100442
launch fast_auto 0.00020 2048 8 1110442
launch conservative 0.00005 2048 8 1120442
launch update_heavy 0.00010 4096 16 1130442
