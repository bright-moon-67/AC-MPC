#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_entry_long_saturated_20260822/b_conservative"
run_root="$project_root/runs/manisoft_waypoint_sac_table_local_pilot_20260822"
resume="$source_root/final_model.zip"
vec_normalize="$source_root/vecnormalize.pkl"

test -f "$resume"
test -f "$vec_normalize"
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
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_physical.yaml \
      --output "$output" --curriculum table_local \
      --total-timesteps 20000 --num-envs 40 \
      --checkpoint-freq 10000 --eval-freq 10000 --eval-episodes 12 \
      --learning-rate "$learning_rate" --batch-size "$batch_size" \
      --gradient-steps 8 --device cuda --seed "$seed" \
      --resume "$resume" --vec-normalize "$vec_normalize" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch c_fast_large 0.000050 16384 860442
launch d_slow_large 0.000025 16384 870442
launch main_balanced 0.000025 2048 880442
launch b_conservative 0.000010 2048 890442
