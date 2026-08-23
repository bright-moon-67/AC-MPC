#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_entry_mid_long_20260821/d_slow_wide"
run_root="$project_root/runs/manisoft_waypoint_sac_entry_long_saturated_20260822"
resume="$source_root/final_model.zip"
vec_normalize="$source_root/vecnormalize.pkl"

test -f "$resume"
test -f "$vec_normalize"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 90); do
    kill -0 "$pid" 2>/dev/null || { echo "$name exited" >&2; return 1; }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 40 )); then
      echo "$name initialized with $children children"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch() {
  local name="$1" learning_rate="$2" batch_size="$3" seed="$4"
  local output="$run_root/$name" log="$run_root/$name.log" pid_file="$run_root/$name.pid"
  if [[ -e "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    local existing_pid
    existing_pid="$(<"$pid_file")"
    echo "$name already running: PID=$existing_pid"
    wait_for_workers "$existing_pid" "$name"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "refusing to reuse existing output: $output" >&2
    return 1
  fi
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_physical.yaml \
      --output "$output" --curriculum entry \
      --total-timesteps 700000 --num-envs 40 \
      --checkpoint-freq 50000 --eval-freq 25000 --eval-episodes 24 \
      --learning-rate "$learning_rate" --batch-size "$batch_size" \
      --gradient-steps 8 --device cuda --seed "$seed" \
      --resume "$resume" --vec-normalize "$vec_normalize" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size envs=40 gradients=8"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
# Start the slower large-batch branches first so all four overlap for most of
# the wall clock instead of letting the fast pilot branches finish early.
launch c_large_batch 0.0000050 16384 840442
launch d_slow_large_batch 0.0000025 16384 850442
launch main_stable 0.0000050 2048 820442
launch b_conservative 0.0000025 2048 830442
