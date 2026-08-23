#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_entry_mid_long_20260821/d_slow_wide"
run_root="$project_root/runs/manisoft_waypoint_sac_entry_long_20260822"
resume="$source_root/final_model.zip"
vec_normalize="$source_root/vecnormalize.pkl"

test -f "$resume"
test -f "$vec_normalize"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 45); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited during initialization" >&2
      return 1
    fi
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 20 )); then
      echo "$name initialized with $children child processes"
      return 0
    fi
    sleep 1
  done
  echo "$name did not create 20 environments within 45 seconds" >&2
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
      --total-timesteps 700000 --num-envs 20 \
      --checkpoint-freq 50000 --eval-freq 25000 --eval-episodes 24 \
      --learning-rate "$learning_rate" --batch-size "$batch_size" \
      --gradient-steps 4 --device cuda --seed "$seed" \
      --resume "$resume" --vec-normalize "$vec_normalize" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size envs=20 gradients=4"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch main_stable 0.0000050 2048 620442
launch b_conservative 0.0000025 2048 630442
launch c_large_batch 0.0000050 16384 640442
launch d_slow_large_batch 0.0000025 16384 650442
