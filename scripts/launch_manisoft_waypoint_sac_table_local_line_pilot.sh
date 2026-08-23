#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_table_cartesian_pilot_v3_20260822"
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
  local name="$1" learning_rate="$2" batch_size="$3" ent_coef="$4" target_entropy="$5" seed="$6"
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_local_line.yaml \
      --output "$output" --curriculum table_local_line \
      --total-timesteps 100000 --num-envs 40 \
      --checkpoint-freq 25000 --eval-freq 25000 --eval-episodes 48 \
      --learning-rate "$learning_rate" --batch-size "$batch_size" \
      --gradient-steps 8 --ent-coef "$ent_coef" \
      --target-entropy "$target_entropy" --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size entropy=$ent_coef/$target_entropy"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch d_fixed_large 0.00010 8192 0.02 -1.0 910442
launch c_fast_auto 0.00020 2048 auto_0.05 -1.0 900442
launch main_auto 0.00010 2048 auto_0.05 -1.0 920442
launch b_low_entropy 0.00005 2048 auto_0.02 -0.5 930442
