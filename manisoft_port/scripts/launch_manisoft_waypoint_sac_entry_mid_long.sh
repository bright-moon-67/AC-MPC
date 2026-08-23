#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_entry_mid_long_20260821"
source_root="$project_root/runs/manisoft_waypoint_sac_entry_mid_pilot_20260821/anchor75_lr1e5_env8_g1_b1024"
resume="$source_root/final_model.zip"
vec_normalize="$source_root/vecnormalize.pkl"

test -f "$resume"
test -f "$vec_normalize"
mkdir -p "$run_root"

launch() {
  local name="$1"
  local learning_rate="$2"
  local batch_size="$3"
  local seed="$4"
  local output="$run_root/$name"
  local log="$run_root/$name.log"
  local pid_file="$run_root/$name.pid"

  if [[ -e "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    echo "$name already running: PID $(<"$pid_file")" >&2
    return 1
  fi
  if [[ -e "$output" ]]; then
    echo "refusing to reuse existing output: $output" >&2
    return 1
  fi

  nohup setsid env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_physical.yaml \
      --output "$output" \
      --curriculum entry_mid \
      --total-timesteps 700000 \
      --num-envs 8 \
      --checkpoint-freq 50000 \
      --eval-freq 25000 \
      --eval-episodes 24 \
      --learning-rate "$learning_rate" \
      --batch-size "$batch_size" \
      --gradient-steps 1 \
      --device cuda \
      --seed "$seed" \
      --resume "$resume" \
      --vec-normalize "$vec_normalize" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size envs=8 gradients=1"
}

cd "$project_root"
launch main_stable 0.00001 1024 120442
launch b_conservative 0.000005 1024 130442
launch c_wide_batch 0.00001 2048 140442
launch d_slow_wide 0.000005 2048 150442
