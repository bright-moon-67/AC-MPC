#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_entry_bridge_long_20260821"
resume="$project_root/runs/manisoft_waypoint_sac_entry_bridge_pilot_20260821/e300k_transfer_safety300/final_model.zip"
vec_normalize="$project_root/runs/manisoft_waypoint_sac_entry_bridge_pilot_20260821/e300k_transfer_safety300/vecnormalize.pkl"

mkdir -p "$run_root"

launch() {
  local name="$1"
  local learning_rate="$2"
  local batch_size="$3"
  local gradient_steps="$4"
  local seed="$5"
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

  nohup setsid "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
    --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
    --config configs/manisoft_waypoint_sac_physical.yaml \
    --output "$output" \
    --curriculum entry_bridge \
    --total-timesteps 400000 \
    --num-envs 4 \
    --checkpoint-freq 50000 \
    --eval-freq 25000 \
    --eval-episodes 24 \
    --learning-rate "$learning_rate" \
    --batch-size "$batch_size" \
    --gradient-steps "$gradient_steps" \
    --device cuda \
    --seed "$seed" \
    --resume "$resume" \
    --vec-normalize "$vec_normalize" \
    >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "$name PID=$pid lr=$learning_rate batch=$batch_size gradients=$gradient_steps"
}

cd "$project_root"
launch b_balanced 0.00006 1024 2 50442
launch c_plasticity 0.00012 512 1 60442
launch d_gpu_quality 0.00005 2048 4 70442
