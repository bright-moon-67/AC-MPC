#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_entry_bridge_long_20260821"
output="$run_root/main"
log="$run_root/main.log"
pid_file="$run_root/main.pid"
resume="$project_root/runs/manisoft_waypoint_sac_entry_bridge_pilot_20260821/e300k_transfer_safety300/final_model.zip"
vec_normalize="$project_root/runs/manisoft_waypoint_sac_entry_bridge_pilot_20260821/e300k_transfer_safety300/vecnormalize.pkl"

mkdir -p "$run_root"
if [[ -e "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
  echo "entry_bridge long training is already running: PID $(<"$pid_file")"
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to reuse existing output: $output" >&2
  exit 1
fi

cd "$project_root"
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
  --learning-rate 0.00008 \
  --batch-size 512 \
  --gradient-steps 1 \
  --device cuda \
  --seed 40442 \
  --resume "$resume" \
  --vec-normalize "$vec_normalize" \
  >"$log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$pid_file"
echo "started PID $pid"
echo "log: $log"
echo "output: $output"
