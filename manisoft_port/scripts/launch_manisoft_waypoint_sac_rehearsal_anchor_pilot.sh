#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_table_cartesian_long_20260822/c_fast_auto"
run_root="$project_root/runs/manisoft_waypoint_sac_rehearsal_anchor_pilot_20260822"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2" expected="$3"
  for _ in $(seq 1 120); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -80 "$run_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= expected )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch() {
  local name="$1" anchor_coef="$2" replay_probability="$3" seed="$4"
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_waypoint_rehearsal.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,2 \
      --waypoint-maximum-turn-degrees 90 \
      --waypoint-single-line-probability "$replay_probability" \
      --eval-waypoint-single-line-probability 0 \
      --resume "$source_root/final_model.zip" \
      --vec-normalize "$source_root/vecnormalize.pkl" \
      --total-timesteps 150000 --num-envs 28 \
      --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 24 \
      --learning-rate 0.00002 --batch-size 2048 \
      --gradient-steps 8 --learning-starts 30000 \
      --actor-anchor-coef "$anchor_coef" \
      --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid anchor=$anchor_coef line_replay=$replay_probability"
  wait_for_workers "$pid" "$name" 28
}

cd "$project_root"
test -f "$source_root/final_model.zip"
test -f "$source_root/vecnormalize.pkl"
launch anchor10_r70 10 0.70 2208221
launch anchor30_r70 30 0.70 2208222
launch anchor10_r50 10 0.50 2208223
launch no_anchor_r70 0 0.70 2208224
