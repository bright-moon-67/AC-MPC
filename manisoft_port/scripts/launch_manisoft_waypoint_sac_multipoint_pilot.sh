#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_range_pilot_20260822/feasible_l008_e035"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_4299864_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4299864_steps.pkl"
run_root="$project_root/runs/manisoft_waypoint_sac_multipoint_pilot_20260822"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 120); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -80 "$run_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 28 )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch() {
  local name="$1" line_probability="$2" anchor="$3" precision="$4" distance_penalty="$5" guard="$6" seed="$7"
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_multipoint_local.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,3 \
      --waypoint-maximum-turn-degrees 90 \
      --waypoint-maximum-extent 0.035 \
      --waypoint-single-line-probability "$line_probability" \
      --eval-waypoint-single-line-probability 0 \
      --terminal-precision-scale "$precision" \
      --terminal-distance-penalty-scale "$distance_penalty" \
      --tracking-guard "$guard" \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 120000 --num-envs 28 \
      --checkpoint-freq 30000 --eval-freq 30000 --eval-episodes 36 \
      --learning-rate 0.00001 --batch-size 2048 \
      --gradient-steps 8 --learning-starts 20000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid line=$line_probability anchor=$anchor precision=$precision"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
test -f "$source_model"
test -f "$source_vecnorm"
launch balanced_gate 0.40 10 2.0 1.0 0.020 2208251
launch more_multipoint 0.25 10 2.0 1.0 0.020 2208252
launch plastic_gate 0.40 5 2.0 1.0 0.020 2208253
launch precision_gate 0.40 10 4.0 2.0 0.018 2208254
