#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
geometric_root="$project_root/runs/manisoft_waypoint_sac_range_pilot_20260822/feasible_l008_e035"
gated_root="$project_root/runs/manisoft_waypoint_sac_multipoint_pilot_20260822/plastic_gate"
run_root="$project_root/runs/manisoft_waypoint_sac_multipoint_long_20260823"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2"
  for _ in $(seq 1 180); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -100 "$run_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= 48 )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch() {
  local name="$1" seed="$2" source_model="$3" source_vecnorm="$4" anchor="$5"
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_multipoint_long.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,4 \
      --waypoint-maximum-turn-degrees 90 \
      --waypoint-maximum-extent 0.035 \
      --waypoint-single-line-probability 0.35 \
      --eval-waypoint-single-line-probability 0 \
      --resume "$source_model" \
      --vec-normalize "$source_vecnorm" \
      --total-timesteps 500000 --num-envs 48 \
      --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 48 \
      --learning-rate 0.000005 --batch-size 4096 \
      --gradient-steps 12 --learning-starts 30000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid seed=$seed"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
geometric_model="$geometric_root/checkpoints/sac_table_waypoint_polyline_4299864_steps.zip"
geometric_vecnorm="$geometric_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4299864_steps.pkl"
gated_model="$gated_root/final_model.zip"
gated_vecnorm="$gated_root/vecnormalize.pkl"
test -f "$geometric_model"
test -f "$geometric_vecnorm"
test -f "$gated_model"
test -f "$gated_vecnorm"
launch source_seed 2308231 "$geometric_model" "$geometric_vecnorm" 3
launch gated_seed 2308232 "$gated_model" "$gated_vecnorm" 5
