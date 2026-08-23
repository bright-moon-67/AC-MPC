#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_multipoint_long_20260823/gated_seed"
source_step="4669712"
run_root="$project_root/runs/manisoft_waypoint_sac_stage3_turn_pilot_20260823"
mkdir -p "$run_root"

source_model="$source_root/checkpoints/sac_table_waypoint_polyline_${source_step}_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_${source_step}_steps.pkl"
test -f "$source_model"
test -f "$source_vecnorm"

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
    if (( children >= 32 )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch() {
  local name="$1" seed="$2" train_turn="$3" anchor="$4"
  local output="$run_root/$name" log="$run_root/$name.log"
  test ! -e "$output"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-segment-count-range 2,3 \
      --waypoint-segment-count-probabilities 0.20,0.80 \
      --waypoint-maximum-turn-degrees "$train_turn" \
      --waypoint-maximum-extent 0.035 \
      --waypoint-single-line-probability 0.20 \
      --internal-waypoint-capture-radius 0.010 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 45 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-panel-seed 840000 --eval-num-envs 4 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 120000 --num-envs 28 \
      --checkpoint-freq 30000 --eval-freq 30000 --eval-episodes 48 \
      --learning-rate 0.000002 --batch-size 2048 \
      --gradient-steps 8 --learning-starts 10000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid seed=$seed train_turn=$train_turn anchor=$anchor"
  wait_for_workers "$pid" "$name"
}

cd "$project_root"
launch easy30_a20 2308251 30 20
launch balanced45_a20 2308252 45 20
launch balanced45_a100 2308253 45 100
launch broad60_a100 2308254 60 100

echo "launched four parallel-eval turn-curriculum pilots under $run_root"
