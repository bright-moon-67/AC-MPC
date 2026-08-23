#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_turn_pilot_20260823/balanced45_a20"
run_root="$project_root/runs/manisoft_waypoint_sac_stage3_long60_20260823"
output="$run_root/main_a30_lr1e6"
log="$run_root/main_a30_lr1e6.log"
source_model="$source_root/best/best_model.zip"
source_vecnorm="$source_root/vecnormalize.pkl"

test -f "$source_model"
test -f "$source_vecnorm"
test ! -e "$output"
mkdir -p "$run_root"

cd "$project_root"
nohup setsid env \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
    --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
    --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
    --output "$output" --curriculum table_waypoint_polyline \
    --waypoint-segment-count-range 2,3 \
    --waypoint-segment-count-probabilities 0.20,0.80 \
    --waypoint-maximum-turn-degrees 60 \
    --waypoint-maximum-extent 0.035 \
    --waypoint-single-line-probability 0.20 \
    --internal-waypoint-capture-radius 0.010 \
    --eval-waypoint-single-line-probability 0 \
    --eval-waypoint-segment-count-range 3,3 \
    --eval-waypoint-maximum-turn-degrees 60 \
    --eval-internal-waypoint-capture-radius 0.010 \
    --eval-panel-seed 860000 --eval-num-envs 8 \
    --resume "$source_model" --vec-normalize "$source_vecnorm" \
    --total-timesteps 500000 --num-envs 64 \
    --checkpoint-freq 50000 --eval-freq 50000 --eval-episodes 96 \
    --learning-rate 0.000001 --batch-size 4096 \
    --gradient-steps 16 --learning-starts 30000 \
    --actor-anchor-coef 30 \
    --device cuda --seed 2308261 \
    >"$log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$run_root/main_a30_lr1e6.pid"
echo "stage3 long60 PID=$pid"

for _ in $(seq 1 240); do
  kill -0 "$pid" 2>/dev/null || {
    echo "long60 exited during initialization" >&2
    tail -120 "$log" >&2 || true
    exit 1
  }
  children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
  if (( children >= 72 )); then
    echo "long60 initialized with $children workers"
    exit 0
  fi
  sleep 1
done

echo "long60 worker initialization timed out" >&2
exit 1
