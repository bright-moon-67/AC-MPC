#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100"
source_step="4879664"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_${source_step}_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_${source_step}_steps.pkl"
run_root="$project_root/runs/manisoft_waypoint_sac_stage3_residual_pilot_20260823"

test -f "$source_model"
test -f "$source_vecnorm"
test ! -e "$run_root"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2" minimum="$3"
  for _ in $(seq 1 300); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -160 "$run_root/$name.log" >&2 || true
      return 1
    }
    local children
    children="$(pgrep -P "$pid" 2>/dev/null | wc -l || true)"
    if (( children >= minimum )); then
      echo "$name initialized with $children workers"
      return 0
    fi
    sleep 1
  done
  echo "$name worker initialization timed out" >&2
  return 1
}

launch_train() {
  local name="$1" seed="$2" residual_scale="$3" learning_rate="$4" penalty="$5"
  local output="$run_root/$name" log="$run_root/$name.log"
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
      --internal-waypoint-bonus 1 \
      --entry-sampling-weights 0.08,0.14,0.08,0.20,0.22,0.28 \
      --cartesian-prior-weight 0.40 \
      --cartesian-prior-proportional-gain 20 \
      --cartesian-prior-feedforward-scale 1 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-panel-seed 870000 --eval-num-envs 4 \
      --frozen-base-model "$source_model" \
      --frozen-base-vec-normalize "$source_vecnorm" \
      --residual-action-scale "$residual_scale" \
      --residual-action-penalty-scale "$penalty" \
      --total-timesteps 90000 --num-envs 10 \
      --checkpoint-freq 30000 --eval-freq 30000 --eval-episodes 48 \
      --learning-rate "$learning_rate" --batch-size 1024 \
      --gradient-steps 8 --learning-starts 10000 \
      --ent-coef auto_0.02 --target-entropy -1 \
      --net-arch 256,256 --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid scale=$residual_scale lr=$learning_rate penalty=$penalty"
  wait_for_workers "$pid" "$name" 14
}

cd "$project_root"
launch_train residual05_lr3e5 2308241 0.05 0.00003 0.005
launch_train residual10_lr3e5 2308242 0.10 0.00003 0.010
launch_train residual10_lr1e4 2308243 0.10 0.00010 0.010
launch_train residual15_lr3e5 2308244 0.15 0.00003 0.020

echo "launched four frozen-base residual SAC pilots under $run_root"
