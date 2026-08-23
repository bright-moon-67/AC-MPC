#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100"
source_step="4879664"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_${source_step}_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_${source_step}_steps.pkl"
run_root="$project_root/runs/manisoft_waypoint_sac_stage3_prior_learning_pilot_20260823"

test -f "$source_model"
test -f "$source_vecnorm"
test ! -e "$run_root"
mkdir -p "$run_root"

wait_for_workers() {
  local pid="$1" name="$2" minimum="$3"
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || {
      echo "$name exited during initialization" >&2
      tail -120 "$run_root/$name.log" >&2 || true
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
  local name="$1" seed="$2" weight="$3" anchor="$4"
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
      --cartesian-prior-weight "$weight" \
      --cartesian-prior-proportional-gain 20 \
      --cartesian-prior-feedforward-scale 1 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-panel-seed 868000 --eval-num-envs 4 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 90000 --num-envs 20 \
      --checkpoint-freq 30000 --eval-freq 30000 --eval-episodes 48 \
      --learning-rate 0.000001 --batch-size 2048 \
      --gradient-steps 8 --learning-starts 10000 \
      --actor-anchor-coef "$anchor" \
      --device cuda --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid prior=$weight anchor=$anchor"
  wait_for_workers "$pid" "$name" 24
}

launch_source_panel() {
  local name="$1" weight="$2" seed="$3"
  local output="$run_root/$name"
  nohup setsid env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
      --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
      --output "$output" --curriculum table_waypoint_polyline \
      --waypoint-maximum-turn-degrees 60 \
      --internal-waypoint-capture-radius 0.010 \
      --cartesian-prior-weight "$weight" \
      --cartesian-prior-proportional-gain 20 \
      --cartesian-prior-feedforward-scale 1 \
      --eval-waypoint-single-line-probability 0 \
      --eval-waypoint-segment-count-range 3,3 \
      --eval-waypoint-maximum-turn-degrees 60 \
      --eval-internal-waypoint-capture-radius 0.010 \
      --eval-panel-seed 868000 --eval-num-envs 8 \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --total-timesteps 1 --num-envs 1 \
      --checkpoint-freq 1 --eval-freq 1 --eval-episodes 48 \
      --learning-starts 100 --device cuda --seed "$seed" \
      >"$run_root/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$run_root/$name.pid"
  echo "$name PID=$pid prior=$weight"
  wait_for_workers "$pid" "$name" 9
}

cd "$project_root"
launch_train prior40_a100 2308301 0.40 100
launch_train prior40_a20 2308302 0.40 20
launch_train prior40_a0 2308303 0.40 0
launch_train prior60_a20 2308304 0.60 20
launch_source_panel source_prior40 0.40 2308305
launch_source_panel source_prior60 0.60 2308306

echo "launched four prior-learning pilots and two matched source panels under $run_root"
