#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
model_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100"
model="$model_root/checkpoints/sac_table_waypoint_polyline_4879664_steps.zip"
vecnorm="$model_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4879664_steps.pkl"
eval_root="$project_root/runs/manisoft_waypoint_sac_stage3_prior_gain_heldout_20260823"

test -f "$model"
test -f "$vecnorm"
test ! -e "$eval_root"
mkdir -p "$eval_root"

launch_variant() {
  local name="$1" weight="$2" gain="$3"
  mkdir -p "$eval_root/$name"
  for chunk in $(seq 0 3); do
    local seed=$((869000 + chunk * 48))
    nohup setsid env \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" \
        --run-config "$model_root/run_config.json" \
        --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$eval_root/$name/chunk_$chunk.npz" --episodes 48 \
        --families waypoint_polyline --curriculum table_waypoint_polyline \
        --waypoint-maximum-extent 0.035 \
        --waypoint-segment-count-range 3,3 \
        --waypoint-maximum-turn-degrees 60 \
        --cartesian-prior-weight "$weight" \
        --cartesian-prior-proportional-gain "$gain" \
        --cartesian-prior-feedforward-scale 1 \
        --successful-only --device cpu --seed "$seed" \
        >"$eval_root/$name/chunk_$chunk.log" 2>&1 </dev/null &
    echo "$!" >"$eval_root/$name/chunk_$chunk.pid"
  done
}

cd "$project_root"
launch_variant pure_sac 0.00 0
launch_variant w40_kp20 0.40 20
launch_variant w40_kp30 0.40 30
launch_variant w50_kp20 0.50 20
launch_variant w50_kp30 0.50 30
launch_variant w60_kp20 0.60 20
launch_variant w60_kp30 0.60 30

echo "launched 28 fresh held-out prior-gain workers under $eval_root"
