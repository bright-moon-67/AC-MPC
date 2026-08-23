#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_turn_pilot_20260823/balanced45_a20"
long_root="$project_root/runs/manisoft_waypoint_sac_stage3_long60_20260823/main_a30_lr1e6"
eval_root="$project_root/runs/manisoft_waypoint_sac_stage3_long60_heldout_eval_20260823"

test ! -e "$eval_root"
mkdir -p "$eval_root"

launch_model() {
  local name="$1" model="$2" vecnorm="$3" run_config="$4"
  local model_root="$eval_root/$name"
  mkdir -p "$model_root"
  for chunk in $(seq 0 3); do
    local seed=$((861000 + chunk * 48))
    local output="$model_root/chunk_$chunk.npz"
    local log="$model_root/chunk_$chunk.log"
    nohup setsid env \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" --run-config "$run_config" \
        --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$output" --episodes 48 \
        --families waypoint_polyline --curriculum table_waypoint_polyline \
        --waypoint-maximum-extent 0.035 \
        --waypoint-segment-count-range 3,3 \
        --waypoint-maximum-turn-degrees 60 \
        --device cpu --seed "$seed" \
        >"$log" 2>&1 </dev/null &
    echo "$!" >"$model_root/chunk_$chunk.pid"
  done
}

cd "$project_root"

launch_model source \
  "$source_root/best/best_model.zip" \
  "$source_root/vecnormalize.pkl" \
  "$source_root/run_config.json"

for spec in \
  "step150k:4939616" \
  "step200k:4989600" \
  "step300k:5089568"; do
  name="${spec%%:*}"
  step="${spec##*:}"
  launch_model "$name" \
    "$long_root/checkpoints/sac_table_waypoint_polyline_${step}_steps.zip" \
    "$long_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_${step}_steps.pkl" \
    "$long_root/run_config.json"
done

echo "launched 16 held-out evaluation workers under $eval_root"
