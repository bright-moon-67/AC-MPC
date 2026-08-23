#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
eval_root="$project_root/runs/manisoft_waypoint_sac_e035_standard_eval_20260822"
source_root="$project_root/runs/manisoft_waypoint_sac_rehearsal_anchor_pilot_20260822/anchor10_r70"
range_root="$project_root/runs/manisoft_waypoint_sac_range_pilot_20260822/feasible_l008_e035"
mkdir -p "$eval_root"

launch_model() {
  local name="$1" model="$2" vecnorm="$3" run_config="$4"
  mkdir -p "$eval_root/$name"
  for chunk in $(seq 0 5); do
    local seed=$((4242 + chunk * 18))
    nohup setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" --run-config "$run_config" \
        --config configs/manisoft_waypoint_sac_table_waypoint_rehearsal.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$eval_root/$name/chunk_$chunk.npz" --episodes 18 \
        --families waypoint_polyline --curriculum table_waypoint_polyline \
        --waypoint-maximum-extent 0.035 \
        --waypoint-segment-count-range 2,2 \
        --waypoint-maximum-turn-degrees 90 \
        --device cpu --seed "$seed" \
        >"$eval_root/$name/chunk_$chunk.log" 2>&1 </dev/null &
    echo "$!" >"$eval_root/$name/chunk_$chunk.pid"
  done
}

cd "$project_root"
launch_model source_100k \
  "$source_root/checkpoints/sac_table_waypoint_polyline_4199960_steps.zip" \
  "$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4199960_steps.pkl" \
  "$source_root/run_config.json"
for step in 4249912 4274888 4299864; do
  launch_model "feasible_$step" \
    "$range_root/checkpoints/sac_table_waypoint_polyline_${step}_steps.zip" \
    "$range_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_${step}_steps.pkl" \
    "$range_root/run_config.json"
done

echo "launched 24 paired e035 evaluation workers under $eval_root"
