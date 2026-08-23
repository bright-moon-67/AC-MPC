#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
eval_root="$project_root/runs/manisoft_waypoint_sac_rehearsal_standard_eval_20260822"
source_root="$project_root/runs/manisoft_waypoint_sac_table_cartesian_long_20260822/c_fast_auto"
pilot_root="$project_root/runs/manisoft_waypoint_sac_rehearsal_anchor_pilot_20260822"
mkdir -p "$eval_root"

launch_model() {
  local name="$1" model="$2" vecnorm="$3" run_config="$4"
  mkdir -p "$eval_root/$name"
  for chunk in $(seq 0 5); do
    local seed=$((4242 + chunk * 18))
    local output="$eval_root/$name/chunk_$chunk.npz"
    local log="$eval_root/$name/chunk_$chunk.log"
    test ! -e "$output"
    nohup setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" \
        --run-config "$run_config" \
        --config configs/manisoft_waypoint_sac_table_waypoint_rehearsal.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$output" --episodes 18 \
        --families waypoint_polyline \
        --curriculum table_waypoint_polyline \
        --device cpu --seed "$seed" \
        >"$log" 2>&1 </dev/null &
    echo "$!" >"$eval_root/$name/chunk_$chunk.pid"
  done
}

cd "$project_root"
launch_model source \
  "$source_root/final_model.zip" \
  "$source_root/vecnormalize.pkl" \
  "$source_root/run_config.json"
launch_model anchor10_r50_final \
  "$pilot_root/anchor10_r50/final_model.zip" \
  "$pilot_root/anchor10_r50/vecnormalize.pkl" \
  "$pilot_root/anchor10_r50/run_config.json"
launch_model anchor10_r70_100k \
  "$pilot_root/anchor10_r70/checkpoints/sac_table_waypoint_polyline_4199960_steps.zip" \
  "$pilot_root/anchor10_r70/checkpoints/sac_table_waypoint_polyline_vecnormalize_4199960_steps.pkl" \
  "$pilot_root/anchor10_r70/run_config.json"

echo "launched 18 standardized evaluation workers under $eval_root"
