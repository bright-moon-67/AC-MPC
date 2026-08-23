#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
eval_root="$project_root/runs/manisoft_waypoint_sac_stage3_source_eval_20260823"
pilot_root="$project_root/runs/manisoft_waypoint_sac_multipoint_pilot_20260822/plastic_gate"
long_root="$project_root/runs/manisoft_waypoint_sac_multipoint_long_20260823"
mkdir -p "$eval_root"

launch_model() {
  local name="$1" model="$2" vecnorm="$3" run_config="$4"
  mkdir -p "$eval_root/$name"
  for chunk in $(seq 0 5); do
    local seed=$((6242 + chunk * 12))
    nohup setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" --run-config "$run_config" \
        --config configs/manisoft_waypoint_sac_table_multipoint_local.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$eval_root/$name/chunk_$chunk.npz" --episodes 12 \
        --families waypoint_polyline --curriculum table_waypoint_polyline \
        --waypoint-maximum-extent 0.035 \
        --waypoint-segment-count-range 3,3 \
        --waypoint-maximum-turn-degrees 90 \
        --device cpu --seed "$seed" \
        >"$eval_root/$name/chunk_$chunk.log" 2>&1 </dev/null &
    echo "$!" >"$eval_root/$name/chunk_$chunk.pid"
  done
}

cd "$project_root"
launch_model plastic_final \
  "$pilot_root/final_model.zip" \
  "$pilot_root/vecnormalize.pkl" \
  "$pilot_root/run_config.json"
launch_model gated_250k \
  "$long_root/gated_seed/checkpoints/sac_table_waypoint_polyline_4669712_steps.zip" \
  "$long_root/gated_seed/checkpoints/sac_table_waypoint_polyline_vecnormalize_4669712_steps.pkl" \
  "$long_root/gated_seed/run_config.json"
launch_model gated_450k \
  "$long_root/gated_seed/checkpoints/sac_table_waypoint_polyline_4869584_steps.zip" \
  "$long_root/gated_seed/checkpoints/sac_table_waypoint_polyline_vecnormalize_4869584_steps.pkl" \
  "$long_root/gated_seed/run_config.json"
launch_model source_300k \
  "$long_root/source_seed/checkpoints/sac_table_waypoint_polyline_4599672_steps.zip" \
  "$long_root/source_seed/checkpoints/sac_table_waypoint_polyline_vecnormalize_4599672_steps.pkl" \
  "$long_root/source_seed/run_config.json"

echo "launched 24 fixed three-segment evaluation workers under $eval_root"
