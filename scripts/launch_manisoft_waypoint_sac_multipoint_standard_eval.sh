#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
eval_root="$project_root/runs/manisoft_waypoint_sac_multipoint_standard_eval_20260823"
source_root="$project_root/runs/manisoft_waypoint_sac_range_pilot_20260822/feasible_l008_e035"
candidate_root="$project_root/runs/manisoft_waypoint_sac_multipoint_pilot_20260822/plastic_gate"
mkdir -p "$eval_root"

launch_model() {
  local name="$1" model="$2" vecnorm="$3" run_config="$4"
  mkdir -p "$eval_root/$name"
  for chunk in $(seq 0 5); do
    local seed=$((5242 + chunk * 18))
    nohup setsid env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" --run-config "$run_config" \
        --config configs/manisoft_waypoint_sac_table_multipoint_local.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$eval_root/$name/chunk_$chunk.npz" --episodes 18 \
        --families waypoint_polyline --curriculum table_waypoint_polyline \
        --waypoint-maximum-extent 0.035 \
        --waypoint-segment-count-range 2,3 \
        --waypoint-maximum-turn-degrees 90 \
        --device cpu --seed "$seed" \
        >"$eval_root/$name/chunk_$chunk.log" 2>&1 </dev/null &
    echo "$!" >"$eval_root/$name/chunk_$chunk.pid"
  done
}

cd "$project_root"
launch_model source \
  "$source_root/checkpoints/sac_table_waypoint_polyline_4299864_steps.zip" \
  "$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4299864_steps.pkl" \
  "$source_root/run_config.json"
launch_model plastic_final \
  "$candidate_root/final_model.zip" \
  "$candidate_root/vecnormalize.pkl" \
  "$candidate_root/run_config.json"

echo "launched 12 paired multi-point evaluation workers under $eval_root"
