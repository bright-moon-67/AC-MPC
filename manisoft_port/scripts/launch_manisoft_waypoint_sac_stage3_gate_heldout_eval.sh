#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_root="$project_root/runs/manisoft_waypoint_sac_stage3_turn_pilot_20260823/balanced45_a20"
pilot_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100"
eval_root="$project_root/runs/manisoft_waypoint_sac_stage3_gate_heldout_eval_20260823"

test ! -e "$eval_root"
mkdir -p "$eval_root"

launch_model() {
  local name="$1" model="$2" vecnorm="$3" run_config="$4"
  mkdir -p "$eval_root/$name"
  for chunk in $(seq 0 3); do
    local seed=$((864000 + chunk * 48))
    nohup setsid env \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" --run-config "$run_config" \
        --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$eval_root/$name/chunk_$chunk.npz" --episodes 48 \
        --families waypoint_polyline --curriculum table_waypoint_polyline \
        --waypoint-maximum-extent 0.035 \
        --waypoint-segment-count-range 3,3 \
        --waypoint-maximum-turn-degrees 60 \
        --device cpu --seed "$seed" \
        >"$eval_root/$name/chunk_$chunk.log" 2>&1 </dev/null &
    echo "$!" >"$eval_root/$name/chunk_$chunk.pid"
  done
}

cd "$project_root"
launch_model source \
  "$source_root/best/best_model.zip" \
  "$source_root/vecnormalize.pkl" \
  "$source_root/run_config.json"
launch_model step90k \
  "$pilot_root/checkpoints/sac_table_waypoint_polyline_4879664_steps.zip" \
  "$pilot_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_4879664_steps.pkl" \
  "$pilot_root/run_config.json"
launch_model step120k \
  "$pilot_root/final_model.zip" \
  "$pilot_root/vecnormalize.pkl" \
  "$pilot_root/run_config.json"

echo "launched 12 gate-aware held-out evaluation workers under $eval_root"
