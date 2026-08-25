#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_stage35_potential_pilot_20260823"
checkpoint_steps="${1:-4889664}"
eval_name="${2:-eval10k}"
variants=(control_p0 potential_p15 potential_p25 potential_p25_stall250)

cd "$project_root"
for variant in "${variants[@]}"; do
  model_root="$run_root/$variant"
  model="$model_root/checkpoints/sac_table_waypoint_polyline_${checkpoint_steps}_steps.zip"
  vecnorm="$model_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_${checkpoint_steps}_steps.pkl"
  output_root="$run_root/$eval_name/$variant"
  test -f "$model"
  test -f "$vecnorm"
  test ! -e "$output_root"
  mkdir -p "$output_root"
  for chunk in 0 1 2 3; do
    seed=$((871000 + chunk * 12))
    nohup setsid env \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
        --model "$model" --vec-normalize "$vecnorm" \
        --run-config "$model_root/run_config.json" \
        --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
        --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
        --output "$output_root/chunk_$chunk.npz" --episodes 12 \
        --families waypoint_polyline --curriculum table_waypoint_polyline \
        --waypoint-maximum-extent 0.035 \
        --waypoint-segment-count-range 3,3 \
        --waypoint-maximum-turn-degrees 60 \
        --successful-only --device cpu --seed "$seed" \
        >"$output_root/chunk_$chunk.log" 2>&1 </dev/null &
    echo "$!" >"$output_root/chunk_$chunk.pid"
  done
done

echo "launched 16 matched-seed evaluation workers under $run_root/$eval_name"
