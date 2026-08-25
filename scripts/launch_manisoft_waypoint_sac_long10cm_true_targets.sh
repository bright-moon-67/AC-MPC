#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
scenario="/root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml"
source_root="${SOURCE_ROOT:-$project_root/runs/manisoft_waypoint_sac_long10cm_pilot_v2_20260823/r015_p005}"
run_root="${RUN_ROOT:-$project_root/runs/manisoft_waypoint_sac_long10cm_true_targets_20260823}"
source_model="$source_root/checkpoints/sac_table_long_waypoints_50000_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_long_waypoints_vecnormalize_50000_steps.pkl"

variants=(r005_p005 r010_p005 r015_p020 r020_p020)
residual_scales=(0.005 0.010 0.015 0.020)
penalty_scales=(0.005 0.005 0.020 0.020)
seeds=(20268201 20268202 20268203 20268204)

cd "$project_root"
test -f "$source_model"
test -f "$source_vecnorm"
mkdir -p "$run_root"

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  output="$run_root/$variant"
  test ! -e "$output"
  mkdir -p "$output"
  nohup setsid env \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
      --config configs/manisoft_waypoint_sac_table_long10cm.yaml \
      --scenario "$scenario" --output "$output" \
      --allow-existing-output \
      --curriculum table_long_waypoints --device cuda \
      --resume "$source_model" --vec-normalize "$source_vecnorm" \
      --seed "${seeds[$index]}" --eval-panel-seed 996500 \
      --total-timesteps 30000 --num-envs 4 \
      --checkpoint-freq 10000 --eval-freq 10000 \
      --eval-episodes 15 --eval-num-envs 3 \
      --learning-rate 0.00001 --learning-starts 4000 \
      --batch-size 512 --gradient-steps 2 \
      --ent-coef 0.005 --target-entropy -2.0 \
      --equilibrium-path-prior-weight 1.0 \
      --equilibrium-path-residual-scale "${residual_scales[$index]}" \
      --policy-action-penalty-scale "${penalty_scales[$index]}" \
      >"$output/train.log" 2>&1 </dev/null &
  pid=$!
  echo "$pid" >"$output/train.pid"
  echo "$variant pid=$pid residual=${residual_scales[$index]} penalty=${penalty_scales[$index]}"
done

echo "launched ${#variants[@]} detached true-target SAC refinements under $run_root"
