#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run_root="$project_root/runs/manisoft_waypoint_sac_long10cm_pilot_v2_20260823"
scenario="/root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml"

variants=(r015_p005 r030_p005 r030_p020 r050_p020)
residual_scales=(0.015 0.030 0.030 0.050)
penalty_scales=(0.005 0.005 0.020 0.020)
seeds=(20268101 20268102 20268103 20268104)

cd "$project_root"
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
      --seed "${seeds[$index]}" --eval-panel-seed 995000 \
      --total-timesteps 50000 --num-envs 4 \
      --checkpoint-freq 10000 --eval-freq 10000 \
      --eval-episodes 12 --eval-num-envs 2 \
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

echo "launched ${#variants[@]} detached long-10cm SAC pilots under $run_root"
