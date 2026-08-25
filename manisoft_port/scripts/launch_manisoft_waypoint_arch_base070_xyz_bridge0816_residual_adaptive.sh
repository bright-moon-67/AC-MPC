#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
stage1_run="$project_root/runs/manisoft_waypoint_arch_base070_xyz_stage1_fresh_pilot_v1_20260824"
stage2_run="$project_root/runs/manisoft_waypoint_arch_base070_xyz_stage2_frozen_residual_agile_v2_20260824"
output="$project_root/runs/manisoft_waypoint_arch_base070_xyz_bridge0816_residual_adaptive_v1_20260824"

test ! -e "$output"
mkdir -p "$output"
cd "$project_root"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table_arch_base070.yaml \
  --config configs/manisoft_waypoint_sac_table_arch_base070_xyz_stage3_goal.yaml \
  --output "$output" \
  --curriculum table_waypoint_polyline \
  --total-timesteps 75000 \
  --num-envs 6 \
  --checkpoint-freq 25000 \
  --eval-freq 25000 \
  --eval-episodes 12 \
  --eval-num-envs 4 \
  --eval-panel-seed 965000 \
  --eval-waypoint-single-line-probability 0 \
  --eval-waypoint-segment-count-range 9,13 \
  --eval-waypoint-maximum-turn-degrees 175 \
  --eval-internal-waypoint-capture-radius 0.010 \
  --eval-waypoint-stall-steps 3000 \
  --waypoint-segment-count-range 9,13 \
  --waypoint-segment-length-range 0.080,0.160 \
  --waypoint-maximum-turn-degrees 175 \
  --waypoint-maximum-extent 0.320 \
  --learning-starts 20000 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --learning-rate 0.000002 \
  --ent-coef 0.001 \
  --target-entropy -9 \
  --net-arch 256,256,256 \
  --resume "$stage2_run/checkpoints/sac_table_waypoint_polyline_74988_steps.zip" \
  --vec-normalize "$stage2_run/checkpoints/sac_table_waypoint_polyline_vecnormalize_74988_steps.pkl" \
  --frozen-base-model "$stage1_run/checkpoints/sac_table_waypoint_polyline_24996_steps.zip" \
  --frozen-base-vec-normalize "$stage1_run/checkpoints/sac_table_waypoint_polyline_vecnormalize_24996_steps.pkl" \
  --residual-action-scale 0.050 \
  --residual-action-penalty-scale 0.020 \
  --actor-anchor-coef 30 \
  --source-policy-warmup \
  --actor-learning-delay-steps 10000 \
  --freeze-vec-normalize \
  --device cuda \
  --seed 20264077
