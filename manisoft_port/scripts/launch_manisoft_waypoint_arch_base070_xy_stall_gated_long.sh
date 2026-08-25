#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
stage1_run="$project_root/runs/manisoft_waypoint_arch_base070_xyz_stage1_fresh_pilot_v1_20260824"
output="$project_root/runs/manisoft_waypoint_arch_base070_xy_stall_gated_long_v1_20260824"

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
  --config configs/manisoft_waypoint_sac_table_arch_base070_xy_tolerant_generalization.yaml \
  --output "$output" \
  --total-timesteps 300000 \
  --num-envs 12 \
  --checkpoint-freq 25000 \
  --eval-freq 50000 \
  --eval-episodes 12 \
  --eval-num-envs 4 \
  --eval-panel-seed 965000 \
  --eval-waypoint-single-line-probability 0 \
  --eval-waypoint-hard-turn-probability 0 \
  --eval-waypoint-segment-count-range 9,13 \
  --eval-waypoint-minimum-turn-degrees 0 \
  --eval-waypoint-maximum-turn-degrees 175 \
  --eval-internal-waypoint-capture-radius 0.010 \
  --eval-waypoint-stall-steps 3000 \
  --learning-starts 20000 \
  --batch-size 2048 \
  --gradient-steps 4 \
  --learning-rate 0.0000015 \
  --ent-coef 0.001 \
  --target-entropy -9 \
  --net-arch 256,256,256 \
  --vec-normalize "$stage1_run/checkpoints/sac_table_waypoint_polyline_vecnormalize_24996_steps.pkl" \
  --frozen-base-model "$stage1_run/checkpoints/sac_table_waypoint_polyline_24996_steps.zip" \
  --frozen-base-vec-normalize "$stage1_run/checkpoints/sac_table_waypoint_polyline_vecnormalize_24996_steps.pkl" \
  --residual-action-scale 0.50 \
  --residual-action-penalty-scale 0.005 \
  --residual-stall-activation-steps 100 \
  --residual-stall-ramp-steps 300 \
  --actor-anchor-coef 5 \
  --source-policy-warmup \
  --actor-learning-delay-steps 5000 \
  --freeze-vec-normalize \
  --zero-init-actor \
  --device cuda \
  --seed 20268131
