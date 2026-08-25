#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec /root/miniconda3/envs/manisoft/bin/python -u \
  scripts/train_manisoft_waypoint_sac.py \
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table_arch_damped.yaml \
  --config configs/manisoft_waypoint_sac_table_vertical_arch_precision.yaml \
  --output runs/manisoft_waypoint_vertical_arch_precision_ft25_20260824 \
  --resume runs/manisoft_waypoint_vertical_arch_long_anchor100_20260823/checkpoints/sac_table_long_waypoints_700000_steps.zip \
  --vec-normalize runs/manisoft_waypoint_vertical_arch_long_anchor100_20260823/checkpoints/sac_table_long_waypoints_vecnormalize_700000_steps.pkl \
  --total-timesteps 100000 \
  --num-envs 8 \
  --checkpoint-freq 25000 \
  --eval-freq 25000 \
  --eval-episodes 8 \
  --eval-num-envs 4 \
  --learning-starts 10000 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --learning-rate 0.00001 \
  --actor-anchor-coef 25 \
  --device cuda \
  --seed 20260870
