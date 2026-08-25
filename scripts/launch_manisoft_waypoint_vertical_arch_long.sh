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
  --config configs/manisoft_waypoint_sac_table_vertical_arch.yaml \
  --output runs/manisoft_waypoint_vertical_arch_long_anchor100_20260823 \
  --total-timesteps 1200000 \
  --num-envs 16 \
  --checkpoint-freq 100000 \
  --eval-freq 100000 \
  --eval-episodes 8 \
  --eval-num-envs 4 \
  --learning-starts 10000 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --learning-rate 0.00003 \
  --equilibrium-path-residual-scale 0.015 \
  --policy-action-penalty-scale 0.20 \
  --ent-coef 0.001 \
  --zero-init-actor \
  --actor-anchor-coef 100 \
  --device cuda \
  --seed 20260860
