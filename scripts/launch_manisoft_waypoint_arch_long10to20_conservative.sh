#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
output="$project_root/runs/manisoft_waypoint_arch_long10to20_conservative_v1_20260824"
source_model="$project_root/runs/manisoft_waypoint_arch_generalized10_long_v3_flat_strict_20260824/best/best_model.zip"
source_vecnorm="$project_root/runs/manisoft_waypoint_arch_generalized10_long_v3_flat_strict_20260824/checkpoints/sac_table_waypoint_polyline_vecnormalize_900000_steps.pkl"

test ! -e "$output"
test -f "$source_model"
test -f "$source_vecnorm"
mkdir -p "$output"
cd "$project_root"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table_arch_damped.yaml \
  --config configs/manisoft_waypoint_sac_table_arch_generalized_10point.yaml \
  --output "$output" \
  --curriculum table_waypoint_polyline \
  --total-timesteps 3000000 \
  --num-envs 8 \
  --checkpoint-freq 100000 \
  --eval-freq 100000 \
  --eval-episodes 16 \
  --eval-num-envs 4 \
  --eval-panel-seed 923000 \
  --eval-waypoint-single-line-probability 0 \
  --eval-waypoint-segment-count-range 8,12 \
  --eval-waypoint-maximum-turn-degrees 175 \
  --eval-internal-waypoint-capture-radius 0.010 \
  --eval-waypoint-stall-steps 0 \
  --learning-starts 30000 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --learning-rate 0.000005 \
  --ent-coef 0.001 \
  --target-entropy -9 \
  --net-arch 256,256,256 \
  --resume "$source_model" \
  --vec-normalize "$source_vecnorm" \
  --actor-anchor-coef 3000 \
  --device cuda \
  --seed 20260954
