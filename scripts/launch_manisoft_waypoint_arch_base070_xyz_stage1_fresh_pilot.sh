#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
output="$project_root/runs/manisoft_waypoint_arch_base070_xyz_stage1_fresh_pilot_v1_20260824"

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
  --config configs/manisoft_waypoint_sac_table_arch_base070_xyz_stage1.yaml \
  --output "$output" \
  --curriculum table_waypoint_polyline \
  --total-timesteps 100000 \
  --num-envs 6 \
  --checkpoint-freq 25000 \
  --eval-freq 25000 \
  --eval-episodes 6 \
  --eval-num-envs 2 \
  --eval-panel-seed 950000 \
  --eval-waypoint-single-line-probability 0 \
  --eval-waypoint-segment-count-range 4,8 \
  --eval-waypoint-maximum-turn-degrees 175 \
  --eval-internal-waypoint-capture-radius 0.010 \
  --eval-waypoint-stall-steps 0 \
  --waypoint-segment-count-range 4,8 \
  --waypoint-maximum-turn-degrees 175 \
  --waypoint-maximum-extent 0.250 \
  --learning-starts 15000 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --learning-rate 0.00001 \
  --ent-coef 0.001 \
  --target-entropy -9 \
  --net-arch 256,256,256 \
  --zero-init-actor \
  --actor-anchor-coef 100 \
  --device cuda \
  --seed 20261061
