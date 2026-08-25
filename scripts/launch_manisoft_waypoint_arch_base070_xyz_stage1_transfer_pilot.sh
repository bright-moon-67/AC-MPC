#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
output="$project_root/runs/manisoft_waypoint_arch_base070_xyz_stage1_transfer_pilot_v1_20260824"
source_root="$project_root/runs/manisoft_waypoint_arch_base070_angle30_poseclosure_final_zeroresidual_v2_20260824"
source_model="$source_root/checkpoints/sac_table_waypoint_polyline_200000_steps.zip"
source_vecnorm="$source_root/checkpoints/sac_table_waypoint_polyline_vecnormalize_200000_steps.pkl"

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
  --resume "$source_model" \
  --vec-normalize "$source_vecnorm" \
  --actor-anchor-coef 50 \
  --device cuda \
  --seed 20261062
