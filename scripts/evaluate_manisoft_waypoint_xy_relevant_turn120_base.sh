#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
stage1_run="$project_root/runs/manisoft_waypoint_arch_base070_xyz_stage1_fresh_pilot_v1_20260824"
output="$project_root/runs/manisoft_waypoint_arch_base070_xy_relevant_turn120_base_eval_v1_20260824"

test ! -e "$output"
mkdir -p "$output"
cd "$project_root"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

exec "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table_arch_base070.yaml \
  --config configs/manisoft_waypoint_sac_table_arch_base070_xy_tolerant_generalization.yaml \
  --output "$output" \
  --total-timesteps 6 --num-envs 6 --checkpoint-freq 6 --eval-freq 6 \
  --eval-episodes 12 --eval-num-envs 3 --eval-panel-seed 975000 \
  --eval-waypoint-single-line-probability 0 \
  --eval-waypoint-hard-turn-probability 0 \
  --eval-waypoint-segment-count-range 9,13 \
  --eval-waypoint-minimum-turn-degrees 0 \
  --eval-waypoint-maximum-turn-degrees 120 \
  --eval-internal-waypoint-capture-radius 0.010 \
  --eval-waypoint-stall-steps 3000 \
  --resume "$stage1_run/checkpoints/sac_table_waypoint_polyline_24996_steps.zip" \
  --vec-normalize "$stage1_run/checkpoints/sac_table_waypoint_polyline_vecnormalize_24996_steps.pkl" \
  --freeze-vec-normalize --device cuda --seed 20268124
