#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_run="$project_root/runs/manisoft_xy47x18_curriculum_20260825/stage1a_two_segment"
source_step="799856"
output="${1:-$project_root/runs/manisoft_ground015_stage1b_frozen_residual_pilot_20260825}"
total_timesteps="${TOTAL_TIMESTEPS:-75000}"
num_envs="${NUM_ENVS:-12}"

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
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
  --config configs/manisoft_waypoint_sac_table_xy47x18_stage1b_control.yaml \
  --output "$output" \
  --total-timesteps "$total_timesteps" \
  --num-envs "$num_envs" \
  --checkpoint-freq 25000 \
  --eval-freq 25000 \
  --eval-episodes 24 \
  --eval-num-envs 8 \
  --eval-panel-seed 741000 \
  --eval-waypoint-single-line-probability 0 \
  --eval-waypoint-segment-count-range 2,2 \
  --eval-waypoint-minimum-turn-degrees 60 \
  --eval-waypoint-maximum-turn-degrees 120 \
  --eval-internal-waypoint-capture-radius 0.020 \
  --eval-waypoint-stall-steps 1500 \
  --learning-starts 10000 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --learning-rate 0.000003 \
  --ent-coef 0.001 \
  --target-entropy -9 \
  --net-arch 256,256,256 \
  --frozen-base-model "$source_run/checkpoints/sac_table_waypoint_polyline_${source_step}_steps.zip" \
  --frozen-base-vec-normalize "$source_run/checkpoints/sac_table_waypoint_polyline_vecnormalize_${source_step}_steps.pkl" \
  --residual-action-scale 0.40 \
  --residual-action-penalty-scale 0.010 \
  --residual-stall-activation-steps 40 \
  --residual-stall-ramp-steps 160 \
  --zero-init-actor \
  --actor-anchor-coef 30 \
  --source-policy-warmup \
  --actor-learning-delay-steps 5000 \
  --freeze-vec-normalize \
  --device cuda \
  --seed 202608252
