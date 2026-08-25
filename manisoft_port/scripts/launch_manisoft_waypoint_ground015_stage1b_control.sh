#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
source_run="$project_root/runs/manisoft_xy47x18_curriculum_20260825/stage1a_two_segment"
source_step="799856"
output="${1:-$project_root/runs/manisoft_ground015_stage1b_control_20260825}"
total_timesteps="${TOTAL_TIMESTEPS:-300000}"

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
  --resume "$source_run/checkpoints/sac_table_waypoint_polyline_${source_step}_steps.zip" \
  --vec-normalize "$source_run/checkpoints/sac_table_waypoint_polyline_vecnormalize_${source_step}_steps.pkl" \
  --total-timesteps "$total_timesteps" \
  --num-envs 16 \
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
  --equilibrium-path-residual-scale 0.010 \
  --actor-anchor-coef 20 \
  --source-policy-warmup \
  --actor-learning-delay-steps 5000 \
  --freeze-vec-normalize \
  --device cuda \
  --seed 202608251
