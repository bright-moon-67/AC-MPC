#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
output="$project_root/runs/manisoft_waypoint_arch_base070_angle30_long10to20_transfer_v1_20260824"
source_model="$project_root/runs/manisoft_waypoint_arch_long10to20_transfer_v1_20260824/best/best_model.zip"
source_vecnorm="$project_root/runs/manisoft_waypoint_arch_long10to20_transfer_v1_20260824/checkpoints/sac_table_waypoint_polyline_vecnormalize_1200000_steps.pkl"

test ! -e "$output"
test -f "$source_model"
test -f "$source_vecnorm"
mkdir -p "$output"
cd "$project_root"

# One simulator worker per physical CPU. Keep per-process BLAS single-threaded;
# policy/critic updates are placed on the RTX 4090.
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec "$python_bin" -u scripts/train_manisoft_waypoint_sac.py \
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table_arch_base070.yaml \
  --config configs/manisoft_waypoint_sac_table_arch_base070_flat30cm_angle30.yaml \
  --output "$output" \
  --curriculum table_waypoint_polyline \
  --total-timesteps 3000000 \
  --num-envs 16 \
  --checkpoint-freq 100000 \
  --eval-freq 100000 \
  --eval-episodes 16 \
  --eval-num-envs 4 \
  --eval-panel-seed 934000 \
  --eval-waypoint-single-line-probability 0 \
  --eval-waypoint-segment-count-range 8,12 \
  --eval-waypoint-maximum-turn-degrees 175 \
  --eval-internal-waypoint-capture-radius 0.010 \
  --eval-waypoint-stall-steps 0 \
  --waypoint-segment-count-range 8,12 \
  --waypoint-maximum-turn-degrees 175 \
  --waypoint-maximum-extent 0.250 \
  --learning-starts 30000 \
  --batch-size 1024 \
  --gradient-steps 4 \
  --learning-rate 0.00001 \
  --ent-coef 0.001 \
  --target-entropy -9 \
  --net-arch 256,256,256 \
  --resume "$source_model" \
  --vec-normalize "$source_vecnorm" \
  --actor-anchor-coef 1000 \
  --device cuda \
  --seed 20261024
