#!/usr/bin/env bash
# Phase 4: assemble MuJoCo datasets, train Koopman per contact config (CPU),
# then compare contact-dim predictability vs the PhysX baseline.
#
# Usage: bash experiments/hopper_hop_mujoco/train_mujoco_koopman.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=/root/miniconda3/bin/python
DATA_ROOT=runs/hopper_hop_mujoco/data
KOOPMAN_ROOT=runs/hopper_hop_mujoco/koopman

echo "== [1/3] assembling datasets =="
$PY -m experiments.hopper_hop_mujoco.collect.build_mujoco_datasets \
  --contacts mujoco_default,mujoco_compliant,mujoco_hard \
  --data-root "$DATA_ROOT"

echo "== [2/3] training Koopman per config (GPU, sequential to stay gentle on"
echo "     the user's PPO jobs; tiny model, <1 GB, 27 GB free) =="
mkdir -p "$KOOPMAN_ROOT"
for contact in mujoco_default mujoco_compliant mujoco_hard; do
  echo "-- $contact --"
  $PY -m experiments.hopper_hop.train_hopper_hop_koopman \
    --dataset "$DATA_ROOT/$contact/hopperhop_koopman.npz" \
    --output-dir "$KOOPMAN_ROOT/$contact" \
    --epochs 500 --batch-size 2048 --learning-rate 3e-4 \
    --lift-dim 48 --k-step 20 --seed 43 \
    --checkpoint-every 25 --patience 40 --max-windows 1000000 \
    --device cuda
done
echo "== all Koopman trainings done =="

echo "== [3/3] contact-dim comparison =="
$PY -m experiments.hopper_hop_mujoco.eval.compare_contact_dims \
  --koopman-root "$KOOPMAN_ROOT"

echo "== DONE =="
