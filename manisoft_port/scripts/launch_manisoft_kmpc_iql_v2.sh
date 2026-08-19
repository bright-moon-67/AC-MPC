#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "40" && "$1" != "60" ) ]]; then
  echo "usage: $0 {40|60}" >&2
  exit 2
fi

solver_iterations="$1"
repo_root="/root/autodl-tmp/AC-MPC"
run_dir="$repo_root/runs/manisoft_kmpc_iql_v2/k${solver_iterations}_seed42"
lock_file="${run_dir}.lock"

mkdir -p "$(dirname "$run_dir")"
if [[ -e "$run_dir/history.jsonl" || -e "$run_dir/last.pt" ]]; then
  echo "refusing to overwrite existing run: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir"
exec 9>"$lock_file"
flock -n 9 || { echo "run is already locked: $run_dir" >&2; exit 1; }

cd "$repo_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec /root/miniconda3/envs/manisoft/bin/python -u \
  scripts/train_manisoft_kmpc_iql.py \
  --dataset data/processed/manisoft_kmpc_offline/combined_v4_1498_v5_7109/dataset.npz \
  --initial-policy-checkpoint runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt \
  --candidate-cost-parameterization structured_v2 \
  --candidate-solver-iterations "$solver_iterations" \
  --distillation-steps 10000 \
  --critic-warmup-steps 20000 \
  --selection-behavior-mse-penalty 10 \
  --gradient-steps 500000 \
  --batch-size 256 \
  --validation-batch-size 1024 \
  --validation-interval 5000 \
  --checkpoint-interval 25000 \
  --log-interval 100 \
  --device cuda \
  --seed 42 \
  --output "$run_dir"
