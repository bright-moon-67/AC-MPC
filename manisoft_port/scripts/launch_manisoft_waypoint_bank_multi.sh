#!/usr/bin/env bash
# Launch N independent waypoint-bank generation processes (one shard each).
# Pure-CPU physics simulation -> run as many processes as CPU cores.
# Each shard uses a distinct seed so triplets do not repeat across shards.
# Stop anytime with: pkill -f generate_manisoft_waypoint_bank
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/processed/manisoft_waypoint_bank_v4_full
SCENARIO=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
N=16
TRIPLETS_PER_SHARD=2000
MAX_ATTEMPTS=500000
BASE_SEED=42
SEED_STRIDE=137

# Output dirs are created by the generator itself; it refuses to overwrite.

echo "launching $N shards x $TRIPLETS_PER_SHARD triplets (seed $BASE_SEED stride $SEED_STRIDE)"
for i in $(seq 0 $((N - 1))); do
  seed=$((BASE_SEED + i * SEED_STRIDE))
  shard=$(printf "shard%02d" "$i")
  # Pin each worker to a single thread so 16 processes fill the 16 cores
  # without BLAS/OpenMP thread contention.
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nohup /root/miniconda3/envs/manisoft/bin/python \
    scripts/generate_manisoft_waypoint_bank.py \
    --scenario "$SCENARIO" \
    --output "$OUT/$shard" \
    --triplets "$TRIPLETS_PER_SHARD" \
    --max-attempts "$MAX_ATTEMPTS" \
    --seed "$seed" \
    > "runs/waypoint_bank_v4_${shard}.log" 2>&1 &
done
echo "all $N processes launched"
