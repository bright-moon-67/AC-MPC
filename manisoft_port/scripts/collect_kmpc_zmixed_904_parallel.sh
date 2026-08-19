#!/usr/bin/env bash
# Launch N independent KMPC offline-collection processes (one per part),
# each writing its own shard directory so collection can run in parallel.
# Afterwards run scripts/merge_kmpc_offline_parts.py to combine all parts.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/processed/manisoft_kmpc_offline/zmixed_904_stochastic
CKPT=runs/manisoft_ppo_compare_v15_zmixed_24h/a_kmpc_r8_lr3_std15_md015/last.pt
N=8
EP_TOTAL=904
EP_PER_PART=$((EP_TOTAL / N))
EPISODE_STEPS=300
BASE_SEED=42

if [ $((EP_PER_PART * N)) -ne "$EP_TOTAL" ]; then
  echo "EP_TOTAL must be divisible by N (got $EP_TOTAL / $N)" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "launching $N parts x $EP_PER_PART episodes"
for i in $(seq 0 $((N - 1))); do
  seed=$((BASE_SEED + i))
  nohup /root/miniconda3/envs/manisoft/bin/python \
    scripts/collect_manisoft_kmpc_offline_dataset.py \
    --checkpoint "$CKPT" \
    --output "$OUT/part_$i" \
    --episodes "$EP_PER_PART" \
    --episode-steps "$EPISODE_STEPS" \
    --device cuda \
    --seed "$seed" \
    --no-merged-dataset \
    > "runs/collect_kmpc_zmixed_904_part_$i.log" 2>&1 &
done
echo "all $N processes launched (seed $BASE_SEED..$((BASE_SEED + N - 1)))"
