#!/usr/bin/env bash
# Launch N independent KMPC offline-collection processes on the v5_10k
# waypoint bank (7109 triplets) with the e_kmpc (v15e) checkpoint.
# 8 processes is the empirically-tuned sweet spot on this box (RTX 4090 + 16
# cores): 12 processes caused contention and LOWER aggregate throughput.
# Each process writes its own shard directory; afterwards run
# scripts/merge_kmpc_offline_parts.py to combine all parts.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/processed/manisoft_kmpc_offline/v5_7109_stochastic
CKPT=runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt
WAYPOINT_ROOT=data/processed/manisoft_waypoint_bank_v5_10k
N=8
EP_TOTAL=7109
EPISODE_STEPS=300
BASE_SEED=42

mkdir -p "$OUT"
echo "launching $N parts for $EP_TOTAL episodes on $WAYPOINT_ROOT"
for i in $(seq 0 $((N - 1))); do
  if [ "$i" -eq $((N - 1)) ]; then
    # distribute the remainder to the last part
    n_ep=$((EP_TOTAL - (N - 1) * (EP_TOTAL / N)))
  else
    n_ep=$((EP_TOTAL / N))
  fi
  seed=$((BASE_SEED + i))
  nohup /root/miniconda3/envs/manisoft/bin/python \
    scripts/collect_manisoft_kmpc_offline_dataset.py \
    --checkpoint "$CKPT" \
    --waypoint-root "$WAYPOINT_ROOT" \
    --output "$OUT/part_$i" \
    --episodes "$n_ep" \
    --episode-steps "$EPISODE_STEPS" \
    --device cuda \
    --seed "$seed" \
    --allow-other-waypoint-bank \
    --no-merged-dataset \
    > "runs/collect_kmpc_v5_7109_part_$i.log" 2>&1 &
  echo "  part_$i: $n_ep eps (seed $seed)"
done
echo "all $N processes launched"
