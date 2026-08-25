#!/usr/bin/env bash
set -euo pipefail

project_root="/root/autodl-tmp/AC-MPC"
python_bin="/root/miniconda3/envs/manisoft/bin/python"
run="$project_root/runs/manisoft_waypoint_arch_base070_angle60_wide_long_v1_20260824"
output="$run/selection_eval_20260825/random24_parallel"
parallel_jobs="${PARALLEL_JOBS:-16}"
checkpoints="${CHECKPOINTS:-249960 424932 499920}"
case_start="${CASE_START:-0}"
case_count="${CASE_COUNT:-24}"

if ((case_start < 0 || case_count < 1)); then
  echo "CASE_START must be non-negative and CASE_COUNT must be positive" >&2
  exit 2
fi
case_end=$((case_start + case_count - 1))

mkdir -p "$output"
cd "$project_root"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

run_case() {
  local checkpoint="$1"
  local case_index="$2"
  local speed_index=$((case_index % 3))
  local speeds=(0.006 0.010 0.015)
  local seed=$((971000 + case_index))
  local prefix="$output/${checkpoint}_case_$(printf '%02d' "$case_index")"

  "$python_bin" -u scripts/evaluate_manisoft_waypoint_sac.py \
    --model "$run/checkpoints/sac_table_waypoint_polyline_${checkpoint}_steps.zip" \
    --vec-normalize "$run/checkpoints/sac_table_waypoint_polyline_vecnormalize_${checkpoint}_steps.pkl" \
    --run-config "$run/run_config.json" \
    --output "${prefix}.npz" \
    --episodes 1 \
    --episode-steps 12000 \
    --curriculum table_waypoint_polyline \
    --families polyline \
    --speeds "${speeds[$speed_index]}" \
    --waypoint-segment-count-range 8,12 \
    --waypoint-maximum-extent 0.33 \
    --waypoint-maximum-turn-degrees 170 \
    --device cpu \
    --seed "$seed" \
    > "${prefix}.log" 2>&1
}
export -f run_case
export project_root python_bin run output

for checkpoint in $checkpoints; do
  for case_index in $(seq "$case_start" "$case_end"); do
    printf '%s %s\n' "$checkpoint" "$case_index"
  done
done | xargs -n 2 -P "$parallel_jobs" bash -c 'run_case "$1" "$2"' _

echo "completed $output"
