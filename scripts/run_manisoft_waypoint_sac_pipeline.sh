#!/usr/bin/env bash
set -euo pipefail

# Curriculum launcher for the physically certified obstacle-free SAC tracker.

STAGE=${1:-help}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/AC-MPC}
MANISOFT_ROOT=${MANISOFT_ROOT:-/root/autodl-tmp/ManiSoft}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/runs/manisoft_waypoint_sac_formal}
SCENARIO=${SCENARIO:-$MANISOFT_ROOT/configs/sac_waypoint_tracking_table.yaml}
CONFIG=${CONFIG:-$PROJECT_ROOT/configs/manisoft_waypoint_sac_physical.yaml}
DEVICE=${DEVICE:-cuda}
NUM_ENVS=${NUM_ENVS:-8}

train_fresh() {
  local curriculum=$1
  local timesteps=$2
  local output=$3
  python -u "$PROJECT_ROOT/scripts/train_manisoft_waypoint_sac.py" \
    --scenario "$SCENARIO" \
    --config "$CONFIG" \
    --output "$output" \
    --curriculum "$curriculum" \
    --total-timesteps "$timesteps" \
    --num-envs "$NUM_ENVS" \
    --device "$DEVICE"
}

train_resume() {
  local curriculum=$1
  local timesteps=$2
  local previous=$3
  local output=$4
  test -f "$previous/final_model.zip"
  test -f "$previous/vecnormalize.pkl"
  python -u "$PROJECT_ROOT/scripts/train_manisoft_waypoint_sac.py" \
    --scenario "$SCENARIO" \
    --config "$CONFIG" \
    --output "$output" \
    --curriculum "$curriculum" \
    --total-timesteps "$timesteps" \
    --num-envs "$NUM_ENVS" \
    --device "$DEVICE" \
    --resume "$previous/final_model.zip" \
    --vec-normalize "$previous/vecnormalize.pkl"
}

generate_bank() {
  python -u "$PROJECT_ROOT/scripts/generate_manisoft_table_entry_bank.py" \
    --scenario "$SCENARIO" \
    --seeds "$PROJECT_ROOT/configs/manisoft_table_entry_seeds.yaml" \
    --output "$PROJECT_ROOT/data/processed/manisoft_table_entry_bank_v1/entry_bank.npz"
}

analyze_local() {
  python -u "$PROJECT_ROOT/scripts/analyze_manisoft_table_local_reachability.py" \
    --scenario "$SCENARIO" \
    --bank "$PROJECT_ROOT/data/processed/manisoft_table_entry_bank_v1/entry_bank.npz" \
    --output "$OUTPUT_ROOT/validation/local_reachability.json" \
    --workers "${PROBE_WORKERS:-8}"
}

run_entry_tail() {
  test -f "$PROJECT_ROOT/data/processed/manisoft_table_entry_bank_v1/entry_bank.npz"
  train_fresh entry_tail 500000 "$OUTPUT_ROOT/01_entry_tail"
}

run_entry_mid() {
  train_resume entry_mid 700000 "$OUTPUT_ROOT/01_entry_tail" "$OUTPUT_ROOT/02_entry_mid"
}

run_entry() {
  train_resume entry 900000 "$OUTPUT_ROOT/02_entry_mid" "$OUTPUT_ROOT/03_entry"
}

run_table_local() {
  train_resume table_local 1000000 "$OUTPUT_ROOT/03_entry" "$OUTPUT_ROOT/04_table_local"
}

run_recovery() {
  train_resume recovery 800000 "$OUTPUT_ROOT/04_table_local" "$OUTPUT_ROOT/05_recovery"
}

run_entry_local() {
  train_resume entry_local 1500000 "$OUTPUT_ROOT/05_recovery" "$OUTPUT_ROOT/06_entry_local"
}

run_mixed() {
  train_resume mixed 1500000 "$OUTPUT_ROOT/06_entry_local" "$OUTPUT_ROOT/07_mixed"
}

run_evaluate() {
  local stage
  for stage in entry table_local entry_local mixed; do
    python -u "$PROJECT_ROOT/scripts/evaluate_manisoft_waypoint_sac.py" \
      --model "$OUTPUT_ROOT/07_mixed/final_model.zip" \
      --vec-normalize "$OUTPUT_ROOT/07_mixed/vecnormalize.pkl" \
      --run-config "$OUTPUT_ROOT/07_mixed/run_config.json" \
      --output "$OUTPUT_ROOT/evaluation/policy_${stage}.npz" \
      --episodes 100 \
      --families line,polyline,bezier,s_curve,reverse \
      --curriculum "$stage" \
      --successful-only \
      --device "$DEVICE"
  done
}

case "$STAGE" in
  generate) generate_bank ;;
  analyze_local) analyze_local ;;
  entry_tail) run_entry_tail ;;
  entry_mid) run_entry_mid ;;
  entry) run_entry ;;
  table_local) run_table_local ;;
  recovery) run_recovery ;;
  entry_local) run_entry_local ;;
  mixed) run_mixed ;;
  evaluate) run_evaluate ;;
  all)
    generate_bank
    analyze_local
    run_entry_tail
    run_entry_mid
    run_entry
    run_table_local
    run_recovery
    run_entry_local
    run_mixed
    run_evaluate
    ;;
  *)
    echo "Usage: $0 {generate|analyze_local|entry_tail|entry_mid|entry|table_local|recovery|entry_local|mixed|evaluate|all}"
    exit 2
    ;;
esac
