#!/usr/bin/env bash
set -euo pipefail

# Detached pure-SAC entry-tail sweep sized for this container's 16-CPU quota
# and one RTX 4090. Logs and PID files stay outside the SB3 output directories
# so the training script's nonempty-output guard remains effective.

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/AC-MPC}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/manisoft/bin/python}
SCENARIO=${SCENARIO:-/root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml}
CONFIG=${CONFIG:-$PROJECT_ROOT/configs/manisoft_waypoint_sac_physical.yaml}
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/runs/manisoft_waypoint_sac_long_20260821_pure}
TIMESTEPS=${TIMESTEPS:-500000}

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pids"
test -f "$PROJECT_ROOT/data/processed/manisoft_table_entry_bank_v1/entry_bank.npz"

launch() {
  local name=$1
  local envs=$2
  local seed=$3
  local learning_rate=$4
  local batch_size=$5
  local gradient_steps=$6
  local net_arch=$7
  local output="$RUN_ROOT/$name"
  local log="$RUN_ROOT/logs/$name.log"
  if [[ -e "$output" ]] && [[ -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Skipping existing nonempty output: $output" >&2
    return 0
  fi
  nohup setsid env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" -u "$PROJECT_ROOT/scripts/train_manisoft_waypoint_sac.py" \
      --scenario "$SCENARIO" \
      --config "$CONFIG" \
      --output "$output" \
      --curriculum entry_tail \
      --total-timesteps "$TIMESTEPS" \
      --num-envs "$envs" \
      --learning-rate "$learning_rate" \
      --batch-size "$batch_size" \
      --gradient-steps "$gradient_steps" \
      --net-arch "$net_arch" \
      --device cuda \
      --seed "$seed" \
      >"$log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$RUN_ROOT/pids/$name.pid"
  printf '%s pid=%s log=%s\n' "$name" "$pid" "$log"
}

# 6 + 5 + 5 simulator workers consume the effective 16-CPU container quota.
launch a_baseline 6 42 0.0002 512 1 256,256,256
launch b_balanced 5 142 0.00012 1024 4 384,384,256
launch c_gpu_heavy 5 242 0.00008 2048 8 512,512,512
# These two profiles fill the alternating CPU-simulation and GPU-update gaps
# left by the first three runs.
launch d_gpu_saturating 3 342 0.00005 4096 16 768,768,512
launch e_cpu_focused 7 442 0.00018 512 1 256,256,256
# PyElastica workers average below one full core, so an additional compact
# policy is needed to consume the remaining cgroup CPU quota.
launch f_cpu_throughput 10 542 0.00025 256 1 128,128
