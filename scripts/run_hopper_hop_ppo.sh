#!/usr/bin/env bash
# Multi-seed parallel PPO training for MS-HopperHop with periodic budgeted
# transition collection for a later global Koopman model.
#
#   HOPPER_SEEDS          space-separated seeds (default: 20240201 20240202 20240203)
#   HOPPER_NUM_ENVS       parallel envs per seed (default 2048)
#   HOPPER_ROLLOUT_STEPS  rollout steps per update (default 100)
#   HOPPER_TOTAL_TIMESTEPS 20M
#   HOPPER_MAX_PARALLEL   max concurrent seeds (default 3)
#   HOPPER_OUTPUT_ROOT    runs/hopper_hop/ppo_v2
#   HOPPER_COLLECT_ROOT   runs/hopper_hop/data_v2 (budgeted, <2GB total)
#
# Detached: tmux new-session -d -s hopper "bash scripts/run_hopper_hop_ppo.sh"
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${HOPPER_OUTPUT_ROOT:-${project_root}/runs/hopper_hop/ppo_v2}"
collect_root="${HOPPER_COLLECT_ROOT:-${project_root}/runs/hopper_hop/data_v2}"
num_envs="${HOPPER_NUM_ENVS:-2048}"
rollout_steps="${HOPPER_ROLLOUT_STEPS:-100}"
total_timesteps="${HOPPER_TOTAL_TIMESTEPS:-20000000}"
max_parallel="${HOPPER_MAX_PARALLEL:-3}"
read -r -a seeds <<< "${HOPPER_SEEDS:-20240201 20240202 20240203}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 2
fi
if (( max_parallel < 1 )); then
    echo "HOPPER_MAX_PARALLEL must be positive" >&2
    exit 2
fi

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"

running=0
failure=0
for seed in "${seeds[@]}"; do
    while (( running >= max_parallel )); do
        if ! wait -n; then
            failure=1
        fi
        running=$((running - 1))
    done
    method_dir="${output_root}/seed_${seed}"
    collect_dir="${collect_root}/seed_${seed}"
    mkdir -p "${method_dir}"
    (
        cd "${project_root}"
        exec "${python_bin}" -u -m \
            experiments.hopper_hop.train_hopper_hop_ppo \
            --output-dir "${method_dir}" \
            --num-envs "${num_envs}" \
            --rollout-steps "${rollout_steps}" \
            --total-timesteps "${total_timesteps}" \
            --seed "${seed}" \
            --collect-dir "${collect_dir}"
    ) > "${method_dir}/console.log" 2>&1 &
    printf '%s\n' "$!" > "${method_dir}/training.pid"
    running=$((running + 1))
done

while (( running > 0 )); do
    if ! wait -n; then
        failure=1
    fi
    running=$((running - 1))
done

if (( failure != 0 )); then
    printf '%s\n' "failed" > "${output_root}/launcher.status"
    exit 1
fi
printf '%s\n' "complete" > "${output_root}/launcher.status"
echo "HopperHop PPO done. Output: ${output_root}"
