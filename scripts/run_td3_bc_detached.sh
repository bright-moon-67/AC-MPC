#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
    echo "Usage: $0 KOOPMAN_CHECKPOINT OUTPUT [SEED] [DEVICE] [GRADIENT_STEPS] [BATCH_SIZE]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
koopman_checkpoint="$1"
output="$2"
seed="${3:-0}"
device="${4:-cuda}"
gradient_steps="${5:-500000}"
batch_size="${6:-256}"
python_bin="${TD3_BC_PYTHON:-${AC_MPC_PYTHON:-python}}"
pid_file="${output}/td3_bc_train.pid"
console_log="${output}/console.log"

if [[ ! -f "${koopman_checkpoint}" ]]; then
    echo "Missing Koopman checkpoint: ${koopman_checkpoint}" >&2
    exit 1
fi
mkdir -p "${output}"
if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "TD3+BC training is already running with PID ${existing_pid}."
        exit 0
    fi
fi

resume_args=()
if [[ -f "${output}/last.pt" ]]; then
    resume_args=(--resume "${output}/last.pt")
elif [[ -f "${output}/history.jsonl" ]]; then
    echo "History exists without last.pt; refusing an unrecoverable restart." >&2
    exit 1
fi

command=(
    "${python_bin}" -u "${project_root}/scripts/train_td3_bc.py"
    --config "${project_root}/configs/antmaze_umaze.yaml"
    --koopman-checkpoint "${koopman_checkpoint}"
    --data "${project_root}/data/processed/antmaze-umaze-v2"
    --output "${output}"
    --device "${device}"
    --seed "${seed}"
    --gradient-steps "${gradient_steps}"
    --batch-size "${batch_size}"
    "${resume_args[@]}"
)
cd "${project_root}"
nohup setsid "${command[@]}" >> "${console_log}" 2>&1 < /dev/null &
training_pid=$!
echo "${training_pid}" > "${pid_file}"
sleep 2
if ! kill -0 "${training_pid}" 2>/dev/null; then
    echo "TD3+BC process exited during startup; inspect ${console_log}." >&2
    exit 1
fi
echo "Started detached TD3+BC training with PID ${training_pid}."
echo "Console: ${console_log}"
