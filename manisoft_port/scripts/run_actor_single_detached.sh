#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 8 ]]; then
    echo "Usage: $0 KOOPMAN_CHECKPOINT OUTPUT [SEED] [DEVICE] [NUM_ENVS] [MINIBATCH_SIZE] [TOTAL_TIMESTEPS] [TD3_BC_ACTOR_INIT]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
koopman_checkpoint="$1"
output="$2"
seed="${3:-0}"
device="${4:-cuda}"
num_envs="${5:-16}"
minibatch_size="${6:-256}"
total_timesteps="${7:-1000000}"
actor_init="${8:-}"
pid_file="${output}/single_train.pid"
console_log="${output}/console.log"

if [[ ! -f "${koopman_checkpoint}" ]]; then
    echo "Missing Koopman checkpoint: ${koopman_checkpoint}" >&2
    exit 1
fi
if (( total_timesteps % num_envs != 0 )); then
    echo "TOTAL_TIMESTEPS must be divisible by NUM_ENVS." >&2
    exit 2
fi
mkdir -p "${output}"
if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Single-seed actor training is already running with PID ${existing_pid}."
        exit 0
    fi
fi

resume_args=()
initialization_args=()
if [[ -f "${output}/last.pt" ]]; then
    resume_args=(--resume "${output}/last.pt")
elif [[ -f "${output}/history.jsonl" ]]; then
    echo "History exists without last.pt; refusing an unrecoverable restart." >&2
    exit 1
elif [[ -n "${actor_init}" ]]; then
    if [[ ! -f "${actor_init}" ]]; then
        echo "Missing TD3+BC actor initialization checkpoint: ${actor_init}" >&2
        exit 1
    fi
    initialization_args=(--actor-init "${actor_init}")
fi

command=(
    "${project_root}/scripts/run_legacy.sh"
    python "${project_root}/scripts/train_actor.py"
    --koopman-checkpoint "${koopman_checkpoint}"
    --output "${output}"
    --backend legacy
    --device "${device}"
    --seed "${seed}"
    --num-envs "${num_envs}"
    --minibatch-size "${minibatch_size}"
    --total-timesteps "${total_timesteps}"
    "${resume_args[@]}"
    "${initialization_args[@]}"
)
cd "${project_root}"
nohup setsid "${command[@]}" >> "${console_log}" 2>&1 < /dev/null &
training_pid=$!
echo "${training_pid}" > "${pid_file}"
sleep 2
if ! kill -0 "${training_pid}" 2>/dev/null; then
    echo "Single-seed actor training exited during startup; inspect ${console_log}." >&2
    exit 1
fi
echo "Started optimized single-seed actor training with PID ${training_pid}."
echo "Console: ${console_log}"
