#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 KOOPMAN_CHECKPOINT PPO_OUTPUT [cuda|cpu]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
koopman_checkpoint="$1"
ppo_output="$2"
device="${3:-cuda}"
pid_file="${ppo_output}/post_ppo_pipeline.pid"
console_log="${ppo_output}/post_ppo_pipeline.log"
python_bin="${POST_PPO_PYTHON:-${AC_MPC_PYTHON:-python}}"

mkdir -p "${ppo_output}"
if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Post-PPO pipeline is already running with PID ${existing_pid}."
        exit 0
    fi
fi

command=(
    "${python_bin}" -u
    "${project_root}/scripts/post_ppo_pipeline.py"
    --koopman-checkpoint "${koopman_checkpoint}"
    --ppo-output "${ppo_output}"
    --device "${device}"
)
cd "${project_root}"
nohup setsid "${command[@]}" >> "${console_log}" 2>&1 < /dev/null &
pipeline_pid=$!
echo "${pipeline_pid}" > "${pid_file}"
sleep 2
if ! kill -0 "${pipeline_pid}" 2>/dev/null; then
    echo "Post-PPO pipeline exited during startup; inspect ${console_log}." >&2
    exit 1
fi
echo "Started post-PPO pipeline with PID ${pipeline_pid}."
echo "Console: ${console_log}"
