#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 actor|delta_ppo KOOPMAN_CHECKPOINT OUTPUT_ROOT [cuda|cpu]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
method="$1"
koopman_checkpoint="$2"
output_root="$3"
device="${4:-cuda}"
pid_file="${output_root}/formal_launcher.pid"
console_log="${output_root}/launcher.log"

if [[ "${method}" != "actor" && "${method}" != "delta_ppo" ]]; then
    echo "Method must be actor or delta_ppo." >&2
    exit 2
fi
if [[ ! -f "${koopman_checkpoint}" ]]; then
    echo "Missing Koopman checkpoint: ${koopman_checkpoint}" >&2
    exit 1
fi
mkdir -p "${output_root}"
if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Formal ${method} launcher is already running with PID ${existing_pid}."
        exit 0
    fi
fi

command=(
    "${project_root}/scripts/run_legacy.sh"
    python "${project_root}/scripts/run_ppo_seeds.py"
    --method "${method}"
    --koopman-checkpoint "${koopman_checkpoint}"
    --output-root "${output_root}"
    --backend legacy
    --device "${device}"
)
cd "${project_root}"
nohup setsid "${command[@]}" >> "${console_log}" 2>&1 < /dev/null &
launcher_pid=$!
echo "${launcher_pid}" > "${pid_file}"
sleep 2
if ! kill -0 "${launcher_pid}" 2>/dev/null; then
    echo "Formal launcher exited during startup; inspect ${console_log}." >&2
    exit 1
fi
echo "Started formal ${method} launcher with PID ${launcher_pid}."
echo "Console: ${console_log}"
