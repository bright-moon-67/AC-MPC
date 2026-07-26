#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 actor|delta_ppo INPUT_ROOT [GAIN_INTERVAL] [DEVICE]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
method="$1"
input_root="$2"
gain_interval="${3:-1}"
device="${4:-cuda}"
pid_file="${input_root}/formal_evaluation.pid"
console_log="${input_root}/formal_evaluation.log"

mkdir -p "${input_root}"
if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Formal evaluation is already running with PID ${existing_pid}."
        exit 0
    fi
fi

command=(
    "${project_root}/scripts/run_legacy.sh"
    python "${project_root}/scripts/evaluate_ppo_seeds.py"
    --input-root "${input_root}"
    --method "${method}"
    --episodes 100
    --backend legacy
    --device "${device}"
    --gain-update-interval "${gain_interval}"
)
cd "${project_root}"
nohup setsid "${command[@]}" >> "${console_log}" 2>&1 < /dev/null &
evaluation_pid=$!
echo "${evaluation_pid}" > "${pid_file}"
sleep 2
if ! kill -0 "${evaluation_pid}" 2>/dev/null; then
    echo "Formal evaluation exited during startup; inspect ${console_log}." >&2
    exit 1
fi
echo "Started formal evaluation with PID ${evaluation_pid}."
echo "Console: ${console_log}"
