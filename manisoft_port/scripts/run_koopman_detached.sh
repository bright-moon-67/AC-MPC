#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${project_root}/runs/antmaze_umaze_fulla_formal}"
checkpoint="${2:-}"
python_bin="${KOOPMAN_PYTHON:-${AC_MPC_PYTHON:-python}}"
run_dir="${output_root}/koopman"
pid_file="${run_dir}/formal_train.pid"
console_log="${project_root}/runs/koopman_fulla_formal_console.log"

mkdir -p "${run_dir}"
if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Koopman training is already running with PID ${existing_pid}."
        exit 0
    fi
fi
if [[ -f "${run_dir}/training_status.json" ]]; then
    echo "Refusing to restart a completed run: ${run_dir}/training_status.json" >&2
    exit 1
fi

if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(
        find "${run_dir}" -maxdepth 1 -type f \
            \( -name 'recovery_epoch_*.pt' -o -name 'best_validation.pt' -o -name 'last.pt' \) \
            -printf '%T@ %p\n' \
            | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
fi
if [[ -z "${checkpoint}" && -f "${run_dir}/history.jsonl" ]]; then
    echo "Run history exists but no resumable checkpoint was found in ${run_dir}." >&2
    exit 1
fi

command=(
    "${python_bin}" -u "${project_root}/scripts/train_koopman.py"
    --config "${project_root}/configs/antmaze_umaze.yaml"
    --data "${project_root}/data/processed/antmaze-umaze-v2"
    --output "${output_root}"
    --device cuda
    --wandb-mode offline
)
if [[ -n "${checkpoint}" ]]; then
    command+=(--resume "${checkpoint}")
fi

cd "${project_root}"
nohup setsid "${command[@]}" >> "${console_log}" 2>&1 < /dev/null &
training_pid=$!
echo "${training_pid}" > "${pid_file}"
sleep 2
if ! kill -0 "${training_pid}" 2>/dev/null; then
    echo "Koopman process exited during startup; inspect ${console_log}." >&2
    exit 1
fi
echo "Started detached Koopman training with PID ${training_pid}."
echo "Console: ${console_log}"
