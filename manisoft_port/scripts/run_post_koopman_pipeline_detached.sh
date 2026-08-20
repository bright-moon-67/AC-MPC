#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
koopman_output="${1:-${project_root}/runs/antmaze_umaze_fulla_formal}"
ppo_output="${2:-${project_root}/runs/antmaze_umaze_formal/actor}"
run_root="${koopman_output}/koopman"
pid_file="${run_root}/post_koopman_pipeline.pid"
console_log="${run_root}/post_koopman_pipeline.log"
python_bin="${POST_KOOPMAN_PYTHON:-${AC_MPC_PYTHON:-python}}"

mkdir -p "${run_root}"
if [[ -f "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Post-Koopman pipeline is already running with PID ${existing_pid}."
        exit 0
    fi
fi

command=(
    "${python_bin}" -u
    "${project_root}/scripts/post_koopman_pipeline.py"
    --koopman-output "${koopman_output}"
    --ppo-output "${ppo_output}"
)
cd "${project_root}"
nohup setsid "${command[@]}" >> "${console_log}" 2>&1 < /dev/null &
pipeline_pid=$!
echo "${pipeline_pid}" > "${pid_file}"
sleep 2
if ! kill -0 "${pipeline_pid}" 2>/dev/null; then
    echo "Post-Koopman pipeline exited during startup; inspect ${console_log}." >&2
    exit 1
fi
echo "Started post-Koopman pipeline with PID ${pipeline_pid}."
echo "Console: ${console_log}"
