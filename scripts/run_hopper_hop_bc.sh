#!/usr/bin/env bash
# Parallel Behavior-Cloning for the four HopperHop actors
# (PPO / KLQR / AB-PQ / BC-KMPC), one process per actor.
#
#   HOPPER_BC_EPOCHS       per-actor BC epochs (default 250)
#   HOPPER_BC_BATCH        batch size (default 2048)
#   HOPPER_BC_DATASET      expert dataset npz
#   HOPPER_BC_KOOPMAN      koopman best.pt
#   HOPPER_BC_OUTPUT_ROOT  runs/hopper_hop/bc_v2 (per-actor subdirs)
#   HOPPER_BC_ACTORS       space-separated actor names
#
# Detached: tmux new-session -d -s hopper_bc "bash scripts/run_hopper_hop_bc.sh"
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${HOPPER_BC_OUTPUT_ROOT:-${project_root}/runs/hopper_hop/bc_v2}"
dataset="${HOPPER_BC_DATASET:-${project_root}/runs/hopper_hop/data/hopperhop_expert.npz}"
koopman="${HOPPER_BC_KOOPMAN:-${project_root}/runs/hopper_hop/koopman_v2/best.pt}"
epochs="${HOPPER_BC_EPOCHS:-250}"
batch="${HOPPER_BC_BATCH:-2048}"
actors="${HOPPER_BC_ACTORS:-PPO KLQR AB-PQ BC-KMPC}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${dataset}" ]]; then
    echo "Expert dataset not found: ${dataset}" >&2
    exit 2
fi
if [[ ! -f "${koopman}" ]]; then
    echo "Koopman checkpoint not found: ${koopman}" >&2
    exit 2
fi

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"

pids=()
for actor in ${actors}; do
    actor_dir="${output_root}/${actor}"
    mkdir -p "${actor_dir}"
    (
        cd "${project_root}"
        exec "${python_bin}" -u -m \
            experiments.hopper_hop.train_hopper_hop_bc \
            --actor "${actor}" \
            --dataset "${dataset}" \
            --koopman "${koopman}" \
            --output-dir "${actor_dir}" \
            --epochs "${epochs}" \
            --batch-size "${batch}"
    ) > "${actor_dir}/console.log" 2>&1 &
    pids+=("$!")
    printf '%s\n' "$!" > "${actor_dir}/training.pid"
done

failure=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failure=1
    fi
done

if (( failure != 0 )); then
    printf '%s\n' "failed" > "${output_root}/launcher.status"
    exit 1
fi
printf '%s\n' "complete" > "${output_root}/launcher.status"
echo "HopperHop parallel BC done. Output: ${output_root}"
