#!/usr/bin/env bash
# BC pretraining for the four PandaReach3 actor routes, run in PARALLEL
# (one process per actor sharing the output directory), then merged into a
# single report.json.
#
#   PPO        256x256 Koopman-free standard PPO MLP baseline
#   KLQR      cost-map (Q diag + p) -> differentiable DARE closed-loop law
#   AB-PQ     low-rank value actor (P route, control rate via frozen A,B)
#   BC-KMPC   Koopman MPC actor
#
# Detached: nohup bash scripts/run_pandareach_bc_pretrain.sh \
#             > runs/pandareach_threewaypoint/bc_launcher.out 2>&1 &
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${BC_OUTPUT_ROOT:-${project_root}/runs/pandareach_threewaypoint/bc}"
koopman="${BC_KOOPMAN:-${project_root}/runs/pandareach_threewaypoint/koopman/best.pt}"
dataset="${BC_DATASET:-${project_root}/runs/pandareach_threewaypoint/data/pandareach_dls_500.npz}"
episodes="${BC_EPISODES:-500}"
epochs="${BC_EPOCHS:-250}"
early_stopping_patience="${BC_EARLY_STOPPING_PATIENCE:-50}"
evaluation_episodes="${BC_EVALUATION_EPISODES:-100}"
seed="${BC_SEED:-47}"
max_parallel="${BC_MAX_PARALLEL:-5}"
read -r -a actors <<< "${BC_ACTORS:-PPO KLQR AB-PQ BC-KMPC}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${koopman}" ]]; then
    echo "Koopman checkpoint not found: ${koopman}" >&2
    exit 2
fi
if (( max_parallel < 1 )); then
    echo "BC_MAX_PARALLEL must be positive" >&2
    exit 2
fi

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"
if [[ -f "${output_root}/report.json" ]]; then
    echo "BC comparison already exists: ${output_root}/report.json" >&2
    exit 1
fi

if [[ ! -f "${dataset}" ]]; then
    echo "Dataset missing; collecting ${episodes} episodes to ${dataset} ..."
    (
        cd "${project_root}"
        "${python_bin}" -m \
            experiments.state_only_feasibility.collect_pandareach_threewaypoint \
            --output "${dataset}" \
            --episodes "${episodes}"
    )
fi

running=0
failure=0
for method in "${actors[@]}"; do
    # Per-method epoch override, e.g. BC_EPOCHS_KLQR=50 while BC_EPOCHS=100.
    method_epochs_var="BC_EPOCHS_${method//-/_}"
    method_epochs="${!method_epochs_var:-${epochs}}"
    while (( running >= max_parallel )); do
        if ! wait -n; then
            failure=1
        fi
        running=$((running - 1))
    done
    (
        cd "${project_root}"
        exec "${python_bin}" -u -m \
            experiments.state_only_feasibility.train_pandareach_threewaypoint_bc \
            --dataset "${dataset}" \
            --koopman "${koopman}" \
            --output-dir "${output_root}" \
            --epochs "${method_epochs}" \
            --early-stopping-patience "${early_stopping_patience}" \
            --evaluation-episodes "${evaluation_episodes}" \
            --seed "${seed}" \
            --actors "${method}"
    ) > "${output_root}/${method}.console.log" 2>&1 &
    printf '%s\n' "$!" > "${output_root}/${method}.pid"
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
    echo "At least one BC actor failed; per-actor logs in ${output_root}/*.console.log" >&2
    exit 1
fi

# Merge per-actor reports into the aggregate comparison report.json.
(
    cd "${project_root}"
    "${python_bin}" -u -m \
        experiments.state_only_feasibility.train_pandareach_threewaypoint_bc \
        --output-dir "${output_root}" \
        --merge-only
) > "${output_root}/merge.log" 2>&1
printf '%s\n' "complete" > "${output_root}/launcher.status"
echo "BC pretraining complete."
echo "Output:  ${output_root}"
echo "Report:  ${output_root}/report.json"
