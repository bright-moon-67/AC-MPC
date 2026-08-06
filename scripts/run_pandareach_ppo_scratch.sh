#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/envs/acmpc_train/bin/python}"
output_root="${PPO_OUTPUT_ROOT:-${project_root}/runs/pandareach_threewaypoint/ppo_formal_scratch}"
num_envs="${PPO_NUM_ENVS:-32}"
rollout_steps="${PPO_ROLLOUT_STEPS:-256}"
minibatch_size="${PPO_MINIBATCH_SIZE:-1024}"
update_epochs="${PPO_UPDATE_EPOCHS:-8}"
total_timesteps="${PPO_TOTAL_TIMESTEPS:-3000000}"
max_parallel="${PPO_MAX_PARALLEL:-2}"
max_wall_time_minutes="${PPO_MAX_WALL_TIME_MINUTES:-}"
seed="${PPO_SEED:-20280804}"
koopman="${PPO_KOOPMAN:-${project_root}/runs/pandareach_threewaypoint/koopman/best.pt}"
read -r -a methods <<< "${PPO_METHODS:-B0 H1-min H1-min-raw AB-PQ BC-KMPC}"

if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi

if (( max_parallel < 1 )); then
    echo "PPO_MAX_PARALLEL must be positive" >&2
    exit 2
fi

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"
extra_args=()
if [[ -n "${max_wall_time_minutes}" ]]; then
    extra_args+=(--max-wall-time-minutes "${max_wall_time_minutes}")
fi

running=0
failure=0
for method in "${methods[@]}"; do
    while (( running >= max_parallel )); do
        if ! wait -n; then
            failure=1
        fi
        running=$((running - 1))
    done
    run_name="${method}"
    if [[ "${method}" == "B0" ]]; then
        # The standard raw-state 256x256 B0 is intentionally incompatible
        # with checkpoints from the earlier one-layer/Koopman-critic route.
        run_name="B0-standard-dense"
    elif [[ "${method}" == "H1-min" || "${method}" == "H1-min-raw" ]]; then
        # Preserve earlier zero/unstable final-layer runs. The closed-loop
        # stable nonzero initialization and dense reward use fresh runs.
        run_name="${method}-stable-dense"
    else
        run_name="${method}-dense"
    fi
    method_dir="${output_root}/${run_name}/seed_${seed}"
    mkdir -p "${method_dir}"
    (
        cd "${project_root}"
        exec "${python_bin}" -m \
            experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo \
            --actor "${method}" \
            --koopman "${koopman}" \
            --output-dir "${method_dir}" \
            --num-envs "${num_envs}" \
            --rollout-steps "${rollout_steps}" \
            --minibatch-size "${minibatch_size}" \
            --update-epochs "${update_epochs}" \
            --total-timesteps "${total_timesteps}" \
            --seed "${seed}" \
            "${extra_args[@]}"
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
