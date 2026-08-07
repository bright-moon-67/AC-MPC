#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${KLQR_SWEEP_OUTPUT_ROOT:-${project_root}/runs/pandareach_threewaypoint/ppo_klqr_actor_lr_sweep}"
koopman="${PPO_KOOPMAN:-${project_root}/runs/pandareach_threewaypoint/koopman/best.pt}"

# Keep every PPO setting equal to the formal scratch launcher. The only
# sweep variable is KLQR's actor learning rate (from-scratch diagnostic).
num_envs="${PPO_NUM_ENVS:-32}"
rollout_steps="${PPO_ROLLOUT_STEPS:-256}"
minibatch_size="${PPO_MINIBATCH_SIZE:-1024}"
update_epochs="${PPO_UPDATE_EPOCHS:-8}"
total_timesteps="${PPO_TOTAL_TIMESTEPS:-3000000}"
critic_learning_rate="${KLQR_CRITIC_LEARNING_RATE:-3e-4}"
entropy_coefficient="${PPO_ENTROPY_COEFFICIENT:-1e-3}"
initial_std_rad="${PPO_INITIAL_STD_RAD:-0.05}"
reward_mode="${PPO_REWARD_MODE:-dense}"
dense_distance_penalty_scale="${PPO_DENSE_DISTANCE_PENALTY_SCALE:-0.05}"
dense_waypoint_completion_reward="${PPO_DENSE_WAYPOINT_COMPLETION_REWARD:-1.0}"
checkpoint_interval_updates="${PPO_CHECKPOINT_INTERVAL_UPDATES:-10}"
seed="${PPO_SEED:-20280804}"
max_parallel="${KLQR_MAX_PARALLEL:-4}"

# From-scratch actor learning-rate sweep for the DARE-based KLQR route.
read -r -a actor_learning_rates <<< \
    "${KLQR_ACTOR_LRS:-1e-5 3e-5 1e-4 3e-4}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found or not executable: ${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${koopman}" ]]; then
    echo "Koopman checkpoint not found: ${koopman}" >&2
    exit 2
fi
if (( max_parallel < 1 )); then
    echo "KLQR_MAX_PARALLEL must be positive" >&2
    exit 2
fi
if (( ${#actor_learning_rates[@]} == 0 )); then
    echo "KLQR_ACTOR_LRS must contain at least one learning rate" >&2
    exit 2
fi

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"
printf '%s\n' "${actor_learning_rates[*]}" > "${output_root}/actor_learning_rates.txt"

running=0
failure=0
for actor_learning_rate in "${actor_learning_rates[@]}"; do
    while (( running >= max_parallel )); do
        if ! wait -n; then
            failure=1
        fi
        running=$((running - 1))
    done

    run_name="actor_lr_${actor_learning_rate}"
    method_dir="${output_root}/${run_name}/seed_${seed}"
    mkdir -p "${method_dir}"
    (
        cd "${project_root}"
        exec "${python_bin}" -m \
            experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo \
            --actor KLQR \
            --koopman "${koopman}" \
            --output-dir "${method_dir}" \
            --num-envs "${num_envs}" \
            --rollout-steps "${rollout_steps}" \
            --minibatch-size "${minibatch_size}" \
            --update-epochs "${update_epochs}" \
            --total-timesteps "${total_timesteps}" \
            --learning-rate "${critic_learning_rate}" \
            --actor-learning-rate "${actor_learning_rate}" \
            --entropy-coefficient "${entropy_coefficient}" \
            --initial-std-rad "${initial_std_rad}" \
            --reward-mode "${reward_mode}" \
            --dense-distance-penalty-scale "${dense_distance_penalty_scale}" \
            --dense-waypoint-completion-reward "${dense_waypoint_completion_reward}" \
            --checkpoint-interval-updates "${checkpoint_interval_updates}" \
            --seed "${seed}"
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
