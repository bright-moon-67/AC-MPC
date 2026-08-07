#!/usr/bin/env bash
# PPO fine-tuning of the PandaReach3 actors from their BC-pretrained
# checkpoints (runs/pandareach_threewaypoint/bc/<NAME>.pt).
#
# This is the replacement for the failed from-scratch H1-min actor-LR sweep:
# the policy is initialized with real BC weights (no 2e-6 g_u gain), the BC
# dataset normalizers are reused, and the actor is fine-tuned with a small
# PPO learning rate. The "PPO" route is the standard raw-state PPO actor
# (256x256 Tanh MLP, linear mean output, no output tanh). The "KLQR" route
# is the DARE-based time-varying closed-loop actor (cost-map -> Q diag + p).
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${PPO_OUTPUT_ROOT:-${project_root}/runs/pandareach_threewaypoint/ppo_finetune}"
bc_root="${PPO_BC_ROOT:-${project_root}/runs/pandareach_threewaypoint/bc}"
koopman="${PPO_KOOPMAN:-${project_root}/runs/pandareach_threewaypoint/koopman/best.pt}"
num_envs="${PPO_NUM_ENVS:-32}"
rollout_steps="${PPO_ROLLOUT_STEPS:-256}"
minibatch_size="${PPO_MINIBATCH_SIZE:-1024}"
update_epochs="${PPO_UPDATE_EPOCHS:-8}"
total_timesteps="${PPO_TOTAL_TIMESTEPS:-3000000}"
actor_learning_rate="${PPO_FINETUNE_ACTOR_LR:-1e-4}"
critic_learning_rate="${PPO_CRITIC_LEARNING_RATE:-3e-4}"
entropy_coefficient="${PPO_ENTROPY_COEFFICIENT:-1e-3}"
initial_std_rad="${PPO_INITIAL_STD_RAD:-0.05}"
reward_mode="${PPO_REWARD_MODE:-dense}"
dense_distance_penalty_scale="${PPO_DENSE_DISTANCE_PENALTY_SCALE:-0.05}"
dense_waypoint_completion_reward="${PPO_DENSE_WAYPOINT_COMPLETION_REWARD:-1.0}"
goal_threshold="${PPO_GOAL_THRESHOLD:-0.01}"
success_goal_threshold="${PPO_SUCCESS_GOAL_THRESHOLD:-${goal_threshold}}"
require_robot_static="${PPO_REQUIRE_ROBOT_STATIC:-true}"
checkpoint_interval_updates="${PPO_CHECKPOINT_INTERVAL_UPDATES:-10}"
seed="${PPO_SEED:-20280804}"
max_parallel="${PPO_MAX_PARALLEL:-2}"
read -r -a methods <<< "${PPO_METHODS:-PPO KLQR AB-PQ BC-KMPC}"

if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${koopman}" ]]; then
    echo "Koopman checkpoint not found: ${koopman}" >&2
    exit 2
fi
if (( max_parallel < 1 )); then
    echo "PPO_MAX_PARALLEL must be positive" >&2
    exit 2
fi

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"
printf '%s\n' "${actor_learning_rate}" > "${output_root}/actor_learning_rate.txt"

running=0
failure=0
for method in "${methods[@]}"; do
    bc_checkpoint="${bc_root}/${method}.pt"
    if [[ ! -f "${bc_checkpoint}" ]]; then
        echo "BC checkpoint not found: ${bc_checkpoint}; run BC pretraining first." >&2
        exit 2
    fi
    while (( running >= max_parallel )); do
        if ! wait -n; then
            failure=1
        fi
        running=$((running - 1))
    done

    method_dir="${output_root}/${method}/seed_${seed}"
    mkdir -p "${method_dir}"
    extra_args=()
    if [[ "${require_robot_static}" == "false" ]]; then
        extra_args+=(--no-require-robot-static)
    fi
    (
        cd "${project_root}"
        exec "${python_bin}" -m \
            experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo \
            --actor "${method}" \
            --bc-checkpoint "${bc_checkpoint}" \
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
            --goal-threshold "${goal_threshold}" \
            --success-goal-threshold "${success_goal_threshold}" \
            --checkpoint-interval-updates "${checkpoint_interval_updates}" \
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
