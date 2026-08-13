#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "Usage: $0 KOOPMAN_CHECKPOINT SCENARIO WAYPOINT_ROOT EXPERT_DATA BC_OUTPUT PPO_OUTPUT [cuda|cpu]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-python}"
koopman_checkpoint="$1"
scenario="$2"
waypoint_root="$3"
expert_data="$4"
bc_output="$5"
ppo_output="$6"
device="${7:-cuda}"
horizon="${BC_KMPC_HORIZON:-10}"
solver_iterations="${BC_KMPC_SOLVER_ITERATIONS:-20}"
quadratic_log_scale="${BC_KMPC_QUADRATIC_LOG_SCALE:-1.5}"
linear_scale="${BC_KMPC_LINEAR_SCALE:-10.0}"
action_quadratic_scale="${BC_KMPC_ACTION_QUADRATIC_SCALE:-1.0}"

for path in "${koopman_checkpoint}" "${scenario}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
done
if [[ ! -d "${waypoint_root}" ]]; then
    echo "Missing waypoint root: ${waypoint_root}" >&2
    exit 1
fi

cd "${project_root}"
if [[ ! -f "${expert_data}" ]]; then
    "${python_bin}" scripts/collect_manisoft_bc_kmpc_expert.py \
        --koopman-checkpoint "${koopman_checkpoint}" \
        --scenario "${scenario}" \
        --waypoint-root "${waypoint_root}" \
        --output "${expert_data}" \
        --episodes "${BC_KMPC_EXPERT_EPISODES:-10}" \
        --episode-steps "${BC_KMPC_EPISODE_STEPS:-300}" \
        --horizon "${horizon}" \
        --device "${device}"
elif [[ ! -f "${expert_data%.npz}.json" ]] \
    || ! grep -q 'manisoft_history_bc_kmpc_three_waypoint_expert' \
        "${expert_data%.npz}.json"; then
    echo "Existing expert dataset is not the three-waypoint schema: ${expert_data}" >&2
    echo "Use a new EXPERT_DATA path." >&2
    exit 1
fi
if ! grep -q '"schema_version": 5' "${expert_data%.npz}.json"; then
    echo "Existing expert dataset predates randomized waypoint-bank BC-KMPC: ${expert_data}" >&2
    echo "Use a new EXPERT_DATA path." >&2
    exit 1
fi

bc_checkpoint="${bc_output}/best_validation.pt"
bc_complete=false
if [[ -f "${bc_output}/training_status.json" ]] \
    && grep -q '"state": "complete"' "${bc_output}/training_status.json"; then
    bc_complete=true
fi
if [[ "${bc_complete}" != true ]]; then
    resume_bc=()
    if [[ -f "${bc_output}/last.pt" ]]; then
        resume_bc=(--resume "${bc_output}/last.pt")
    fi
    "${python_bin}" scripts/train_manisoft_bc_kmpc_bc.py \
        --koopman-checkpoint "${koopman_checkpoint}" \
        --dataset "${expert_data}" \
        --output "${bc_output}" \
        --epochs "${BC_KMPC_BC_EPOCHS:-150}" \
        --batch-size "${BC_KMPC_BC_BATCH_SIZE:-256}" \
        --horizon "${horizon}" \
        --solver-iterations "${solver_iterations}" \
        --quadratic-log-scale "${quadratic_log_scale}" \
        --linear-scale "${linear_scale}" \
        --action-quadratic-scale "${action_quadratic_scale}" \
        --sequence-weight "${BC_KMPC_SEQUENCE_WEIGHT:-0.25}" \
        --device "${device}" \
        "${resume_bc[@]}"
fi
if [[ ! -f "${bc_checkpoint}" ]]; then
    echo "BC training did not produce ${bc_checkpoint}" >&2
    exit 1
fi

ppo_initialization=(--bc-checkpoint "${bc_checkpoint}")
if [[ -f "${ppo_output}/last.pt" ]]; then
    ppo_initialization=(--resume "${ppo_output}/last.pt")
fi
"${python_bin}" scripts/train_manisoft_bc_kmpc_ppo.py \
    --koopman-checkpoint "${koopman_checkpoint}" \
    --scenario "${scenario}" \
    --waypoint-root "${waypoint_root}" \
    --output "${ppo_output}" \
    --episode-steps "${BC_KMPC_EPISODE_STEPS:-300}" \
    --horizon "${horizon}" \
    --solver-iterations "${solver_iterations}" \
    --quadratic-log-scale "${quadratic_log_scale}" \
    --linear-scale "${linear_scale}" \
    --action-quadratic-scale "${action_quadratic_scale}" \
    --num-envs "${BC_KMPC_NUM_ENVS:-1}" \
    --actor-learning-rate "${BC_KMPC_ACTOR_LEARNING_RATE:-0.0001}" \
    --target-kl "${BC_KMPC_TARGET_KL:-0.02}" \
    --device "${device}" \
    "${ppo_initialization[@]}"
