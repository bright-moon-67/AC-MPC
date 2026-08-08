#!/usr/bin/env bash
# Parallel from-scratch PPO for the four HopperHop actors
# (PPO / KLQR / AB-PQ / KMPC), one process per actor.
#
#   PPO4_NUM_ENVS        parallel envs per actor process (default 2048, the
#                         official PPO baseline config)
#   PPO4_ROLLOUT_STEPS   rollout steps per update (default 100)
#   PPO4_TOTAL_TIMESTEPS per-actor budget (default 100M)
#   PPO4_INITIAL_STD     Gaussian exploration std at start (default 1.0, the
#                         official PPO baseline config)
#   PPO4_SEED            shared seed across actors (paired comparison, default 20240801)
#   PPO4_LR              learning rate (default 3e-4)
#   PPO4_OUTPUT_ROOT     runs/hopper_hop/ppo_fair
#   PPO4_KOOPMAN         runs/hopper_hop/koopman_v2/best.pt
#
# Detached: tmux new-session -d -s ppo4 "bash scripts/run_hopper_hop_ppo4.sh"
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${PPO4_OUTPUT_ROOT:-${project_root}/runs/hopper_hop/ppo_fair}"
koopman="${PPO4_KOOPMAN:-${project_root}/runs/hopper_hop/koopman_v2/best.pt}"
num_envs="${PPO4_NUM_ENVS:-2048}"
rollout_steps="${PPO4_ROLLOUT_STEPS:-100}"
total_timesteps="${PPO4_TOTAL_TIMESTEPS:-100000000}"
initial_std="${PPO4_INITIAL_STD:-1.0}"
seed="${PPO4_SEED:-20240801}"
learning_rate="${PPO4_LR:-0.0003}"
actors="${PPO4_ACTORS:-PPO KLQR AB-PQ KMPC}"
# minibatch = num_envs * rollout / 32 minibatches (matches the PPO baseline)
minibatch_size=$(( num_envs * rollout_steps / 32 ))

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${koopman}" ]]; then
    echo "Koopman checkpoint not found: ${koopman}" >&2
    exit 2
fi

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"
echo "num_envs=${num_envs} minibatch=${minibatch_size} total=${total_timesteps} std=${initial_std}" > "${output_root}/launcher.conf"

pids=()
for actor in ${actors}; do
    actor_dir="${output_root}/${actor}"
    mkdir -p "${actor_dir}"
    (
        cd "${project_root}"
        exec "${python_bin}" -u -m \
            experiments.hopper_hop.train_hopper_hop_ppo_actors \
            --actor "${actor}" \
            --koopman "${koopman}" \
            --output-dir "${actor_dir}" \
            --num-envs "${num_envs}" \
            --rollout-steps "${rollout_steps}" \
            --minibatch-size "${minibatch_size}" \
            --total-timesteps "${total_timesteps}" \
            --learning-rate "${learning_rate}" \
            --initial-std "${initial_std}" \
            --seed "${seed}"
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
echo "HopperHop 4-method PPO done. Output: ${output_root}"
