#!/usr/bin/env bash
# Full visual PickCube PPO runs: PPO (official ManiSkill params) -> KMPC -> AB-PQ.
#
# The PPO route uses the official ManiSkill3 PPO RGB parameters
# (examples/baselines/ppo/ppo_rgb.py): gamma=0.8, gae_lambda=0.9,
# update_epochs=4, rollout_steps=50, minibatch=batch/32, target_kl=0.2,
# NO lr annealing, lr=3e-4, log_std=-0.5.  KMPC / AB-PQ share the same base
# PPO hyperparameters (only the actor head differs), matching the HopperHop
# fairness principle.
#
# Hardware-bound knobs are configurable:
#   VPICK_NUM_ENVS         parallel envs (official 512; this host's Vulkan
#                          camera-group limit is ~128, so default 128)
#   VPICK_TOTAL_TIMESTEPS  official 10M
#   VPICK_ACTORS           space-separated, run sequentially (default PPO KMPC AB-PQ)
#   VPICK_KOOPMAN          runs/pickcube_robot_koopman_coverage/best.pt
#   VPICK_OUTPUT_ROOT      runs/visual_pickcube_ppo
#   VPICK_SEED             training seed
#
# Detached: tmux new-session -d -s vpick "bash scripts/run_visual_pickcube_ppo.sh"
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${VPICK_OUTPUT_ROOT:-${project_root}/runs/visual_pickcube_ppo}"
koopman="${VPICK_KOOPMAN:-${project_root}/runs/pickcube_robot_koopman_coverage/best.pt}"
num_envs="${VPICK_NUM_ENVS:-128}"
total_timesteps="${VPICK_TOTAL_TIMESTEPS:-10000000}"
seed="${VPICK_SEED:-20280804}"
actors="${VPICK_ACTORS:-PPO KMPC AB-PQ}"

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
{
    echo "num_envs=${num_envs} total=${total_timesteps} seed=${seed}"
    echo "actors=${actors}"
} > "${output_root}/launcher.conf"

# Official ManiSkill PPO RGB algorithm knobs (shared by all actor routes).
common=(
    --koopman "${koopman}"
    --num-envs "${num_envs}"
    --total-timesteps "${total_timesteps}"
    --seed "${seed}"
    --rollout-steps 50
    --update-epochs 4
    --minibatch-size 0   # placeholder; real value = batch/32 computed below
    --learning-rate 3e-4
    --discount 0.8
    --gae-lambda 0.9
    --target-kl 0.2
    --anneal-lr false
    --clip-vloss false
)

# Replace the placeholder minibatch with the official batch/32 (computed here
# because the launcher knows num_envs before the trainer does).
batch_size=$(( num_envs * 50 ))
official_minibatch=$(( batch_size / 32 ))
if (( official_minibatch < 1 )); then official_minibatch=1; fi
for i in "${!common[@]}"; do
    if [[ "${common[$i]}" == "--minibatch-size" ]]; then
        common[$((i + 1))]="${official_minibatch}"
    fi
done

failure=0
for actor in ${actors}; do
    actor_dir="${output_root}/${actor}"
    mkdir -p "${actor_dir}"
    echo "===== $(date +%H:%M:%S) launching ${actor} ====="
    if ! (
        cd "${project_root}"
        exec "${python_bin}" -u -m \
            experiments.maniskill_pick_visual.train_visual_pickcube_ppo \
            --actor "${actor}" \
            --output-dir "${actor_dir}" \
            "${common[@]}"
    ) > "${actor_dir}/console.log" 2>&1; then
        echo "actor ${actor} FAILED" >&2
        failure=1
    else
        echo "===== $(date +%H:%M:%S) ${actor} DONE ====="
    fi
done

if (( failure != 0 )); then
    printf '%s\n' "failed" > "${output_root}/launcher.status"
    exit 1
fi
printf '%s\n' "complete" > "${output_root}/launcher.status"
echo "Visual PickCube PPO done. Output: ${output_root}"
