#!/usr/bin/env bash
# Protected four-way visual PickCube comparison using the published v3.0.1
# PPO optimization budget.  This host is limited to 128 camera envs/process,
# so several frozen-policy 16-step chunks are aggregated to batch 16,384.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AC_MPC_PYTHON:-/root/miniconda3/bin/python}"
output_root="${VPICK_OUTPUT_ROOT:-${project_root}/runs/visual_pickcube_official50m}"
koopman="${VPICK_KOOPMAN:-${project_root}/runs/pickcube_robot_koopman_coverage/best.pt}"
num_envs="${VPICK_NUM_ENVS:-128}"
total_timesteps="${VPICK_TOTAL_TIMESTEPS:-50000000}"
seed="${VPICK_SEED:-20280804}"
actors="${VPICK_ACTORS:-PPO KMPC AB-PQ AC-MPC-MPVE}"
max_restarts="${VPICK_MAX_RESTARTS:-20}"
target_batch=16384
rollout_steps=16

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${koopman}" ]]; then
    echo "Koopman checkpoint not found: ${koopman}" >&2
    exit 2
fi
chunk_size=$(( num_envs * rollout_steps ))
if (( target_batch % chunk_size != 0 )); then
    echo "num_envs*16 must divide the published batch 16384" >&2
    exit 2
fi
collection_chunks=$(( target_batch / chunk_size ))

mkdir -p "${output_root}"
printf '%s\n' "$$" > "${output_root}/launcher.pid"
{
    echo "published=1024env x 16step x 1chunk = 16384"
    echo "adapted=${num_envs}env x ${rollout_steps}step x ${collection_chunks}chunks = ${target_batch}"
    echo "epochs=8 minibatches=32 minibatch_size=512 total=${total_timesteps}"
    echo "gamma=0.8 gae_lambda=0.9 target_kl=0.2 anneal_lr=false"
    echo "actors=${actors} seed=${seed}"
    echo "checkpoint=atomic latest every 10; 2 rolling recovery; milestones every 250"
} > "${output_root}/launcher.conf"

common=(
    --koopman "${koopman}"
    --num-envs "${num_envs}"
    --total-timesteps "${total_timesteps}"
    --seed "${seed}"
    --rollout-steps "${rollout_steps}"
    --collection-chunks "${collection_chunks}"
    --update-epochs 8
    --minibatch-size 512
    --learning-rate 3e-4
    --discount 0.8
    --gae-lambda 0.9
    --target-kl 0.2
    --anneal-lr false
    --clip-vloss false
    --eval-interval-updates 25
    --num-eval-envs 16
    --num-eval-steps 50
    --checkpoint-interval-updates 10
    --recovery-checkpoints-to-keep 2
    --milestone-interval-updates 250
    --mpve-horizon 5
    --mpve-value-loss-coefficient 1.0
)

run_actor() {
    local actor="$1"
    local actor_dir="${output_root}/${actor}"
    local attempt=0
    mkdir -p "${actor_dir}"
    if [[ -f "${actor_dir}/PAUSED" ]]; then
        printf '%s\n' "paused by marker ${actor_dir}/PAUSED" \
            > "${actor_dir}/supervisor.status"
        return 0
    fi
    while (( attempt <= max_restarts )); do
        attempt=$((attempt + 1))
        printf '%s\n' "running attempt=${attempt} pid=$$ time=$(date --iso-8601=seconds)" \
            > "${actor_dir}/supervisor.status"
        if (
            cd "${project_root}"
            exec "${python_bin}" -u -m \
                experiments.maniskill_pick_visual.train_visual_pickcube_ppo \
                --actor "${actor}" \
                --output-dir "${actor_dir}" \
                "${common[@]}"
        ) >> "${actor_dir}/console.log" 2>&1; then
            printf '%s\n' "complete attempt=${attempt} time=$(date --iso-8601=seconds)" \
                > "${actor_dir}/supervisor.status"
            return 0
        fi
        printf '%s\n' "retrying attempt=${attempt} time=$(date --iso-8601=seconds)" \
            > "${actor_dir}/supervisor.status"
        sleep 10
    done
    printf '%s\n' "failed attempts=${max_restarts} time=$(date --iso-8601=seconds)" \
        > "${actor_dir}/supervisor.status"
    return 1
}

pids=()
for actor in ${actors}; do
    run_actor "${actor}" &
    pids+=("$!")
    sleep 5
done

failure=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failure=1
    fi
done
if (( failure )); then
    printf '%s\n' "failed" > "${output_root}/launcher.status"
    exit 1
fi
printf '%s\n' "complete" > "${output_root}/launcher.status"
