#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/AC-MPC
python_bin=/root/miniconda3/envs/manisoft/bin/python
output_root=runs/manisoft_ppo_kmpc_sweep_v13_structured_sustained

common_args=(
  -u scripts/train_manisoft_ppo_comparison.py
  --actor ppo_kmpc
  --koopman-checkpoint work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
  --waypoint-root data/processed/manisoft_waypoint_bank_v1_merged
  --episode-steps 300
  --absolute-action-limit 0.30
  --max-delta 0.015
  --kmpc-cost-parameterization structured
  --horizon 10
  --solver-iterations 80
  --solver-diagnostic-iterations 320
  --normalized-delta-curvature 0
  --total-timesteps 600000
  --rollout-steps 4096
  --num-envs 16
  --parallel-env-processes
  --minibatch-size 512
  --learning-rate 1e-4
  --std-learning-rate 1e-6
  --freeze-log-std
  --no-anneal-learning-rate
  --gamma 0.99
  --gae-lambda 0.95
  --clip-range 0.2
  --clip-value-loss
  --value-coefficient 0.5
  --entropy-coefficient 1e-4
  --minimum-action-std 0.001
  --maximum-action-std 0.20
  --max-grad-norm 0.5
  --target-kl 0.02
  --kl-soft-stop-multiplier 1.5
  --kl-hard-rollback-multiplier 3.0
  --normalize-advantages-globally
  --checkpoint-interval-updates 5
  --max-wall-time-hours 12
  --device cuda
  --seed 42
)

launch_one() {
  local experiment_name=$1
  shift
  local screen_name="kmpc_v13_${experiment_name}"
  local output_dir="${output_root}/${experiment_name}"
  local launcher_log="${repo_root}/${output_dir}/launcher.log"

  if screen -list | grep -q "[.]${screen_name}[[:space:]]"; then
    echo "screen already exists: ${screen_name}" >&2
    return 1
  fi
  if [[ -e "${repo_root}/${output_dir}/history.jsonl" ]]; then
    echo "refusing to overwrite existing run: ${output_dir}" >&2
    return 1
  fi

  mkdir -p "${repo_root}/${output_dir}"
  local command=(
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    "${python_bin}" "${common_args[@]}"
    --output "${output_dir}"
    "$@"
  )
  local quoted_command
  printf -v quoted_command ' %q' "${command[@]}"
  local quoted_root quoted_log
  printf -v quoted_root '%q' "${repo_root}"
  printf -v quoted_log '%q' "${launcher_log}"
  screen -dmS "${screen_name}" bash -lc \
    "cd ${quoted_root} && exec${quoted_command} >>${quoted_log} 2>&1"
  echo "started ${screen_name} -> ${output_dir}"
}

# Central candidate: more expressive [0.25, 4] multipliers and a larger actor step.
launch_one a_range4_lr5_std10 \
  --structured-log-scale 1.3862943611198906 \
  --kmpc-hidden-dims 128 \
  --actor-learning-rate 5e-5 \
  --update-epochs 4 \
  --initial-action-std 0.10 \
  --progress-reward-scale 1.0

# Larger per-step policy movement with fewer repeated epochs over one rollout.
launch_one b_range4_lr7_e3_std10 \
  --structured-log-scale 1.3862943611198906 \
  --kmpc-hidden-dims 128 \
  --actor-learning-rate 7e-5 \
  --update-epochs 3 \
  --initial-action-std 0.10 \
  --progress-reward-scale 1.0

# More state-space exploration while preserving the same deterministic mean actor.
launch_one c_range4_lr5_std15 \
  --structured-log-scale 1.3862943611198906 \
  --kmpc-hidden-dims 128 \
  --actor-learning-rate 5e-5 \
  --update-epochs 4 \
  --initial-action-std 0.15 \
  --progress-reward-scale 1.0

# Wider [0.125, 8] structured cost range, with a conservative actor step.
launch_one d_range8_lr3_std10 \
  --structured-log-scale 2.0794415416798357 \
  --kmpc-hidden-dims 128 \
  --actor-learning-rate 3e-5 \
  --update-epochs 4 \
  --initial-action-std 0.10 \
  --progress-reward-scale 1.0

# Higher-capacity context-to-cost map; the QP still has only five learned outputs.
launch_one e_range4_h256x128_lr5 \
  --structured-log-scale 1.3862943611198906 \
  --kmpc-hidden-dims 256 128 \
  --actor-learning-rate 5e-5 \
  --update-epochs 4 \
  --initial-action-std 0.10 \
  --progress-reward-scale 1.0

# Stronger dense progress signal to test whether sparse stage credit caused the plateau.
launch_one f_range4_lr5_prog2 \
  --structured-log-scale 1.3862943611198906 \
  --kmpc-hidden-dims 128 \
  --actor-learning-rate 5e-5 \
  --update-epochs 4 \
  --initial-action-std 0.10 \
  --progress-reward-scale 2.0
