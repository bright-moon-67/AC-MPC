#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/AC-MPC
python_bin=/root/miniconda3/envs/manisoft/bin/python
output_root=runs/manisoft_ppo_kmpc_long_v14_sustained_12h

common_args=(
  -u scripts/train_manisoft_ppo_comparison.py
  --actor ppo_kmpc
  --koopman-checkpoint work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
  --waypoint-root data/processed/manisoft_waypoint_bank_v1_merged
  --episode-steps 300
  --absolute-action-limit 0.30
  --kmpc-cost-parameterization structured
  --kmpc-hidden-dims 128
  --horizon 10
  --solver-iterations 80
  --solver-diagnostic-iterations 320
  --normalized-delta-curvature 0
  --progress-reward-scale 1.0
  --total-timesteps 5000000
  --rollout-steps 4096
  --num-envs 16
  --parallel-env-processes
  --minibatch-size 512
  --update-epochs 4
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
  --checkpoint-interval-updates 10
  --max-wall-time-hours 12.5
  --device cuda
  --seed 42
)

launch_one() {
  local experiment_name=$1
  shift
  local screen_name="kmpc_v14_${experiment_name}"
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
  local quoted_command quoted_root quoted_log
  printf -v quoted_command ' %q' "${command[@]}"
  printf -v quoted_root '%q' "${repo_root}"
  printf -v quoted_log '%q' "${launcher_log}"
  screen -dmS "${screen_name}" bash -lc \
    "cd ${quoted_root} && exec${quoted_command} >>${quoted_log} 2>&1"
  echo "started ${screen_name} -> ${output_dir}"
}

# Short-sweep winner: strongest late success/waypoint trend with stable KL.
launch_one a_best_r4_lr5_std15_md015 \
  --structured-log-scale 1.3862943611198906 \
  --actor-learning-rate 5e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# Slightly smaller actor step to reduce rare KL spikes over a 12.5-hour run.
launch_one b_r4_lr4_std15_md015 \
  --structured-log-scale 1.3862943611198906 \
  --actor-learning-rate 4e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# Intermediate exploration between the old std=0.10 and the winning std=0.15.
launch_one c_r4_lr5_std13_md015 \
  --structured-log-scale 1.3862943611198906 \
  --actor-learning-rate 5e-5 \
  --initial-action-std 0.13 \
  --max-delta 0.015

# Intermediate multiplier range [1/6, 6] between the good range-4 and range-8 runs.
launch_one d_r6_lr4_std15_md015 \
  --structured-log-scale 1.791759469228055 \
  --actor-learning-rate 4e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# Preserve the late-improving wide-range branch, with the winning exploration level.
launch_one e_r8_lr3_std15_md015 \
  --structured-log-scale 2.0794415416798357 \
  --actor-learning-rate 3e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# Larger physical action-rate envelope. std=0.1125 keeps sigma*max_delta
# equal to 0.15*0.015, isolating the rate-limit change from exploration scale.
launch_one f_r4_lr4_std1125_md020 \
  --structured-log-scale 1.3862943611198906 \
  --actor-learning-rate 4e-5 \
  --initial-action-std 0.1125 \
  --max-delta 0.020
