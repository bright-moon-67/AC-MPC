#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/AC-MPC
python_bin=/root/miniconda3/envs/manisoft/bin/python
output_root=runs/manisoft_ppo_compare_v15_zmixed_24h

# All six runs share the same Koopman lift, 904-triplet z-mixed waypoint bank,
# reward, PPO batch/update settings, critic learning rate, and random seed.
common_args=(
  -u scripts/train_manisoft_ppo_comparison.py
  --koopman-checkpoint work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
  --waypoint-root data/processed/manisoft_waypoint_bank_v2_zmixed_merged
  --episode-steps 300
  --absolute-action-limit 0.30
  --progress-reward-scale 1.0
  --horizon 10
  --total-timesteps 100000000
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
  --checkpoint-interval-updates 50
  --max-wall-time-hours 24
  --device cuda
  --seed 42
)

launch_one() {
  local experiment_name=$1
  shift
  local screen_name="ppo_v15_${experiment_name}"
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
    env
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    PYTHONUNBUFFERED=1
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

kmpc_common=(
  --actor ppo_kmpc
  --kmpc-cost-parameterization structured
  --kmpc-hidden-dims 128
  --solver-iterations 80
  --solver-diagnostic-iterations 320
  --normalized-delta-curvature 0
)

# Required baseline: the strongest late-training branch from v14.
launch_one a_kmpc_r8_lr3_std15_md015 \
  "${kmpc_common[@]}" \
  --structured-log-scale 2.0794415416798357 \
  --actor-learning-rate 3e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# Reduce actor movement by 1/6 to suppress the baseline's remaining KL spikes.
launch_one b_kmpc_r8_lr25_std15_md015 \
  "${kmpc_common[@]}" \
  --structured-log-scale 2.0794415416798357 \
  --actor-learning-rate 2.5e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# Keep the winning actor/cost range while reducing rollout exploration noise.
launch_one c_kmpc_r8_lr3_std13_md015 \
  "${kmpc_common[@]}" \
  --structured-log-scale 2.0794415416798357 \
  --actor-learning-rate 3e-5 \
  --initial-action-std 0.13 \
  --max-delta 0.015

# Interpolate between the strong range-6/lr4e-5 and winning range-8/lr3e-5 runs.
launch_one d_kmpc_r7_lr35_std15_md015 \
  "${kmpc_common[@]}" \
  --structured-log-scale 1.9459101490553132 \
  --actor-learning-rate 3.5e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# Tighten the physical rate limit while preserving baseline physical noise:
# 0.18 * 0.0125 == 0.15 * 0.015 == 0.00225.
launch_one e_kmpc_r8_lr3_std18_md0125 \
  "${kmpc_common[@]}" \
  --structured-log-scale 2.0794415416798357 \
  --actor-learning-rate 3e-5 \
  --initial-action-std 0.18 \
  --max-delta 0.0125

# MLP control: same lifted history/context, dataset, PPO settings, actor LR and
# numerical std as the required KMPC baseline. Its actor emits absolute action,
# so max-delta is intentionally not an operative constraint in this branch.
launch_one f_mlp_lr3_std15 \
  --actor ppo_mlp \
  --kmpc-cost-parameterization full \
  --mlp-hidden-dims 256 256 \
  --actor-learning-rate 3e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015
