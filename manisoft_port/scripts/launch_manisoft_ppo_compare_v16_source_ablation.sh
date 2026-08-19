#!/usr/bin/env bash
# Three paired, single-factor ablations relative to v15e.
#
# Q2: keep v15e structured q and terminal multiplier; replace only explicit
#     -Q*x_ref/-R*u_ref with upstream-style learned stage-wise p.
# Q3: keep the v15e cost/reference/terminal; replace only normalized-delta
#     rate-constrained MPC with direct absolute-U box MPC.  The trainer maps
#     sigma_D=.18 through max_delta=.0125 so physical sigma remains .00225.
# Q8: keep v15e cost/reference/decision space; fix the extra terminal
#     multiplier to one while retaining the unused fifth output head.
set -euo pipefail

repo_root=/root/autodl-tmp/AC-MPC
python_bin=/root/miniconda3/envs/manisoft/bin/python
output_root=runs/manisoft_ppo_compare_v16_source_ablation
wait_for_v15=false

if [[ "${1:-}" == "--wait-for-v15" ]]; then
  wait_for_v15=true
  shift
fi
if (($#)); then
  echo "usage: $0 [--wait-for-v15]" >&2
  exit 2
fi

cd "$repo_root"

if $wait_for_v15; then
  echo "waiting for v15 a/b/e continuation screens to finish"
  while screen -list 2>/dev/null \
    | grep -Eq '[.]ppo_v15_cont_(a|b|e)_'; do
    sleep 60
  done
fi

# Pinned after the implementation passed its test suite.  This prevents a
# queued run from silently starting with later runtime-critical edits.
expected_source_sha256=6babb38f29227a57654423d78ec7c4b15a9093d1f8d76c755063f8917218a7c9
source_files=(
  antmaze_ac/rl/koopman_mpc_actor.py
  antmaze_ac/rl/history_koopman_mpc_policy.py
  antmaze_ac/rl/manisoft_ppo_policies.py
  scripts/train_manisoft_ppo_comparison.py
)
actual_source_sha256=$(
  sha256sum "${source_files[@]}" | sha256sum | awk '{print $1}'
)
if [[ "$actual_source_sha256" != "$expected_source_sha256" ]]; then
  echo "source hash mismatch; refusing to start queued ablations" >&2
  echo "expected=$expected_source_sha256" >&2
  echo "actual=$actual_source_sha256" >&2
  exit 1
fi

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
  --max-wall-time-hours 48
  --device cuda
  --seed 42
  --actor ppo_kmpc
  --kmpc-cost-parameterization structured
  --kmpc-hidden-dims 128
  --solver-iterations 80
  --solver-diagnostic-iterations 320
  --normalized-delta-curvature 0
  --structured-log-scale 2.0794415416798357
  --actor-learning-rate 3e-5
  --initial-action-std 0.18
  --max-delta 0.0125
)

launch_one() {
  local experiment_name=$1
  shift
  local screen_name="ppo_v16_${experiment_name}"
  local output_dir="${output_root}/${experiment_name}"
  local absolute_output="${repo_root}/${output_dir}"
  local launcher_log="${absolute_output}/launcher.log"

  if screen -list 2>/dev/null \
    | grep -q "[.]${screen_name}[[:space:]]"; then
    echo "screen already exists: ${screen_name}" >&2
    return 1
  fi
  if ! mkdir "$absolute_output"; then
    echo "refusing to reuse existing run directory: ${output_dir}" >&2
    return 1
  fi

  local command=(
    env
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    PYTHONUNBUFFERED=1
    "$python_bin" "${common_args[@]}"
    --output "$output_dir"
    "$@"
  )
  local quoted_command quoted_root quoted_log
  printf -v quoted_command ' %q' "${command[@]}"
  printf -v quoted_root '%q' "$repo_root"
  printf -v quoted_log '%q' "$launcher_log"
  screen -dmS "$screen_name" bash -lc \
    "cd ${quoted_root} && exec${quoted_command} >>${quoted_log} 2>&1"
  echo "started ${screen_name} -> ${output_dir}"
}

launch_one q2_source_implicit_reference \
  --kmpc-reference-mode implicit

launch_one q3_source_absolute_box \
  --kmpc-decision-space absolute

launch_one q8_source_no_terminal \
  --no-structured-terminal-multiplier
