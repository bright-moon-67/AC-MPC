#!/usr/bin/env bash
# Resume/continue PPO-KMPC v15 experiments a / b / e from their last.pt
# checkpoints for another 24h (wall-time 24h -> 48h).
#
# These are the three best-performing KMPC configs from the v15 zmixed run.
# Each command reuses the exact original hyperparameters (runtime +
# training_signature checks in train_manisoft_ppo_comparison.py must match)
# and only changes: --resume <last.pt>, --max-wall-time-hours 48.
#
# Detached: bash scripts/launch_manisoft_ppo_compare_v15_abc_continue.sh
set -euo pipefail

cd /root/autodl-tmp/AC-MPC

PY=/root/miniconda3/envs/manisoft/bin/python
K=work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt
S=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
W=data/processed/manisoft_waypoint_bank_v2_zmixed_merged
OUT=runs/manisoft_ppo_compare_v15_zmixed_24h

COMMON=(
  --koopman-checkpoint "$K"
  --scenario "$S"
  --waypoint-root "$W"
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
)

launch() {
  local name="$1"; shift
  local out_dir="$OUT/$name"
  screen -dmS "ppo_v15_cont_${name}" bash -lc "cd /root/autodl-tmp/AC-MPC && exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 $PY -u scripts/train_manisoft_ppo_comparison.py ${COMMON[*]} $* --output $out_dir --resume $out_dir/last.pt >>$out_dir/launcher_continue.log 2>&1"
  echo "launched: ppo_v15_cont_${name}"
}

# a: r8 / lr3e-5 / std15 / md015
launch a_kmpc_r8_lr3_std15_md015 \
  --actor-learning-rate 3e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# b: r8 / lr2.5e-5 / std15 / md015
launch b_kmpc_r8_lr25_std15_md015 \
  --actor-learning-rate 2.5e-5 \
  --initial-action-std 0.15 \
  --max-delta 0.015

# e: r8 / lr3e-5 / std18 / md0125
launch e_kmpc_r8_lr3_std18_md0125 \
  --actor-learning-rate 3e-5 \
  --initial-action-std 0.18 \
  --max-delta 0.0125

echo "all launched"
