#!/usr/bin/env bash
set -euo pipefail

# Override these variables to scale or relocate a run.  The default values are
# the recommended first complete experiment, not a smoke test.
WORKSPACE=${WORKSPACE:-/root/autodl-tmp}
AC_ROOT=${AC_ROOT:-$WORKSPACE/AC-MPC}
MANISOFT_ROOT=${MANISOFT_ROOT:-$WORKSPACE/ManiSoft}
RUN_ROOT=${RUN_ROOT:-$AC_ROOT/runs/manisoft_kinematic_push}
KOOPMAN_OUTPUT=${KOOPMAN_OUTPUT:-$RUN_ROOT/koopman}
KOOPMAN_CHECKPOINT=${KOOPMAN_CHECKPOINT:-$KOOPMAN_OUTPUT/koopman_history/best_validation.pt}
SCENARIO=${SCENARIO:-$MANISOFT_ROOT/configs/push_around_obstacle_kinematic.yaml}
EXPERT_EPISODES=${EXPERT_EPISODES:-40}
BC_EPOCHS=${BC_EPOCHS:-100}
PPO_TIMESTEPS=${PPO_TIMESTEPS:-200000}
STAGES=${STAGES:-koopman,expert,bc,ppo,evaluate,render}

EXPERT_DATASET=${EXPERT_DATASET:-$RUN_ROOT/expert/expert.npz}
BC_CHECKPOINT=${BC_CHECKPOINT:-$RUN_ROOT/bc/best_validation.pt}
PPO_CHECKPOINT=${PPO_CHECKPOINT:-$RUN_ROOT/ppo/best.pt}
TRAJECTORY=${TRAJECTORY:-$RUN_ROOT/evaluation/trajectory.npz}
VIDEO=${VIDEO:-$RUN_ROOT/evaluation/trajectory.mp4}

export PYTHONPATH=$MANISOFT_ROOT${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$RUN_ROOT"

run_stage() {
  [[ ",$STAGES," == *",$1,"* ]]
}

cd "$AC_ROOT"
if run_stage koopman; then
  conda run -n manisoft python scripts/train_koopman_history.py \
    --config configs/manisoft_coll.yaml \
    --data "$MANISOFT_ROOT/work_dirs/koopman_workspace_5m_v1" \
    --data "$MANISOFT_ROOT/work_dirs/koopman_table_plane_supplement_v1" \
    --data "$MANISOFT_ROOT/work_dirs/koopman_table_plane_supplement_v2" \
    --output "$KOOPMAN_OUTPUT" \
    --history-steps 10 \
    --tip-position-weight 10000 \
    --rate-stratified-sampling \
    --max-wall-time-hours "${KOOPMAN_HOURS:-5}" \
    --device auto
fi

if run_stage expert; then
  conda run -n manisoft python scripts/collect_manisoft_kinematic_push_expert.py \
    --koopman-checkpoint "$KOOPMAN_CHECKPOINT" \
    --scenario "$SCENARIO" \
    --output "$EXPERT_DATASET" \
    --episodes "$EXPERT_EPISODES" \
    --episode-steps 600 \
    --route-sides both \
    --minimum-contact-fraction 0.10 \
    --minimum-success-fraction 0.05 \
    --device auto
fi

if run_stage bc; then
  conda run -n manisoft python scripts/train_manisoft_kinematic_push_bc.py \
    --koopman-checkpoint "$KOOPMAN_CHECKPOINT" \
    --dataset "$EXPERT_DATASET" \
    --output "$RUN_ROOT/bc" \
    --epochs "$BC_EPOCHS" \
    --phase-balanced-sampling \
    --device auto
fi

if run_stage ppo; then
  conda run -n manisoft python scripts/train_manisoft_kinematic_push_ppo.py \
    --koopman-checkpoint "$KOOPMAN_CHECKPOINT" \
    --scenario "$SCENARIO" \
    --initial-checkpoint "$BC_CHECKPOINT" \
    --expert-dataset "$EXPERT_DATASET" \
    --output "$RUN_ROOT/ppo" \
    --total-timesteps "$PPO_TIMESTEPS" \
    --bc-coefficient 0.1 \
    --bc-updates-per-rollout 1 \
    --device auto
fi

if run_stage evaluate; then
  conda run -n manisoft python scripts/evaluate_manisoft_kinematic_push.py \
    --policy-checkpoint "$PPO_CHECKPOINT" \
    --scenario "$SCENARIO" \
    --output "$TRAJECTORY" \
    --episodes 2 \
    --device auto
fi

if run_stage render; then
  cd "$MANISOFT_ROOT"
  conda run -n manisoft python scripts/render_push_scene.py \
    --config "$SCENARIO" \
    --trajectory "$TRAJECTORY" \
    --episode 0 \
    --output "$VIDEO" \
    --speed 4 \
    --width 640 \
    --height 480 \
    --fps 24 \
    --legend
fi

printf '%s\n' "pipeline output: $RUN_ROOT" "video: $VIDEO"
