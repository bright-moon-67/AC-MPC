# Smoke test report — 2026-07-26

These checks validate interfaces and numerical health only. Short runs and
zero-success episodes are not formal learning results.

## Data and Koopman

- Rebuilt all 999,999 D4RL transitions with schema version 2.
- Verified every old/new field pair is byte-equivalent:
  `x/state`, `delta_action/action`, `next_x/next_state`, and
  `action/current_action`.
- Verified `done == terminal OR timeout`.
- Maximum reconstruction error for
  `current_action == previous_action + action`:
  `5.960464477539063e-08`.
- Loaded the schema-v2 test split through the unchanged Koopman window API.
- Full horizon-1 test inference evaluated 97,942 valid non-boundary
  transitions from 98,956 rows; all reported errors were finite.

## Legacy environment and fixed controller

- D4RL 1.1, Gym 0.23.1, mujoco-py 2.1.2.14.
- Raw/augmented/action dimensions: 29 / 37 / 8.
- Episode limit: 700.
- Reset and zero-action previous-action checks passed.
- One complete fixed-cost Koopman-LQR episode was finite.
- DARE retry/fallback: 0 / 0.
- Maximum relative DARE residual:
  `8.052172495710942e-10`.
- Closed-loop spectral radius: approximately `0.998231465`.

## Offline TD3+BC

- Unit update covered BC-only warm-up and active TD3+BC Q objective.
- Real CUDA training passed at batch sizes 16 and 256.
- Checkpoint save and resume reproduced steps 1, 2, 3 without duplicate
  history.
- A 10-step batch-256 profile reached approximately 0.688 updates/s after
  removing per-step GPU synchronization.
- Validation BC MSE moved from 1.3294 to 1.3220 in the 10-step interface
  smoke; this is not a convergence claim.
- DARE retry/fallback remained 0 / 0.
- Maximum validation relative DARE residual:
  `3.2954072148656e-09`.
- Maximum validation closed-loop spectral radius:
  `0.9982375126594561`.
- A deterministic legacy 700-step evaluation and trajectory PNG/NPZ
  generation completed.

## PPO paths

- Fresh Koopman-LQR PPO: one 16-transition CUDA/legacy update completed with
  finite gradients and zero DARE retry/fallback.
- TD3+BC-initialized Koopman-LQR PPO: actor import, one 16-transition update,
  checkpoint, and initialization provenance all completed.
- Delta-PPO: one 16-transition CUDA/legacy update completed.

## Static and unit checks

- Python compile check: passed.
- All shell launchers: `bash -n` passed.
- YAML/config and schema/action support assertions: passed.
- Python package dependency check: no broken requirements.
- Test suite: **35 passed in 5.19 seconds** using
  `python -m pytest -q`.

Formal offline training, long PPO training, five seeds, and 100-episode model
selection were intentionally not run as part of this smoke gate.
