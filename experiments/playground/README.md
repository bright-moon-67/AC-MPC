# MuJoCo Playground GPU comparison

This branch ports the DMC comparison to the official GPU-native MuJoCo
Playground environments.  It deliberately avoids the DMC approval/fingerprint
pipeline: runs have one config source, atomic framework checkpoints, JSONL
curves, and deterministic post-training evaluation.

The isolated environment is reproducible with:

```bash
python -m venv .venv
.venv/bin/pip install -r experiments/playground/requirements.txt
```

Always disable JAX's large default memory reservation so the A100 can be
shared with Koopman training and other methods:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=.
```

The pinned stack is Playground commit `d5e6b475`, Brax 0.14.2, JAX 0.11.0,
MuJoCo 3.11.0, and Warp 1.16.0.  `jax_compat.py` supplies the single removed
JAX replication helper still called by Brax 0.14.2; it is restricted to the
one-GPU runner.

## Training-free audit

```bash
.venv/bin/python -m experiments.playground.audit \
  --output runs/playground/audits/five_task_gpu.json

MUJOCO_GL=egl .venv/bin/python -m experiments.playground.parity \
  --output runs/playground/audits/cartpole_dmc_parity.json
```

## Training

Standard PPO uses Playground's task-specific `brax_ppo_config` directly:

```bash
.venv/bin/python -m experiments.playground.train_ppo \
  --task CartpoleSwingup \
  --seed 20260812 \
  --output-dir runs/playground/train/CartpoleSwingup/seed_20260812/PPO
```

Export a completed PyTorch Koopman checkpoint and train a structured peer:

```bash
python -m experiments.playground.export_koopman \
  --source runs/dmc/koopman/cartpole_swingup/development_v2_3m/best.pt \
  --output runs/playground/koopman/cartpole_swingup/full.npz

.venv/bin/python -m experiments.playground.train_structured \
  --task CartpoleSwingup \
  --method KMPC \
  --koopman runs/playground/koopman/cartpole_swingup/full.npz \
  --seed 20260812 \
  --output-dir runs/playground/train/CartpoleSwingup/seed_20260812/KMPC
```

`AB-PQ` and `AC-MPC-MPVE` use the same command.  KMPC plans 20 steps;
AC-MPC-MPVE consumes the first 10 steps as detached critic targets.  The
default critic consumes the running-standardized raw observation.  A focused
critic-input ablation is available through `--critic-input lifted_state`; it
uses the frozen Koopman-normalized lifted state and does not change the
actor/controller.  The Cartpole Koopman model uses 10 learned lift
coordinates, so its lifted state has 5 + 10 = 15 dimensions.

Use `evaluate.py` on an existing numeric checkpoint directory for fixed-seed,
deterministic per-episode returns.  Playground/Warp results are reported as a
separate benchmark from CPU dm_control because their float precision and some
task simulation timesteps differ.

The task-scaled formal settings are centralized in `tasks.py`: Koopman uses
roughly 1--2 observation dimensions of additional lift and a 0.5 s rollout;
KMPC plans for 0.2 s; MPVE uses the first 0.1 s.  ReacherHard and HumanoidRun
have observation-sufficient exact reward oracles (checked against live GPU
transitions); HopperHop and WalkerRun use the jointly trained reward model.

After the Cartpole supervisor completes, the remaining official PPO → 3M data
→ Koopman → three structured-peer jobs can be run resumably with:

```bash
.venv/bin/python -m experiments.playground.run_remaining_tasks \
  --root runs/playground \
  --seed 20260812 \
  --tasks ReacherHard HopperHop WalkerRun HumanoidRun
```

## Reacher global-coverage Koopman data

PPO-only Reacher trajectories quickly concentrate inside the sparse-reward
target.  The collector therefore supports an opt-in episode-level behavior
mixture while retaining PPO-only collection as its default.  For example, a
global-coverage run can combine 40% stochastic PPO, 30% independent uniform
actions, and 30% uniform actions held for 10 control steps:

```bash
.venv/bin/python -m experiments.playground.collect_koopman \
  --task ReacherHard \
  --checkpoint EARLY_CHECKPOINT \
  --checkpoint MIDDLE_CHECKPOINT \
  --checkpoint LATE_CHECKPOINT \
  --output-dir runs/playground/data/ReacherHard/SEED/global_mixed \
  --num-envs 3000 \
  --ppo-fraction 0.4 \
  --uniform-iid-fraction 0.3 \
  --uniform-held-fraction 0.3 \
  --action-hold-steps 10
```

Every archive records its per-episode `behavior_mode`; the manifest records
behavior counts and reward coverage.  Compare exports fairly with a common
dataset scale and the held-out `episode_index % 10 == 9` split:

```bash
.venv/bin/python -m experiments.playground.evaluate_koopman \
  --task ReacherHard \
  --data-dir runs/playground/data/ReacherHard/SEED/global_mixed \
  --model old=OLD_EXPORT.npz \
  --model global=GLOBAL_EXPORT.npz \
  --output runs/playground/koopman/ReacherHard/SEED/comparison.json
```

The report separates PPO, independent-random, held-random, first-50-step,
and not-yet-reached windows.  A low aggregate error alone is not sufficient
for claiming global coverage.
