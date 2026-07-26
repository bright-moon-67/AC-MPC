# Server training runbook

This runbook keeps source control, large datasets, and experiment artifacts
separate. `data/`, `runs/`, `.references/`, W&B files, and model checkpoints
are intentionally ignored by Git.

## 1. Clone and install

```bash
git clone YOUR_REPOSITORY_URL AC-MPC
cd AC-MPC

conda env create -f environment-legacy.yml
conda run -n antmaze_legacy python -m pip install -e .

git clone https://github.com/Farama-Foundation/D4RL.git .references/d4rl
git -C .references/d4rl checkout 89141a689b0353b0dac3da5cba60da4b1b16254d
conda run -n antmaze_legacy python -m pip install \
  --no-deps -e .references/d4rl
```

For Koopman and offline TD3+BC training, create a Python 3.10/3.11 PyTorch
environment appropriate for the server CUDA driver and install:

```bash
python -m pip install -e ".[test,d4rl-data,plots,tracking]"
```

Do not blindly install the local CUDA 12.4 PyTorch wheel if the server uses a
different driver/toolkit. Install the matching PyTorch build first.

Detached training launchers use the active `python` by default. Pin the
interpreter explicitly before starting a long job:

```bash
export AC_MPC_PYTHON="$(command -v python)"
```

The legacy launcher discovers the `antmaze_legacy` Conda prefix and defaults
to `${HOME}/.mujoco/mujoco210`. Override `AC_MPC_LEGACY_ENV`,
`AC_MPC_LEGACY_PREFIX`, `MUJOCO_PY_MUJOCO_PATH`, or
`AC_MPC_NVIDIA_LIB` when the server layout differs.

## 2. Rebuild schema-v2 D4RL data

```bash
mkdir -p data/raw
curl -L --output data/raw/antmaze-umaze-v2.hdf5 \
  https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5

python scripts/build_d4rl_sequences.py \
  --config configs/antmaze_umaze.yaml \
  --input data/raw/antmaze-umaze-v2.hdf5 \
  --output data/processed/antmaze-umaze-v2 \
  --expected-sha256 5ef15257771c50ef4d23c7de001750e96c8bb5d9b6a5e4a821dcfb3065fbd130
```

Expected metadata:

- 999,999 transitions and 10,154 episodes;
- train/validation/test rows: 805,244 / 95,799 / 98,956;
- augmented state 37 and delta action 8;
- maximum action reconstruction error
  `5.960464477539063e-08`;
- `dataset_schema_version=2`.

## 3. Transfer or retrain Koopman

The locally retained controller checkpoint is not committed to Git. Transfer
it separately if it should initialize server experiments:

```bash
rsync -av \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  SERVER:AC-MPC/runs/antmaze_umaze_fulla_formal/koopman/

sha256sum runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt
```

Expected local SHA256:

```text
4327e5def374bff6a3b3fb644de24387c01156bb98dfadfc58b44c019322b74e
```

Alternatively, launch a new Koopman run with the server GPU:

```bash
python -u scripts/train_koopman.py \
  --config configs/antmaze_umaze.yaml \
  --data data/processed/antmaze-umaze-v2 \
  --output runs/antmaze_umaze_fulla_formal \
  --device cuda --wandb-mode offline
```

## 4. Mandatory server smoke gate

Run these before a long job:

```bash
python -m pytest -q

scripts/run_legacy.sh python scripts/check_legacy_env.py \
  --output /tmp/acmpc_legacy_env_check.json

python scripts/evaluate_koopman.py \
  --checkpoint runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  --data data/processed/antmaze-umaze-v2 \
  --split test --horizons 1 --device cuda \
  --output /tmp/acmpc_koopman_smoke.json --no-save-curves

python scripts/train_td3_bc.py \
  --koopman-checkpoint runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  --data data/processed/antmaze-umaze-v2 \
  --output /tmp/acmpc_td3_bc_smoke \
  --device cuda --seed 0 --gradient-steps 4 --batch-size 256 \
  --bc-warmup-steps 4 --log-interval 2 --validation-interval 4 \
  --checkpoint-interval 4 --max-wall-time-hours 0.1
```

The TD3+BC smoke must report finite losses, zero DARE fallback, a relative
DARE residual near numerical tolerance, and a closed-loop spectral radius
below one.

## 5. Offline TD3+BC

```bash
scripts/run_td3_bc_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_td3_bc/seed_0 \
  0 cuda 500000 256
```

The launcher resumes `last.pt`. Configuration also stops at five hours, so the
first reached limit wins. Monitor:

```bash
tail -f runs/antmaze_umaze_td3_bc/seed_0/console.log
tail -f runs/antmaze_umaze_td3_bc/seed_0/history.jsonl
nvidia-smi
```

Before increasing batch size, profile 256/512/1024. The differentiable DARE
uses float64 internally, so server FP64 throughput matters.

Evaluate without online fine-tuning:

```bash
scripts/run_legacy.sh python scripts/evaluate_actor.py \
  --checkpoint runs/antmaze_umaze_td3_bc/seed_0/last.pt \
  --method td3_bc --episodes 100 --backend legacy --device cuda \
  --plot-paths 10
```

## 6. PPO initialized from TD3+BC

```bash
scripts/run_actor_single_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_ppo_from_td3_bc/seed_0 \
  0 cuda 16 256 1000000 \
  runs/antmaze_umaze_td3_bc/seed_0/last.pt
```

This copies only CostActor weights. PPO creates a fresh state-value critic,
optimizer, and Gaussian exploration parameter. After the single seed passes a
100-episode legacy evaluation, launch multi-seed experiments with one seed per
GPU rather than competing processes on one GPU.

## 7. Artifact policy

Keep for formal runs:

- resolved config and source commit;
- console and JSONL history;
- `last.pt`, best checkpoint, sparse recovery checkpoints, status JSON;
- evaluation JSON/PNG/NPZ;
- checkpoint and dataset SHA256 values.

Do not commit these artifacts to normal Git history. Store them in an artifact
service, W&B offline/online storage, a release asset, or a dedicated Git LFS
repository.
