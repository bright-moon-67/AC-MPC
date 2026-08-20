# ManiSoft H=10 History-MLP baseline

This baseline uses exactly the same environment information and action support
as the history BC-KMPC policy, but it does not use the Koopman lift, learned
dynamics, or an MPC solver.

```text
physical state:       45-D
absolute action:      18-D
history steps:        10
history context:      10 * (45 + 18) = 630-D
task context:         normalized target tip + normalized tip error = 6-D
MLP actor input:      636-D
environment obs:      45 + 630 + 3 = 678-D
```

The MLP outputs an action median inside the intersection of the absolute and
rate boxes. PPO uses the same rate-limited squashed Normal distribution as
BC-KMPC, so the action whose log probability is stored is also the action the
environment executes.

## Full BC to PPO run

Use the same expert dataset as BC-KMPC:

```bash
K=work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt
S=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
R=/root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz
D=data/processed/manisoft_bc_kmpc/expert.npz

scripts/run_manisoft_history_mlp.sh \
  "$K" "$S" "$R" "$D" \
  runs/manisoft_history_mlp/bc \
  runs/manisoft_history_mlp/ppo/seed_42 cuda
```

The launcher collects the fixed-cost history MPC demonstrations only when the
shared dataset is missing, resumes BC from `last.pt`, and resumes PPO from
`last.pt`.

## Individual stages

```bash
python scripts/train_manisoft_history_mlp_bc.py \
  --koopman-checkpoint "$K" --dataset "$D" \
  --output runs/manisoft_history_mlp/bc \
  --epochs 150 --batch-size 256 --max-delta 0.001 --device cuda

python scripts/train_manisoft_history_mlp_ppo.py \
  --koopman-checkpoint "$K" \
  --bc-checkpoint runs/manisoft_history_mlp/bc/best_validation.pt \
  --scenario "$S" --reference "$R" \
  --output runs/manisoft_history_mlp/ppo/seed_42 \
  --max-delta 0.001 --actor-learning-rate 0.0001 \
  --target-kl 0.02 --device cuda

python scripts/evaluate_manisoft_history_mlp.py \
  --checkpoint runs/manisoft_history_mlp/ppo/seed_42/last.pt \
  --scenario "$S" --reference "$R" \
  --output runs/manisoft_history_mlp/evaluation/seed_42 \
  --episodes 10 --episode-steps 300 --device cuda
```

For the from-scratch PPO ablation, omit `--bc-checkpoint` and use a fresh
output directory. Keep the expert dataset split, environment, reference,
training timesteps, evaluation seeds, action limits, and PPO budget identical
when comparing against BC-KMPC. Actor learning rates may be tuned separately
with the same-size search grid.
