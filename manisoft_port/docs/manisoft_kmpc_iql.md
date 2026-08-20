# ManiSoft KMPC-IQL

This is an independent offline training route for the existing history
Koopman-MPC actor. It leaves `antmaze_ac/rl/ppo.py` and every PPO training
script unchanged. The implementation follows the PyTorch IQL update in
[rail-berkeley/rlkit](https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/iql_trainer.py): twin action-value networks, expectile value
regression, and exponentially advantage-weighted behavior cloning.

The policy remains KMPC at deployment time:

```text
history observation -> frozen Koopman lift -> learned KMPC cost map
                    -> differentiable finite-horizon QP -> normalized delta
```

The IQL Q and V networks are training-only. Their state now explicitly includes
the latest applied absolute action (normalized by the actuator box):

$$
f_t = [z_t,\ c_t,\ \bar u_{t-1}],\qquad
\bar u_{t-1}=\frac{u_{t-1}-(u_{\max}+u_{\min})/2}
{(u_{\max}-u_{\min})/2}.
$$

This matters because the normalized-delta feasible set and the physical next
action both depend on $u_{t-1}$. The actor likelihood uses the existing
state-dependent truncated Normal distribution, and deterministic evaluation
uses the KMPC mean.

## Dataset and important limitation

The current combined dataset is:

```text
data/processed/manisoft_kmpc_offline/combined_v4_1498_v5_7109/dataset.npz
```

It contains 8,607 episodes and 1,940,193 transitions. It also contains
`behavior_action_means`, which are the deterministic normalized-delta means of
the PPO behavior actor. Timeouts are bootstrapped by default because every stored transition has
a valid `next_observations`; pass `--treat-timeouts-as-terminal` for episodic
horizon semantics.

Both sources use physical `max_delta=0.0125` and the same behavior checkpoint.

The roughly 2.2 GiB compressed NPZ expands to about 10.2 GiB for the required fields.
On first use, the loader creates a read-only NPY cache beside the dataset and
memory-maps it; later runs reuse that cache without loading all observations
into RAM. Set `--cache-dir` to put it on another disk.

## Train

The default candidate is structured-v2. It transfers the compatible v15e
cost-map rows, then distills the candidate against `behavior_action_means`
before any IQL update. The Q/V critics warm up for 20,000 updates while the
actor remains frozen. Example K=60 run:

```bash
python scripts/train_manisoft_kmpc_iql.py \
  --dataset data/processed/manisoft_kmpc_offline/combined_v4_1498_v5_7109/dataset.npz \
  --initial-policy-checkpoint runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt \
  --candidate-cost-parameterization structured_v2 \
  --candidate-solver-iterations 60 \
  --distillation-steps 10000 \
  --critic-warmup-steps 20000 \
  --output runs/manisoft_kmpc_iql_v2/k60_seed42 \
  --device cuda \
  --seed 42 \
  --gradient-steps 500000 \
  --batch-size 256
```

The trainer checks that this checkpoint's SHA256 exactly matches the behavior
checkpoint recorded in every dataset source. This prevents silently distilling
means produced by a different PPO snapshot.

Defaults follow rlkit's AntMaze IQL settings where applicable: expectile 0.9,
temperature 0.1, advantage-weight clip 100, discount 0.99, and target update
coefficient 0.01. Unlike rlkit's sparse AntMaze example, `reward_bias` defaults
to 0 because this dataset already contains signed dense progress rewards and
waypoint bonuses. To reproduce the rlkit shift explicitly, pass
`--reward-bias -1`.

Resume without repeating the initialization checkpoint:

```bash
python scripts/train_manisoft_kmpc_iql.py \
  --resume runs/manisoft_kmpc_iql_v2/k60_seed42/last.pt \
  --output runs/manisoft_kmpc_iql_v2/k60_seed42 \
  --device cuda
```

`last.pt`, `best_offline.pt`, and recovery checkpoints contain the policy,
twin Q networks and targets, V network, optimizers, RNG states, exact dataset
signature, source provenance, and all IQL hyperparameters.

`best_offline.pt` is no longer selected by the moving IQL policy loss. At the
end of critic warm-up, a frozen copy of the target twin-Q is captured. Candidate
checkpoints maximize

$$
S(\pi)=\mathbb E[\min_i Q_i^{\rm frozen}(s,\mu_\pi(s))
-\min_i Q_i^{\rm frozen}(s,\mu_\beta(s))]
-\lambda\,\mathbb E\|\mu_\pi(s)-\mu_\beta(s)\|_2^2.
$$

This makes scores comparable over training and penalizes leaving behavior
support. It remains an offline proxy, not a replacement for closed-loop model
selection.

For the paired 40/60-iteration experiment, use the guarded launcher twice:

```bash
screen -dmS iql_v2_k40_seed42 scripts/launch_manisoft_kmpc_iql_v2.sh 40
screen -dmS iql_v2_k60_seed42 scripts/launch_manisoft_kmpc_iql_v2.sh 60
```

Both runs use seed 42, the same episode split, 10,000 distillation steps,
20,000 critic-only warm-up steps, and 500,000 IQL updates. Only the deployed
FISTA iteration count differs.

## Evaluate

The existing comparison evaluator now auto-detects PPO and IQL checkpoints:

```bash
python scripts/evaluate_manisoft_ppo_comparison.py \
  --checkpoint runs/manisoft_kmpc_iql_v2/k60_seed42/best_offline.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --output runs/manisoft_kmpc_iql/seed_42/evaluation_100ep \
  --episodes 100 \
  --device cuda
```

Offline validation loss is a diagnostic, not an unbiased environment-return
estimate. Use the simulator evaluation above to select the final policy, and
compare it against the retained PPO checkpoint on the same waypoint schedule.
