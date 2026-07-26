# Offline Koopman-LQR TD3+BC

TD3+BC is an independent offline training path. It does not replace or modify
the PPO implementation required by `prompt.md`. Both methods share the frozen
Koopman model, `CostActor`, differentiable affine DARE-LQR, state
normalization, incremental-action environment interface, and evaluator.

## Transition schema

Processed D4RL files use schema version 2:

\[
\begin{aligned}
\texttt{state}_t &= [\texttt{observation}_t,u_{t-1}],\\
\texttt{action}_t &= \Delta u_t=u_t-u_{t-1},\\
\texttt{next\_state}_t &= [\texttt{observation}_{t+1},u_t],\\
\texttt{current\_action}_t &= u_t,\\
\texttt{done}_t &= \texttt{terminal}_t\lor\texttt{timeout}_t.
\end{aligned}
\]

`next_state` never stores `delta_action` in its final block. Its final action
block is the absolute action that was applied at time `t`. The aliases `x`,
`delta_action`, and `next_x` remain available in memory so the existing
Koopman pipeline is source compatible. The loader also upgrades old NPZ files
on read; newly written files contain only the unambiguous schema-v2 names.

## Model and losses

The deterministic policy retains the project controller:

```text
state
  -> CostActor stage_hessian/stage_linear
  -> differentiable affine DARE
  -> raw delta = -K z - d
  -> 2 * tanh(raw delta / 2)
```

The smooth final bound matches the observed D4RL delta-action support
`[-2,2]`. It is enabled only for TD3+BC checkpoints; PPO keeps its existing
Gaussian mean semantics.

TD3 uses two separate action-value critics named `q_value_1` and `q_value_2`.
They are not the local quadratic stage cost produced by `CostActor`.

\[
y=r+\gamma(1-d)\min_i Q'_i(s',\pi'(s')+\epsilon)
\]

\[
L_\pi=-\lambda Q_1(s,\pi(s))
      +w_{BC}\|\pi(s)-a_{\mathrm{D4RL}}\|^2,\qquad
\lambda=\frac{\alpha}{E|Q_1|}
\]

The configured BC warm-up sets `lambda=0` before introducing the learned
action-value objective. Koopman parameters stay frozen. Target actor and
critics use Polyak averaging. The default training reward remains the original
D4RL sparse `0/1` reward (`reward_scale=1`, `reward_bias=0`).

## Commands

Rebuild the canonical dataset:

```bash
conda run -n soft_vla_cuda python scripts/build_d4rl_sequences.py \
  --config configs/antmaze_umaze.yaml \
  --input data/raw/antmaze-umaze-v2.hdf5 \
  --output data/processed/antmaze-umaze-v2 \
  --expected-sha256 5ef15257771c50ef4d23c7de001750e96c8bb5d9b6a5e4a821dcfb3065fbd130
```

Run or resume detached offline training:

```bash
scripts/run_td3_bc_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_td3_bc/seed_0 \
  0 cuda 500000 256
```

During training, the default configuration pauses updates at gradient step 1
and every 2,500 steps, evaluates five fixed-seed episodes in the original
legacy `antmaze-umaze-v2` simulator, and plots all five paths. This produces:

```text
periodic_evaluation/
  history.jsonl
  trend.png
  step_00000001.pt
  step_00000001_legacy_5ep.json
  step_00000001_legacy_5ep_paths.png
  step_00000001_legacy_5ep_paths.npz
  ...
```

`trend.png` is overwritten atomically after each evaluation and shows success
rate, goal-progress fraction, and minimum goal distance against process wall
time. Step labels are printed beside each point. Per-step JSON/PNG/NPZ files
are retained as key diagnostic nodes. Evaluation uses the same fixed episode
seeds at every checkpoint, so trajectory changes are directly comparable.
Its duration is recorded separately and excluded from `updates_per_second`,
while remaining part of the five-hour process wall-time budget.

Override the cost/precision tradeoff from the command line:

```bash
python scripts/train_td3_bc.py ... \
  --environment-evaluation-interval 5000 \
  --environment-evaluation-episodes 10 \
  --environment-evaluation-plot-paths 10
```

Set `--environment-evaluation-interval 0` only for interface smoke tests. A
failed simulator subprocess is recorded with its error and does not discard
the offline training state.

Evaluate the deterministic offline policy on the original legacy environment:

```bash
scripts/run_legacy.sh python scripts/evaluate_actor.py \
  --checkpoint runs/antmaze_umaze_td3_bc/seed_0/last.pt \
  --method td3_bc --episodes 100 --backend legacy --device cuda \
  --plot-paths 10
```

Initialize a fresh PPO run from only the learned TD3+BC `CostActor` weights:

```bash
scripts/run_actor_single_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_ppo_from_td3_bc/seed_0 \
  0 cuda 16 256 1000000 \
  runs/antmaze_umaze_td3_bc/seed_0/last.pt
```

The PPO value critic, optimizer and Gaussian exploration parameter are created
fresh. TD3 action-value critics are not copied into PPO's state-value critic.

## Checkpoints and diagnostics

`last.pt`, `best_bc_validation.pt`, and periodic
`recovery_step_XXXXXXXX.pt` contain actor/target actor, twin critics/targets,
both optimizers, RNG state, Koopman checkpoint hash, config, and dataset schema.
`history.jsonl` records TD3 losses, BC error, Q values, gradient norms, DARE
retry/fallback rates, relative residual, closed-loop spectral radius, and
throughput. Rows at rollout-evaluation steps also contain the compact legacy
evaluation summary and cumulative evaluation overhead. BC validation error is
a supervised diagnostic rather than an offline estimate of environment
return; the periodic legacy rollouts provide learning diagnostics, while
formal model selection still requires the independent 100-episode evaluation.
