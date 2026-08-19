# ManiSoft KMPC offline dataset collection

`collect_manisoft_kmpc_offline_dataset.py` rolls out a trained `ppo_kmpc`
checkpoint and writes transition-complete episode shards. The scenario and
certified waypoint bank default to the paths recorded in the checkpoint.

```bash
python scripts/collect_manisoft_kmpc_offline_dataset.py \
  --checkpoint runs/<kmpc-run>/best.pt \
  --output data/processed/manisoft_kmpc_offline/<dataset-name> \
  --episodes 1000 \
  --episode-steps 300 \
  --device cuda
```

Collection is stochastic by default, using the action distribution stored in
the checkpoint. This is normally preferable for offline RL because it retains
action diversity and the checkpoint's natural successful/failed trajectory
mixture. Add `--deterministic` to collect only the KMPC mean action.

The output contains:

- `episodes/episode_XXXXXX.npz`: crash-safe, transition-aligned episode shards;
- `dataset.npz`: the same episodes flattened in D4RL-style layout;
- `collection_config.json`: immutable provenance and action semantics;
- `summary.json`: success rate, returns, transition count, and episode summaries.

The standard fields are `observations`, `actions`, `rewards`,
`next_observations`, `terminals`, and `timeouts`. For PPO-KMPC, `actions` are
the normalized delta actions consumed by `HistoryContextTrackingWrapper`, not
absolute muscle activations. The physical controls are retained separately as
`requested_absolute_actions`, `applied_actions`, and `applied_delta_actions`.
Behavior means, log probabilities, value estimates, waypoint diagnostics, and
MPC residuals are also stored.

If collection is interrupted, rerun the same command with `--resume`. Episode
seeds are independent, so resumed stochastic collection reproduces the same
dataset as an uninterrupted run. For very large datasets, pass
`--no-merged-dataset` and consume the episode shards directly to avoid building
one large compressed archive.
