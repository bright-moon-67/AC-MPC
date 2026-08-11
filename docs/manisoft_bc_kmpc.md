# ManiSoft three-waypoint history BC-KMPC

这条管线把有限时域 BC-KMPC 迁移到当前软体仿真，并严格沿用已经验证过的
history Koopman 语义：

```text
s_t:       45-D 物理状态
u_t:       18-D 绝对肌肉激活
context_t: [normalized s[t-H+1:t+1], u[t-H:t]], H=10
z[t+1]:    A z[t] + B u[t]
```

任务按 `ref_4cm -> ref_8cm -> ref_12cm` 连续跟踪，中间 waypoint 到达后只切换
阶段，不重置仿真；第三个 waypoint 稳定到达后才结束回合。策略观测沿用
PandaReach3 的 three-waypoints 语义：

```text
[s_t, context_t, G1_xyz, G2_xyz, G3_xyz, one_hot(active_stage)]
```

共 `45 + 10*(45+18) + 12 = 687` 维。历史和上一动作都显式包含在观测中，因此
PPO 打乱 minibatch 后仍能重建同一个 MPC，不依赖隐藏 controller state。

## 学习和控制链

`KoopmanMPCActor` 根据当前 lift、三个归一化目标和阶段 one-hot 输出每个预测步的
物理状态/动作对角二次权重及线性项。冻结的 `A/B/C` 将问题凝缩为有限时域
QP，固定展开的投影 FISTA 求动作序列，只执行第一步。投影同时执行：

- 绝对激活 `-0.30 <= u <= 0.30`；
- 默认逐步变化率 `|u_k-u_{k-1}| <= 0.001`；
- 可配置的动作平滑正则。

FISTA 是可微近似求解，不等价于验证脚本中的精确 OSQP；训练和评估会记录
`projected_gradient_residual`。固定代价 OSQP 只作为 BC 专家，不进入 PPO
反向传播。

训练期的硬约束投影使用 straight-through backward：forward 仍是精确的绝对值/
变化率裁剪，但避免 BC 初始化在错误约束边界时因零梯度无法翻转动作方向。PPO
不再采样后交给环境二次裁剪，而是用 rate-limited squashed Normal 直接在当前
`u[t-1]` 对应的可行区间内采样并计算同一动作的 `log_prob`。

## 推荐运行顺序

公共文件：

```bash
K=work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt
S=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
W=/root/autodl-tmp/ManiSoft/work_dirs/smooth_reference_45d
```

1. 用已验证的 fixed-cost history MPC 采集专家数据：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/collect_manisoft_bc_kmpc_expert.py \
  --koopman-checkpoint "$K" --scenario "$S" --waypoint-root "$W" \
  --output data/processed/manisoft_bc_kmpc/expert.npz \
  --episodes 10 --episode-steps 300 --horizon 10 \
  --max-delta 0.001 --rollout-noise-std 0.0002 --device cuda
```

小幅 rollout noise 让确定性复位下的专家数据覆盖相邻状态；保存的监督标签仍是
专家在实际 history 上重新求得的动作，而不是加噪后的执行动作。

2. Behavior cloning：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/train_manisoft_bc_kmpc_bc.py \
  --koopman-checkpoint "$K" \
  --dataset data/processed/manisoft_bc_kmpc/expert.npz \
  --output runs/manisoft_bc_kmpc/bc \
  --epochs 150 --batch-size 256 --horizon 10 \
  --max-delta 0.001 --device cuda
```

如果确定性 BC 闭环与专家状态分布发生偏移，可以用当前 BC 驱动仿真、由 OSQP
专家对访问到的状态重新标注，并与原数据合并（DAgger）：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/collect_manisoft_bc_kmpc_expert.py \
  --koopman-checkpoint "$K" --scenario "$S" --waypoint-root "$W" \
  --base-dataset data/processed/manisoft_bc_kmpc/expert.npz \
  --rollout-checkpoint runs/manisoft_bc_kmpc/bc/best_validation.pt \
  --output data/processed/manisoft_bc_kmpc/expert_dagger.npz \
  --episodes 3 --episode-steps 300 --rollout-noise-std 0.0001 \
  --max-delta 0.001 --device cuda
```

3. 用独立 value critic 做 PPO 精调。有限时域 MPC 均值在变化率边界附近对代价
参数非常敏感，因此默认采用较小 actor 学习率，并用 target-KL 阻止单次 rollout
上的过度更新：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/train_manisoft_bc_kmpc_ppo.py \
  --koopman-checkpoint "$K" \
  --bc-checkpoint runs/manisoft_bc_kmpc/bc/best_validation.pt \
  --scenario "$S" --waypoint-root "$W" \
  --output runs/manisoft_bc_kmpc/ppo/seed_42 \
  --horizon 10 --max-delta 0.001 --num-envs 1 \
  --actor-learning-rate 0.0000003 --target-kl 0.02 --device cuda
```

4. 确定性 mean policy 评估：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/evaluate_manisoft_bc_kmpc.py \
  --checkpoint runs/manisoft_bc_kmpc/ppo/seed_42/last.pt \
  --scenario "$S" --waypoint-root "$W" \
  --output runs/manisoft_bc_kmpc/evaluation/seed_42 \
  --episodes 10 --episode-steps 300 --device cuda
```

也可以用一条命令顺序执行前三步（请使用新的输出目录，旧单目标数据/checkpoint
不会被当成三 waypoint 资产复用）：

```bash
AC_MPC_PYTHON=/root/miniconda3/envs/manisoft/bin/python \
  scripts/run_manisoft_bc_kmpc.sh "$K" "$S" "$W" \
  data/processed/manisoft_bc_kmpc_three_waypoint/expert.npz \
  runs/manisoft_bc_kmpc_three_waypoint/bc \
  runs/manisoft_bc_kmpc_three_waypoint/ppo/seed_42 cuda
```

以下参数必须在
专家数据、BC 和 PPO 中保持一致，否则 checkpoint 会拒绝加载：

- `horizon`
- `solver_iterations`
- `max_delta`
- `absolute_action_limit`
- waypoint reference/action 文件的 SHA256

`--waypoint-root` 固定读取并交叉校验以下文件；JSON 中的 `u` 必须与对应 NPZ 的
`reference_action` 一致，确保缩放动作可复现：

```text
ref_4cm/reference.npz   <-> actions/u_scale_0p25.json
ref_8cm/reference.npz   <-> actions/u_scale_0p50.json
ref_12cm/reference.npz  <-> actions/u_scale_0p75.json
```

重点监控 `action_saturation_rate`、`distance_minimum`、
`completed_success_rate`、`waypoints_completed_mean`、`approx_kl` 和
`ppo_early_stopped`。可行分布下
`action_saturation_rate` 应接近零；`approx_kl` 超过 target 时本轮会提前结束
PPO minibatch 更新，而不是继续破坏 BC 初始化。
