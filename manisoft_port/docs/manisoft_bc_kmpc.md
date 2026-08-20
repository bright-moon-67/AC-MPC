# ManiSoft three-waypoint history BC-KMPC

这条管线把有限时域 BC-KMPC 迁移到当前软体仿真，并严格沿用已经验证过的
history Koopman 语义：

```text
s_t:       45-D 物理状态
u_t:       18-D 绝对肌肉激活
context_t: [normalized s[t-H+1:t+1], u[t-H:t]], H=10
z[t+1]:    A z[t] + B u[t]
```

每个 episode 从经过稳定性认证的参考库中确定性抽取一个三路点组，三个目标距初始
末端分别为 4--8 cm、8--14 cm、12--20 cm；同组目标来自同一随机动作方向和三个
递增幅值。中间 waypoint 到达后只切换阶段，不重置仿真；第三个 waypoint 稳定
到达后才结束回合。策略观测沿用
PandaReach3 的 three-waypoints 语义：

```text
[s_t, context_t, G1_xyz, G2_xyz, G3_xyz, one_hot(active_stage)]
```

共 `45 + 10*(45+18) + 12 = 687` 维。历史和上一动作都显式包含在观测中，因此
PPO 打乱 minibatch 后仍能重建同一个 MPC，不依赖隐藏 controller state。

三路点环境使用简单的进度/时间奖励：

```text
r = (previous_distance - distance) / waypoint_initial_distance
    - 0.01
    - 0.001 * mean((action / 0.30)^2)
    + 3 * passed_waypoint
    + 5 * completed_all_waypoints
```

距离进展项在一个 waypoint 内近似望远镜求和，不能靠停留持续刷分；每步固定
`-0.01` 使完成相同路点数量时用时越短、累计回报越高。

## 学习和控制链

`KoopmanMPCActor` 根据当前 lift、三个归一化目标和阶段 one-hot 输出每个预测步的
物理状态/动作对角二次权重及线性项。冻结的 `A/B/C` 将问题直接凝缩为 absolute
action QP。固定展开的投影 FISTA 与参考仓库一致，对每个绝对动作使用逐元素 box
projection，并只执行动作序列的第一步：

- 绝对激活 `-0.30 <= u <= 0.30`；

BC-KMPC 不加入 fixed smoothness 或动作变化率约束。FISTA 仍是与参考实现相同的
有限次可微近似求解，训练和评估会记录
`projected_gradient_residual`。固定代价 OSQP 只作为 BC 专家，不进入 PPO 反向传播。

无动作变化率约束后的专家 MPC 使用验证过的固定代价参数：

```text
state_weight=200, tip_state_scale=5,
action_weight=8000, control_weight=1
```

这组参数在 106 个认证 triplet、每回合 `rollout_noise_std=0.0002` 的全库测试中
实现 106/106 三路点成功且无动作元素饱和；搜索证据保存在
`work_dirs/bc_kmpc_weight_search/final_all106.json`。

PPO 与参考仓库一致，在 BC-KMPC 绝对动作均值上使用对角 Gaussian；环境执行前只
施加 `[-0.30,0.30]` 绝对动作裁剪，不施加相邻时刻变化率限制。

## 推荐运行顺序

公共文件：

```bash
K=work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt
S=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
W=data/processed/manisoft_waypoint_bank_v1
```

1. 生成参考库；每个路点最后 250 步必须满足 1 mm 位置稳定性和速度阈值，并在
独立的新仿真中复验：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/generate_manisoft_waypoint_bank.py \
  --scenario "$S" --output "$W" --triplets 100 --seed 42 \
  --distance-ranges-cm 4 8 8 14 12 20 --stable-steps 250
```

2. 用已验证的 fixed-cost history MPC 采集专家数据：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/collect_manisoft_bc_kmpc_expert.py \
  --koopman-checkpoint "$K" --scenario "$S" --waypoint-root "$W" \
  --output data/processed/manisoft_bc_kmpc/expert.npz \
  --episodes 10 --episode-steps 300 --horizon 10 \
  --rollout-noise-std 0.0002 --device cuda
```

小幅 rollout noise 让确定性复位下的专家数据覆盖相邻状态；保存的监督标签仍是
专家在实际 history 上重新求得的动作，而不是加噪后的执行动作。

3. Behavior cloning：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/train_manisoft_bc_kmpc_bc.py \
  --koopman-checkpoint "$K" \
  --dataset data/processed/manisoft_bc_kmpc/expert.npz \
  --output runs/manisoft_bc_kmpc/bc \
  --epochs 150 --batch-size 256 --horizon 10 \
  --device cuda
```

除当前动作外，BC 还按参考仓库监督后续 receding-horizon expert actions，默认
sequence weight 为 `0.25`。future target 不跨越 episode 或 active waypoint
边界，避免把当前 waypoint 的计划错误延伸到下一个 waypoint。

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
  --device cuda
```

4. 用独立 value critic 做 PPO 精调。有限时域 MPC 均值对代价
参数非常敏感，因此默认采用较小 actor 学习率，并用 target-KL 阻止单次 rollout
上的过度更新：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/train_manisoft_bc_kmpc_ppo.py \
  --koopman-checkpoint "$K" \
  --bc-checkpoint runs/manisoft_bc_kmpc/bc/best_validation.pt \
  --scenario "$S" --waypoint-root "$W" \
  --output runs/manisoft_bc_kmpc/ppo/seed_42 \
  --horizon 10 --num-envs 1 \
  --actor-learning-rate 0.0001 --target-kl 0.02 --device cuda
```

5. 确定性 mean policy 评估：

```bash
conda run --no-capture-output -n manisoft python \
  scripts/evaluate_manisoft_bc_kmpc.py \
  --checkpoint runs/manisoft_bc_kmpc/ppo/seed_42/last.pt \
  --scenario "$S" --waypoint-root "$W" \
  --output runs/manisoft_bc_kmpc/evaluation/seed_42 \
  --episodes 10 --episode-steps 300 --device cuda
```

也可以用一条命令顺序执行前三步。无 fixed smoothness、absolute-box FISTA 的
随机参考库数据和 checkpoint 格式版本为 5；旧数据/checkpoint 会被明确拒绝，请使用新的
输出路径：

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
- `absolute_action_limit`
- waypoint-bank manifest 的 SHA256

`--waypoint-root` 读取 `manifest.json` 及其中列出的 NPZ。加载器校验 manifest、
每个参考文件和 scenario 的 SHA256，并保证同一 episode 的环境目标和 MPC
reference state/action 使用同一个 `waypoint_triplet_index`：

```text
manifest.json
triplet_0000/waypoint_1.npz
triplet_0000/waypoint_2.npz
triplet_0000/waypoint_3.npz
...
```

重点监控 `action_saturation_rate`、`distance_minimum`、
`completed_success_rate`、`waypoints_completed_mean`、`approx_kl` 和
`ppo_early_stopped`。可行分布下
`action_saturation_rate` 应接近零；`approx_kl` 超过 target 时本轮会提前结束
PPO minibatch 更新，而不是继续破坏 BC 初始化。
