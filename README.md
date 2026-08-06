# AC-MPC: Deep Koopman and Structured Actors

本仓库包含两条实验路径：

1. AntMaze 的 Deep Koopman、differentiable LQR、PPO 和离线 BC/TD3+BC 组件；
2. ManiSkill PandaReach 的无接触 state-space 三 waypoint 对照实验。

## PandaReach 核心实验

当前正式环境是 `ACMPC-PandaReach3-v0`。每个 episode 在三个互不重叠的小区域中分别随机生成 `G1 -> G2 -> G3`，专家使用分段 minimum-jerk Cartesian reference 和 DLS 依次通过三个目标。

Koopman 状态为 `x=[q, qdot, tcp_xyz]`；当前绝对 waypoint `G_j` 是单独的 actor context，不并入动力学状态。三个 BC actor 为：

- `B0`：参数匹配的 MLP action actor；
- `H1-min`：lifted state + goal context 的 minimal direct-H actor；
- `BC-KMPC`：lifted state + goal context 输出时变 diagonal `Q,p`，通过 horizon-10 box-constrained Koopman MPC 得到 action。

可复现实验入口：

```bash
python -m experiments.state_only_feasibility.collect_pandareach_threewaypoint
python -m experiments.state_only_feasibility.train_pandareach_koopman
python -m experiments.state_only_feasibility.train_pandareach_threewaypoint_bc
python -m experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo_smoke \
  --actor runs/pandareach_threewaypoint/bc/BC-KMPC.pt
```

PPO smoke 支持五种 BC checkpoint。无需 BC checkpoint、从随机 actor 初始化的
正式 PPO 使用 ManiSkill 原生 GPU vector env；RTX 5090 本机配置可直接启动：

```bash
AC_MPC_PYTHON=/root/miniconda3/envs/acmpc_train/bin/python \
  PPO_NUM_ENVS=32 PPO_MAX_PARALLEL=2 \
  nohup bash scripts/run_pandareach_ppo_scratch.sh \
  > runs/pandareach_threewaypoint/ppo_formal_scratch/launcher.log 2>&1 &
```

默认依次覆盖 `B0`、`H1-min`、`H1-min-raw`、`AB-PQ` 和 `BC-KMPC`，最多
同时训练两种方法；每种方法使用 32 个环境、256-step rollout、1024 minibatch、
8 PPO epochs。可用 `PPO_METHODS`、`PPO_TOTAL_TIMESTEPS` 和
`PPO_MAX_WALL_TIME_MINUTES` 调整方法、步数和时限。正式结果和 smoke JSON 默认写入
`runs/`；该目录被 Git 忽略。

正式 PPO 中的 `B0` 是标准 raw-state baseline：actor 和 value critic 都输入
`[normalize(x), normalize(task context)]`，均使用 `256x256 Tanh` MLP。它不把
Koopman lift 作为网络特征；输出目录使用 `B0-standard-dense/`，避免误加载早期的
单隐层 actor / Koopman-feature critic checkpoint。

`H1-min` 与 `H1-min-raw` 的 PPO MLP 同样使用 orthogonal initialization。
Minimal-H 解析层含状态依赖矩阵求解，只对齐 reset 时的单步动作会漏掉闭环正反馈：
旧的 `0.001` final gain 在 220 步确定性轨迹中约 90% 的动作触及边界；即使降至
`1.75e-4`，标准 `0.05 rad` Gaussian exploration 仍会把解析 actor 推入放大区。
当前 final gain 使用 `2e-6`（B0 为 `0.01`）：在正式 32-env rollout 中，H1 的
actor-mean 平均绝对值约 `4.2e-4 rad`，与 B0 同量级且无饱和，同时保持非零
actor mean。新 checkpoint 使用 v3 架构标识隔离旧初始化。
H1 actor learning rate 相应标定为 `5e-8`；4 个完整 PPO updates 中最大观测
KL 为 `0.0093`（guard 阈值 `0.03`），未触发提前停止。

正式 PPO 默认使用稠密外部奖励：每步奖励为当前 active waypoint 距离的 `-0.05` 倍，
每通过一个中间 waypoint 或完成最终 waypoint 额外奖励 `+1`。因此普通远离目标的
轨迹持续提供负样本，而三个有序 waypoint 都会产生明确正事件，即
`r=-0.05*d_active+1*completion`。命令行可用
`--dense-distance-penalty-scale`、`--dense-waypoint-completion-reward` 调节，或用
`--reward-mode sparse` 恢复原稀疏奖励。成功率和完成 waypoint 数仍独立记录。

## AntMaze / offline pipeline

核心模块位于：

- `antmaze_ac/koopman/`：identity-skip Deep Koopman 与 rollout losses；
- `antmaze_ac/control/`：DARE、affine LQR 和 quadratic greedy control；
- `antmaze_ac/rl/`：PPO、TD3+BC、structured quadratic actors 和 Koopman MPC actor；
- `scripts/`：D4RL 数据转换、Koopman/actor 训练与评估入口。

D4RL、MuJoCo 和服务器环境依赖请根据本地环境单独安装。数据、checkpoint、W&B 输出和训练日志均不纳入版本控制。

## Tests

```bash
pytest -q
```

当前本地测试覆盖 Koopman rollout、quadratic actor、PPO smoke、数据边界和控制器数值稳定性。

## License

本仓库代码使用 MIT License。外部算法和数据集仍需遵守各自的许可证与使用条件。
