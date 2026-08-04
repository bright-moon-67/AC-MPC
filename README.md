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

PPO smoke 支持 `B0.pt`、`H1-min.pt` 和 `BC-KMPC.pt` 三种 BC checkpoint。正式结果和 smoke JSON 默认写入 `runs/`；该目录被 Git 忽略。

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
