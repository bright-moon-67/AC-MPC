# AC-MPC: Deep Koopman + 结构化 Actor 控制（PandaReach3）

本仓库是 **ManiSkill `ACMPC-PandaReach3-v0`**（state-only 三 waypoint 顺序到达任务）的完整实验管线。经多轮迭代，**BC 预训练 + PPO 精调**是最终验证成功的流程，共 4 种方法，各有特点。

## 任务与环境

- 环境 `ACMPC-PandaReach3-v0`：机械臂按顺序到达 `G1 → G2 → G3` 三个随机 waypoint。
- 控制：`pd_joint_delta_pos`，20Hz 控制 / 100Hz 仿真 / 5 物理步，`goal_threshold=0.01m`，`max_episode_steps=220`，动作限幅 ±0.1 rad（7 关节，夹爪常开）。
- Koopman 状态 `x=[q(7), qdot(7), tcp_xyz(3)]`；当前 waypoint 作为单独 task context（12 维：3×3 目标 + one-hot 阶段），不并入动力学。
- 成功 = 到达第 3 阶段且到当前目标 ≤ `success_goal_threshold`（默认 0.01m，可配置是否要求机械臂静止）。

## 四种方法（统一 BC → PPO 流程）

| 方法 | 策略结构 | 特点 |
|---|---|---|
| **PPO** | 标准 256×256 Tanh MLP，线性 mean 输出，不用 Koopman（75K 参数） | 收敛快、动作平滑稳健 |
| **KLQR** | cost-map（lift+context）只输出**物理 24 维** `Q,p`（几何均值归一，KMPC 式），经冻结 `A,B` 与**可微 DARE**（隐式微分加速梯度）得到**时变闭环控制率** `u=-Kz-d`（14K 参数，最小） | 参数最少、最"控制味"，但收敛慢 |
| **AB-PQ**（P 路线） | 学习升维空间 low-rank value `P`，借助冻结 `A,B` 贪心求控制率（46K 参数） | 收敛最快、回合最短 |
| **BC-KMPC** | cost-map 输出物理 24 维 `Q,p`，horizon-10 box-constrained Koopman MPC（FISTA 投影）得动作（70K 参数） | 部署最激进（动作常到限幅）、回合短 |

## 管线（推荐完整流程）

```bash
# 1) 采集数据（DLS 专家数据集，默认 500 episodes）
python -m experiments.state_only_feasibility.collect_pandareach_threewaypoint \
  --output runs/pandareach_threewaypoint/data/pandareach_dls_500.npz --episodes 500

# 2) 训练全局 Koopman（用覆盖数据集，保证全状态空间泛化）
python -m experiments.state_only_feasibility.train_pandareach_koopman \
  --dataset runs/pandareach_threewaypoint/data/pandareach_coverage_600k.npz \
  --output-dir runs/pandareach_threewaypoint/koopman_coverage

# 3) BC 预训练（4 方法并行；支持 per-method epochs，如 BC_EPOCHS_KLQR=50）
BC_KOOPMAN=runs/pandareach_threewaypoint/koopman_coverage/best.pt \
BC_OUTPUT_ROOT=runs/pandareach_threewaypoint/bc_v3 \
bash scripts/run_pandareach_bc_pretrain.sh

# 4) PPO 精调（4 方法并行，从 BC checkpoint 微调）
PPO_KOOPMAN=runs/pandareach_threewaypoint/koopman_coverage/best.pt \
PPO_BC_ROOT=runs/pandareach_threewaypoint/bc_v3 \
PPO_OUTPUT_ROOT=runs/pandareach_threewaypoint/ppo_finetune_v3 \
bash scripts/run_pandareach_ppo_finetune.sh

# 5) 评估与渲染部署 GIF（确定性 mean policy）
python scripts/evaluate_ppo_checkpoint.py \
  --checkpoint runs/pandareach_threewaypoint/ppo_finetune_v3/<M>/seed_20280804/latest.pt
python -m experiments.state_only_feasibility.render_pandareach_bc_gif \
  --checkpoint runs/pandareach_threewaypoint/ppo_finetune_v3/<M>/seed_20280804/latest.pt \
  --output-dir runs/pandareach_threewaypoint/ppo_finetune_v3/gifs
```

长时训练建议放 tmux（`tmux new-session -d -s <name> '...'`），断开 SSH 不中断；PPO 每 10 updates 自动落盘 `latest.pt`，可断点续训。

## 关键实现要点 / 优化

- **KLQR**（`antmaze_ac/rl/quadratic_actors.py::KoopmanLQRActor`）：
  - cost-map 只输出物理 17+7=24 维的 `Q`（对角）与 `p`，经读出矩阵 `C` 映射到升维空间（`Q_z=C'diag(Q_x)C`、`q_z=p_xC`、`R=diag(Q_u)`、`r=p_u`）再解 DARE。
  - `Q` 采用 **per-sample 几何均值归一**（与 BC-KMPC 一致）：LQR 策略对成本整体缩放不变，去掉冗余自由度、数值更稳。
  - 使用项目优化的**结构化倍增 DARE + 隐式微分反向**（`antmaze_ac/control/differentiable_dare.py`）；构造时检查 `(A,B)` 可镇定与 `(A,C)` 可检测。
- **全局 Koopman**：早期仅用专家数据训练的模型在 OOD 状态完全失效（rollout normalized MSE 达 1.7e6）；改用 600K 状态覆盖数据重训后，覆盖 test 集 one-step rmse 0.013，专家分布上 0.0004（比旧模型更准）。
- **BC 写盘**：`BCConfig.checkpoint_interval` 每 N epoch 原子落盘 `{name}.pt` + `status.json`，中断不丢模型；支持 `--eval-only` 对已存 checkpoint 手动评估。
- **向量化闭环评估**：`BCConfig.evaluation_num_envs` 并行 GPU env（`ManiSkillVectorEnv`），把 KLQR 的 DARE 从 batch-1（43ms/样本）批到 16（~1ms/样本），100 集评估从 ~16 分钟降到 ~1 分钟。
- **探索/成功条件（宽松定稿）**：PPO 初始 std=0.015、entropy=1e-4；成功半径 0.01m、不要求机械臂静止（`--no-require-robot-static`）。

## 实验结果（新全局 Koopman + BC→PPO，seed 20280804）

| 方法 | BC 闭环成功率 | PPO 训练 success | 平均回合长度 | 备注 |
|---|---|---|---|---|
| PPO | 9% | ~100% | 37 步 | 收敛快、稳健 |
| KLQR | 1% | 0%（wp≈2，仍在收敛） | 220（超时） | 需要更多步数 |
| AB-PQ | 4% | ~100% | 19 步 | 最快最省 |
| BC-KMPC | 25% | 100%（3M 完整跑完） | 23 步 | 部署激进高效 |

## 目录结构

```
experiments/state_only_feasibility/   # 主实验（采集/训练/评估/渲染）
antmaze_ac/
  control/    # 可微 DARE / affine LQR / 物理成本映射
  koopman/    # DeepKoopman 模型 / 损失 / checkpoint
  rl/         # KLQR / KMPC / AB-PQ 等结构化 actor
scripts/      # 训练/评估启动脚本与工具
tests/        # DARE / actor / Koopman 核心单测
runs/pandareach_threewaypoint/
  data/             # 数据集（DLS 专家 + 600K 覆盖）
  koopman_coverage/ # 全局 Koopman 模型（best.pt + report.json）
  bc_v3/            # 4 方法 BC checkpoint + 报告
  ppo_finetune_v3/  # 4 方法 PPO checkpoint + 指标 + 部署 GIF
```


## Tests

```bash
pytest -q
```

当前本地测试覆盖 Koopman rollout、quadratic actor、PPO smoke、数据边界和控制器数值稳定性。

## License

本仓库代码使用 MIT License。外部算法和数据集仍需遵守各自的许可证与使用条件。
