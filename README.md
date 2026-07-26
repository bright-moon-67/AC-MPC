# AntMaze Deep Koopman + Actor-Critic Koopman-LQR

本项目实现 `prompt.md` 指定的 U-Maze 首版。动力学模型、Actor-Critic
控制器和主 baseline 统一使用

\[
x_t=[s_t,u_{t-1}],\qquad v_t=\Delta u_t,\qquad
u_t=\operatorname{clip}(u_{t-1}+v_t,-1,1).
\]

本项目中的 `history=1` 严格表示 `(x_t,v_t) -> x_{t+1}`，不额外拼接
`s_{t-1}`。`K_step=20` 只表示 Koopman loss 的多步预测长度。稳态控制器没有
`lqr_horizon`；`dare_max_iterations` 是 Riccati 数值预算，
`gain_update_interval` 是增益重算周期。

## 实现范围

- `DeltaActionWrapper`：reset 清零上一动作，累加增量，裁剪实际动作，返回
  `[s_t,u_{t-1}]`，记录 requested/applied delta、实际动作和饱和率。
- D4RL HDF5 转换：使用 terminal/timeout 划分完整 episode，首步
  `u_{-1}=0`，禁止跨 episode 构造转移或 K-step window，只用 train split
  计算 normalization。
- full-A identity-skip Deep Koopman：
  `z=[x,phi(x)]`、`z_{t+1}=Az_t+Bv_t`、`C=[I,0]`。
- 与 `fullA_history_v2` 一致的 one-step lifted linear loss、K-step rollout
  loss、谱半径软惩罚和 latent regularization；prompt 指定的维度、history
  和 K-step 参数优先。
- 可微 PyTorch structured-doubling DARE（SDA）：低精度输入内部提升为
  float64，不调用 SciPy 训练路径，输出 absolute/relative residual、条件数和
  闭环谱半径。
- 完整 affine LQR：对
  `z'Qz+v'Rv+q'z+r'v` 求 `v=-Kz-d`，保留 actor 的线性成本。
- `CostActor -> Q/p -> differentiable DARE-LQR -> Gaussian delta action`，
  独立 critic，PPO 期间冻结 Koopman。
- 独立的离线 Koopman-LQR TD3+BC：同一个 CostActor/DARE 控制链，新增 twin
  action-value critics、target networks 和 BC 约束；可以初始化但不替换正式 PPO。
- 同接口的 Delta-PPO、固定成本 Koopman-LQR、100-episode legacy 评估和分组件
  latency benchmark。

## 环境

开发/Koopman 环境：

```bash
conda run -n soft_vla_cuda python -m pip install -e \
  '.[test,d4rl-data,modern-env,plots,tracking]'
conda run -n soft_vla_cuda pytest -q
```

当前机器为 Python 3.11、PyTorch 2.6.0+cu124 和 RTX 4060 Laptop。

脱离终端的训练脚本默认使用当前 `PATH` 中的 `python`。服务器上可以统一指定
训练环境解释器，避免依赖本机目录结构：

```bash
export AC_MPC_PYTHON="$(command -v python)"
```

正式 D4RL 结果必须使用 legacy `antmaze-umaze-v2`。独立 Python 3.10 环境由
`environment-legacy.yml` 定义；`scripts/run_legacy.sh` 会设置
`LD_LIBRARY_PATH`、MuJoCo 2.1 和无显示渲染所需变量：

```bash
conda env create -f environment-legacy.yml
conda run -n antmaze_legacy python -m pip install -e .
git clone https://github.com/Farama-Foundation/D4RL.git .references/d4rl
git -C .references/d4rl checkout 89141a689b0353b0dac3da5cba60da4b1b16254d
conda run -n antmaze_legacy python -m pip install \
  --no-deps -e .references/d4rl
scripts/run_legacy.sh python scripts/check_legacy_env.py \
  --output runs/legacy_env_check.json
```

legacy launcher 默认使用 Conda 环境 `antmaze_legacy` 和
`${HOME}/.mujoco/mujoco210`。非默认安装位置可通过
`AC_MPC_LEGACY_ENV`、`AC_MPC_LEGACY_PREFIX`、
`MUJOCO_PY_MUJOCO_PATH` 和 `AC_MPC_NVIDIA_LIB` 覆盖。

当前机器已经用真实 D4RL env 验证：原 observation 29 维、action 8 维、增广
observation 37 维、episode limit 700、reset 后 previous action 为零。现代
Gymnasium-Robotics backend 只用于接口 smoke，不可冒充 legacy v2 正式结果。

## 数据

```bash
mkdir -p data/raw
curl -L --output data/raw/antmaze-umaze-v2.hdf5 \
  https://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5

conda run -n soft_vla_cuda python scripts/build_d4rl_sequences.py \
  --expected-sha256 5ef15257771c50ef4d23c7de001750e96c8bb5d9b6a5e4a821dcfb3065fbd130
```

真实构建记录在 `data/processed/antmaze-umaze-v2/metadata.json`。磁盘格式为
schema v2：

```text
state       = [observation_t, previous_action=u_{t-1}]  # 37维
action      = delta_action=u_t-u_{t-1}                  # 8维
next_state  = [observation_{t+1}, current_action=u_t]   # 37维
current_action = D4RL absolute action u_t               # 8维
done        = terminal OR timeout
```

这里 `next_state` 的动作块是当前时刻施加的绝对动作 `u_t`，不是
`delta_action`。内存中的 `x/delta_action/next_x` 兼容别名和旧 NPZ loader
仍保留，不破坏 Koopman/PPO。

- 原始 `observations=(1000000,29)`、`actions=(1000000,8)`；
- 增广状态 37 维，共 999999 条转移；
- action reconstruction 最大误差 `5.960464477539063e-08`；
- train/validation/test episode 数为 8124/1015/1015；
- train split 有 748469 个合法 K=20 windows。

原 HDF5 没有 `next_observations`。转换遵循 D4RL q-learning dataset 的相邻行
定义，并显式排除 terminal/timeout 边界上的跨 episode window。

## Koopman 正式训练

配置唯一来源为 `configs/antmaze_umaze.yaml`。当前正式设置是 batch/eval batch
均为 4096，最多 1000 epochs 或累计训练 5 小时（先到者停止），W&B offline：

```bash
conda run --no-capture-output -n soft_vla_cuda \
  python -u scripts/train_koopman.py \
  --config configs/antmaze_umaze.yaml \
  --data data/processed/antmaze-umaze-v2 \
  --output runs/antmaze_umaze_fulla_formal \
  --device cuda --wandb-mode offline
```

正式控制台记录：
`runs/koopman_fulla_formal_console.log`。训练每 25 epochs 保存包含 optimizer 和
累计 wall time 的 recovery checkpoint；validation 改善时保存
`best_validation.pt`；正常结束保存 `last.pt` 和 `training_status.json`。

会话中断后使用持久化 launcher 恢复；它检查已有 PID、拒绝覆盖已完成 run，并
默认在 recovery/best/last 中选择修改时间最新的可恢复 checkpoint：

```bash
scripts/run_koopman_detached.sh \
  runs/antmaze_umaze_fulla_formal \
  runs/antmaze_umaze_fulla_formal/koopman/recovery_epoch_0049.pt
```

W&B offline 不依赖网络。每次进程段写本地 offline run；训练指标同时追加到
`history.jsonl`，checkpoint 是恢复和结果归档的权威来源。

训练结束后：

```bash
conda run -n soft_vla_cuda python scripts/evaluate_koopman.py \
  --checkpoint runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  --data data/processed/antmaze-umaze-v2
```

README 不记录运行中的临时性能数字。正式完成与否以
`training_status.json`、checkpoint SHA256 和 evaluation JSON 为准。

当前正式链路可由持久化 gate runner 串联。它等待 Koopman 正常结束；异常退出
时最多从最新 checkpoint 自动恢复三次。完成后核对 checkpoint SHA256，依次
执行 full test-split Koopman evaluation、legacy CUDA/env 检查和完整 episode
固定 LQR；全部通过才启动 5-seed Actor-Critic PPO：

```bash
scripts/run_post_koopman_pipeline_detached.sh \
  runs/antmaze_umaze_fulla_formal \
  runs/antmaze_umaze_formal/actor
```

状态与诊断位于
`runs/antmaze_umaze_fulla_formal/koopman/post_koopman_pipeline_status.json`
和同目录的 `post_koopman_pipeline.log`。任何 gate 失败时 stage 会停在
`failed_before_ppo`，不会启动 PPO。

## DARE 失败恢复

初始化 policy 时先对冻结的 `(A,B)` 做 stabilizability PBH 检查，对
`(A,C)` 做 detectability 检查；这两项失败属于结构错误，不能靠数值 fallback
掩盖。

运行时每个 actor cost 的处理为：

1. 用配置的 SDA 参数求解，并验证 convergence、relative DARE residual 和
   `rho(A-BK)<1`；
2. 失败时增加迭代预算和 R regularization 后重试；
3. 仍失败的样本切换到 policy 构造时已求解并验证的固定稳定增益 `K_safe`，
   对当前 observation 的 `z(x)` 计算动作，并限制 fallback delta；
4. 记录 `dare_retry_fraction`、`dare_fallback_fraction` 和 failure count。

fallback 不是“缓存上一次成功的 K”。它是 checkpoint 中的固定 buffer，
因此动作仍是当前 observation 的确定函数；PPO 的乱序 `evaluate_actions`
能够精确重建 rollout 时的策略分布。失败分支不把无效 Riccati 梯度传回 cost
actor。

默认允许孤立失败继续训练。只有 fallback 比例超过单个 rollout 的 5%，且连续
3 个 rollout 发生，训练才保存 `emergency.pt` 和带明确原因的
`training_status.json` 后终止。actor/gradient NaN 也保存 emergency
checkpoint；结构 PBH 失败在训练开始前直接报告。

这参考了 `mpc.pytorch` “保留 batch 历史最好候选、可选择 detach 未收敛样本”
的容错思想，但没有照搬 AC-MPC drone 中
`exit_unconverged=False, detach_unconverged=False, lqr_iter=1` 的激进近似：
有限时域 iLQR 的可执行候选轨迹不等价于无限时域 DARE 的稳定化解。

## 控制与 RL

```bash
# 固定成本、完整 episode 的控制链检查
scripts/run_legacy.sh python scripts/test_fixed_lqr.py \
  --koopman-checkpoint runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  --backend legacy

# 五个正式 seed 的 Actor-Critic Koopman-LQR PPO，脱离终端并自动续接 last.pt
scripts/run_ppo_formal_detached.sh actor \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_formal/actor cuda

# 本地先做单 seed；16 个独立环境批量求 DARE，minibatch=256
scripts/run_actor_single_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_single/actor/seed_0 0 cuda 16 256 1000000

# 守护 5-seed 训练；异常退出时续训，全部完成后自动做 5×100-episode 评估
scripts/run_post_ppo_pipeline_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_formal/actor cuda

# 同状态/同 delta-action 接口的 Delta-PPO
scripts/run_ppo_formal_detached.sh delta_ppo \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_formal/delta_ppo cuda

# D4RL 离线 Koopman-LQR TD3+BC；已有 last.pt 时自动恢复
scripts/run_td3_bc_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_td3_bc/seed_0 0 cuda 500000 256

# 默认第 1 步及之后每 2500 步做 5 个固定 seed 的真实 legacy episode；
# 每个节点保存路径，trend.png 按训练墙钟时间持续更新

# 评估 TD3+BC，并保存前 10 个 episode 路径
scripts/run_legacy.sh python scripts/evaluate_actor.py \
  --checkpoint runs/antmaze_umaze_td3_bc/seed_0/last.pt \
  --method td3_bc --episodes 100 --backend legacy --device cuda \
  --plot-paths 10

# 五个 checkpoint 各 100 个 legacy episodes，随后自动聚合
scripts/run_evaluation_formal_detached.sh actor \
  runs/antmaze_umaze_formal/actor 1 cuda
```

TD3+BC 的训练期真实评估保存在输出目录的 `periodic_evaluation/`：每个
gradient-step 节点有轻量策略 checkpoint、JSON、路径 PNG/NPZ；
`trend.png` 汇总 success rate、目标进度和最小目标距离。训练 smoke 可用
`--environment-evaluation-interval 0` 显式关闭该阶段。

feed-forward PPO 训练强制 `gain_update_interval=1`。interval 2/5/10 的
gain-hold 是有显式 controller state 的评估/latency 选项，不在普通 PPO
minibatch 中缓存旧增益。

本地训练的 rollout/DARE batching、minibatch 实测、单 seed 启动方式和
高性能 GPU 迁移建议见 [`docs/performance.md`](docs/performance.md)。
`evaluate_actor.py --plot-paths N` 会额外保存 U-Maze 全局轨迹、局部放大窗
和原始 XY 路径；正式评估默认保存每个 seed 的前 10 条。

离线 transition 数学语义、TD3+BC loss、checkpoint、恢复训练以及
TD3+BC → PPO actor 初始化方式见
[`docs/offline_td3_bc.md`](docs/offline_td3_bc.md)。
服务器环境、数据重建、checkpoint 传输、smoke gate 和正式启动顺序见
[`docs/server_runbook.md`](docs/server_runbook.md)。
本地提交前逐环节 smoke 结果见
[`docs/smoke_test_report.md`](docs/smoke_test_report.md)。

## 测试与 benchmark

```bash
conda run -n soft_vla_cuda pytest -q

scripts/run_legacy.sh python scripts/benchmark_inference.py \
  --actor-checkpoint runs/antmaze_umaze_formal/actor/seed_0/last.pt \
  --iterations 1000 --warmup 50 --control-episodes 1 \
  --backend legacy
```

当前单元/集成测试为 35 passed，覆盖 wrapper、episode 数据边界、schema-v1
兼容读取、动作重构、
K=20 Koopman rollout/backward、SDA 与 SciPy reference、near-unit-circle
float32 case、PBH/detectability、non-convergence 两种模式、DARE gradcheck、
affine linear cost、actor 梯度、固定无状态 fallback、gain-hold、PPO smoke
、TD3+BC twin-Q/BC 更新以及正式训练/评估守护流程的 checkpoint 完整性检查。

正式 benchmark 必须重新测 encoder、actor、DARE、feedback 和 total 的
mean/P95/max，并同时报告 interval 1/2/5/10 的性能结果；旧 fixed-point
solver 的 smoke 延迟不是当前 SDA 的结果，因此不保留。

## 参考来源与许可证

实际拉取并固定：

1. Deep Koopman
   `yuej0422-dev/skill_learning_soft_robot@fe8b5dabd7de1778f1919be2b88015f13e7a0460`，
   重点检查
   `motion_control_training/koopman/experiments/fullA_history_v2/`。
2. Actor-Critic MPC
   `uzh-rpg/acmpc_public@c59e53aec11c1fffa8b69d99b0ee7879ba7ccb28`，
   其 stable-baselines3 submodule 为
   `152c353863d3b05fb5feed4deb37b952bb4beb7b`，mpc.pytorch submodule 为
   `63732fa85ab2a151045493c4e67653210ca3d7ff`。重点检查
   `training_modules/mlp_mpc_policy.py`、`diff_mpc_drones/il_env.py` 和
   `mpc.pytorch/mpc/mpc.py`。

本项目参考公开算法和数据流，没有复制 AC-MPC 或 `mpc.pytorch` 源文件；
Koopman/DARE/affine LQR/policy/PPO 均为针对本 prompt 的独立实现。本仓库使用
MIT License。AC-MPC 与 mpc.pytorch 的许可证条件仍适用于其各自代码；
D4RL 数据按发布页条件使用。

## 尚未完成的正式实验

- 累计 5 小时 full-data Koopman 正式训练及最终 test evaluation；
- Actor-Critic Koopman-LQR 和 Delta-PPO 各 5 seeds；
- 每个正式 checkpoint 的 legacy D4RL 100-episode evaluation；
- 固定 LQR、gain interval 与 latency/性能对照汇总。

这些任务耗时较长，运行期间只保留正式 console/history、recovery、best/last、
status 和最终 evaluation/benchmark；不把短 smoke 或失败中间文件当作实验
结果，也不伪造尚未产生的数值。
