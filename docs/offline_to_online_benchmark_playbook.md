# Offline-to-Online AC-KMPC 新 Benchmark 迁移手册

更新时间：2026-08-18。

本文把 Cartpole Swingup + ExORL 阶段已经验证和踩坑后的流程整理成可复用手册。目标是迁移到
Reacher、Hopper、Walker、Humanoid 或其他 benchmark 时，从数据、Koopman、baseline、结构化
方法、训练预算到最终评估都使用同一套科学边界，而不是复制 Cartpole 的数字。

本文使用最终展示名：`Cal-QL-KMPC`、`Cal-RLPD-KMPC`、`Cal-QL-MPVE`、
`Cal-RLPD-MPVE`。部分历史代码和 checkpoint 仍使用带 `AC` 的内部名称，例如
`Cal-QL-AC-KMPC`；它们表示同一类 differentiable AC-KMPC actor。新实验及论文图表统一省略
名称中的 `AC`。

## 1. 先冻结比较问题

主矩阵包含三个 raw baseline 和四个提出的方法：

| 方法 | 离线阶段 | 在线阶段 | actor 输入 | critic 输入 | actor | MPVE |
| --- | --- | --- | --- | --- | --- | --- |
| `Cal-QL-Raw` | Cal-QL | Cal-QL | 标准化 raw observation | raw observation + action | MLP | 无 |
| `RLPD-Raw` | 无离线梯度预训练 | RLPD，50:50 offline/online replay | 标准化 raw observation | raw observation + action | MLP | 无 |
| `Cal-RLPD-Raw` | Cal-QL regularized RLPD | pure RLPD，50:50 replay | 标准化 raw observation | raw observation + action | MLP | 无 |
| `Cal-QL-KMPC` | Cal-QL | Cal-QL | frozen lifted state | lifted state + action | stochastic KMPC | 无 |
| `Cal-RLPD-KMPC` | Cal-QL regularized RLPD | pure RLPD，50:50 replay | frozen lifted state | lifted state + action | stochastic KMPC | 无 |
| `Cal-QL-MPVE` | Cal-QL + MPVE | Cal-QL + MPVE | frozen lifted state | lifted state + action | stochastic KMPC | 离线+在线 |
| `Cal-RLPD-MPVE` | Cal-QL regularized RLPD + MPVE | pure RLPD + MPVE | frozen lifted state | lifted state + action | stochastic KMPC | 离线+在线 |

必须保持以下边界：

- 三个 `*-Raw` baseline 的 actor 和 critic 只接收 raw observation。它们的 run metadata 中
  `koopman` 必须为 `null`。只要 baseline 使用 learned lift，就不再是标准 baseline。
- 四个提出的方法共享同一份冻结 Koopman。当前选择的 lifted-critic 版本中，actor 与 critic
  都使用 `z=[normalized raw observation, learned lift]`。
- `Cal-QL-KMPC` 相对 `Cal-QL-Raw`、`Cal-RLPD-KMPC` 相对 `Cal-RLPD-Raw` 改变的是完整
  Koopman-structured package，包括 representation 和 actor；不能把差值只解释为 actor 增益。
- MPVE 没有独立 actor 范式。`Cal-QL-MPVE` 和 `Cal-RLPD-MPVE` 的 actor 仍是 stochastic
  KMPC；MPVE 只改变 critic target/loss，而且离线、在线两个阶段都启用。
- MPVE rollout、reward 和 bootstrap target 必须 `stop_gradient/detach`，不得通过模型轨迹
  反向改变 frozen Koopman，也不得把 imagined transition 写进真实 replay。

## 2. 新任务的环境协议审计

训练任何模型之前先写一页 task protocol，至少冻结：

| 项目 | 必须记录的内容 |
| --- | --- |
| 状态 | 实际 policy observation 的字段、顺序、维数、单位；不要只引用论文中的 qpos/qvel 维数 |
| 动作 | 维数、范围、是否 normalized、是否经过 clipping 或 action repeat |
| 时间 | physics timestep、control timestep、action repeat、episode step limit、物理时长 |
| reward | 官方公式、逐 transition 范围、episode return 上界、是否能由 observation/action 精确复现 |
| 终止 | true terminal、time-limit truncation、environment discount 和 bootstrap 语义 |
| 评估 | 官方 episode 数、deterministic/stochastic、seed 规则、训练/评估交互是否计预算 |
| 软件 | 环境、MuJoCo、DMC/Playground、JAX/PyTorch 版本及 source commit |

所有 horizon 首先用秒定义，再换成 control step：

```text
step_horizon = round(time_horizon_seconds / control_timestep_seconds)
```

Cartpole 的 `K=50`、KMPC `H=20`、MPVE `H=10` 分别表示 0.50、0.20、0.10 秒；换成
20 Hz 或 action repeat 不同的任务后，绝不能继续照抄 50/20/10。

如果把 ExORL/DMC 离线数据用于当前 DMC 在线环境，应记录历史 simulator 与当前 simulator
的版本漂移。如果在线阶段迁移到 MuJoCo Playground，则已经变成跨引擎 distribution shift；
应重新生成匹配 Playground 的离线数据，或者明确标成 cross-engine ablation，不能和原生
DMC/ExORL 主结果混报。

## 3. 数据端：来源、覆盖与 canonical schema

### 3.1 数据不是越“专家”越好

Koopman 需要动力学空间覆盖，offline RL 还关心数据质量。二者要分别审计：

- expert-only 数据可能 return 高，但动作和失败状态覆盖窄，不一定适合学习全局动力学。
- random 数据覆盖部分动作，但很少访问任务相关稳定区域。
- reward-free exploration、多个训练阶段或多策略混合数据通常更适合当前目标。
- 主数据源应优先使用 benchmark 官方公开数据；额外自采数据必须作为单独数据协议和消融。

Cartpole 使用的 ExORL Proto 数据不是专家数据，也不是 expert+random 人工混合。它来自
ProtoRL reward-free exploration policy 随训练演化产生的非平稳 replay，之后按任务 reward
重标注。最初只取时间顺序前 1,000 个 episode 会偏向探索早期阶段，因此 Koopman 主模型采用
Proto-Stratified-1M：把完整 10M 数据分成十个时间段，每段确定性取 100 个完整 episode，
合计 1M transition。

当前缩减版 offline-RL development 另用 Proto-Stratified-50k：仍把完整 10M 分成十个时间段，
但每段按 200-episode microstratum 取起始 episode，共取 5 个完整 episode，即 50 个 episode、
50,000 transitions。它用于检查数据规模下降后的策略学习，不用于重训或替换上述 1M-data
Koopman。两份数据的用途和 SHA 必须分别记录，不能把“policy 只用 50k”误写成“Koopman
也只用 50k”。

### 3.2 每个新任务都做覆盖审计

至少报告：

- 每个 observation 分量的 min/max、均值、标准差及高分位数；
- action 各维范围、标准差、边界饱和率；
- reward/episode return 分布，成功、失败和中间状态比例；
- 按 collector、训练时间段、return 桶或行为策略分层后的覆盖差异；
- 低维任务的 occupancy/角度 histogram；高维任务的 PCA、kNN 距离或分层 holdout；
- 极端 episode 对 Koopman multi-step error 的贡献，不允许只报告去除 outlier 的结果。

Cartpole 的经验是：时间分层 1M 在相同分层 test 上的 H50 weighted NMSE 为约 `0.0555`，
仅取最早 1M 约为 `0.1119`，PPO replay 3M 模型约为 `0.1567`。数据分布匹配比单纯增加
transition 数更重要。

### 3.3 canonical transition 契约

统一保存：

```text
observation, action, reward, discount, next_observation,
episode_id, episode_step, terminated, truncated, mc_return
```

必须验证：

- transition 不跨 episode；episode id/step 连续且完整；
- reset dummy 只跳过一次。ExORL 长度 1,001 的 episode 中，有效 transition `i=1..1000`
  对齐为 `obs[i-1], action[i], reward[i], discount[i], obs[i]`；
- time-limit truncation 保留 DMC 的 bootstrap discount；finite-horizon MC return 在文件 episode
  边界停止递推；
- 数值 finite、动作在范围内，reward 与 exact oracle 或官方环境逐 transition 对齐；
- canonical dataset、原始 archive、选择 episode 列表和转换 manifest 都保存 SHA256。

数据划分必须按完整 episode 进行。推荐 80/10/10 train/validation/test，并让每个 split 都覆盖
所有 collector/时间层。normalizer 只能用 train split 拟合。

## 4. Koopman 训练与验证

### 4.1 架构选择

初始搜索范围：

- learned lift dimension 取 raw observation dimension 的约 1–2 倍；
- 高维 Humanoid 先从约 1 倍开始，避免 lifted critic 和 MPC cost head 失控；
- 小型 smooth task 可以用 2 倍左右；contact task 优先增加数据覆盖，而不是先堆 lift；
- lifted state 必须显式包含标准化 raw observation，使 representation 保留原始 Markov 信息。

Cartpole 当前冻结模型使用 raw observation 5 维、learned lift 10 维，总 lifted dimension 15；
它由 Proto-Stratified-1M 训练，而当前 50k offline-RL 实验只复用该模型。训练参数为：

```yaml
lift_dim: 10
k_step: 50                 # 0.50 s
batch_size: 2048
max_windows_per_epoch: 500000
validation_windows: 10000
epochs: 500
patience: 40
learning_rate: 3.0e-4
spectral_radius_limit: 0.95
stability_reference_dt: 0.04
```

谱半径约束与 reference timestep 绑定。当前实现按
`rho_effective = 0.95 ** (control_dt / 0.04)` 换算；Cartpole `control_dt=0.01s`，因此实际每个
控制步使用的上限约为 `0.9872585`。迁移任务时必须用新任务的 control timestep 重新换算，
不能把 `0.95` 直接当成任意离散步长下相同的单步约束。

### 4.2 horizon 选择规则

建议从物理时间出发：

| 模块 | 初始范围 | 选择依据 |
| --- | --- | --- |
| Koopman training rollout | 0.5–1.0 s | 覆盖 MPC/MPVE，并能暴露长期漂移 |
| KMPC planning horizon | 约 0.2 s | 控制响应与模型可信区间折中 |
| MPVE total TD horizon | 0.1–0.2 s | 必须不超过可靠模型区间和 KMPC horizon |

smooth Cartpole/Reacher 可先用区间上部；Hopper/Walker/Humanoid 接触越强，越应先缩短模型
horizon并检查跨接触 multi-step error，再决定是否延长。

### 4.3 reward 路径

对每个任务先检查 predicted observation + action 是否足以精确复现官方 reward：

1. 在真实 dataset transition 上比较解析 oracle 和保存 reward；
2. 在模型预测 observation 上确认字段和单位仍满足公式；
3. 逐 transition 数值一致时，主实验使用 exact oracle；
4. 无法由 observation 恢复时不用训练 reward model，暂时不进行MPVE版本训练。

Cartpole 主实验使用 exact reward oracle，暂不使用 learned reward model。

### 4.4 Koopman gate

至少在完全未参与训练的 episode 上报告：

- K=1/5/10、KMPC horizon、MPVE horizon 和完整 training K 的 rollout/endpoint NMSE；
- 每个 physical observation 分量的 RMSE；
- exact reward rollout RMSE/MAE/bias；
- 每个数据层/collector 的指标及 worst-case episode；
- 与旧模型或 raw linear dynamics 的同窗对照；没有可比 artifact 时明确记录缺失，不伪造对照。

只有模型在 KMPC/MPVE 实际 horizon 上通过 gate，才能进入 structured policy 训练。训练后
Koopman 完全冻结；后续同一任务涉及 Koopman 的方法使用同一个模型 artifact 和 SHA。

当前冻结模型 SHA256 为
`7d61b4b13417b70a9b51d55638d4437e05a018e1888af3ab19cbb0e2093e9edc`。在分层 holdout
同窗评估中，weighted rollout NMSE 在 K=10/20/50 分别约为 `0.0133/0.0191/0.0555`，
K=50 exact-reward rollout RMSE 约为 `0.0559`。这些数值证明当前 Cartpole horizon 可用，
但不自动证明它适用于新任务或新的 observation 协议。

## 5. Baseline 参数和网络构建规则

参数来源按以下优先级选择：

1. 作者官方代码中同一个 benchmark、同一个 observation/action 协议的推荐配置；
2. 作者官方代码中最接近的 state-based 连续控制配置；
3. 论文表格或官方 example；
4. 若都不存在，在 development seed 上做小型、预先声明的 sweep，冻结后再跑正式 seed。

“官方最优”必须满足同任务或同协议证据。否则应写
`official-implementation-derived`、`official-style` 或 `task-adapted`，不能写成官方 best/SOTA。

不同 baseline 不需要强行使用相同 batch、UTD、Q 数或网络宽度。公平性来自相同 dataset、
真实 online transition 预算、训练 seed、评估 seed 和环境协议；算法内部则尽量保留各自官方
recipe；若没有可靠的 method-specific 配置，可以采用相同 batch、UTD、Q 数或网络宽度，
但要在 metadata 中标为 task-adapted 公共配置。

### 5.1 Cartpole 当前 baseline 配置

| 参数 | Cal-QL-Raw | RLPD-Raw | Cal-RLPD-Raw |
| --- | ---: | ---: | ---: |
| representation | raw-5 | raw-5 | raw-5 |
| actor/critic hidden | 2×1024 | 2×256 | 2×256 |
| Q ensemble | 2 | 10 | 10 |
| target Q subset | 2 | random 2 | random 2 |
| batch size | 1024 | 256 | 256 |
| actor/critic/temp LR | 1e-4 | 3e-4 | 3e-4 |
| target tau | 0.01 | 0.005 | 0.005 |
| offline updates | 当前 50k | 0 | 当前 50k |
| online UTD | 1 | 20 | 20 |
| offline:online replay | online 50:50 | online 50:50 | online 50:50 |
| random online warmup | 0 | 5k transitions | 0 |
| env workers | 1 | 5 | 5 |
| CQL proposals/weight | 3 / 0.01 | 无 | offline 10 / 0.01 |
| target entropy | -1 | -0.5 | -0.5 |

Cal-QL 的 2×1024、batch 1024、K=3、weight 0.01 主要来自 ExORL DMC CQL recipe，再加入
Cal-QL calibration；它不是 Cal-QL 作者仓库对 DMC Cartpole 的官方最优。Cal-QL 在线阶段
继续 calibrated CQL，完整 1,000-step episode 收齐后计算 MC return，再把 online transition
加入 replay。

RLPD 使用 10-Q LayerNorm ensemble、random min-2 target、UTD20、50:50 offline/online
replay。Cal-RLPD 的主体是 RLPD：离线阶段加入 Cal-QL regularizer，在线阶段关闭 CQL，
使用 pure RLPD。这是项目组合算法，不是作者发布的标准 Cal-QL 或 RLPD。

## 6. 提出方法的构建规则

### 6.1 共同 stochastic KMPC actor

Koopman 给出 lifted linear dynamics：

```text
z_{t+1} = A z_t + B a_t
```

actor 网络根据 `z_t` 输出 horizon 内的正定 diagonal Q 和线性项 p，KMPC 解出动作序列。
第一个规划动作转换成 tanh-Gaussian 的 pre-tanh mean，并使用可训练 log-std 采样，因此仍能
使用 SAC/Cal-QL/RLPD 的 reparameterized actor loss 与 entropy objective。

critic 是独立的 `Q(z,a)` ensemble，不与 MPC cost head 共享参数。Koopman A/B/C、encoder、
normalizer 全部冻结。

### 6.2 网络容量

网络容量按任务和 baseline 重新决定，不要求参数量逐个相等，但应避免数量级不对称：

- 优先让结构化 actor 使用和对应 baseline 相同的 hidden depth/width；
- output head 由 `2 × horizon × (physical_state_dim + action_dim)` 决定，参数量天然不同；
- 报告 actor、critic、总 trainable 参数量；frozen Koopman 参数单独列出；
- 如果容量可能解释结果，增加 small-controller 消融，而不是让主方法长期使用明显偏小网络。

Cartpole 当前按算法 core 对齐结构化 actor 容量，而不是给四种方法统一使用同一宽度：

```text
Cal-QL-KMPC / Cal-QL-MPVE:
  lifted-15 -> 1024 -> 1024 -> 240

Cal-RLPD-KMPC / Cal-RLPD-MPVE:
  lifted-15 -> 256 -> 256 -> 240

240 = 2 * KMPC_H20 * (physical_state_5 + action_1)
```

Cal-QL 结构化 actor 约 1.312M trainable 参数，与 Cal-QL-Raw actor 的约 1.060M 处于同一
量级。Cal-RLPD 结构化 actor 约 132k 参数，宽度和深度与 RLPD/Cal-RLPD 的 2×256 MLP
recipe 对齐；其输出头更大，因此不追求逐参数完全相等。历史 `15->128->240` actor 仅约
33k trainable 参数，不再作为当前主配置。

### 6.3 四个方法的唯一差异

- `Cal-QL-KMPC`：完整 Cal-QL core + lifted critic + stochastic KMPC actor。
- `Cal-RLPD-KMPC`：Cal-RLPD core + lifted critic + stochastic KMPC actor。
- `Cal-QL-MPVE`：在 `Cal-QL-KMPC` 上，offline 和 online 都增加 MPVE auxiliary。
- `Cal-RLPD-MPVE`：在 `Cal-RLPD-KMPC` 上，offline 和 online 都增加 MPVE auxiliary。

MPVE total horizon `H` 包含 replay 中第一条真实 transition。Cartpole `H=10` 的含义是
`1 real + 9 model`，不是 `1 real + 10 model`。模型 rollout 使用 KMPC proposal action 和
exact reward，terminal soft-Q bootstrap 使用 target critic；整个 target detach。MPVE 作为
辅助 loss 保留真实 one-step Bellman 主损失，不能完全替换真实 TD。

为了严格比较 KMPC 和 MPVE，使用相同 training seed 和 method-specific 初始化规则；由于
MPVE 从 offline 第一个 update 就改变 critic loss，不应从“已经完成的无 MPVE offline
checkpoint”继续并称为完整 MPVE。二者应从同一初始随机状态独立训练。

## 7. 训练预算与阶段决策

### 7.1 Cartpole development 口径

#### 7.1.1 当前 50k 缩减实验（2026-08-18）

为降低 CPU/DMC 在线链路以外的试验成本，当前 Cartpole 缩减实验单独使用一份
50,000-transition canonical 数据集：从 ExORL Proto 的 10 个时间 decile 各取 5 个
完整 episode（共 50 个、每个 1,000 transitions），而不是截取最早的 50 个 episode。
该选择保留时间分层覆盖；reward oracle 对齐、episode 边界和 timeout discount 契约
必须重新校验。Koopman 模型暂不重训，继续使用已通过 K=50 holdout gate 的
`proto_stratified_1m_lift10` 模型，并在 artifact 中绑定其 SHA。

本轮实际执行口径如下：

```yaml
dataset_transitions: 50000
dataset_episodes: 50
dataset_selection: 10 temporal deciles x 5 episodes
dataset_sha256: 127b0b2fd80f561438f32a05d9a6dbc50b4814173dbdafed47e6278fb1060553
koopman_training_dataset: Proto-Stratified-1M
koopman_sha256: 7d61b4b13417b70a9b51d55638d4437e05a018e1888af3ab19cbb0e2093e9edc
offline_updates: 50000
offline_eval_interval_updates: 10000
online_steps: 50000                 # 仅写入配置，当前 stop-after-offline
diagnostic_eval_episodes: 10
final_evaluation: 10 groups x 10 deterministic episodes = 100 episodes
training_seed: 20260826
```

本轮启动六个包含离线阶段的方法：`Cal-QL-Raw`、`Cal-RLPD-Raw`、
`Cal-QL-KMPC`、`Cal-QL-MPVE`、`Cal-RLPD-KMPC`、`Cal-RLPD-MPVE`。每个方法达到
50k offline updates 后立即停止 optimizer/环境交互，并由独立 watcher 启动
`evaluate_10x10`，输出 100 个 episode 的逐 seed、逐 episode 明细。`RLPD-Raw`
没有 offline gradient-pretraining 阶段，故本轮不启动其 online 训练；它应在单独的
online-only 实验中按 RLPD warmup/UTD20 协议运行，不能把 offline=0 当作离线结果。

这是一轮 development 筛选，不替代后续正式多 training-seed 统计；10×10 是单个
checkpoint 的评估覆盖，不可用于估计 training-seed 间的标准误或置信区间。

这里的“50k 数据、50k updates”不是把数据顺序扫一遍。每个 update 都从 canonical dataset
随机采样一个 batch：Cal-QL 系 batch=1024，Cal-RLPD 系 batch=256；因此名义采样量分别为
51.2M 和 12.8M transition draws，允许重复。所有方法 offline 外层均为每次 iteration 做一次
learner update；RLPD 的 `UTD20` 只描述后续 online 阶段每个真实 transition 的更新强度。

训练器会保存 step-0 内部诊断，并在 10k/20k/30k/40k/50k 做 10-episode offline
diagnostic；绘图器只画 artifact 中实际存在的点，不会为缺失的 0k 人工补点。最终 watcher
在方法完成后读取 `latest.pt`，运行 100 个唯一 reset seed，并按 10 组×每组 10 episode
保存逐回合明细。这里的“10×10”是单个 training seed 下的 robustness evaluation，不是
10 个独立 training seeds。

#### 7.1.2 先前 1M/60k development 口径（历史对照）

```yaml
offline_dataset: 1_000_000 transitions / 1_000 complete episodes
offline_updates:
  Cal-QL: 60_000
  RLPD-Raw: 0
  Cal-RLPD: 60_000
offline_eval_interval: 10_000 updates
online_steps: 50_000 real transitions
online_eval_grid: [0, 1000, 2500, 5000, 10000, ..., 50000]
diagnostic_eval_episodes: 10
training_seed_count_development: 1
```

ExORL 公开协议中的 500k gradient updates 是重要参考，但 Cartpole development 在 20k–60k
已经接近高 return，因此先前使用 60k 做快速算法筛选；当前缩减实验改为 50k 数据和 50k
updates。迁移到新任务时不要自动沿用 50k 或 60k：

- 若官方 baseline 明确给出 update/environment-step 预算，主实验优先遵循官方预算；
- 先在 development seed 设置 10k 或固定比例的诊断点；
- loss 稳定但 return 尚升时延长所有可比方法；
- 只有数值崩坏或基础设施故障可以单独停止；正式性能早停必须对同组方法使用共同规则；
- online 初始预算可取官方预算的 5%–10% 做 sample-efficiency pilot，曲线仍上升时统一扩展。

并行环境只加速 online data collection。`num_envs=5`、50k online steps 表示五个环境共采
50k transition，即约 10k vector rounds，不是每个环境各 50k。offline update 数完全不受
并行环境数量影响。

### 7.2 下一任务的预算模板

| 阶段 | development | formal |
| --- | --- | --- |
| 数据 | 先用官方主数据规模；另做 coverage stress | 冻结 dataset SHA |
| Koopman | 小 lift/horizon sweep，1 seed | 冻结 best architecture 与 SHA |
| Offline RL | 1 seed，10%/25%/50%/100% 诊断 | 官方预算或冻结后的统一预算 |
| Online RL | 1 seed，早期密集评估 | 官方 seed 数与统一终点 |
| 阶段诊断 | 10 deterministic episodes | 固定 seed 协议，不计训练预算 |
| 最终评估 | 10 seed blocks × 10 episodes | 优先官方协议；正式统计仍以多个 training seeds 为轴 |

## 8. 评估和图表

必须区分三类指标：

1. Koopman prediction：state/reward multi-step error，不是 policy return。
2. Offline policy diagnostic：横轴是 gradient updates，online step 恒为 0。
3. Online evaluation：横轴只统计真实 environment transitions，评估 rollout 不进 replay、
   不计训练预算。

训练曲线至少生成两个 panel：

- 左图：offline return vs offline gradient updates；
- 右图：online return vs real online transitions。

当前绘图入口会递归发现各方法的 `metrics.jsonl`，只画实际存在的 evaluation 行；缺失 0k
时直接跳过，不补零、不做插值。默认交付格式只保留 PNG 和 CSV，CSV 保存画图使用的原始点，
不再生成 PDF。当前 offline-only 轮次只有左图数据；在真正启动 online 前，右图应为空或省略，
不能把 offline diagnostic 接到 online 横轴上。

最终表格同时给出：

- step-0 offline return；
- online 1k/5k/25k/终点 return；
- online AUC、cumulative regret；
- final latest checkpoint 的固定 seed 独立评估；
- action saturation、episode length、成功率等任务指标；
- 每个 training seed 的独立结果。

统计推断轴是 training seed，不是 eval episode。development 单 seed 只能报告描述统计；正式
多 seed 才报告跨 training-seed mean/std/SE/CI。当前 final 10×10 会报告 100 episodes 的总体
mean/population std、每组均值和逐 episode return，但这些仍然只是同一个 training seed 的
描述统计。若官方 benchmark 的评估规则不同，官方规则作为 primary，本项目 10×10 robustness
作为 secondary，不能用 secondary 冒充官方分数。

## 9. 运行保护与 artifact 契约

推荐每个方法独立提交，而不是把所有方法强绑定为一个不可拆矩阵；这样可以单独调整或停止
development run，同时 formal aggregate 仍检查公共协议。

每个 run 至少保留：

```text
run.json                 # resolved config + identities
latest.pt                # 可恢复训练状态
best.pt                  # 若使用预注册 best 选择
metrics.jsonl            # offline/online/episode/evaluation 曲线
train.log
evaluation_10x10.json    # completed latest checkpoint 的 100-episode 总评估
evaluation_10x10.log
```

checkpoint 应保存 actor/critic/target、optimizers、temperature、replay、normalizers、
Python/NumPy/Torch/CUDA RNG、阶段和真实 online counter。写入使用 tempfile + fsync + atomic
replace。run identity 绑定 dataset SHA、Koopman SHA、config fingerprint、source commit/dirty
identity 和环境协议。

长训练使用 detached tmux/systemd/调度器保护 SSH 断线。停止时按确切 session/pane/PID 操作；
多个 tmux session 可能共享同一个 server，不能为了停止单个训练而终止整个 tmux server。
当每个方法独立提交时，可再启动一个只读 watcher：它只在 `run.json` 明确
`completed=true`、`execution_scope=offline_only` 且 offline counter 等于批准预算后调用
10×10 evaluator，不负责恢复、停止或修改训练进程。

## 10. 新 Benchmark 启动清单

按顺序完成，任何一项失败都先修协议，不进入下一阶段：

- [ ] 冻结 task、software、control timestep、action repeat、episode limit、reward、discount。
- [ ] 找到官方 baseline implementation/config/evaluation；记录“官方同任务”或“task-adapted”。
- [ ] 审计离线数据来源、episode 对齐、coverage、reward parity 和版本漂移。
- [ ] 生成 canonical dataset + manifest + SHA；episode-level train/val/test split。
- [ ] 按 1–2× raw dimension 起步选择 lift；以秒选择 Koopman/KMPC/MPVE horizon。
- [ ] 训练 Koopman；完成 state/reward multi-step、分层和 worst-case test。
- [ ] 验证三个 raw baseline 完全不加载 Koopman。
- [ ] 验证四个提出方法共享同一 frozen Koopman，actor/critic 输入维数正确。
- [ ] 核对 stochastic KMPC log-prob、Q(z,a)、MPVE detach、1-real+N-model horizon。
- [ ] 对齐网络容量并打印 actor/critic/frozen 参数量。
- [ ] 只跑 smoke/single-update/resume；不把 smoke 结果当性能结果。
- [ ] 单 training seed development：离线分段评估，随后统一 online budget。
- [ ] 根据预注册规则冻结参数；再跑官方 seed 数或 10-seed formal evaluation。
- [ ] 绘制完整 offline+online 曲线，生成 final table，保存 source/data/model/config identity。

### 10.1 每个任务填写一份 resolved spec

以下模板中的值必须在 development 启动前解析完成；正式训练只读取冻结后的 spec：

```yaml
task:
  name: TASK_NAME
  engine_and_version: null
  observation_fields: []
  observation_dim: null
  action_dim: null
  action_bounds: null
  physics_dt: null
  control_dt: null
  action_repeat: null
  episode_steps: null
  reward_contract: exact_oracle | mpve_not_supported
  timeout_bootstrap_discount: null

data:
  source_and_version: null
  behavior_policy: expert | random | exploration | mixture
  policy_episode_count: null
  policy_transition_count: null
  koopman_episode_count: null
  koopman_transition_count: null
  policy_and_koopman_share_dataset: null
  temporal_or_policy_strata: []
  split_by_episode: [0.8, 0.1, 0.1]
  archive_sha256: null
  policy_canonical_sha256: null
  koopman_canonical_sha256: null

koopman:
  raw_dim: null
  lift_dim: null
  rollout_seconds: null
  k_step: null
  batch_size: null
  learning_rate: null
  spectral_radius_limit: null
  stability_reference_dt: null
  model_sha256: null
  test_gate: null

control_and_mpve:
  kmpc_seconds: null
  kmpc_steps: null
  mpve_seconds: null
  mpve_total_steps: null
  controller_hidden_layers: null
  controller_hidden_dim: null
  reward_source: null

baseline_provenance:
  calql: official_same_task | official_derived | task_adapted
  rlpd: official_same_task | official_derived | task_adapted
  source_links: []
  frozen_dev_sweep: null

budget:
  offline_updates: null
  offline_eval_interval: null
  online_real_transitions: null
  online_eval_grid: []
  development_train_seeds: []
  formal_train_seeds: []
  final_eval_seed_groups: null
  episodes_per_eval_group: null
```

## 11. 任务类型的默认调整方向

| 任务类型 | 数据重点 | Koopman/MPVE | 网络与训练重点 |
| --- | --- | --- | --- |
| Cartpole smooth | 角度、速度、swing-up 与平衡区 | 可用较长 model horizon | 小 lift 即可，重点看容量对照 |
| Reacher nonlinear | 全工作空间、目标分布、低回报轨迹 | reward 是否由 observation+target 精确恢复 | 按官方 target sampling/eval，避免只采专家末端区域 |
| Hopper contact | 起跳、腾空、触地、失败姿态 | 先缩短 horizon，分 contact phase 报误差 | 覆盖接触和动作饱和，必要时增加 lift |
| Walker locomotion | gait phase、速度、跌倒与恢复 | 分 gait/contact phase validation | online budget通常高于 Cartpole |
| Humanoid high-dimensional | 多姿态、多速度、跌倒、稀有关节状态 | lift 先约 1× raw，短 horizon 起步 | 控制 ensemble/UTD 的显存和吞吐，不能照搬大宽度 |

这张表只是起点。最终数值必须由官方配置、环境物理时间、数据覆盖和 development 证据共同
决定。

## 12. 一手参考

- ExORL：[论文](https://arxiv.org/abs/2201.13425)、[官方代码与数据](https://github.com/denisyarats/exorl)
- RLPD：[论文](https://proceedings.mlr.press/v202/ball23a.html)、[官方实现](https://github.com/ikostrikov/rlpd)
- Cal-QL：[论文](https://arxiv.org/abs/2303.05479)、[官方实现](https://github.com/nakamotoo/Cal-QL)
- DMC：[论文](https://arxiv.org/abs/1801.00690)、[`dm_control` 官方仓库](https://github.com/google-deepmind/dm_control)
- 本项目 Cartpole O2O 实现说明：[`experiments/dmc/o2o/README.md`](../experiments/dmc/o2o/README.md)
- DMC 迁移总计划：[`docs/dmc_migration_plan.md`](dmc_migration_plan.md)
- MuJoCo Playground Cartpole 阶段结果：[`docs/mujoco_playground_cartpole_results.md`](mujoco_playground_cartpole_results.md)
