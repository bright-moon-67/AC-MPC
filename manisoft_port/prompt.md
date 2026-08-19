# VS Code Codex 实施 Prompt：D4RL AntMaze + Deep Koopman + Actor-Critic Koopman-LQR

请在当前项目中完整实现基于 D4RL AntMaze 的增量控制实验。CUDA、MuJoCo、D4RL 和 Python 环境你可以检查配置，可以重新安装或更换环境。

## 一、参考仓库

实现前必须检查以下两个仓库，并在最终 README 中记录实际参考文件和 commit：

1. Deep Koopman
   <https://github.com/yuej0422-dev/skill_learning_soft_robot.git>

   重点参考：

   - history=1 的数据组织；
   - encoder、lifting 和 Koopman 矩阵 \(A,B\)；
   - 单步和多步预测损失；
   - 状态重构与归一化；
   - checkpoint、训练和评估流程。

2. Actor-Critic MPC
   <https://github.com/uzh-rpg/acmpc_public.git>

   重点参考：

   - `training_modules/mlp_mpc_policy.py`；
   - `actor → Q/p → differentiable solver → action` 的数据流；
   - critic、PPO 和策略分布的连接方式；
   - actor 输出内部最优控制代价，而不是直接输出最终动作的结构。

不要整仓复制。针对 AntMaze 和 Koopman-LQR 独立实现，并在 README 中注明
参考来源。AC-MPC 是 GPL-3.0；如复制其源码，必须遵守许可证。默认只参考算法
与公开接口。

## 二、总体流程

```text
D4RL AntMaze 数据
  → x_t=[s_t,u_{t-1}]
  → delta_u_t=u_t-u_{t-1}
  → 离线训练 history=1 Deep Koopman
  → 冻结 Koopman
  → actor 根据当前状态输出一组局部 Q(x_t)、p(x_t)
  → 可微稳态 DARE-LQR 求闭环增益 K 和 affine/feedforward 项
  → delta_u_t=-K z_t-d
  → u_t=clip(u_{t-1}+delta_u_t,-1,1)
  → 使用 AntMaze 原始稀疏奖励通过 PPO 训练 actor/critic
```

先完成：

```text
antmaze-umaze-v2
```

通过全部底层测试后再扩展 Medium/Large。

## 三、固定状态与动作接口

原始状态与动作：

\[
s_t\in\mathbb R^{n_s},\qquad u_t\in[-1,1]^8.
\]

legacy D4RL AntMaze 的预期原始状态维度为 29，但代码必须读取实际 observation
和 action space 并进行断言。

所有 Koopman、actor、critic 和主要基线统一使用增广状态：

\[
x_t=
\begin{bmatrix}
s_t\\
u_{t-1}
\end{bmatrix}.
\]

环境 reset 时：

\[
u_{-1}=0.
\]

所有控制器统一输出动作增量：

\[
v_t=\Delta u_t.
\]

实际环境动作：

\[
u_t=\operatorname{clip}(u_{t-1}+v_t,-1,1).
\]

下一增广状态：

\[
x_{t+1}=
\begin{bmatrix}
s_{t+1}\\
u_t
\end{bmatrix}.
\]

history=1 定义为：

\[
(x_t,v_t)\rightarrow x_{t+1}.
\]

不要额外拼接 \(s_{t-1}\)。如果参考仓库中的 history=1 含义不同，在 README
中说明，但优先遵守本 Prompt。

## 四、统一使用增广状态与增量动作

Koopman 的训练接口必须与闭环控制一致：

- Koopman 使用 \(x_t=[s_t,u_{t-1}]\) 和 \(v_t=\Delta u_t\)；
- actor 和 critic 输入相同的增广状态；
- actor/Koopman-LQR 输出 \(\Delta u_t\)；
- 主要 RL baseline 也使用增广状态和 \(\Delta u_t\)；
- 不使用“原始状态 → 绝对动作 PPO”作为主基线；
- 如果研究绝对动作参数化，必须单独训练匹配该接口的动力学模型，并标记为
  独立消融。

\(u_{t-1}\) 拼入状态的主要作用是恢复增量控制下的 Markov 性，不代表需要把
上一动作惩罚到 0。过强动作惩罚会使 Ant 趋向零动作而无法运动。因此：

- 主实验 reward 不加入较大的
  \(-\|u_t\|^2\) 或 \(-\|\Delta u_t\|^2\)；
- 增广状态中 \(u_{t-1}\) 的固定代价权重设为 0 或很小；
- LQR 的 \(\Delta u\) Hessian 只保留保证可解所需的小正定下界；
- 非零控制偏好通过 actor 输出的状态相关 \(Q,p\) 表达；
- 动作安全通过环境边界、裁剪和饱和率监控；
- 强动作惩罚只能作为明确标注的消融实验。

## 五、固定首版参数

```yaml
history: 1
lift_dim: 32
K_step: 20
encoder_hidden_dims: [256, 256]
encoder_activation: silu
```

全部参数必须放入 YAML，不能散落硬编码。

`K_step=20` 只表示 Deep Koopman 训练时，多步 rollout/prediction loss 覆盖
20 个状态转移：

\[
\hat x_{t+1},\ldots,\hat x_{t+20}.
\]

它不是 LQR horizon，也不是 DARE 求解的迭代次数。代码、配置和日志中必须明确
区分：

```text
K_step              # Koopman 多步训练长度
dare_max_iterations # DARE 数值求解的最大迭代次数
gain_update_interval # 闭环增益重新计算间隔
```

不要再定义 `lqr_horizon`。

## 六、任务一：增量动作环境

实现 `DeltaActionWrapper`：

- reset 后 `prev_action=zeros(action_dim)`；
- observation 返回 `[original_obs, prev_action]`；
- step 输入为 `delta_u`；
- 计算并裁剪总动作；
- `prev_action` 更新为真正施加到环境的动作；
- next observation 返回 `[next_obs, applied_action]`；
- `info` 记录 requested delta、applied delta、applied action 和饱和比例；
- 正确处理 episode reset；
- 正确处理 vectorized env 中不同子环境分别 reset。

测试 reset、连续动作累加、动作裁剪、episode 隔离、vectorized env 和 shape。

## 七、任务二：构造 D4RL 增广数据

从原始轨迹生成：

\[
\begin{aligned}
x_t &= [s_t,u_{t-1}],\\
v_t &= u_t-u_{t-1},\\
x_{t+1} &= [s_{t+1},u_t].
\end{aligned}
\]

要求：

- 使用 `terminals` 和 `timeouts` 识别 episode 边界；
- 每个 episode 首步使用 \(u_{-1}=0\)；
- 禁止跨 episode 使用上一动作；
- train/validation/test 按完整 episode 划分；
- 归一化统计量只由 train split 计算；
- 保留 reward、terminal、timeout、episode_id 和 step index。

硬性一致性测试：

\[
\max_t\|u_t-(u_{t-1}+v_t)\|_\infty<10^{-6}.
\]

在数据测试通过前，不要训练 Koopman。

## 八、任务三：实现 Deep Koopman

采用带 identity skip 的 lifting：

\[
z_t=
\begin{bmatrix}
x_t\\
\phi_\theta(x_t)
\end{bmatrix},
\qquad
z_{t+1}=Az_t+Bv_t.
\]

`lift_dim=32` 指新增的 neural lifted features 数量。定义读取矩阵 \(C\)，满足：

\[
x_t=Cz_t.
\]

损失至少包括：

\[
\mathcal L_{\mathrm{linear}}
=
\|z_{t+1}-(Az_t+Bv_t)\|^2,
\]

\[
\mathcal L_{\mathrm{rollout}}
=
\sum_{j=1}^{K_{\mathrm{step}}}
w_j\|\hat x_{t+j}-x_{t+j}\|^2,
\qquad K_{\mathrm{step}}=20,
\]

以及必要的状态重构和正则项。

必须完成：

- shape 测试；
- one-step loss 前向测试；
- 20-step rollout 测试；
- backward/gradient 测试；
- 短程训练 smoke test；
- checkpoint 保存/加载测试；
- 归一化/反归一化一致性测试；
- 评估脚本 smoke test；
- NaN/Inf 检查。

## 九、Koopman 正式训练

只有环境、数据、shape、loss、梯度、checkpoint 和 smoke test 全部通过后，才
启动正式训练。

固定训练预算：

```yaml
history: 1
lift_dim: 32
K_step: 20
max_epochs: 1000
max_wall_time_hours: 5
stop_condition: first_reached
```

达到 1000 epochs 或累计正常训练时间达到 5 小时，二者先到即停止。

必须：

- 持续保存验证集最佳 `best_validation` checkpoint；
- 保存停止时的 `last` checkpoint；
- 定期保存可恢复 checkpoint；
- 记录实际 epochs、总耗时、最佳 epoch 和停止原因；
- 崩溃、NaN、数据错误或设备错误不视为训练完成，必须修复后重新运行。

训练结束后立即：

1. 加载 `best_validation` checkpoint；
2. 运行 1、5、10、20、25 步 rollout 评估；
3. 保存完整指标、预测曲线、配置和归一化参数；
4. 与“保持状态不变”naive baseline 对比；
5. 分别报告 \(x,y\)、其他状态和上一动作预测误差；
6. 冻结 encoder、\(A\)、\(B\)；
7. 继续 Koopman-LQR，不在 Koopman 调参上无限停留。

## 十、任务四：可微稳态 Koopman-LQR

在当前真实时刻 \(t\)，actor 根据 \(x_t\) 输出一组局部、冻结于当前控制段的
二次代价参数。定义：

\[
y_j=
\begin{bmatrix}
x_j\\
v_j
\end{bmatrix}.
\]

局部无限时域目标：

\[
J_t=
\sum_{j=0}^{\infty}
\left(
y_j^\top Q(x_t)y_j+
p(x_t)^\top y_j
\right).
\]

这里：

- \(Q(x_t),p(x_t)\) 在一次增益求解和对应控制段内保持固定；
- 下一次更新增益时，actor 根据新状态重新产生 \(Q,p\)；

第一版使用结构化/对角代价：

\[
Q(x_t)=
\operatorname{diag}
\left(
\operatorname{softplus}(\hat q(x_t))+\epsilon
\right).
\]

控制增量块 \(R_{\Delta u}\) 必须严格正定。利用 \(x=Cz\) 将物理增广状态代价
映射到 lifted space。

对固定 Koopman 系统：

\[
z_{j+1}=Az_j+Bv_j,
\]

求解离散代数 Riccati 方程 DARE，得到稳态 value Hessian \(P\) 和闭环增益：

\[
K=
\left(
R+B^\top PB
\right)^{-1}
B^\top PA.
\]

因为存在线性代价项 \(p\)，控制律一般不是纯 \(-Kz\)，必须同时求 affine/
feedforward 项：

\[
v_j=-Kz_j-d.
\]

实现必须正确处理状态线性项和控制线性项。不能忽略 \(p\) 后仍声称复现了
原 AC-MPC 的 \(Q,p\) 参数化。

### DARE 实现要求

- 使用 PyTorch；
- batch 支持；
- actor 训练路径可微；如果最后迭代求K的梯度计算开销太大，也可以考虑actor更新时不要\(Q,p\)到\(\Delta u\)的梯度
- 禁止在训练路径中使用不可微的 SciPy DARE；
- SciPy `solve_discrete_are` 只可作为单元测试参考；
- 使用稳定的可微 DARE/fixed-point/Kleinman 或隐式微分实现；
- `dare_tolerance` 和 `dare_max_iterations` 为数值求解参数；
- 它们不是控制 horizon；
- 禁止显式 matrix inverse，使用 `torch.linalg.solve` 或 Cholesky；
- 支持 jitter、收敛标志、残差和条件数诊断；
- 检查 \((A,B)\) stabilizability 和 DARE 可解性；
- 梯度能从 \(v=-Kz-d\) 回传到 actor 输出的 \(Q,p\)；
- Koopman 参数默认冻结。

必须测试：

- 标量解析系统；
- 与 SciPy DARE 结果对比；
- DARE residual；
- batch 与逐样本一致；
- affine/linear cost；
- `torch.autograd.gradcheck`；
- actor 参数收到有限、非零梯度；
- 不可稳定或 DARE 不收敛时给出明确诊断。

## 十一、闭环增益的使用方式

标准稳态 LQR 得到一个 \(K\)，它可以在一小段时域内重复使用。即使 \(K\) 不
更新，控制仍是闭环的，因为每一步都使用新的 lifted state：

\[
v_j=-Kz_j-d.
\]

增加配置：

```yaml
gain_update_interval: 1
```

含义：

- `1`：每个环境 step 重新由 actor 产生 \(Q,p\) 并求解新的 \(K,d\)；
- `m>1`：每隔 \(m\) 个环境 step 更新一次 \(Q,p,K,d\)，中间使用相同
  \(K,d\) 对新状态做闭环反馈。

正确性阶段默认 `gain_update_interval=1`。之后可测试 2、5、10，以降低平均
求解开销。

注意：若 PPO 训练时跨多个 step 复用由旧状态产生的 \(K,d\)，策略包含隐藏的
控制器状态。必须采用以下一种正确方式：

1. 默认每步重算，使动作是当前 observation 的确定函数；或
2. 将 segment 作为 macro-action，由 PPO 只在 segment 边界决策，并累计该段
   reward；或
3. 将当前 \(Q,p,K,d\) 和 segment phase 纳入可重建的策略状态。

禁止在没有处理该问题时，直接把旧 \(K\) 缓存在普通 feed-forward PPO policy
内部，否则 PPO 的 `evaluate_actions` 无法重建采样时策略。

## 十二、任务五：固定代价闭环

接 PPO 前，使用人工固定 \(Q,p\) 验证：

- DARE 收敛；
- 闭环谱半径合理；
- 输出有限 `delta_u`；
- AntMaze 能运行完整 episode；
- reset、动作累加和状态拼接正确；
- 动作不会立即全部饱和；
- 短期 Koopman rollout 与真实环境趋势一致；
- 日志能区分 Koopman 误差、DARE 问题和动作裁剪问题。

固定代价不要求直接解决 U-Maze，但闭环控制链必须稳定。

## 十三、任务六：Actor-Critic Koopman-LQR

参考 AC-MPC 的结构：

- actor 输入增广状态 \(x_t\)；
- actor 输出一组局部 `Q_diag` 和 `p`；
- 可微 DARE-LQR 产生 \(K,d\)；
- `delta_u=-Kz-d` 作为高斯策略均值；
- critic 使用独立 MLP 输出状态价值；
- PPO 根据环境 reward 更新 actor 和 critic；
- 不要在策略内部隐藏 `u_prev`；
- 保留探索噪声；
- 限制 \(Q,p\) 输出尺度；
- 记录 \(Q,p\)、DARE residual、条件数、闭环谱半径、动作饱和率和梯度范数；
- 第一阶段冻结 Koopman；
- 联合微调 Koopman 只能作为独立消融。

数学中沿用 \(Q,p\)，代码中建议将代价矩阵命名为 `stage_hessian`，避免与 RL
action-value \(Q(s,a)\) 混淆。

## 十四、与时变价值函数的关系

该结构可理解为状态条件、局部的二次价值/代价模型，但代价矩阵 \(Q(x_t)\)
不是 RL 的 action-value \(Q(s,a)\)。

每次增益更新：

1. actor 根据当前 \(x_t\) 产生局部 \(Q(x_t),p(x_t)\)；
2. DARE 得到稳态二次 value：

   \[
   V(z)=z^\top Pz+2r^\top z+c;
   \]

3. 得到 \(K,d\)；
4. 在当前控制段中使用 \(v=-Kz-d\)；
5. 下一次更新时根据新状态重新生成局部代价和增益。

因此整体上是 gain-scheduled/state-conditioned LQR：

- actor 学习局部二次代价；
- DARE 将局部代价变为稳定反馈增益；
- PPO critic 根据真实稀疏回报估计长时域 \(V^\pi(x)\)；
- 两种 value 的作用不同，不要混淆命名。

## 十五、任务七：奖励与训练

主实验保持 AntMaze 默认稀疏奖励：

\[
r_t=
\begin{cases}
1,&\text{到达目标},\\
0,&\text{否则}.
\end{cases}
\]

内部二次型是 actor 学习出的控制代价，不是环境 reward。不要把主实验替换成
普通欧氏距离平方奖励。

若纯稀疏训练没有成功样本，依次尝试：

1. 调整探索；
2. 使用目标位置代价初始化 \(Q,p\)；
3. curriculum；
4. 基于迷宫最短路距离的 potential-based shaping。

塑形必须使用独立配置；最终评估始终报告原始 sparse success rate 和
D4RL normalized score。

## 十六、主要基线

所有主要方法统一使用增广状态和增量动作：

1. 增广状态 → `delta_u` 的 Delta-PPO；
2. 固定代价、固定 DARE 增益的 Koopman-LQR；
3. 稀疏奖励 Actor-Critic Koopman-LQR；
4. 可选的 shaping Actor-Critic Koopman-LQR。

先用一个 seed 做 smoke test，再使用至少 5 个 seeds 做正式实验。每个正式
checkpoint 至少评估 100 episodes。

报告：

- success rate；
- sparse return；
- D4RL normalized score；
- episode length；
- \(\|\Delta u\|^2\)；
- \(\|u\|^2\)；
- 动作饱和率；
- Koopman 多步误差；
- DARE residual/failure；
- 闭环谱半径；
- 多 seed 均值和标准差。

## 十七、推理时间

稳态 Koopman-LQR ，当 actor 更新 \(Q,p\) 时需要求解
一次 DARE。若在多个环境 step 内复用 \(K,d\)，则 DARE 成本可由该控制段摊销；
每个 step 只需 encoder 和矩阵反馈 \(v=-Kz-d\)。

实现：

```text
scripts/benchmark_inference.py
```

要求：

- warm-up 后测试至少 1000 次；
- 分别测量 encoder、actor、DARE、单步反馈和总耗时；
- 报告 mean、P95、max；
- 测试 CPU/GPU、batch=1；
- 不包含 render；
- 对比 `gain_update_interval=1,2,5,10`；
- 同时报告控制性能，不能只追求速度；
- P95 必须小于环境控制周期。

## 十八、建议代码结构

```text
antmaze_koopman_ac/
├── configs/
├── antmaze_ac/
│   ├── envs/delta_action_wrapper.py
│   ├── data/build_sequences.py
│   ├── koopman/model.py
│   ├── koopman/losses.py
│   ├── control/differentiable_dare.py
│   ├── control/steady_state_lqr.py
│   ├── control/quadratic_cost.py
│   ├── rl/cost_actor.py
│   ├── rl/critic.py
│   └── rl/ac_koopman_policy.py
├── scripts/
│   ├── build_d4rl_sequences.py
│   ├── train_koopman.py
│   ├── evaluate_koopman.py
│   ├── benchmark_inference.py
│   ├── test_fixed_lqr.py
│   ├── train_actor.py
│   └── evaluate_actor.py
└── tests/
    ├── test_delta_action_wrapper.py
    ├── test_dataset_boundaries.py
    ├── test_action_reconstruction.py
    ├── test_koopman_rollout.py
    ├── test_dare_reference.py
    ├── test_dare_gradients.py
    ├── test_affine_lqr.py
    └── test_policy_smoke.py
```

环境、数据、Koopman、DARE-LQR 和 RL 必须解耦。

## 十九、严格执行顺序

1. 阅读当前项目和两个参考仓库；
2. 确认实际状态、动作和 D4RL 数据字段；
3. 实现并测试 `DeltaActionWrapper`；
4. 构造并验证增广 D4RL 数据；
5. 完成 Koopman 模型、loss、梯度和 smoke test；
6. 正式训练 history=1、lift=32、K_step=20 的 Koopman，最多 1000 epochs
   或 5 小时；
7. 加载最佳 checkpoint，完成多步评估并冻结 Koopman；
8. 实现并验证可微稳态 DARE-LQR；
9. 完成固定代价闭环；
10. 运行推理时间 benchmark；
11. 接入 actor/critic 和 PPO；
12. 对照原github项目检查actor内部训练实现是否正确（只有底层控制器实现替换，其余保持原有训练框架）
13. 在 U-Maze 上训练稀疏奖励主实验；
14. 加入统一接口的基线和 shaping 对照；
15. U-Maze 通过后再扩展 Medium/Large。

每完成一个阶段都必须运行测试、报告改动文件和实际结果，再进入下一阶段。
不要在底层测试未通过时同时修改 Koopman、DARE-LQR 和 PPO。

## 二十、最终交付

- 增量动作 AntMaze wrapper；
- D4RL 增广数据转换脚本；
- history=1、lift=32、K_step=20 的 Koopman；
- Koopman 正式训练记录：
  - 实际 epochs；
  - 实际时长；
  - 停止原因；
  - best/last checkpoint；
  - 多步评估结果；
- 通过参考测试和梯度测试的可微 DARE；
- 稳态 Koopman-LQR 和 affine/feedforward 控制；
- gain update/hold 机制及 PPO 一致性处理；
- 推理耗时 benchmark；
- Actor-Critic Koopman-LQR；
- 稀疏奖励训练和评估脚本；
- 统一使用增广状态与增量动作的基线；
- 单元测试；
- README，说明数学定义、命令、参考仓库 commit、checkpoint hash、
  许可证和已知限制。

不要伪造训练结果。如果完整 PPO 训练耗时较长，至少完成 smoke test，再尝试少量训练看结果.并提供
继续训练的命令、checkpoint 和日志路径。
