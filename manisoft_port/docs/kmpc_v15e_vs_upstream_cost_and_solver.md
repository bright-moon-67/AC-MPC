# 当前 v15e KMPC 与上游 AC-MPC 的代价及求解差异

## 1. 比较范围

本文比较两条具体实现，而不是泛指所有实验分支：

- **当前工程**：ManiSoft `manisoft-port` 分支上的 v15e structured/history-KMPC，代码基线为 `8fa752474354831a3aaf120f8155bb260279b960`，运行配置见 `runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/run_config.json`。其主要维度为物理状态 $n_x=45$、lifted state $n_z=77$、动作 $n_u=18$、任务 context $n_c=12$、预测步数 $N=10$。
- **远端仓库**：`yuej0422-dev/AC-MPC` 的 canonical state-only PandaReach3 BC-KMPC，固定到 commit [`3549f1ff`](https://github.com/yuej0422-dev/AC-MPC/commit/3549f1ff9b94d0a1eb9a265c806158cf2182cc72)。其主要维度为 $n_x=17,n_z=49,n_u=7,n_c=12,N=10$。

两者共享相同的大框架：冻结 Koopman 模型

$$
z_{k+1}=Az_k+Bu_k,\qquad x_k=Cz_k,
$$

将预测动力学凝缩成稠密凸 QP，固定次数展开 FISTA，每个控制周期只执行预测序列的第一拍。主要差异不在“是否使用 Koopman MPC”，而在**如何生成代价、是否显式指定参考、优化什么动作变量，以及零初始化在物理上代表什么**。

## 2. 总览

| 项目 | 当前 v15e | 远端 canonical PandaReach3 |
|---|---|---|
| cost-map 输出 | 5 个结构化标量 | 每阶段、每状态/动作维度的完整 $q,p$，共 480 个标量 |
| actor MLP | $89\to128\to5$ | $61\to128\to480$ |
| actor 参数量 | 12,165 | 69,856 |
| 二次权重 | tip 三轴、共享动作权重、terminal tip 倍率 | 所有 stage、所有状态/动作维度分别输出 |
| 线性项 | 由显式参考严格确定：$p_x=-Qx_{\rm ref},p_u=-Ru_{\rm ref}$ | 网络自由输出有界 $p$，无显式 $x_{\rm ref}$ |
| 决策变量 | 归一化动作增量 $D\in\mathbb R^{180}$ | 绝对动作序列 $U\in\mathbb R^{70}$ |
| 动作约束 | 增量限制与绝对动作 box 同时生效 | 只有逐拍绝对动作 box |
| FISTA 零初值 | $D^{(0)}=0$，对应整段保持上一动作 | $U^{(0)}=0$，对应整段零动作 |
| production 迭代数 | 80（另以 320 次作诊断） | 20 |

## 3. 共同的阶段代价和凝缩 QP

令

$$
w_k=\begin{bmatrix}x_{k+1}\\u_k\end{bmatrix},\qquad
Q_k=\operatorname{diag}(q_k),
$$

两边最终都构造如下阶段代价：

$$
J=\sum_{k=0}^{N-1}
\left(\frac12 w_k^\top Q_k w_k+p_k^\top w_k\right).
$$

把状态预测写成

$$
X=Sz_0+TU,
$$

即可得到绝对动作坐标中的稠密 QP：

$$
\min_U\ \frac12U^\top H_UU+f_U^\top U.
$$

因此，“当前 cost-map 只输出 5 维”并不意味着 QP 中只有五个代价项。它输出五个**生成完整 $q_k,p_k$ 的结构化参数**；生成后，当前工程仍具有 $N(n_x+n_u)=630$ 个二次对角项和同样数量的线性项。

## 4. 代价映射与 actor 参数量

### 4.1 远端：完整、逐阶段的自由 $q,p$

远端 cost-map 以当前 lifted state 和任务 context 为输入：

$$
g_\theta([z_0,c])\in
\mathbb R^{2N(n_x+n_u)}.
$$

PandaReach3 中

$$
2\times10\times(17+7)=480.
$$

网络的 raw 输出在每个 stage 被转换为

$$
q_{k,j}
=\exp\!\left[
1.5\left(
\tanh r^q_{k,j}
-\frac{1}{n_x+n_u}\sum_\ell\tanh r^q_{k,\ell}
\right)
\right],
$$

$$
p_{k,j}=10\tanh r^p_{k,j}.
$$

所以每个 stage 的 $q$ 几何均值为 1，理论范围约为

$$
e^{-3}\le q_{k,j}\le e^3,
$$

而 $p$ 可以独立于 $q$ 自由改变隐式最优点。其 MLP 为

$$
(49+12)\to128\to480,
$$

参数量为

$$
(61\times128+128)+(128\times480+480)=69{,}856.
$$

这使网络可以分别表达每个预测时刻、每个状态维度和每个动作维度的代价，但也带来更高维的 PPO 优化问题、更弱的物理可解释性以及更大的过拟合空间。

### 4.2 当前 v15e：5 个结构化正权重

当前 actor 输入同样是 $[z_0,c]$，但只输出

$$
s=(s_x,s_y,s_z,s_u,s_T)\in\mathbb R^5.
$$

经过

$$
\ell_i=\log(8)\tanh(s_i),\qquad m_i=e^{\ell_i}\in[1/8,8]
$$

后，五项分别表示：

1. tip 的 $x,y,z$ 三轴权重倍率；
2. 全部 18 个动作共享的权重倍率；
3. 最后一个预测 stage 的 tip 权重倍率。

设 $b_j$ 为固定的物理维度基准权重，当前配置中 tip 三维的 $b_j=1$，其余 42 维为 $b_j=10^{-8}$。于是除最后 stage 外，

$$
q^x_{k,j}=
\begin{cases}
b_jm_j,&j\in\{\text{tip-}x,\text{tip-}y,\text{tip-}z\},\\
b_j,&\text{其他物理维度},
\end{cases}
$$

$$
q^u_{k,j}=m_u,\qquad j=1,\ldots,18.
$$

在 $k=N-1$ 时，三个 tip 权重再乘 $m_T$。注意 $m_T$ 只是 learned terminal-stage multiplier，不是由 Riccati 方程得到的 Lyapunov terminal matrix，也没有配套 terminal invariant set。

当前 MLP 为

$$
(77+12)\to128\to5,
$$

参数量为

$$
(89\times128+128)+(128\times5+5)=12{,}165.
$$

相较远端，它减少了约 82.6% 的 actor 参数，归纳偏置更强、从零 PPO 通常更容易优化；代价是无法学习逐阶段权重、单独的动作通道权重，且非 tip 状态几乎不参与控制目标。

## 5. 显式参考与线性项

### 5.1 当前 v15e 的显式参考构造

当前策略先构造归一化物理参考

$$
x_{\rm ref}
=\left[
x_{\rm current,non\mbox{-}tip},\;
g_{\rm active,tip}
\right],
\qquad u_{\rm ref}=0,
$$

即非 tip 状态保持为当前值，仅把 tip 三维替换为当前 active waypoint。该单个参考在整个 $N=10$ horizon 上重复，并没有提供逐阶段的参考轨迹。

线性项不由网络独立学习，而是绑定为

$$
p^x_k=-Q^x_kx_{\rm ref},\qquad
p^u_k=-R_ku_{\rm ref}=0.
$$

因此每一维都有

$$
\frac12 qx^2-qx_{\rm ref}x
=\frac12q(x-x_{\rm ref})^2
-\frac12q x_{\rm ref}^2.
$$

最后一项与决策变量无关，可以从优化中省略。这使 v15e 在 cost-map 末层全零时已经是一个有效的 tip-reference tracker，不需要先由 PPO 猜出“目标对应哪个线性项”。

不过，这个参考还有两个明显限制：

- waypoint bank 中保存的完整 45 维平衡状态和 18 维平衡动作没有被使用；特别是 $u_{\rm ref}=0$ 可能不是真实柔性体目标形状的保持动作；
- horizon 内只跟踪当前 active waypoint，下一 waypoint 没有变成 stage-wise reference，因此不能显式规划转弯或阶段切换。

### 5.2 远端的隐式目标

远端在线 KMPC 不接收 $x_{{\rm ref},0:N}$ 或 $u_{{\rm ref},0:N-1}$。目标坐标和 stage one-hot 只作为 context 输入 cost-map，网络直接输出自由的 $q_k,p_k$。若忽略耦合并看单个维度，代价的隐式中心为

$$
x^*_{k,j}=-\frac{p^x_{k,j}}{q^x_{k,j}},\qquad
u^*_{k,j}=-\frac{p^u_{k,j}}{q^u_{k,j}}.
$$

因此远端可以学习非零动作偏置、逐阶段变化的隐式目标和更复杂的前馈行为；但 $q,p$ 由动作模仿或 PPO 间接辨识，隐式中心不受物理参考约束，也没有唯一的“真实代价”解释。末层零初始化时 $q=1,p=0$，初始策略是归一化原点调节器，通常既不是目标跟踪器，也不保证输出零动作。

## 6. 动作决策变量和约束

### 6.1 远端：直接优化绝对动作 $U$

远端的决策变量是

$$
U=[u_0^\top,\ldots,u_{N-1}^\top]^\top\in\mathbb R^{70},
$$

约束只有

$$
-0.1\le u_{k,j}\le0.1.
$$

因此 proximal projection 就是逐元素 clamp。没有动作变化率约束，连续两拍的最优动作可以从一个 box 边界跳到另一个边界。

### 6.2 当前 v15e：优化归一化动作增量 $D$

当前决策变量是

$$
D=[d_0^\top,\ldots,d_{N-1}^\top]^\top\in\mathbb R^{180},
$$

绝对动作由

$$
U=Eu_{-1}+\delta(L\otimes I_{18})D,
$$

重构，其中

$$
E=\mathbf1_N\otimes I_{18},\qquad
L=\begin{bmatrix}
1&0&\cdots&0\\
1&1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
1&1&\cdots&1
\end{bmatrix},\qquad
\delta=0.0125.
$$

也就是

$$
u_k=u_{-1}+\delta\sum_{i=0}^{k}d_i.
$$

约束同时包含

$$
-1\le d_{k,j}\le1,
\qquad
-0.30\le u_{k,j}\le0.30.
$$

优点是确定性计划天然限制每拍物理变化量，适合柔性执行器并减少超出 Koopman 训练动作率分布的情况。代价是：

- $L$ 引入跨 stage 耦合，QP 维度由远端的 70 增至 180；
- 在 50 Hz 下，动作从 0 增加到 0.30 至少需要 $0.30/0.0125=24$ 拍，即 0.48 s，长于当前 $N=10$ 的 0.2 s 物理视野；
- 当前实现用逐时刻 causal clamp 使 $D$ 可行。它保证增量和绝对动作不越界，但一般不是上述耦合可行集的精确欧氏投影，因此不能直接继承标准 projected-FISTA 的严格收敛/KKT 解释。

将绝对动作 QP 代入该仿射变换后，求解的是

$$
\min_D\ \frac12D^\top H_DD+f_D^\top D,
$$

其中

$$
H_D=M^\top H_UM,\qquad
f_D=M^\top(H_UEu_{-1}+f_U),qquad
M=\delta(L\otimes I_{18}).
$$

所以增量坐标不仅改变约束，也通过 $M$ 改变 Hessian 的尺度和条件数。

## 7. FISTA 初值：代码上都为零，物理含义不同

两边每次 MPC forward 都重新令决策向量为零，没有把上一周期的最优序列平移后 warm-start。对一般 QP

$$
\min_{y\in\mathcal C}\frac12y^\top Hy+f^\top y,
$$

FISTA 从

$$
y^{(0)}=\tilde y^{(0)}=0,
\qquad
\alpha=\frac{0.95}{\lVert H\rVert_\infty+10^{-6}}
$$

开始迭代。

但是两边的 $y$ 不同：

- **远端**：$y=U$，所以 $U^{(0)}=0$ 表示未来整段均为零绝对动作；
- **当前 v15e**：$y=D$，所以 $D^{(0)}=0$ 经仿射重构后得到
  $$
  u_0^{(0)}=\cdots=u_{N-1}^{(0)}=u_{-1},
  $$
  表示未来整段保持上一拍动作。

因此，当前实现虽然没有 warm-start 优化变量，却具有“以上一动作作为物理基线”的效果。对于需要非零稳态激励的软体系统，这通常比从零绝对动作起步更连续；但固定 80 次迭代的误差会围绕该 hold plan 展开，策略结果仍依赖迭代预算。

远端 production 使用 20 次 FISTA；当前 v15e 使用 80 次，并额外计算 320 次解作为训练诊断。迭代数增加可减小近似求解误差，但不能修复非精确 projection 的理论问题，也不能替代自适应停止或 KKT/duality-gap 验收。

## 8. 对训练与理论性质的主要影响

### 训练层面

1. **当前更容易从零 PPO 启动。** 显式参考和 $p=-Qx_{\rm ref}$ 在网络零初始化时已经提供目标方向；远端必须通过 BC/PPO 学出自由线性项。
2. **当前 actor 方差更低，但表达力更受限。** 12,165 个参数和 5 维结构化输出降低高维策略优化难度，却无法表达逐阶段 $Q/R$、动作通道差异和自由前馈动作。
3. **当前动作更平滑。** 增量坐标直接限制动作率；远端 box-only 方案更容易产生饱和和突变动作。
4. **远端具有更强的隐式目标表达。** 自由 $p$ 能表示非零 holding action 和 stage-wise target，但其学习更依赖数据覆盖、BC 初始化和 PPO 稳定性。

### 理论层面

1. **显式参考提升可解释性，但不自动给出稳定性证明。** 当前 terminal multiplier 不是 Lyapunov terminal cost，且 learned 权重每拍变化，没有 terminal invariant set。
2. **远端的 box projection 是标准 proximal operator。** 当前增量约束形成跨时刻耦合集，现有 causal clamp 只保证可行，不是精确欧氏投影；这是两者求解理论上最关键的差异。
3. **两者都没有真正的 receding-horizon warm-start。** 当前的 hold-action 物理初值改善连续性，但仍不等同于 shifted optimal sequence。
4. **两者的安全保证都有限。** 远端只保证动作 box；当前额外保证动作率，但两者都没有完整状态、碰撞、接触或 robust uncertainty 约束。

## 9. 代码依据

当前工程：

- 通用代价、凝缩和动作增量变换：`antmaze_ac/rl/koopman_mpc_actor.py:55-650`
- 五维 structured cost-map 与显式参考线性项：`antmaze_ac/rl/koopman_mpc_actor.py:653-828`
- $x_{\rm ref}$、$u_{\rm ref}$ 和 active waypoint 构造：`antmaze_ac/rl/history_koopman_mpc_policy.py:245-346`
- tip-only 基准权重及 actor factory：`antmaze_ac/rl/manisoft_ppo_policies.py:397-430`
- v15e 权威运行配置：`runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/run_config.json`

远端固定版本：

- [KoopmanMPCActor 数学说明、cost-map 与求解器](https://github.com/yuej0422-dev/AC-MPC/blob/3549f1ff9b94d0a1eb9a265c806158cf2182cc72/antmaze_ac/rl/koopman_mpc_actor.py#L52-L340)
- [PandaReach3 actor 构造与维度](https://github.com/yuej0422-dev/AC-MPC/blob/3549f1ff9b94d0a1eb9a265c806158cf2182cc72/experiments/state_only_feasibility/train_pandareach_threewaypoint_bc.py#L559-L604)
- [PandaReach3 状态、动作与任务定义](https://github.com/yuej0422-dev/AC-MPC/blob/3549f1ff9b94d0a1eb9a265c806158cf2182cc72/README.md#L7-L10)
