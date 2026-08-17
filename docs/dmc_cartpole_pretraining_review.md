# DMC Cartpole Swingup 首训前评审单

> 更新：2026-08-11
>
> 状态：数据源 PPO（seed 20260811）、300k dataset 与 Koopman best（epoch 138）已完成；原 1M 四方法试跑已停止并归档。按用户决定，seed 20260812 的四方法已按 batch-aligned 10M 预算从头持久化运行。
>
> 范围：仅评审 `cartpole_swingup / development`；不授权 benchmark、Reacher 或任何后续任务。
> 重要：数据/Koopman lineage 与 actor comparison lineage 已解耦。结构化 actor 必须绑定冻结 Koopman 文件及其自身历史 provenance，但 policy seed/approval 不必等于数据/model seed/approval。

## 1. 本轮需要用户评估的方案

首轮主链固定为：

`data-source PPO → episode-safe dataset → DeepKoopman + TransitionRewardModel → standard PPO / KMPC / AB-PQ / AC-MPC-MPVE（统一新 seed）→ final evaluation`

需要确认的核心决定是：

1. 使用 `dmc_native_v1` 原生 100 Hz / 1000-step Cartpole 协议；
2. PPO 以 DeepMind Acme 当前 continuous PPO 示例为官方参考口径，而不声称是 DMC 官方 best/SOTA；
3. 数据源 PPO/Koopman 使用 seed `20260811`；四个对比 actor 统一使用新 seed `20260812`，每种方法 `9,998,336` environment steps；
4. 第五方法 `AC-MPC-MPVE` 使用与 KMPC 完全相同的 actor，只给 critic 增加 detached MPC 预测轨迹的 TD-k loss；`K=50 / MPC=20 / MPVE=10`；
5. 最终主结果使用固定预算结束的 `latest.pt`，先报告 deterministic 10 episodes，再报告 10×10 robustness；
6. 接受或调整第 6 节的 proposed gates。

数据源 PPO 的控制表现未过 gate，但其 early/mid/late 300k transitions 已用于训练 Koopman。Koopman 在用户指定的额外 5 分钟后于 epoch 138 正常保存 best 并停止。原 1M actor 试跑只用于判断预算不足，不进入正式结果；本轮四方法从头训练 10M。

## 2. 环境与计分协议

| 项 | Cartpole development |
|---|---|
| task | `cartpole:swingup` |
| observation | 官方 state，5D：`position(3) + velocity(2)` |
| action | 1D，`[-1,1]` |
| control / physics timestep | 0.01 s / 0.01 s（100 Hz） |
| action repeat | 1 |
| episode | 1000 control steps = 10 s |
| score | 1000 步官方 reward 之和，最大 1000 |
| timeout | `truncated=true, terminated=false, discount=1`；TD 对 final observation bootstrap，GAE 不跨 autoreset |
| software target | `dm_control==1.0.44`，`mujoco==3.11.0` |

环境来源是 [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)。`dmc_native_v1` 不与 pixel 或 action-repeat 论文混报；若未来复现其他论文，必须另建具名 protocol/config。

## 3. Reference PPO 配置

DMC 本身没有发布一套任务级“官方最优 PPO 参数”。本项目固定 Google DeepMind Acme commit [`770bc75e`](https://github.com/google-deepmind/acme/tree/770bc75e) 的 continuous-control PPO 示例作为 reference，主要来源是 [run_ppo.py](https://github.com/google-deepmind/acme/blob/770bc75e/examples/baselines/rl_continuous/run_ppo.py)、[config.py](https://github.com/google-deepmind/acme/blob/770bc75e/acme/agents/jax/ppo/config.py)、[networks.py](https://github.com/google-deepmind/acme/blob/770bc75e/acme/agents/jax/ppo/networks.py) 和 [learning.py](https://github.com/google-deepmind/acme/blob/770bc75e/acme/agents/jax/ppo/learning.py)。

| 项 | 拟冻结值 |
|---|---:|
| policy / value network | 各 `3 × 256 ReLU`；所有方法的 critic 都只输入标准化原始 observation |
| policy distribution | state-dependent location/scale，tanh-squashed diagonal Gaussian |
| deterministic evaluation | distribution mode |
| sequences / unroll | 256 / 8 |
| transitions per update | 2048 |
| minibatches / epochs | 8 个 256-transition minibatches / 2 |
| updates / total steps | 4,882 / 9,998,336 |
| learning rate / Adam epsilon | constant `3e-4` / `1e-7` |
| discount / GAE lambda | 0.99 / 0.95 |
| PPO clip | 0.2 |
| entropy / value coefficient | `3e-4` / `1.0` |
| max gradient norm | 0.5 |
| value clip / reward clip | off / off |
| diagnostic cadence | 50,000 environment steps |

Normalization 也是算法合同的一部分：标准 PPO observation 使用 Acme running statistics；structured controller 使用 frozen Koopman center/scale 后再 lift，而其 critic 只读取同一 center/scale 标准化后的原始 observation，不读取额外 `phi` 特征。advantage 除以零偏差修正 EMA 的 `mean(abs(A))`（`tau=0.995`，不做 minibatch z-score 或减均值）；value network 输出归一化值，GAE 前反归一化，critic target 再归一化。normalizer 状态必须进入 checkpoint、resume 和 evaluator，不能仅在训练进程内存在。

`diagnostic_every_steps=50,000` 对齐 Acme 示例的评估节奏，但训练中诊断不参与 `latest.pt` 选择。若首训实际启用这些额外 episodes，必须在 fresh preflight/启动清单中单列环境步和耗时。

历史 CleanRL 结果页报告的 [`640.86 ± 11.44`](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/docs/rl-algorithms/ppo.md#ppo_continuous_actionpy) 只作为旧栈 compatibility anchor。其 [历史实现](https://github.com/vwxyzjn/cleanrl/blob/cbd83f623bd1985af5628ff1609b6a3ddd527df6/cleanrl/gymnasium_support/ppo_continuous_action.py) 使用不同网络、分布、归一化、rollout 和旧 `dm-control/Shimmy/Gymnasium` 组合；该数值是在线随机策略 last-100 training episodes 的统计，不能与本项目 deterministic fixed-checkpoint evaluation 直接比较，也不是 proposed gate 的来源。

## 4. 本轮四种方法、初始化与 MPVE 边界

实现仍支持 KLQR，但按用户决定，本轮 actor comparison 只运行：

- `PPO`：Acme-reference MLP actor；
- `AB-PQ`：低秩二次 value + 冻结 Koopman A/B；
- `KMPC`：cost map → horizon-20 box-constrained Koopman MPC；
- `AC-MPC-MPVE`：**复用同一个 KMPC actor**，标准 PPO/GAE 不变，另加 critic-only MPVE loss。

统一 seed 表示可复现的共同随机实验条件，不表示不同 actor 结构必须有相同参数。标准 PPO 使用 Acme/Haiku 对齐初始化；AB-PQ 使用低秩二次模块自身初始化；KMPC 使用 cost-map/MPC controller 初始化。它们的参数形状与含义不同，不做逐元素配对。共同的 raw-observation critic 在 actor 构造后单独重设相同 seed，因此不会因不同 actor 消耗 RNG 而改变初值。只有 `KMPC ↔ AC-MPC-MPVE` 是严格消融对：actor 与 exploration log-std 从相同值开始，并通过专项测试保证一轮 PPO update 后 actor 参数仍逐元素相同；MPVE 只允许改变 critic。

MPVE 依据 [Romero et al., TRO 2025](https://rpg.ifi.uzh.ch/docs/TRO25_ACMPC_Romero.pdf)。在每个真实 rollout state 上复用 horizon-20 KMPC 已计算的 action/state prediction，截取 `H=10`：

\[
\hat y_h=\sum_{j=h}^{H-1}\gamma^{j-h}\hat r_j
          +\gamma^{H-h}V(\hat s_H).
\]

预测 action/state、reward 和末端 target 在 rollout 时全部 detach；预测 lifted state 先经 `koopman.reconstruct()` 还原为标准化原始 observation，再送入与其他方法同构的 critic。附加 TD-k MSE 只更新 critic，不向 actor、Koopman 或 reward source 反传，actor 仍由标准 PPO surrogate 更新。因此 `KMPC ↔ AC-MPC-MPVE` 的唯一区别是 critic 是否利用 MPC 预测轨迹。

Cartpole 主实验不使用 learned reward：解析 oracle 从 Koopman 重建的下一时刻官方 5D observation 与 applied action 直接复现 `dm_control.suite.cartpole.Balance` dense reward。preflight 必须逐 transition 对照 live `TimeStep.reward`，最大绝对误差不超过 `2e-7`，否则禁止 approval。

Koopman 阶段仍同时拟合 `TransitionRewardModel(normalized_state, applied_action, normalized_next_state)`，作为显式 learned-reward 消融和未来 offline/真机接口。其 checkpoint/validation 保留，但 Cartpole 主 `AC-MPC-MPVE` 的 metadata 必须标记 `dmc_official_observation_oracle_v1`；只有另行审批的 ablation config 才能标记 `learned_transition_model_v1`。后续每个任务也必须先做 observation-oracle parity audit，能精确复现就用官方解析 reward，否则明确使用 learned fallback。

## 5. 精确工作量与数据上限

| 项 | 值 |
|---|---:|
| data-source PPO / Koopman seed | `20260811` |
| actor comparison seed | `20260812`（四方法统一） |
| environments / rollout | 256 / 8 |
| batch / minibatch / epochs | 2048 / 256 / 2 |
| updates / actor | 4,882 |
| steps / actor | 9,998,336 |
| 四方法 comparison 训练总额 | **39,993,344 env steps** |
| 主 final evaluation | 4 methods × 10 seeds × 1 episode = 40,000 steps |
| robustness evaluation 上限 | **400,000 env steps**（4 × 10 × 10 × 1000，已包含主评估 episodes） |
| comparison 训练 + robustness evaluation | **40,393,344 env steps** |
| 参考 PPO 数据 hard cap | 300,000 complete-episode transitions |

历史数据源 PPO 按 early/mid/late 配额持久化了 300k complete-episode transitions；其采集合同仍固定为 488 updates，与本轮 actor comparison 的 4,882 updates 解耦。新的 comparison PPO 使用 `--no-collect`，不重复生成 Koopman 数据。**不能按 actor 总步数反推 episode、split 或 K=50 windows 数量**；这些值以现有 dataset artifact 为准。

## 6. Proposed gates

| gate | 建议值 |
|---|---:|
| PPO 10-episode reference mean return | ≥ 750 / 1000 |
| validation K=50 rollout normalized MSE | ≤ 0.25 |
| model RMSE / zero-order-hold RMSE | ≤ 0.70 |
| TransitionRewardModel validation RMSE | ≤ 0.05 |
| KMPC mean return / PPO mean return | ≥ 0.90 |
| AC-MPC-MPVE mean return / KMPC mean return | ≥ 1.00 |
| deterministic applied-action saturation fraction | ≤ 0.50 |

动作饱和沿用 `abs(applied_action) >= 0.99` 口径。Development 只有一个 training seed，只能用于全链诊断；training-seed std/SE/CI 必须标为不可估。Benchmark 才使用 3 个独立 policy-training seeds 和 df=2 的 Student-t 95% CI，evaluation seeds 与 episodes 是嵌套描述层。

`AC-MPC-MPVE / KMPC ≥ 1.00` 是预注册的“不劣于”候选门槛，并不保证一定实现提升；如果未通过，应报告负结果和 reward/dynamics/horizon 诊断，不能删除第五方法或事后换门槛。

## 7. Final evaluation 口径

- 主 checkpoint：固定预算结束的 `latest.pt`；`best.pt` 仅诊断，不看结果择优替换。
- reference summary：deterministic，10 个固定 seeds 各 1 episode，共 10 episodes。
- robustness summary：相同 10 seeds 各 10 episodes，共 100 episodes；每个 seed 的第一条 episode组成 reference summary。
- 每个方法报告 return、动作饱和及任务诊断；另外基于对齐 training-seed 后的 aggregate mean，报告 `KMPC` 与 `AC-MPC-MPVE` 的差值/比值，不把 evaluation episodes 当成独立配对样本。
- benchmark 使用 3 个独立 training seeds。若共享一个 Koopman/reward checkpoint，必须明确 CI 不含动力学模型训练的不确定性。

五方法 `comparison.json` 的 `overall_control_primary_pass` 只汇总 PPO return、两项方法 return ratio 与 action saturation；Koopman rollout、hold ratio 和 reward RMSE 三项模型 gates 必须从 Koopman/reward report 另行判定，不能被该布尔值隐式视为已通过。

## 8. Fresh 无训练验收与 10M 启动（2026-08-11）

| 证据 | fresh 结果 |
|---|---|
| DMC 专项 | `285 passed`，16.22 s；仅无 DISPLAY 的 GLFW warning |
| 整仓回归 | `358 passed`，19.33 s；仅 3 条既有 SAPIEN/Vulkan/GLFW warning |
| 六任务 parity | 每项 passed；observation/reward/discount 最大误差为 0，reward probe 最大误差 `2.78e-17` |
| timeout | 3-step probe：`truncated=true, terminated=false, discount=1.0` |
| 五 actor probe | batch 256；shape、bound、finite forward/backward 与 gradients 全 passed |
| MPVE probe | 旧 lift=32 审查曾通过；重训配置现为 `H=10`、critic 预测 observation `[256,10,5]`、controller 使用 15 维 lift（5 维原始 observation + 10 维 `phi`）。新 checkpoint 生成后需重新做一次轻量 exact-reward、detach 与梯度 smoke。 |
| 四方法 formal dry-run | PPO / KMPC / AB-PQ / AC-MPC-MPVE 均为 `optimization_steps=0`、`environment_steps=0`、`training_approved=false` |

身份 artifact：

- [fresh preflight](../runs/dmc/preflight/cartpole_swingup_development_actor_seed_20260812_10m.json)：SHA-256 `dfa95e23c769451dfa667d6f199f9ba2c1cbb1dc7592200396b7879714adb537`；
- [PPO dry-run manifest](../runs/dmc/dry_runs/cartpole_swingup/development/actor_seed_20260812_10m/PPO/run_manifest.json)：SHA-256 `9f97519500a4d9af3955d2cc07b87033f65e98f0ada6cf80bf36d6ecd7a22af0`；
- [KMPC dry-run manifest](../runs/dmc/dry_runs/cartpole_swingup/development/actor_seed_20260812_10m/KMPC/run_manifest.json)：SHA-256 `7079a4548da25482c9e706c488ea48203b26b57b2a8e992a83fb2a772e98e7ee`；
- [AB-PQ dry-run manifest](../runs/dmc/dry_runs/cartpole_swingup/development/actor_seed_20260812_10m/AB-PQ/run_manifest.json)：SHA-256 `8515dfb1747fd2faaed19166c720ff5c2edb5f23568c40d934eb1a347d436288`；
- [AC-MPC-MPVE dry-run manifest](../runs/dmc/dry_runs/cartpole_swingup/development/actor_seed_20260812_10m/AC-MPC-MPVE/run_manifest.json)：SHA-256 `911069f25135cf4df97e2b4ece06a5594dbd904ddb892fe071f9d7c3156a8283`；
- config fingerprint：`sha256:916d7f740f2b9d6a13899d51d6eb50cb5a4a339a551fded641eaf89e036b2438`；
- environment protocol fingerprint：`9e7bb9cdb24fc553ac2dd2b4f1432c0fe5969cad47d65bd03db718f46f5a0a95`；
- source fingerprint：`8b73c191e4a3d70e5376ca9434a3626188c475ebf714d9fb011b95c0068a4f4b`。

当前机器资源 probe：

| 项 | fresh 结果 |
|---|---:|
| hardware | 80 CPU；A100 40 GB；系统可用内存约 562 GiB |
| preflight total wall time | 51.50 s |
| 256-env construction + reset | 约 44.33 s（其中 reset 20.82 s） |
| exact 256×8 environment step | 2048 transitions / 0.991 s = 2067.0 transitions/s |
| vector-env process peak RSS | 约 6.35 GiB；相对 probe 前增加约 5.03 GiB |
| actor probe peak CUDA allocation | 最大约 53.3 MiB（KLQR synthetic backward） |
| 300k collector raw-array upper bound | 34.3 MiB；不含 checkpoint、临时文件与文件系统开销 |
| workspace free | 约 31.6 GiB |
| environment-only lower bound / actor seed | 10M 配置下约为 1M probe 的 10 倍；仅作 stepping 下界 |

最后一项只是 256-env stepping 下界，明确排除 policy/critic/MPC、backprop、I/O、评估和 Koopman/reward 训练，不能冒充完整 wall-clock 预测。第一条正式 run 才能产生可信的端到端耗时；当前未以任何 synthetic optimizer step 伪造该数字。

preflight 保存 source identity 作为复现记录，但执行硬边界现已收敛为 task/config、环境协议和 artifact 内容哈希；CPU worker 与实现维护不再让历史数据或冻结模型失效。[10M actor-comparison approval](../runs/dmc/approvals/cartpole_swingup_development_actor_seed_20260812_10m.json) 仍精确绑定其 preflight 与 canonical config。

持久化训练进程与日志：

| actor | PID | 日志 / 状态 |
|---|---:|---|
| PPO | `123478` | `runs/dmc/ppo/cartpole_swingup/development_10m/seed_20260812/PPO/{persistent.log,status.json,latest.pt}` |
| KMPC | `123480` | `runs/dmc/ppo/cartpole_swingup/development_10m/seed_20260812/KMPC/{persistent.log,status.json,latest.pt}` |
| AB-PQ | `123482` | `runs/dmc/ppo/cartpole_swingup/development_10m/seed_20260812/AB-PQ/{persistent.log,status.json,latest.pt}` |
| AC-MPC-MPVE | `123484` | `runs/dmc/ppo/cartpole_swingup/development_10m/seed_20260812/AC-MPC-MPVE/{persistent.log,status.json,latest.pt}` |

它们的 PPID 均为 1 且各有独立 session，SSH 断开不会发送 SIGHUP；checkpoint 使用临时文件、fsync 与原子 replace，并且四方法均已产出首个原子 `latest.pt`。本轮 PPO 禁用重复数据采集，四方法每 10 updates 落盘；中断最多丢失未完成的 checkpoint interval，不能留下半写 checkpoint。

## 9. 10M 已批准并执行的启动命令记录

下面的启动顺序已经通过 CLI/static contract 核对；实际长任务使用等价参数通过 `nohup + setsid` 持久化启动。

```bash
set -euo pipefail

DMC_CONFIG=experiments/dmc/configs/cartpole_swingup.yaml
DMC_PREFLIGHT=runs/dmc/preflight/cartpole_swingup_development_actor_seed_20260812_10m.json
DMC_APPROVAL=runs/dmc/approvals/cartpole_swingup_development_actor_seed_20260812_10m.json
DMC_KOOPMAN=runs/dmc/koopman/cartpole_swingup/development/best.pt

# 0) 最终树无 optimizer preflight。
MUJOCO_GL=egl python -m experiments.dmc.preflight \
  --config "$DMC_CONFIG" \
  --profile development \
  --parity-steps 100 \
  --throughput-steps 1000 \
  --output "$DMC_PREFLIGHT"

# 0b) 四方法绑定 fresh preflight 的 formal dry-run；不需要 approval，
#     不构造 optimizer、不执行 environment step。
for DMC_ACTOR in PPO KMPC AB-PQ AC-MPC-MPVE; do
  DMC_KOOPMAN_ARG=()
  DMC_COLLECT_ARG=()
  if [ "$DMC_ACTOR" != PPO ]; then DMC_KOOPMAN_ARG=(--koopman "$DMC_KOOPMAN"); fi
  if [ "$DMC_ACTOR" = PPO ]; then DMC_COLLECT_ARG=(--no-collect); fi
  MUJOCO_GL=egl python -m experiments.dmc.ppo.train_dmc_ppo \
    --config "$DMC_CONFIG" \
    --profile development \
    --train-seed-index 0 \
    --preflight-file "$DMC_PREFLIGHT" \
    --actor "$DMC_ACTOR" \
    --output-dir "runs/dmc/dry_runs/cartpole_swingup/development/actor_seed_20260812_10m/$DMC_ACTOR" \
    --device cuda \
    --dry-run \
    "${DMC_COLLECT_ARG[@]}" \
    "${DMC_KOOPMAN_ARG[@]}"
done

# 1) 只有用户评审 fresh preflight + dry-run 后才能创建；approval 永不覆盖。
python -m experiments.dmc.approval \
  --config "$DMC_CONFIG" \
  --profile development \
  --preflight "$DMC_PREFLIGHT" \
  --output "$DMC_APPROVAL" \
  --approve

# 2--4) seed 20260811 的 data-source PPO、dataset、Koopman 已完成；不重跑。
```

Actor comparison 使用 fresh approval，并统一 seed 20260812：

```bash
set -euo pipefail
DMC_CONFIG=experiments/dmc/configs/cartpole_swingup.yaml
DMC_PREFLIGHT=runs/dmc/preflight/cartpole_swingup_development_actor_seed_20260812_10m.json
DMC_APPROVAL=runs/dmc/approvals/cartpole_swingup_development_actor_seed_20260812_10m.json
DMC_KOOPMAN=runs/dmc/koopman/cartpole_swingup/development/best.pt

for DMC_ACTOR in PPO KMPC AB-PQ AC-MPC-MPVE; do
  DMC_KOOPMAN_ARG=()
  DMC_COLLECT_ARG=()
  if [ "$DMC_ACTOR" != PPO ]; then DMC_KOOPMAN_ARG=(--koopman "$DMC_KOOPMAN"); fi
  if [ "$DMC_ACTOR" = PPO ]; then DMC_COLLECT_ARG=(--no-collect); fi
MUJOCO_GL=egl python -m experiments.dmc.ppo.train_dmc_ppo \
  --config "$DMC_CONFIG" \
  --profile development \
  --train-seed-index 0 \
  --preflight-file "$DMC_PREFLIGHT" \
  --approval-file "$DMC_APPROVAL" \
  --actor "$DMC_ACTOR" \
  "${DMC_COLLECT_ARG[@]}" \
  "${DMC_KOOPMAN_ARG[@]}" \
  --output-dir "runs/dmc/ppo/cartpole_swingup/development_10m/seed_20260812/$DMC_ACTOR" \
  --device cuda
done

# 6) 固定 latest.pt；每个 actor 输出 10-episode reference + 10x10 robustness。
for DMC_ACTOR in PPO KMPC AB-PQ AC-MPC-MPVE; do
  MUJOCO_GL=egl python -m experiments.dmc.eval.aggregate_dmc \
    --config "$DMC_CONFIG" \
    --profile development \
    --actor-checkpoint "runs/dmc/ppo/cartpole_swingup/development_10m/seed_20260812/$DMC_ACTOR/latest.pt" \
    --output "runs/dmc/eval/cartpole_swingup/development/$DMC_ACTOR.json" \
    --device cuda
done

# 四方法 cross-method report 使用同一 seed/config/evaluation plan 汇总；
# KLQR 本轮不训练，不得用缺失行伪造成五方法 comparison.json。
```

## 10. 本次明确不授权

- 未经 fresh preflight 与用户评审就创建 approval 或启动任何 optimizer；
- Cartpole benchmark（3 training seeds）；
- Reacher → Hopper → Walker 的训练；
- Humanoid 55D pure-state 与可选 67D 标准 observation；
- offline RL 与真机。

任何配置变化都必须重新计算 fingerprint、重跑无训练 preflight 并重新评审，不能在训练 CLI 中临时 override。
