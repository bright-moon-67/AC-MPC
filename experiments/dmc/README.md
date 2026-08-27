# AC-MPC DMC benchmark sub-project

状态：Cartpole data-source PPO、300k dataset 与 Koopman 已完成。主 MPVE 使用官方解析 reward oracle，采用 `K=50 / MPC=20 / MPVE=10`；原 1M actor 诊断已归档，本轮四方法按 batch-aligned 10M 预算从头训练。

## Benchmark 梯度

主路线：

`cartpole_swingup → reacher_hard → hopper_hop → walker_run`

Walker 通过后，默认增加用户指定的 `humanoid_run_pure_state`（55D）高维压力测试。`humanoid_run`（67D 标准 observation）保留为需另行确认的公开 benchmark 对照。

除 Hopper Stand 的历史 AR2 适配外，PPO benchmark 路线使用 [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) 的 `dmc_native_v1`：`dm_control==1.0.44`、native control timestep、action repeat 1、官方 state observation/reward、1000 control steps/episode。

Formal offline-to-online（O2O）实验有两套互斥协议：Cartpole Swingup 使用上述 native 协议；Hopper Stand/Hop 使用 TD-MPC2 数据对应的具名 `tdmpc2_action_repeat2_v1`，即每个 outer action 保持两个 20 ms native steps、reward 求和、outer observation 取第二步之后的状态，共 500 outer steps/episode。dataset、训练环境、checkpoint 和 evaluator 的协议必须逐字段一致，否则启动或恢复会直接失败。

## Formal O2O 实验

本分支包含 Cartpole Swingup、Hopper Stand 和 Hopper Hop 的正式七方法矩阵：

`Cal-RLPD-KMPC / Cal-RLPD / Cal-RLPD-Lift / Cal-QL / RLPD / AWAC / IQL`

| 任务 | Offline RL 数据 | Offline/online 预算 | 环境协议 | Koopman / KMPC / MPVE horizon |
|---|---:|---:|---|---:|
| Cartpole Swingup | 100k | 50k updates / 20k transitions | native 100 Hz，1000 steps | 50 / 20 / 10 |
| Hopper Stand | 200k | 50k updates / 20k transitions | AR2 25 Hz，500 steps | 20 / 8 / 4 |
| Hopper Hop | 200k | 50k updates / 20k transitions | AR2 25 Hz，500 steps | 20 / 8 / 4 |

正式训练 seeds 为 `20260851..20260855`。RLPD 没有 offline gradient phase，其余方法执行 50k offline updates；诊断评估每 5k offline updates、每 2.5k online transitions 运行 10 episodes。paper-level boundary 固定为 online step 0/20k 的 10 evaluation seeds × 10 episodes。

Hopper 的 frozen Koopman 使用独立于 200k RL buffer 的 400k Hop+Stand quality-balanced corpus；每个 training seed 对应独立 Koopman 初始化。`Cal-RLPD-KMPC` 和 `Cal-RLPD-Lift` 的 offline actor/critic LR 为 `1e-4`，online 恢复为 `3e-4`。

稳定入口如下：

- `formal_cartpole_dataset.py`、`formal_cartpole_koopman.py`、`formal_cartpole.py`：Cartpole 数据、模型和单 seed/method 正式训练；
- `download_tdmpc2_task.py`、`convert_tdmpc2.py`：Hopper TD-MPC2 数据获取与确定性转换；
- `formal_hopper_koopman.py`、`formal_hopper.py`、`formal_hopper_hop.py`：Hopper 模型与单 seed/method 正式训练；
- `archive_checkpoints.py`：完成后的 10×10 boundary evaluation 与 checkpoint ZIP 归档；
- `formal_cartpole_results.py`、`summarize_hopper_stand.py`：正式结果校验、汇总与作图。

所有入口在写入 optimizer state 前校验任务、dataset hash/selection、Koopman identity、seed、维度、horizon 和环境时间协议。运行产物中的 `run.json`、`protocol.json`、dataset/Koopman manifests 与 SHA256 是最终复现依据；源码不包含机器本地 `runs/` 结果或一次性提交队列。

## 五种方法

| 方法 | actor / value 路径 | 训练差异 |
|---|---|---|
| `PPO` | Acme-reference MLP policy/value | 标准 PPO + GAE |
| `KLQR` | cost map → differentiable DARE → affine LQR | 标准 PPO + GAE |
| `AB-PQ` | low-rank quadratic value + frozen A/B | 标准 PPO + GAE |
| `KMPC` | cost map → finite-horizon box-QP | 标准 PPO + GAE |
| `AC-MPC-MPVE` | **与 KMPC 完全相同的 actor** | 标准 PPO + GAE，并给 critic 加 detached MPC trajectory TD-k loss |

`AC-MPC-MPVE` 是 `KMPC` 的 critic-training 消融，不是第五套 actor。依据 [Romero et al., TRO 2025](https://rpg.ifi.uzh.ch/docs/TRO25_ACMPC_Romero.pdf)，它复用 MPC 已求出的状态/动作预测；预测轨迹、预测 reward 与 bootstrap target 在 rollout 时 detach，附加 loss 只更新 critic，不向 actor、Koopman 或 reward source 反传。

四种 active comparison 方法的 critic 都是同一个 `3×256 ReLU` raw-observation value network：PPO 使用 Acme running-normalized observation，structured 方法使用 frozen Koopman center/scale 标准化后的原始 observation；只有 actor/controller 读取 lifted state。critic 在不同 actor 构造完成后单独用共同 seed 初始化。MPVE 将预测 lift 经 `koopman.reconstruct()` 还原为标准化 observation 后再计算 value expansion。

Cartpole 主 MPVE 使用逐 transition parity 验证的官方解析 reward oracle；`TransitionRewardModel(normalized_state, applied_action, normalized_next_state)` 仍与 Koopman 联训并进入 checkpoint，供显式 learned-reward 消融和未来 offline/真机使用。其他任务在开始前先审计 observation 是否足以精确复现官方 reward，不能通过 parity 才使用并标记 learned fallback。

## Reference PPO 合同

DMC 没有官方任务级 best PPO 超参。主 PPO 固定到 Google DeepMind Acme continuous PPO 示例 commit [`770bc75e`](https://github.com/google-deepmind/acme/tree/770bc75e)，称为“current official reference”，不称为 DMC 官方 best/SOTA。主来源：[example](https://github.com/google-deepmind/acme/blob/770bc75e/examples/baselines/rl_continuous/run_ppo.py)、[config](https://github.com/google-deepmind/acme/blob/770bc75e/acme/agents/jax/ppo/config.py)、[networks](https://github.com/google-deepmind/acme/blob/770bc75e/acme/agents/jax/ppo/networks.py)、[learner](https://github.com/google-deepmind/acme/blob/770bc75e/acme/agents/jax/ppo/learning.py)。

Cartpole v2 候选合同：

- policy/value 各 `3×256 ReLU`；state-dependent tanh-squashed diagonal Gaussian；deterministic evaluation 取 mode；
- observation normalization、EMA mean-absolute-advantage scaling（`tau=0.995`，不做 batch z-score）、value normalization；所有统计进入 checkpoint/resume/eval；
- `256 sequences × unroll 8 = 2048` transitions/update，8 个 minibatches，minibatch 256，2 epochs；
- constant LR `3e-4`、Adam `eps=1e-7`、gamma `0.99`、GAE `0.95`、clip `0.2`、entropy `3e-4`、value coefficient `1.0`、max grad `0.5`；value/reward clipping 关闭；
- Acme 示例参考预算约 1M；当前 Cartpole development 因 1M 诊断不足，按用户决定扩为每 actor/seed `9,998,336 = 4,882×2,048` environment steps。诊断 cadence 50,000 steps；这个延长预算不称为 Acme 官方设置。

CleanRL 结果页的历史 [`640.86±11.44`](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/docs/rl-algorithms/ppo.md#ppo_continuous_actionpy) 仅作旧栈 compatibility anchor；其 [历史实现](https://github.com/vwxyzjn/cleanrl/blob/cbd83f623bd1985af5628ff1609b6a3ddd527df6/cleanrl/gymnasium_support/ppo_continuous_action.py) 的网络、分布、归一化、rollout 和在线随机 last-100 统计均不同，不能直接等同于本项目 deterministic final evaluation。

## 目录结构

```text
experiments/dmc/
├── actors.py                 # PPO/KLQR/AB-PQ/KMPC/AC-MPC-MPVE 构造入口
├── reward_model.py           # TransitionRewardModel + strict checkpoint loader
├── reward_oracle.py          # parity-verified official reward + learned fallback contract
├── config.py                 # versioned YAML contract
├── configs/*.yaml            # 每任务预算、seed、模型与 proposed gates
├── tasks/                     # registry + canonical adapter
├── ppo/                       # batch runner + five-method PPO/MPVE trainer
├── collect/                   # complete-episode collector + dataset builder
├── koopman/                   # lazy K-step Koopman + reward joint trainer
├── eval/                      # evaluation + nested aggregate + five-method gate report
└── preflight.py              # 无 optimizer step 的训练前验收
```

实现尽量复用共享 `antmaze_ac/` 控制与 Koopman 核心；共享 checkpoint writer 只做了可选 `extra_payload` 的向后兼容扩展，DMC 专属环境、数据、训练和身份校验逻辑仍放在本目录。

## Cartpole development 候选预算

| 项 | 值 |
|---|---:|
| data/model seed | `20260811` |
| actor comparison seed | 1 (`20260812`，四方法统一) |
| environments / rollout / batch | 256 / 8 / 2048 |
| minibatch / epochs / updates | 256 / 2 / 4,882 |
| 每 actor 步数 | 9,998,336 |
| 四方法 comparison 训练 | **39,993,344 environment steps** |
| reference PPO data cap | 300,000 complete-episode transitions |
| Koopman | 额外 `phi` 10 维（总 lifted dim 15），K=50 = 0.50 s |
| TransitionRewardModel | hidden 256×256，joint reward MSE weight 1.0 |
| KMPC / MPVE | MPC horizon 20；MPVE horizon 10；Cartpole official exact reward |
| final reference evaluation | deterministic `latest.pt`，10 fixed seeds × 1 episode |
| robustness | 10 fixed seeds × 10 episodes；四方法上限 400,000 steps |
| comparison 训练 + robustness | **40,393,344 environment steps** |

300,000 是历史 data-source PPO 的 hard cap，其 collection contract 保持 488 updates。本轮 comparison PPO 使用 `--no-collect`，不重复生成 Koopman 数据。完整 episode 数、8/1/1 split 和 K-step windows 以既有 builder artifact 为准，不能按 10M actor 预算推算。

## Approval-bound DAG

```text
data/model lineage: seed 20260811 PPO data → dataset → Koopman
actor lineage: fresh source/config → preflight → user review → approval
  → seed 20260812 PPO / KMPC / AB-PQ / AC-MPC-MPVE
  → deterministic latest.pt 10-episode reference + 10×10 robustness
```

数据源 PPO/Koopman 与 actor comparison 是两条独立 lineage。冻结模型严格保留其文件、dataset 和历史审批身份；标准 PPO、KMPC、AB-PQ、AC-MPC-MPVE 共享新的 policy seed/approval。两层 seed 不要求相同。`best.pt` 只作诊断；actor 主结果固定使用预算结束时的 `latest.pt`。

## Fresh 无训练 preflight

最终实现冻结后运行：

```bash
python -m pip install -e '.[dmc,test]'

MUJOCO_GL=egl python -m pytest -q tests/test_dmc_*.py

MUJOCO_GL=egl python -m experiments.dmc.preflight \
  --config experiments/dmc/configs/cartpole_swingup.yaml \
  --profile development \
  --parity-steps 100 \
  --throughput-steps 1000 \
  --output runs/dmc/preflight/cartpole_swingup_development_v2.json
```

整体无训练验收由专项 tests、真实 preflight 和随后绑定该 preflight artifact 的 reference-PPO formal dry-run 三部分组成。preflight 覆盖六任务 live spec/parity、timeout bootstrap、五 actor exact-shape/finite gradient、reward model、MPVE detach/critic-only gradient及环境/批处理 probe；它不构造或执行 optimizer，报告保持 `training_approved=false`。preflight 文件落盘后再单独执行 dry-run，并核对 `optimization_steps=0`、`environment_steps=0`；尚无 dataset/Koopman checkpoint 时不伪造 structured/Koopman dry-run。

preflight/approval 的硬边界是 task、canonical config、环境协议和 artifact 内容哈希。source identity 仍写入报告用于复现审计，但不再跨阶段锁死执行：调整 CPU worker、日志或实现细节不会让已经采集的数据和冻结模型失效。dataset、Koopman、actor 各自保留 provenance；下游只强制 task/protocol/shape/content hash 兼容，不要求它们共享同一 approval 文件。

真实 DMC 训练默认使用 spawn 多进程 vector runner；worker 数是执行参数，不属于算法或 benchmark 身份：

```bash
python -m experiments.dmc.ppo.train_dmc_ppo ... --env-workers 16
```

也可设置 `DMC_ENV_WORKERS`。`1` 表示参考同步实现；本机 Cartpole 256-env probe 中，16 workers 的 stepping 为约 12.5k transitions/s，单 worker 约 2.0k/s。不同 worker 数必须保持相同 seed、protocol、transition 与 timeout 语义，已有逐元素 parity 测试覆盖。

## 数据、checkpoint 与统计契约

主数据来自参考 PPO early/mid/late 阶段且只落盘完整 episode。每条 transition 保存 state、requested/applied action、next state、reward、discount、terminated/truncated、reset seed、episode/step id、policy update/global step、stage、protocol/config/profile/approval lineage。

Builder 按完整 episode 切分，拒绝断链、不完整 episode、重复轨迹及 protocol/seed/approval drift。Koopman/reward checkpoint 保存 dataset hash、normalizer、K-step 物理视野和完整 lineage；actor checkpoint 保存架构、Acme normalization state 和精确 Koopman/reward identity。Evaluator 必须从 checkpoint 恢复，不能从 CLI 默认值猜网络。

Development 只有一个 training seed，training-seed std/SE/CI 报不可估。Benchmark 才用 3 个独立 training seeds；10×10 evaluation episodes 始终是嵌套描述量。若 benchmark 共用一个 Koopman/reward checkpoint，必须声明 CI 不包含动力学建模不确定性。

完整首训评审、proposed gates 与待审批命令见 [`docs/dmc_cartpole_pretraining_review.md`](../../docs/dmc_cartpole_pretraining_review.md)，研究路线见 [`docs/dmc_migration_plan.md`](../../docs/dmc_migration_plan.md)。
