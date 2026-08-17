# DMC Reacher Hard development 训练前评审草案

> 状态：环境与 reward 只读审计已完成；尚未生成 Reacher preflight/approval，尚未执行任何 Reacher optimizer step。
> 当前约束：Cartpole Koopman 正在正式训练。其结束并冻结 artifact 之前，不修改 `experiments/dmc/**/*.py`，避免使正在使用的 source identity 与审批记录失配。

## 1. 任务和原生协议

| 项 | 冻结候选值 |
|---|---:|
| DMC task | `reacher:hard` |
| observation | 官方 state，`position(2) + to_target(2) + velocity(2)`，共 6D |
| action | 2D，逐维 `[-1, 1]` |
| control timestep | 原生 `0.02 s`，50 Hz |
| time limit | 20 s |
| episode length | 1000 control steps |
| action repeat | 1 |
| score | 1000 步官方 reward 之和，最大 1000 |

不使用旧计划中的 20 Hz/400-step 口径。Reacher physics timestep 本身是 `0.02 s`，`0.05 s` 不是其整数倍，不能在 `dmc_native_v1` 中使用。

## 2. 官方 reward observation oracle

`dm_control.suite.reacher.Reacher.get_reward` 使用指尖到目标的二维距离：

```text
r = 1[ ||to_target|| <= target_radius + finger_radius ]
  = 1[ ||to_target|| <= 0.015 + 0.010 ]
  = 1[ ||to_target|| <= 0.025 ]
```

官方 6D observation 已直接包含 `to_target`，因此预测 observation 是 reward 的充分统计量，不需要把 learned reward model 作为主 MPVE reward source。

2026-08-11 的只读 live audit 使用 `dm_control==1.0.44`、10 个 episode、10,000 transitions：

| 检查 | 结果 |
|---|---:|
| observation oracle vs `TimeStep.reward` 最大绝对误差 | `0.0` |
| 随机动作正奖励步数 | `129 / 10,000` |
| 随机动作正奖励比例 | `1.29%` |

正式实现后，preflight 仍须重新执行逐 transition parity，且 checkpoint 必须明确记录 Reacher exact-oracle metadata。`TransitionRewardModel` 继续与 Koopman 联训，但只作为 learned-reward 消融、offline RL 和未来真机 fallback。

## 3. PPO 参考口径

主实验继续采用 Google DeepMind Acme continuous PPO 示例的统一参考实现口径，而不声称它是 Reacher Hard 的官方 best/SOTA：

- policy/value 各 `3 x 256 ReLU`；
- state-dependent tanh-squashed diagonal Gaussian；
- observation normalization、Acme EMA advantage/value normalization；
- `256 envs x rollout 8 = 2048 transitions/update`；8 个 256-transition minibatch，2 epochs；
- constant LR `3e-4`、Adam `eps=1e-7`、gamma `0.99`、GAE `0.95`、clip `0.2`、entropy `3e-4`、value cost `1.0`、global grad norm `0.5`；
- 每 actor/seed `999,424 = 488 x 2,048` environment steps。

历史 CleanRL PPO 表报告 `reacher-hard 443.80 +/- 9.64`，只作为 compatibility anchor。它使用旧 dm-control/Shimmy/Gymnasium 栈、2x64 Tanh、state-independent unsquashed Normal、reward normalization/clipping 和 online stochastic last-100 training return；不能与本项目固定 checkpoint 的 deterministic post-evaluation 直接等同。

Cartpole 的 Acme-aligned PPO 首轮未通过其 return gate，因此 Reacher 正式 PPO 前还需完成同一 PPO 实现的诊断。不能仅因网络和超参来自官方 reference 就假设任务一定收敛。

## 4. Reacher development 候选参数

| 项 | 候选值 | 物理视野 |
|---|---:|---:|
| training seed | `20260821` | - |
| PPO budget | 999,424 steps | 约 19,988.5 s 模拟时间 |
| Koopman 数据 hard cap | 300,000 complete-episode transitions | early/mid/late 各 100,000 |
| Koopman K-step | 50 | 1.0 s |
| spectral radius limit / stability weight | `0.95 / 1.0` | - |
| KMPC horizon | 20 | 0.4 s |
| MPVE horizon | 10 | 0.2 s |
| MPVE reward source | Reacher official observation oracle | sparse exact reward |
| learned reward | 256x256，joint weight 1.0 | 仅消融/fallback |

`K=50` 在 Reacher 上是 1.0 s，而 Cartpole 上是 0.5 s。这是按用户指定的统一步数，不是假装两项任务具有相同物理预测时长；报告必须同时给出 steps 和 seconds。

当前 YAML 中仍是旧占位值（K=20、stability weight=0.1、MPVE H=5、learned reward、500k data cap）。必须在 Cartpole 当前训练结束后统一修改、运行全量测试并生成 fresh Reacher preflight；旧占位配置不得启动正式训练。

## 5. 训练前 Gate 与部分执行顺序

1. Cartpole Koopman 当前进程结束，保存并核对 final artifact；
2. 实现 Reacher exact observation reward oracle，并加入 live parity/unit tests；
3. 将 Reacher YAML 冻结为本评审中的候选参数；
4. 运行全 DMC tests；
5. 运行 Reacher 真实 preflight：suite parity、timeout、256x8 vector probe、exact reward parity、MPVE H=10 detach/critic-only gradient；
6. 运行 PPO formal dry-run，要求 optimizer/environment steps 均为 0；
7. 向用户提交 config/source/protocol fingerprints、资源和时间估计；
8. 用户批准后才创建 Reacher approval artifact；
9. 正式 PPO 数据源训练并采集 early/mid/late 完整 episode；
10. 构建 episode-safe dataset，再训练 Koopman + learned reward ablation。

在第 8 步以前不得执行 Reacher optimizer step。PPO 数据源 checkpoint 的表现、正奖励覆盖率和轨迹多样性必须单独报告；即使 PPO peer 未达最终控制 gate，也不能隐瞒或用 best checkpoint 替代预注册 latest。

## 6. 初步时间估计

在当前机器上，Cartpole PPO 的 999,424 steps 用 GPU 约 14.5 分钟。Reacher 物理和 2D policy 略重，首次 development PPO 暂估 20--35 分钟；300k 数据采集包含在该预算内，不额外 rollout。K=50 Koopman 暂估 1--2 小时，最终以 Reacher preflight 和前 5 个 epoch 的实测吞吐刷新。
