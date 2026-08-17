# MuJoCo Playground CartpoleSwingup 阶段结果

更新时间：2026-08-13。

本页冻结 CartpoleSwingup 当前阶段的可复查结论。模型 checkpoint、训练日志和逐 episode 评估 JSON 保留在本机 `runs/playground/`，不纳入 Git；Git 仅保存实现、配置、测试和本摘要。

下一任务从数据、Koopman、Cal-QL/RLPD baseline 到 KMPC/MPVE 的端到端迁移规则，见
[`offline_to_online_benchmark_playbook.md`](offline_to_online_benchmark_playbook.md)。

## 四方法主实验

以下为 seed `20260812` 的 latest checkpoint、128 episodes 确定性评估。数值是单训练 seed 的描述统计，不应解释为跨 seed 置信区间。

| 方法 | return mean | population std |
|---|---:|---:|
| PPO | 870.7491 | 0.9063 |
| KMPC | 850.7813 | 5.9709 |
| AB-PQ | 465.7972 | 28.0679 |
| AC-MPC-MPVE | 263.7082 | 21.2075 |

## Lifted critic 对照

`KMPC-LiftedCritic` 使用 seed `20260814`：actor/controller 继续使用 Koopman lifted state，critic 也改为输入 frozen normalized lifted state。其最佳诊断 checkpoint 位于训练 step `49,152,000`，128 episodes return 为 `803.8133 ± 0.4443`（population std）；latest checkpoint 为 `588.5563 ± 7.8524`，说明后期发生明显退化，应以 best checkpoint 作为该对照的代表结果。

本地保留目录：

```text
runs/playground/ablation/CartpoleSwingup/seed_20260814/KMPC-LiftedCritic/
```

Fixed-QP prior、重复 ZeroInit、LiftNorm/ValueNorm 学习率扫描和 smoke 临时产物已从工作区清理；这些失败路线不再保留为正式可选训练参数。当前通用 structured trainer 只保留可解释的 `critic_input = raw_observation | lifted_state` 对照。
