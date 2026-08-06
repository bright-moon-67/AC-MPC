# ManiSkill PickCube 视觉 Controlled-Koopman

本实验只使用 ManiSkill 官方 `PickCube-v1` 任务、官方 PPO 离线动作和原始动力学/奖励；不使用 PandaReach、waypoint 或自定义状态。唯一的薄环境派生是让官方绿色 `goal_site` 对 agent camera 可见，因为上游环境默认把它加入 `_hidden_objects`。数值 `goal_pos`、物体位姿和抓取标志等特权量都不会进入模型，目标信息只通过 RGB 暗含在视觉表征中。

## 状态、动作和线性演化

```text
RGB(128×128)
  -> frozen ImageNet ResNet18
  -> b_t (512维 frozen feature)
  -> train-only normalization
  -> MLP encoder E
  -> v_t (16或32维)

r_t = [qpos(9), qvel(9), tcp_xyz(3)]  # 21维
s_t = [r_t, v_t]                     # 37或53维，不再额外升维
z_t = T s_t
z_hat_{t+1} = A z_t + B u_t
s_hat_t = C z_hat_t
b_hat_t = D(v_t)
```

`u_t` 是 8 维 `pd_joint_delta_pos` 外生控制：前 7 维是 Panda 七个手臂关节的 delta position，第 8 维是双指夹爪的单一开合命令。模型使用 action，但不预测 action 自身演化。环境控制频率为 20 Hz，训练与 MPC horizon 都取 `K=20`，即预测未来 1 秒。

`D` 只重构归一化后的冻结 ResNet18 feature，不重构像素。等维 `T` 有三种相关实现：

- `identity`：`T=C=I`。
- `learned_inverse`：学习自由方阵 `T`，通过精确线性求解得到 `C=T^{-1}`。
- `learned_orthogonal`：学习生成元 `S`，使用 `T=exp(S-S^T)` 与精确逆 `C=T^T`。

自由可逆 `T` 在表达能力上可被 `A/B` 吸收，并不会增加模型阶数；同时它与线性一致性 loss 存在坐标缩放自由度。预实验中自由 `T` 的 condition number 达到 `5.44`（latent16）和 `10.86`（latent32），出现了通过缩小坐标降低 loss 的迹象。因此正式对照采用硬约束的 `learned_orthogonal`：它仍检验“可学习等维坐标变换”是否改善优化，但恒有 `cond(T)≈1`，不会发生尺度退化。

## 数据因果性

采集器对每个 episode 只恢复第一个官方 `env_state`，之后连续执行 `env.step(action)` 并记录 `T+1` 帧，保证每个样本满足：

```text
(r_t, image_t), u_t -> (r_{t+1}, image_{t+1})
```

不能用逐帧强制恢复 `env_states` 的 replay 方式做系统辨识，因为相邻状态未必由保存的 action 因果产生。官方 PPO 文件包含未经约束的高斯策略输出，而 ManiSkill 控制器实际执行 `clip(raw_action, -1, 1)`，所以数据同时保存：

```text
actions      # 实际施加的裁剪后 action，用于辨识 B
raw_actions  # 原始策略输出，仅用于审计
```

当前数据包含 200 episodes、9,935 transitions 和 10,135 frames；193 个 replay episode 成功。动作标量中约 19.5% 被裁剪，主要来自夹爪命令。按 episode 固定划分为 160/20/20 个 train/validation/test episodes；`K=20` 窗口数为 4,899/620/620。评估 H1/H5/H10/H20 时分别使用 test split 的全部 1,000/920/820/620 个合法起点，不跨 episode。normalizer 只在 train episodes 上拟合。

训练器还会强制核验 causal replay、goal visibility、applied-action 标志、action bounds，以及 feature sidecar 的 source SHA256，避免轨迹和特征缓存错配。

## 训练目标和日志

原 AC-MPC Koopman 训练约束已迁移到视觉模型：

- 20 步 latent linear consistency 与 robot rollout loss；
- 当前帧和未来帧的 frozen-feature reconstruction loss；
- target latent 方差约束，防止视觉 encoder 塌缩；
- `rho(A)` 软约束，阈值设为 `1.02`；静态目标视觉模态允许接近单位根；
- 小权重 `A≈I` 正则、gradient clipping、weight decay；
- `T/C` 重构误差、奇异值和 condition number 诊断。

模型选择只使用 validation observable metric：

```text
J_obs = 2.0 * robot_rollout
      + 0.2 * current_feature_reconstruction
      + 0.2 * future_feature_reconstruction
```

这样 `T` 坐标系内部的 latent loss 不会直接决定 best checkpoint。学习率可通过 `--learning-rate` 调整，并使用 `ReduceLROnPlateau(factor, patience, min_lr)`；过拟合主要由 validation early stopping、weight decay 和数据规模控制，而不是单独降低学习率。本轮四组 best epoch 都在 195/198，validation 仍在改善，scheduler 未触发，最终 LR 保持 `3e-4`，没有观察到典型过拟合。

每个 epoch 的 train/validation 全部 loss、`rho(A)`、latent std、`T` 奇异值/条件数、LR、next LR 和 gradient norm 都写入 `metrics.jsonl`，同时默认写入本地 W&B offline run。

## 200-episode、K=20 对照结果

以下是 seed 43 的 held-out test 结果；TCP 指标是物理单位 mm，feature RMSE 是冻结 ResNet18 feature 空间中的误差。

| visual latent / T | best epoch | val `J_obs` | TCP H1 | TCP H5 | TCP H10 | TCP H20 | feature H20 | `rho(A)` | min latent std | `cond(T)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 / I | 195 | 0.5432 | 1.39 | 5.47 | 7.45 | 9.58 | 0.3339 | 1.0071 | 0.849 | 1.000 |
| 16 / orthogonal T | 198 | 0.5042 | **1.17** | 4.67 | 6.70 | 9.07 | 0.3305 | 1.0056 | 0.945 | 1.000 |
| 32 / I | 198 | 0.5162 | 1.51 | 5.59 | 7.29 | 9.69 | **0.3262** | 1.0033 | 0.831 | 1.000 |
| **32 / orthogonal T** | **198** | **0.4922** | 1.18 | **4.56** | **6.36** | **8.85** | 0.3266 | 1.0043 | 0.854 | 1.000 |

正式推荐 `latent32_learned_orthogonal`。它的 validation observable metric 最低，H5/H10/H20 TCP 均最好；相对 latent32/I，H20 TCP RMSE 从 `9.69 mm` 降至 `8.85 mm`。四组的 `B` 都具有满列秩 8。推荐模型在 H20 将整段 action 换成其他 episode 的确定性乱序序列或全零序列时，normalized robot RMSE 分别变为正确 action 的 `3.21x` 和 `3.30x`，表明模型确实使用了 `B u_t`。

这是单 seed 模型选择结果，足以进入 AC-MPC 接口验证，但正式统计结论仍应补 3 个以上 seed。

## 复现当前对照

现有缓存可直接运行四组对照：

```bash
.venv/maniskill/bin/python -m experiments.maniskill_pick_visual.run_ablation \
  --trajectory-h5 .data/maniskill/visual_pickcube_causal_v2_200_seed43.h5 \
  --feature-h5 .data/maniskill/visual_pickcube_causal_v2_200_seed43.resnet18.h5 \
  --output-dir runs/visual_pickcube_k20_ablation_orthogonal_200ep_seed43 \
  --horizon 20 --epochs 200 --patience 40 --batch-size 512 \
  --learning-rate 3e-4 --lr-factor 0.5 --lr-patience 10 --min-lr 1e-6 \
  --preload --wandb-offline --wandb-project acmpc-visual-pickcube \
  --wandb-group k20-200ep-orthogonal-main-seed43 --device cuda

.venv/maniskill/bin/python -m experiments.maniskill_pick_visual.evaluate_visual_koopman \
  --checkpoint runs/visual_pickcube_k20_ablation_orthogonal_200ep_seed43/latent32_learned_orthogonal/best.pt \
  --split test --device cpu
```

## 接入现有 AC-MPC

训练结束后先冻结 ResNet、视觉 encoder/decoder 和 `A/B/T`。在线路径为：

```python
b = frozen_resnet(rgb)
b_norm = (b - feature_mean) / feature_std
r_norm = (r - robot_mean) / robot_std
s = koopman.make_state(r_norm, b_norm)
z = koopman.lift(s)

actor = KoopmanMPCActor(
    koopman.A,
    koopman.B,
    koopman.readout_matrix(),  # I、T^-1或T^T，由模式决定
    horizon=20,
    action_low=-1.0,
    action_high=1.0,
)
u = actor(z).action
```

推荐 checkpoint 已完成真实重载和 finite-horizon actor smoke test：`A[53,53]`、`B[53,8]`、`C[53,53]`，20 步 action sequence 形状为 `[batch,20,8]`，数值有限且满足 `[-1,1]`；`C@T` 与单位阵最大误差约 `1.2e-6`。

这个 smoke test 只证明动力学和 QP 接口兼容。当前 AC-MPC cost network 尚未训练，零初始化会输出零 action，不能据此宣称闭环 PickCube 成功。下一步应先用官方离线 action 对 actor/cost network 做 BC warm start，再固定 Koopman 做 PPO 或 TD3+BC fine-tune；稳定后才联合微调 encoder/Koopman。每次更新 Koopman 参数后需重建 actor，因为现有 actor 会 clone/detach `A/B/C`。

若后续发现遮挡或接触阶段残差显著高于其他片段，再比较短 history encoder；输出仍保持固定 16/32 维 `v_t` 和等维线性演化，不改变 AC-MPC 接口。
