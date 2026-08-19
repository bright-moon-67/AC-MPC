# ManiSoft Koopman 参考跟踪验证 —— 最优运行命令

公共参数：
- scenario: `/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml`
- reference: `/root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz`
- delta 模型: `runs/manisoft_45d_824ep_seed42/koopman/best_validation.pt`
- abs 模型: `runs/koopman_45d_abs_seed42/best_validation.pt`
- history 模型: `work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt`
- 最新 targeted history 模型: `work_dirs/manisoft_koopman_history_h10_abs_targeted_v4_tip_seed42/koopman_history/best_validation.pt`

## ⭐ 最新 targeted-v4-tip history Koopman-MPC（≈0.064mm @1000步）

使用 `history_h10_targeted_v4_tip` checkpoint。围绕原 history MPC 参数复调后，`tip-state-scale=50` 最优；1000 步复验的末 500 步平均误差约 0.064mm、std 0.024mm、最大误差约 0.102mm，最终误差约 0.090mm。

```bash
python scripts/validate_koopman_mpc_reference_history.py \
  --checkpoint work_dirs/manisoft_koopman_history_h10_abs_targeted_v4_tip_seed42/koopman_history/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_mpc_history_targeted_v4_tip_best \
  --steps 1000 --horizon 10 --max-delta 0.001 --tip-state-scale 50 \
  --state-weight 200 --action-weight 30 --control-weight 1 \
  --smoothness-weight 10 --success-threshold 0.002 --device cuda
```

## 最新 targeted-v4-tip history Koopman-LQR（综合稳健配置）

新 checkpoint 的有效反馈增益尺度与旧 history 模型不同。综合准确参考动作和 1% 错误前馈动作测试，选用 `feedback-scale=30`。准确参考动作的 1000 步复验中，末 500 步平均误差约 0.157mm、最大误差约 0.233mm，最终误差约 0.083mm；参考动作缩小 1% 时，末 500 步平均误差约 1.217mm、最大误差约 1.286mm，仍满足 2mm 判据。

```bash
python scripts/validate_koopman_lqr_reference_history.py \
  --checkpoint work_dirs/manisoft_koopman_history_h10_abs_targeted_v4_tip_seed42/koopman_history/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_lqr_history_targeted_v4_tip_best \
  --steps 1000 --reference-action-scale 1.0 --feedback-scale 30 \
  --state-weight 0.001 --tip-state-scale 20 --action-weight 0.3 \
  --control-weight 100000 --max-delta 0.002 \
  --required-success-streak 500 --stability-window 500 --device cuda
```

## ⭐ 推荐：history MPC — `validate_koopman_mpc_reference_history.py`（最优 ≈0.91mm @1000步）

当前所有方案中精度最高（末 100 步均值 0.91mm、std 0.073mm、81.8% 时间 <1.5mm），稳定无发散。

```bash
python scripts/validate_koopman_mpc_reference_history.py \
  --checkpoint work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_mpc_reference_45d_history_best \
  --steps 1000 --horizon 10 --max-delta 0.001 --tip-state-scale 20 \
  --state-weight 200 --action-weight 30 --control-weight 1 \
  --smoothness-weight 10 --device cuda
```

## 1. delta MPC — `validate_koopman_mpc_reference.py`（最优 ≈7.98mm @5000步）

```bash
python scripts/validate_koopman_mpc_reference.py \
  --checkpoint runs/manisoft_45d_824ep_seed42/koopman/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_mpc_reference_45d_best \
  --steps 5000 --horizon 10 --state-weight 200 --action-weight 100 \
  --control-weight 1 --smoothness-weight 10 --max-delta 0.002 --device cuda
```

## 2. abs MPC — `validate_koopman_mpc_reference_abs.py`（最优 ≈1.28mm @500步）

```bash
python scripts/validate_koopman_mpc_reference_abs.py \
  --checkpoint runs/koopman_45d_abs_seed42/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_mpc_reference_45d_abs_best \
  --steps 500 --horizon 10 --max-delta 0.002 --tip-state-scale 15 \
  --state-weight 200 --action-weight 30 --control-weight 1 \
  --smoothness-weight 10 --device cuda
```

## 3. delta LQR — `validate_koopman_lqr_reference.py`（最优 ≈0.49mm @500步）

```bash
python scripts/validate_koopman_lqr_reference.py \
  --checkpoint runs/manisoft_45d_824ep_seed42/koopman/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_lqr_reference_45d_best \
  --steps 500 --state-weight 0.001 --tip-state-scale 20 --action-weight 0.3 \
  --control-weight 100000 --max-delta 0.002 --device cuda
```

## 4. abs LQR — `validate_koopman_lqr_reference_abs.py`（最终 ≈0.134mm @1000步）

该配置不修改 Koopman checkpoint；将 LQR 反馈缩放为原增益的 3%，以参考动作作为主要控制、LQR 作为小幅修正。末 100 步平均误差约 0.229mm、最大误差约 0.424mm，并连续 848 步保持在 2mm 以内。

```bash
python scripts/validate_koopman_lqr_reference_abs.py \
  --checkpoint runs/koopman_45d_abs_seed42/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_lqr_abs_best_default_verified \
  --steps 1000 --state-weight 0.001 --tip-state-scale 20 \
  --action-weight 0.3 --control-weight 100000 --max-delta 0.002 \
  --feedback-scale 0.03 --success-threshold 0.002 \
  --required-success-streak 100 --stability-window 100 --device cuda
```

## 5. history abs LQR — `validate_koopman_lqr_reference_history.py`

使用 H=10 history-context Koopman checkpoint。综合准确参考和错误前馈动作测试，最优反馈缩放为 `0.0045`。

准确参考动作的 2000 步验证：末 500 步平均误差约 0.057mm、最大误差约 0.099mm。

```bash
python scripts/validate_koopman_lqr_reference_history.py \
  --checkpoint work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_lqr_history_best_nominal_fs0p0045_s2000 \
  --steps 2000 --reference-action-scale 1.0 --feedback-scale 0.0045 \
  --required-success-streak 500 --stability-window 500 --device cuda
```

将前馈参考动作统一缩小 1% 后，末 500 步平均误差约 1.536mm、最大误差约 1.562mm，仍满足 2mm 判据：

```bash
python scripts/validate_koopman_lqr_reference_history.py \
  --checkpoint work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_lqr_history_robust_ffbias099_default_s2000 \
  --steps 2000 --reference-action-scale 0.99 --feedback-scale 0.0045 \
  --required-success-streak 500 --stability-window 500 --device cuda
```

更强的 5% 错误参考动作测试会稳定在约 7.736mm，不发散但不能达到 2mm：

```bash
python scripts/validate_koopman_lqr_reference_history.py \
  --checkpoint work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt \
  --scenario /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/ManiSoft/work_dirs/random_reference_45d/reference.npz \
  --output runs/koopman_lqr_history_best_ffbias095_fs0p0045_s2000 \
  --steps 2000 --reference-action-scale 0.95 --feedback-scale 0.0045 \
  --required-success-streak 100 --stability-window 500 --device cuda
```
