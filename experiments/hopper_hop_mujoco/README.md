<!--
  Hopper MuJoCo compliant-contact branch — consolidated notes.
  Status: Phase 1-4 running (Koopman training in progress, see §6).
  Last updated: 2026-08-08.
  Detailed audit: docs/hopper-contact-stiffness-audit-20260808.md
  Migration plan/phase report: docs/hopper_mujoco_migration_plan.md
-->
# Hopper MuJoCo 柔顺接触平行分支

在不触碰原 PhysX / AC-MPC 主线的前提下，为 `MS-HopperHop-v1` 建立的 **MuJoCo 原生柔顺接触**严格对照实验分支（"NC-inspired compliant-contact Koopman"）。本文档汇总全部改进、关键结论与运行方法。

---

## 0. 状态速览（2026-08-08）

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | 接触审计 | ✅ 完成（`docs/hopper-contact-stiffness-audit-20260808.md`） |
| Phase 2 | MuJoCo backend + state/action 对齐 | ✅ 完成 |
| Phase 3 | compliant-contact 数值标定 + PD 稳定性修复 | ✅ 完成（41 测试通过） |
| Phase 4 | 数据采集 → Koopman 训练 → 接触维对比 | 🔄 训练中（epoch ~300/500，三配置 val_nMSE 0.334-0.340，全部低于 PhysX 基线 0.4512） |

---

## 1. 目标与铁律

**目标**：验证核心假设——**柔顺接触 ⇒ 接触力连续 ⇒ 线性 Koopman 对接触维更可预测**（受 Nature Commun. 2026 接触论文启发，见 §7）。

**铁律（全部落实）**：
1. 不破坏现有框架：`experiments/hopper_hop/`、`antmaze_ac/`、原 checkpoints、`MS-HopperHop-v1` 零改动。
2. Phase 顺序执行（1→2→3→汇报→4）。
3. 不凭 solref/solimp 名称判断软硬，全部数值标定。
4. 不凭记忆硬编码公式，全部实证验证。
5. 不用自定义 penalty-force 接触，只用 MuJoCo 原生柔顺。
6. 不破坏线性 Koopman/MPC 核心，只在数据/观测层面做 NC-inspired 改造。

---

## 2. 文件结构

```
experiments/hopper_hop_mujoco/
├── __init__.py                    # 导出 MuJoCoHopper / HopperAdapter / ContactConfig / 预设
├── assets/
│   ├── hopper.xml                 # ManiSkill 原 XML（参考）
│   └── hopper_mujoco.xml          # MuJoCo 可用版（删 fixed joint + sensor 块）
├── configs/                       # mujoco_default / compliant / hard 三预设 yaml
├── envs/
│   ├── contact_config.py          # ContactConfig + 三预设 + 解析器
│   ├── mujoco_hopper.py           # MuJoCoHopper（reset/step/state/reward/诊断）
│   └── hopper_adapter.py          # 统一适配器
├── collect/
│   ├── collect_hopperhop_mujoco.py   # 数据采集（PPO-collector 同 schema）
│   └── build_mujoco_datasets.py      # 复用原 build_hopperhop_dataset 装配数据集
├── eval/
│   ├── contact_calibration.py     # 胶囊落体数值标定
│   └── compare_contact_dims.py    # 跨配置接触维对比
├── tests/                         # 47 个测试（Phase 1-4 管线）
└── train_mujoco_koopman.sh        # 装配→训练→对比 一键编排
```

---

## 3. Phase 1-3 关键结论

### 3.1 接触审计（PhysX 侧）
- `MS-HopperHop-v1` 接触为 **near-rigid**（TGS 冲量，k≈1.2e7 N/m、零穿透、单步刹停）。
- SAPIEN 只暴露 `sim_freq / solver_position_iterations / contact_offset / rest_offset`，**无法表达柔顺**。
- Koopman 接触维误差大（one-step RMSE≈1.25、hold≈1.80）。

### 3.2 MuJoCo 后端与对齐
- 关节序、canonical action（`[hip,knee,waist,ankle]`、scale `[2,2,2,0.8]`、kp=100/kd=10、25Hz）、mech13/legacy15 状态、触地力读取（efc_force+frame）——**全部实证对齐 PhysX**。
- 接触力读取经胶囊标定校验（静止 Fn=12N=重量 ✓）。

### 3.3 接触参数与标定（胶囊 0.5m 落体，impact≈2.9-3.0 m/s）

| 配置 | solref | 峰值力 | 最大穿透 | 冲击时长 | 结论 |
|---|---|---|---|---|---|
| PhysX-hard（基线） | — | 223 N | ~0 | 单步 | 近刚性 |
| mujoco_default | (0.02,1) | 399 N | 14 mm | 30-35 ms | MuJoCo 原生软度 |
| **mujoco_compliant** | **(0.08,1)** | **107 N** | **60-64 mm** | **135-140 ms** | **真柔顺（峰值力减半、冲击摊开）** |
| mujoco_hard | (0.005,1) | 790-1478 N | 0.03-3.8 mm | 5-45 ms | 近刚性（需 dt≤0.0025） |

### 3.4 PD 数值稳定性修复（重要工程结论）
- **症状**：MuJoCo 精确求解器 + 显式 PD 会发散（dt=0.005 零 action 自由落体即爆）；PhysX 因 TGS 有损求解器天然稳定。
- **修复**：`implicit_kd=True`——把 kd=10 写入 `dof_damping`（implicitfast 半隐式积分），手动 tau 只留 kp 弹簧。**净力矩定律不变**，且更贴近 PhysX 的隐式 drive。
- 结果：三配置在 dt=0.005/0.0025 下 random(-1..1)×200 步×3 seeds **全稳定**（与 PhysX 同压力协议）。

---

## 4. Phase 4：数据采集 + Koopman 训练

### 4.1 数据采集（完成）
- 3 配置 × 3 seeds，每 seed **40.2 万过渡**（20 万 random + 20 万 **PhysX-PPO 策略**，策略在 MuJoCo 上 6/6 撑满 600 步可迁移）。
- 每配置数据集 120.6 万过渡（1608/201/201 episodes），**与原 PPO-collector / build 管线完全同 schema**。
- **踩坑修复**：9 进程并行时 PPO 阶段因 torch OpenMP 线程过订阅几乎停滞 → `torch.set_num_threads(1)` + 断点续采标记（`{random,ppo}_done.txt`）。

### 4.2 Koopman 训练（进行中）
- 完全复用原 `train_hopper_hop_koopman.py`（epochs500/lift48/kstep20/seed43/patience40/maxwindows1M），3 配置 GPU 并行（tmux `mujoco_koopman` 保护）。
- 训练结束自动跑 `compare_contact_dims.py`（tmux watcher `mujoco_koopman_done`）。

### 4.3 初步结果（训练中的强信号，epoch ~300）

| 配置 | val_nMSE | 相对 PhysX |
|---|---|---|
| PhysX-hard（koopman_v2 最优） | **0.4512** | 基线 |
| mujoco_default | ~0.340 | **低 25%** |
| mujoco_compliant | ~0.339 | **低 25%** |
| mujoco_hard | ~0.334 | **低 26%** |

> 初步发现：三种 MuJoCo 配置（含近刚性）都显著优于 PhysX——差异主要来自**求解器连续性**（MuJoCo Newton 精确 vs PhysX TGS 有损），不只是接触软硬。最终**接触维 RMSE 对比**（训练完成后的核心证据）由 watcher 自动产出至 `contact_dim_comparison.txt`。

---

## 5. 运行方法

```bash
# 标定（重跑覆盖 JSON/CSV）
python -m experiments.hopper_hop_mujoco.eval.contact_calibration

# 数据采集（9 任务并行：3 配置 × 3 seeds，每 seed 40 万过渡）
python -m experiments.hopper_hop_mujoco.collect.collect_hopperhop_mujoco \
  --contact mujoco_compliant --seed-index 0 --transitions-per-stage 200000

# 装配数据集 + 训练 + 对比（一键）
bash experiments/hopper_hop_mujoco/train_mujoco_koopman.sh

# 测试
python -m pytest experiments/hopper_hop_mujoco/tests/ -q
```

---

## 6. 当前运行中任务

| tmux 会话 | 内容 | 状态 |
|---|---|---|
| `mujoco_koopman` | 3 个 Koopman 训练（GPU 并行） | 运行中 |
| `mujoco_koopman_done` | 训练完成后自动接触维对比 | 等待 |
| `ppo_fair` | 用户 PPO 任务（原有） | 运行中 |

GPU 共用（<13GB/40GB），无 OOM 风险；长时训练均在 tmux 中，SSH 断连不中断。

---

## 7. NC 原文 CCK 分析（待实现）

**论文**：O'Neill, Terrones & Asada, *Nat. Commun.* **17**, 7749 (2026), DOI `10.1038/s41467-026-72485-7`。
**代码**：`jterrone1/Koopman-for-Rimless-Wheel`（`make_cck_model.py`、`gurobi_functions.py`）。

**CCK（Control-Coherent Koopman）结构**（从代码逐条还原）：
1. 物理状态含**接触坐标**（辐条穿透 p）——p↔q 双向耦合吸收进单一线性 A。
2. Lifting = 物理状态 + k-means RBF 观测（`lift_dim = 11 + 100`），A 用正则化 EDMD 拟合。
3. **B 解析构造**（`b_val = dt/(Il·R²+Im)`，由致动器物理给出）——"without approximation to control inputs"。
4. **Embedding compensation**：`B[rbf块] = A_gp @ inv(A_pp) @ B[物理块]`，保证输入跨嵌入相干。
5. 控制器：Gurobi 凸 QP。

**对 hopper 的落实路径（不碰 antmaze_ac 核心）**：
- **Level 1（数据级）**：接触维从 `log1p(‖F‖)` 换成 MuJoCo 穿透坐标 p（`margin−contact.dist`），状态 15D→17D，同结构重训。
- **Level 2（输入级，更接近严格 CCK）**：输入改关节力矩 τ（MuJoCo 直接控 actuator），B 解析构造 + embedding compensation。
- compliant 软接触正是 CCK 成立的前提——与我们的核心假设互相印证。

---

## 8. 测试与验证

- `experiments/hopper_hop_mujoco/tests/`：**47 个测试**（reset/state/action 映射/稳定性压力/标定/Phase4 管线/PhysX 未动），全部通过。
- 原有核心测试（`tests/test_affine_lqr.py` 等）：20 passed，无回归。
- `git status`：仅新增本分支 + `docs/` 两个 md；`antmaze_ac/koopman/checkpoint.py` 的未提交修改为**预先存在**（非本次工作）。

---

## 9. 关键注释要点（代码内）

- `mujoco_hopper.py`：`implicit_kd` 为何必须写在 `_zero_joint_passives()` 之后；`control_actuator_ids` 按 actuator 索引而非关节索引；efc_force 读取依赖 elliptic 锥（每接触 3 行）。
- `contact_config.py`：solref 正/负语义、refsafe 对 dt 的钳制（hard 配置需 dt≤0.0025）。
- `collect_hopperhop_mujoco.py`：numpy 2.x `savez_compressed` 会追加 `.npz` 后缀；`torch.set_num_threads(1)` 避免多进程线程过订阅；`{policy}_done.txt` 断点续采。
- `contact_calibration.py`：impulse 只积分冲击窗口 `[i0,i_end]`（含窗口内体重支撑 mg·Δt），否则静止支撑力会虚高。
