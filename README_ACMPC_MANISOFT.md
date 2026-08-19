# AC-MPC + ManiSoft 项目说明

本文档用于向接手同事交接 AC-MPC + ManiSoft 工作，覆盖环境搭建、Koopman
数据收集与训练、单点跟踪、三 waypoint PPO-KMPC、离线数据收集和 KMPC-IQL。
文档同时区分当前推荐主线、仍可复现的旧实验和正在验证的研究分支。除特别
标注外，命令均从 `AC-MPC/` 根目录执行；当前服务器上已配好的环境和数据位置
见 2.1。

## 1. 项目总览与推荐入口

### 1.1 项目要解决的问题

ManiSoft 上游仓库是面向软体连续体机械臂的仿真与视觉语言操作 benchmark；
AC-MPC 上游仓库最初研究 AntMaze 上的增量动作 Deep Koopman 和 Actor-Critic
LQR。本分支把两者结合，当前主任务不是完整的 ManiSoft COLL/ALN/ARR/STK
视觉语言 benchmark，而是：

1. 在纯软体臂 ManiSoft 场景中采集可控的自由运动数据；
2. 学习 45 维软体状态的 history Koopman 动力学；
3. 用 Koopman-LQR/KMPC 完成参考状态和连续三个 waypoint 的闭环跟踪；
4. 比较结构化 PPO-KMPC、MLP、BC/DAgger 和离线 IQL 等学习方法。

两个仓库的职责边界为：

```text
ManiSoft/acmpc-integration
  仿真后端、软体动力学、肌肉力矩、45D 状态抽取、原始数据采集
                  │
                  ▼
AC-MPC/manisoft-port
  数据转换、Koopman 训练、LQR/KMPC、waypoint 环境、BC/PPO/IQL、评估
```

ManiSoft 中仍保留官方的 VLM benchmark、数据生成和 RL executor；这些代码可
独立使用，但不是本文后续训练命令的依赖主线。

### 1.2 当前推荐主线

如果只想运行目前最成熟的三 waypoint 控制策略，应从 **v15e PPO-KMPC**
开始，而不是从第 5 章的早期小规模 BC-KMPC 开始。当前各层推荐选择如下：

| 层级 | 当前推荐项 | 状态与用途 |
| --- | --- | --- |
| 仿真 | `ManiSoft/configs/demo_elastica_fast.yaml` | 纯 Elastica 软体臂，50 Hz 控制主场景 |
| 状态 | 45D 三截面物理状态 | 当前统一状态定义 |
| 动力学 | H=10 absolute-action、谱半径上限 0.95 的 history Koopman | v15 PPO 和 IQL 的冻结动力学 |
| 在线策略 | **v15e**：`e_kmpc_r8_lr3_std18_md0125/last.pt` | 当前默认行为策略与离线数据采集策略 |
| 在线对照 | v15a：`a_kmpc_r8_lr3_std15_md015/last.pt` | v14 最强分支延续；已有独立 20-episode 测试 |
| 非 KMPC 对照 | v15f History-MLP | 同 Koopman lift/context，但直接输出绝对动作 |
| 离线数据 | `combined_v4_1498_v5_7109/dataset.npz` | v15e 随机策略采集，8,607 episodes |
| 离线策略 | structured-v2 KMPC-IQL，K=40/60 | 当前研究分支，尚需闭环正式选型 |

这里把 v15e 称为“当前推荐主线”，依据是：它在 v15 的 a/b/e 三个强配置中
使用更紧的物理变化率限制 `max_delta=0.0125`，同时把 normalized-delta 标准差
设为 0.18，使物理探索标准差仍为 `0.18×0.0125=0.00225`；其 `last.pt`
随后被用于 v4、v5 两批共 8,607 回合的离线数据采集，合并数据中的随机闭环
成功率为 59.56%、平均完成 waypoint 数为 2.507。训练最后一个 rollout 为
12/16 成功，但这是在线训练批次指标，不应当当作独立测试集成功率。

v15a 的 `last.pt` 已在独立的 v4 test20 waypoint 子集上做过确定性评估，结果为
12/20 成功、平均完成 2.5 个 waypoint。因此在报告中应分别写清：

- v15e 是当前默认行为策略、离线数据来源和后续 IQL 的基线；
- v15a 有一份现存的独立 20 回合确定性结果；
- 最终发表级“最佳模型”仍应在相同独立 waypoint bank、相同 seed schedule、
  至少 100 episodes 下比较 v15a/v15e/IQL 后确定，不能只比较 PPO rollout。

### 1.3 核心状态、动作和任务约定

当前 ManiSoft 主线使用三个代表截面，每个截面包含位置 3D、旋转 6D、线速度
3D 和角速度 3D，因此物理状态为 `3×15=45D`。动作是 6 个肌肉控制点的
3 轴激活，共 18D，绝对范围为 `[-0.30,0.30]`。

history Koopman 使用：

```text
s_t ∈ R^45
u_t ∈ R^18
context_t = [normalized s[t-H+1:t+1], u[t-H:t]], H=10
z[t+1] = A z[t] + B u[t]
```

三 waypoint 策略观测为：

```text
current state                         45
history context       10 × (45 + 18) = 630
three waypoint xyz                 3 × 3 = 9
active-stage one-hot                    3
total                                  687
```

v15e 策略输出的不是绝对动作，而是每维位于 `[-1,1]` 的 normalized delta：

```text
u_t = clip(u[t-1] + max_delta * d_t, -0.30, 0.30)
max_delta = 0.0125
```

因此 checkpoint、离线数据和 IQL 中的 `actions` 都必须结合 checkpoint 的
`max_delta` 解释，不能直接当成 18D 肌肉绝对激活。

### 1.4 代码和实验成熟度

| 路线 | 成熟度 | 建议 |
| --- | --- | --- |
| 45D 数据采集与转换 | 稳定 | 新数据按第 3 章生成 |
| Koopman 单点 LQR/MPC | 稳定 | 用第 4 章命令做动力学与控制冒烟 |
| fixed-cost 专家、BC、DAgger | 可复现的历史主线 | 用于理解演进或生成监督数据 |
| v15 structured PPO-KMPC | **当前在线主线** | 新使用者优先从 v15e 评估开始 |
| v16 source ablation | 研究消融 | 不作为部署策略 |
| KMPC-IQL | 当前研究主线 | 训练代码完整，最终模型仍需闭环比较 |
| 41D、随机 waypoint、早期无约束 BC-KMPC | 旧实验 | 仅用于历史对照，不建议新实验使用 |

### 1.5 已具备 artifact 时的最短使用路径

先定义与机器无关的路径。后文出现 `/root/autodl-tmp` 时，均可用相同方式替换：

```bash
export WORKSPACE=/path/to/workspace
export ACMPC_ROOT="$WORKSPACE/AC-MPC"
export MANISOFT_ROOT="$WORKSPACE/ManiSoft"
export AC_MPC_PYTHON=/path/to/conda/env/bin/python
cd "$ACMPC_ROOT"
```

验证代码与依赖：

```bash
"$AC_MPC_PYTHON" -c "import manisoft, antmaze_ac; print('imports: OK')"
"$AC_MPC_PYTHON" -m pytest -q
```

直接评估当前主线 v15e：

```bash
"$AC_MPC_PYTHON" scripts/evaluate_manisoft_ppo_comparison.py \
  --checkpoint runs/manisoft_ppo_compare_v15_zmixed_24h/\
e_kmpc_r8_lr3_std18_md0125/last.pt \
  --scenario "$MANISOFT_ROOT/configs/demo_elastica_fast.yaml" \
  --waypoint-root data/processed/manisoft_waypoint_bank_v2_zmixed_merged \
  --output runs/handoff_smoke/v15e_eval_10ep \
  --episodes 10 --episode-steps 300 --device cuda --seed 42
```

评估器使用确定性 KMPC mean，并保存逐回合轨迹和 `summary.json`。若故意在与
checkpoint 记录不同的 waypoint bank 上测试泛化，必须额外传入
`--allow-other-waypoint-bank`；否则 SHA256 不一致会被拒绝。

## 2. 环境配置

### 2.1 基本要求

- Linux（已在 Ubuntu 上验证）；
- Conda/Miniconda 和 Git；
- Python 3.11（`AC-MPC/pyproject.toml` 要求 `>=3.10,<3.13`，已在 3.11 上验证）；
- 训练推荐使用支持 CUDA 的 NVIDIA GPU，数据生成和小规模测试可使用 CPU。

AC-MPC 依赖本项目修改后的 ManiSoft 仿真环境。两个仓库应放在同一父目录下，
不要用原生 ManiSoft 仓库替代 `acmpc-integration` 分支。

```text
workspace/
├── AC-MPC/
└── ManiSoft/
```

当前服务器上已完成上述搭建，接手时无需重复：

- 仓库：`/root/autodl-tmp/AC-MPC`（`manisoft-port` 分支）与
  `/root/autodl-tmp/ManiSoft`（`acmpc-integration` 分支）；
- Python 环境：名为 `manisoft`（`/root/miniconda3/envs/manisoft`，Python 3.11）；
- GPU：NVIDIA RTX 4090 24 GB（CUDA 12.6），适合训练；数据生成只依赖 CPU。

### 2.2 获取代码

```bash
mkdir workspace && cd workspace

git clone --branch manisoft-port \
  https://github.com/bright-moon-67/AC-MPC.git

git clone --branch acmpc-integration --recurse-submodules \
  https://github.com/bright-moon-67/ManiSoft.git
```

若克隆 ManiSoft 时未加 `--recurse-submodules`，需补充执行：

```bash
git -C ManiSoft submodule update --init --recursive
```

克隆后请确认分支：AC-MPC 为 `manisoft-port`、ManiSoft 为
`acmpc-integration`（`git branch --show-current`）。两个仓库必须配套
使用，ManiSoft 分支或版本不对时，AC-MPC 中的接口将无法工作。

### 2.3 创建环境并安装依赖

两个仓库共用一个 Python 环境，环境名可自行指定（下文以 `acmpc-manisoft`
为例，当前服务器上名为 `manisoft`）。先安装 ManiSoft 及其固定版本依赖，
再安装 AC-MPC 的控制、测试和绘图依赖：

```bash
conda create -n acmpc-manisoft python=3.11 -y
conda activate acmpc-manisoft
python -m pip install --upgrade pip

python -m pip install -e ./ManiSoft
python -m pip install --no-deps \
  -e ./ManiSoft/third_party/pyelastica \
  -e ./ManiSoft/third_party/liegroups

python -m pip install -e \
  './AC-MPC[test,mpc,plots,tracking]'
```

ManiSoft 使用了两处 PyElastica 兼容性修改。在全新克隆的仓库中执行：

```bash
git -C ManiSoft/third_party/pyelastica apply \
  ../../patches/pyelastica_local.patch
```

补丁只需应用一次。如命令提示补丁已应用，不要重复执行（当前服务器已应用）。

### 2.4 下载仿真资源

`assets/` 体积较大，未纳入 Git。下载并解压到 ManiSoft 仓库根目录：

```bash
cd ManiSoft
hf download JobsWei/ManiSoft --local-dir ./ \
  --repo-type dataset --include 'assets.tar'
tar -xf assets.tar
cd ..
```

完成后应存在 `ManiSoft/assets/`（约 3.1 GB，可删除 `assets.tar`）。
若还需要 ManiSoft 官方的完整 benchmark 数据集（本项目不依赖），按其
README 用 `--exclude "assets.tar"` 单独下载到 `work_dirs/Datasets/`。

`work_dirs/`、训练数据和模型权重也不在 Git 中，按后续章节生成或下载。

### 2.5 安装验证

```bash
python -c "import manisoft, antmaze_ac; print('imports: OK')"
python -m pytest AC-MPC/tests -q
(cd ManiSoft && python scripts/demo.py)
```

仿真 demo 的输出位于 `ManiSoft/work_dirs/`，首次运行需编译渲染内核，可能耗时数分钟。

服务器长时间训练时，可显式指定解释器；`AC-MPC/scripts/run_*_detached.sh`
等后台脚本均以 `AC_MPC_PYTHON`（缺省 `python`）启动训练：

```bash
export AC_MPC_PYTHON="$(command -v python)"
```

## 3. 数据集收集

### 3.1 收集流程

Koopman 数据的基本收集流程如下：

1. 在 ManiSoft 中按指定激励生成 18 维绝对动作；
2. 以 50 Hz 推进仿真（物理步长 0.2 ms、每 100 步执行一次控制），
   记录 `(state, action, next_state)`；
3. 过滤触地（任意节点 z<0 即中止）、末端过低（默认 <0.15 m）、
   末端速度过大（默认 >0.5 m/s）或位移越界（默认 >1.0 m）的轨迹；
   rate 类激励还会保留违规前不少于 32 步的安全前缀；
4. 将每个 episode 保存为独立 NPZ，同时写入 `metadata.json`；
5. 在 AC-MPC 中合并 episode，按 episode 划分 train/validation/test
   （默认 80/10/10），并仅使用训练集计算归一化参数。

当前采集器输出 45 维物理状态：在软体机械臂的三个代表截面上，每个截面
记录 15 维（位置 3 + 旋转 6D 表示 + 线速度 3 + 角速度 3），其中旋转 6D
取旋转矩阵前两列，连续且无四元数符号或欧拉角回绕歧义。动作为
`6×3=18` 维肌肉激活量，范围为 `[-0.30, 0.30]`（采集与 MPC 均保持
`|u|≤0.30`）。

### 3.2 激励类型

`ManiSoft/scripts/collect_koopman_data.py` 提供三种主要激励：

- `coverage`：随机 6–20 s 最小加加速度转移、3–8 s 保持段和 10–30 s
  局部多正弦激励，段长与频率连续随机化以避免周期锁定，用于覆盖常规
  状态与动作范围；
- `rate_coverage`：隔离的动作变化脉冲和慢恢复，覆盖
  `max|Δu|=0.002…0.10`；
- `targeted_rate_coverage`：在 `|u|=0.10…0.30` 的非零动作附近施加
  `max|Δu|=0.01…0.10` 的探针，用于补充 BC-KMPC 曾出现的分布外区域。

`balanced`、`fast`、`reference` 和 `control` 为早期或特定频率对照激励，
保留用于复现旧数据，新的 45 维数据优先使用上述三种。

### 3.3 采集脚本及用法

在 ManiSoft 根目录中执行：

```bash
python scripts/collect_koopman_data.py \
  --config configs/demo_elastica_fast.yaml \
  --output-dir work_dirs/koopman_45d_seed42 \
  --episodes 100 \
  --episode-seconds 180 \
  --control-hz 50 \
  --seed 42 \
  --excitation coverage
```

收集 `rate_coverage` 或 `targeted_rate_coverage` 时，只需替换
`--excitation` 和输出目录。输出目录必须为空，若已含 `episode_*.npz`
脚本会直接报错，不会覆盖；中断重跑前需删除残留文件或换新目录。

多进程收集时，每个 worker 必须使用不同的 `--seed` 和 `--output-dir`，
不得共享输出目录（典型做法是为每个 worker 单独建 `worker_XX` 子目录）。

原始 episode 收集完成后，在 AC-MPC 根目录中进行合并和划分：

```bash
python scripts/build_manisoft_sequences.py \
  --config configs/manisoft_coll.yaml \
  --input-root ../ManiSoft/work_dirs/koopman_45d_seed42 \
  --expected-episodes 100 \
  --output data/processed/manisoft_45d_seed42
```

若数据由多个 worker 生成，对每个目录重复传入一次 `--input-root`；
各目录 episode 数不一致时，用 `--episode-counts N1 N2 ...` 按
`--input-root` 的顺序逐个指定。默认按配置（`seed: 42`）以 80/10/10
按 episode 划分 train/validation/test，可用 `--split-seed` 覆盖。

转换后的默认数据格式为：

```text
state          = [45D physical_state_t, 18D previous_action]  # 63D
action         = current_action - previous_action             # 18D 增量动作
next_state     = [45D physical_state_t+1, 18D current_action] # 63D
current_action = 18D 绝对动作
```

每个 episode 首帧的 `previous_action` 取 `u_{-1}=0`，首行增量动作即为
`u_0`。

数据中同时保留 `current_action`，因此同一批轨迹可用于增量动作、绝对动作
和带历史信息的绝对动作 Koopman 模型。

### 3.4 现有数据集

| 数据集 | 内容 | 规模 | 用途 |
| --- | --- | ---: | --- |
| `manisoft_45d_824ep` | 45D 三截面自由运动数据 | 824 episodes，7,416,000 transitions | 当前 45D 模型的基础数据 |
| `koopman_45d_rate_v3_8env` | 动作变化率补充数据 | 448 episodes，1,178,365 transitions | 补充高 `|Δu|` 覆盖 |
| `koopman_45d_targeted_rate_v4_8env` | 高动作幅值与高变化率联合补充 | 1,543 episodes，1,031,173 transitions | 最新定向追加数据 |
| `manisoft_600k` | 早期 41D 自由运动数据 | 567 episodes，567,000 transitions | 41D 增量动作模型 |
| `manisoft_free_motion_275` | 早期 41D 长轨迹数据 | 275 episodes，825,000 transitions | 早期模型与对照 |
| `manisoft_600k_tip11` | 由 41D 数据压缩得到的 11D 末端状态 | 567 episodes，567,000 transitions | 末端状态辅助实验 |
| `manisoft_coll_100` | 100 条 COLL 任务轨迹 | 124,692 transitions | 任务轨迹对照，非主要自由运动数据 |

前三项为 45D 模型的核心数据。`rate_v3` 和 `targeted_rate_v4` 目前保存在
`ManiSoft/work_dirs/` 的原始 worker 目录中；基础合并数据位于
`AC-MPC/data/processed/`。`manisoft_45d_824ep` 由
`ManiSoft/work_dirs/koopman_45d_16env/` 的 16 个 worker 目录合并构建，
构建日志见 `AC-MPC/data/processed/manisoft_45d_824ep_build.log`。
专家轨迹和三 waypoint 测试数据见第 5 章。

### 3.5 数据保存与复现

`ManiSoft/work_dirs/`、`ManiSoft/assets/` 和 `AC-MPC/data/processed/` 均被
Git 忽略，不会随仓库自动下载。对外交付时应单独归档核心数据，并保留：

- 数据集下载地址和 SHA256；
- 原始 `metadata.json` 中的 `scenario_path`/`scenario_sha256`、`seed`、
  `excitation`、安全阈值（`min_tip_height`、`max_tip_speed`、
  `max_tip_displacement`）以及 `state_layout`、`transition_fields` 等字段；
- episode 数、transition 数及 train/validation/test 划分；
- 合并后 `data/processed/<name>/metadata.json` 中的
  `dataset_schema_version`、`transitions` 与 `transition_semantics`，
  用于校验交付数据的格式与规模；
- 用于训练的精确数据版本，避免混用基础数据与追加数据。

## 4. Koopman 模型训练与单点跟踪验证

### 4.1 模型与 checkpoint 总览

训练统一使用 `configs/manisoft_coll.yaml` 的 `koopman` 超参数：`lift_dim=32`、
encoder 256×256 SiLU、`K_step=20`、lr=3×10⁻⁴、batch=4096、`max_epochs=1000`、
`max_wall_time_hours=5`；loss 权重 `linear=10 / reconstruction=1 / rollout=1 /
latent_std=0.1 / stability=0.01 / identity=1e-4`；谱半径上限 1.0。数据按
80/10/10 划分，归一化参数只用训练集。

| 模型 | 状态/动作语义 | 训练数据（episodes） | checkpoint |
| --- | --- | ---: | --- |
| delta 45D | 63D 状态（45D 物理 + 上一动作），动作 = Δu | 824 | `runs/manisoft_45d_824ep_seed42/koopman/best_validation.pt` |
| abs 45D | 45D 物理状态，动作 = 绝对 u_t | 824 | `runs/koopman_45d_abs_seed42/best_validation.pt` |
| history H=10 abs | 同 abs + H=10 历史 context | 824 | `work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt` |
| history H=10 abs（targeted v4 tip） | 同上 | 2,815（824 + rate_v3 + targeted_v4） | `work_dirs/manisoft_koopman_history_h10_abs_targeted_v4_tip_seed42/koopman_history/best_validation.pt` |
| 41D（早期） | 59D 状态（41D 物理 + 上一动作） | 早期 41D 数据 | `runs/manisoft_coll_full_seed42/koopman/best_validation.pt` |

训练结果（`history.jsonl` 中的最佳验证 loss，val total 越低越好）：

| 模型 | 训练时长 | 最佳 epoch | val linear / total |
| --- | ---: | ---: | ---: |
| delta 45D | 47 epochs / 2.64 h | 29 | 0.0127 / 0.176 |
| abs 45D | 80 epochs / 3.94 h | 74 | 0.0373 / 0.464 |
| history H=10 abs | 20 epochs | 18 | 0.0093 / 0.136 |

targeted v4 tip history 在含高变化率追加数据（2,815 episodes）上训练 92
epochs，最佳 epoch 86、val total 0.543，与上表量纲不同，不作直接比较。

### 4.2 训练命令

delta 模型（AC-MPC 根目录，输入为第 3 章合并后的数据集）：

```bash
python scripts/train_koopman.py \
  --config configs/manisoft_coll.yaml \
  --data data/processed/manisoft_45d_824ep \
  --output runs/manisoft_45d_824ep_seed42 \
  --device cuda --wandb-mode offline
```

abs 模型（直接读采集器原始 worker 目录，无需先执行
`build_manisoft_sequences.py`；学习 `z_{t+1} = A z_t + B u_t`，状态为纯
45D 物理状态、不含上一动作块）：

```bash
python scripts/train_koopman_abs_action.py \
  --config configs/manisoft_coll.yaml \
  --input-root ../ManiSoft/work_dirs/koopman_45d_16env/worker_00 \
  --input-root ../ManiSoft/work_dirs/koopman_45d_16env/worker_01 \
  ... （16 个 worker 逐个传入，共 824 episodes）...
  --output runs/koopman_45d_abs_seed42 \
  --device cuda --wandb-mode offline
```

history 模型用 `scripts/train_koopman_history.py`
（`--history-steps 10 --data ...`）。后台训练参考：

```bash
export AC_MPC_PYTHON="/root/miniconda3/envs/manisoft/bin/python"
cd /root/autodl-tmp/AC-MPC
nohup setsid "$AC_MPC_PYTHON" -u scripts/train_koopman_abs_action.py \
  ... >> runs/<name>_train.log 2>&1 < /dev/null &
```

注意 `scripts/run_koopman_detached.sh` 目前硬编码 antmaze-umaze 配置，
ManiSoft 训练请直接调用上述脚本。每个运行目录包含 `best_validation.pt`、
`history.jsonl`（每 epoch 一行 JSON）、`resolved_config.json` 与 wandb
offline 目录；断点续训用 `--resume <recovery_epoch_*.pt>`。

### 4.3 ±5 mm 单点跟踪冒烟测试（早期 41D 模型）

快速冒烟脚本，验证 Koopman + LQR 闭环能把软体臂末端稳定到 ±5 mm 偏移的
setpoint：

- `scripts/smoke_manisoft_lqr_track_5mm.py`：+5 mm x 偏移，100 步；
- `scripts/smoke_manisoft_lqr_track_minus5mm_x_300.py`、
  `smoke_manisoft_lqr_track_plus5mm_y_300.py`、
  `smoke_manisoft_lqr_track_minus5mm_y_300.py`、
  `smoke_manisoft_lqr_track_minus5mm_z_300.py`：各轴 ±5 mm，300 步。

脚本离线求解 LQR 增益 K（末端位置权重 20、末端速度大小 0.1、姿态四元数
2、上一动作 0.01，`R=1000·I`），将参考 setpoint 偏移 ±5 mm 后闭环跟踪，
并与零动作基线对比；Δu 截断 ±0.02，`|u|≤0.30`。使用早期 41D 模型
`runs/manisoft_coll_full_seed42/koopman/best_validation.pt`，环境为 COLL
场景（`ManiSoft/work_dirs/data_gen/COLL/can/scenarios/0/config.yaml` +
`ManiSoft/work_dirs/rl_models/model_1.zip`）。

脚本以 cwd 定位两个仓库，需在 ManiSoft 根目录执行：

```bash
cd ManiSoft
python ../AC-MPC/scripts/smoke_manisoft_lqr_track_5mm.py
```

结果直接打印在控制台：每 10 步输出 `tip_drift` 与最大动作，最后对比 LQR
与零动作基线的平均/最终/最大误差。这是 41D 时代的快速冒烟验证，45D
模型的正式单点跟踪验证见 4.4。

### 4.4 45D 参考跟踪验证（LQR/MPC 参数与结果）

公共设置：

- scenario：`ManiSoft/configs/demo_elastica_fast.yaml`（纯软体臂，无夹爪与物体）；
- reference：`ManiSoft/work_dirs/random_reference_45d/reference.npz`
  （随机稳态参考，初始末端距参考约 158 mm）；
- 成功判据：状态距离 ≤2 mm 连续 ≥100 步（`--success-threshold 0.002
  --required-success-streak 100 --stability-window 100`）；
- 结果写入 `<output>/summary.json` 与 `<output>/trajectory.npz`。

验证脚本分 LQR 与 MPC 两套，各含 delta/abs/history 三个入口：
`validate_koopman_lqr_reference{,_abs,_history}.py` 与
`validate_koopman_mpc_reference{,_abs,_history}.py`。

**abs LQR（已验证，推荐复现起点）**

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

结果：最终误差 0.134 mm，连续 848/1000 步 <2 mm；DARE 20 次迭代收敛、
闭环谱半径 0.99998；反馈耗时均值 0.56 ms（p95 0.57 ms），1000 步总耗时
11.5 s，满足 50 Hz 实时。关键做法是把 LQR 反馈缩放到原增益的 3%
（`feedback-scale 0.03`）：以参考前馈动作为主、LQR 只作小修正。

**参数敏感性**（汇总自 `runs/koopman_lqr_abs_*` 的扫描）：

- `max-delta=0.002` + `control-weight=100000` + `feedback-scale ≤0.1`
  的配置均成功；`feedback-scale=0.3` 失败（最终误差 10.1 mm）；
- `max-delta ≥0.005` 或 `control-weight ≤10000` 全部失败（末端在
  100–680 mm 间振荡，无法收敛）。

**各方案最佳结果**（完整命令与复验步骤见 `docs/best_validation_commands.md`）：

| 方案 | 脚本 | 最佳结果 |
| --- | --- | --- |
| delta MPC | `validate_koopman_mpc_reference.py` | ≈7.98 mm @5000 步 |
| abs MPC | `validate_koopman_mpc_reference_abs.py` | ≈1.28 mm @500 步 |
| delta LQR | `validate_koopman_lqr_reference.py` | ≈0.49 mm @500 步 |
| abs LQR | `validate_koopman_lqr_reference_abs.py` | 0.134 mm @1000 步 |
| history abs LQR | `validate_koopman_lqr_reference_history.py` | ≈0.057 mm @2000 步（feedback-scale 0.0045） |
| history MPC | `validate_koopman_mpc_reference_history.py` | ≈0.91 mm @1000 步（末 100 步均值） |
| targeted v4 tip history MPC | `validate_koopman_mpc_reference_history.py` | ≈0.064 mm @1000 步（tip-state-scale 50） |
| targeted v4 tip history LQR | `validate_koopman_lqr_reference_history.py` | ≈0.157 mm @1000 步（feedback-scale 30） |

history 模型的反馈尺度与其它模型不同（不同 checkpoint 的有效增益尺度不同），
调参时以 `docs/best_validation_commands.md` 中的现成命令为准。

## 5. 专家轨迹与三 waypoint BC-KMPC

本章把有限时域 BC-KMPC 迁移到 ManiSoft 软体仿真，流程为：生成 waypoint
参考库 → 用固定代价 history Koopman-MPC 采集专家轨迹 → BC 克隆 →
（可选 DAgger）→ PPO 精调 → 确定性评估。详细设计见
`docs/manisoft_bc_kmpc.md`。

### 5.1 任务与观测定义

- 模型：history H=10 abs Koopman（`z[t+1] = A z[t] + B u[t]`，45D 物理状态 +
  18D 绝对动作）；
- episode：从稳定性认证的参考库中确定性抽取一个三路点组，三个目标距初始
  末端分别为 4–8 cm、8–14 cm、12–20 cm，同组目标来自同一随机动作方向、
  递增幅值；中间 waypoint 到达后只切换阶段、不重置仿真，第三个 waypoint
  稳定到达才结束回合；
- 观测（687 维，沿用 PandaReach3 的 three-waypoints 语义）：

```text
[s_t, context_t, G1_xyz, G2_xyz, G3_xyz, one_hot(active_stage)]
context_t = [normalized s[t-H+1:t+1], u[t-H:t]], H=10
即 45 + 10*(45+18) + 12 = 687
```

- 奖励（进度/时间/完成组合，不能靠停留刷分）：

```text
r = (previous_distance - distance) / waypoint_initial_distance
    - 0.01
    - 0.001 * mean((action / 0.30)^2)
    + 3 * passed_waypoint
    + 5 * completed_all_waypoints
```

### 5.2 控制与专家 MPC

`KoopmanMPCActor` 根据当前 lift、三个归一化目标与阶段 one-hot 输出每个
预测步的二次权重与线性项；冻结的 `A/B/C` 将问题凝缩为 absolute action QP，
固定展开的投影 FISTA 对绝对动作逐元素 box 投影 `[-0.30, 0.30]`，只执行
序列第一步。BC-KMPC 不加入 smoothness 或动作变化率约束；训练和评估记录
`projected_gradient_residual`，固定代价 OSQP 只作为 BC 专家、不进入 PPO
反向传播。

专家 MPC 使用已验证的固定代价参数：

```text
state_weight=200, tip_state_scale=5, action_weight=8000, control_weight=1
```

该组参数在 106 个认证 triplet、`rollout_noise_std=0.0002` 的全库测试中实现
106/106 三路点成功且无动作元素饱和，搜索证据见
`work_dirs/bc_kmpc_weight_search/final_all106.json`。

### 5.3 参考库（waypoint bank）生成

每个 waypoint 的最后 250 步必须满足 1 mm 位置稳定性和速度阈值，并在独立
新仿真中复验：

```bash
python scripts/generate_manisoft_waypoint_bank.py \
  --scenario "$S" --output "$W" --triplets 100 --seed 42 \
  --distance-ranges-cm 4 8 8 14 12 20 --stable-steps 250
```

当前服务器上的参考库（`AC-MPC/data/processed/`）：

| 目录 | triplets | manifest | 说明 |
| --- | ---: | --- | --- |
| `manisoft_waypoint_bank_v1` | 12（散件） | 无 | 早期未收满，不可直接用 |
| `manisoft_waypoint_bank_v1_merged` | 391 | 有 | 认证参考库，docs 的 106 认证集出自此库 |
| `manisoft_waypoint_bank_v2_zmixed_merged` | 904 | 有 | 混合幅度版本 |
| `manisoft_waypoint_bank_v4_full_merged` | 1,498 | 有 | 主力库（PPO 对比用） |
| `manisoft_waypoint_bank_v5_10k` | 7,109 | 有 | 大规模库 |

注意：早期 three_waypoint 数据集元数据引用的
`ManiSoft/work_dirs/smooth_reference_45d` 已被清理，复现时需重新生成参考库
或改用上表中有 manifest 的库。多进程批量生成参考
`scripts/launch_manisoft_waypoint_bank_multi.sh`（每进程独立 shard + 不同
seed，收满后合并）。

### 5.4 专家数据采集与 DAgger

用已验证的 fixed-cost history MPC 采集专家数据。公共文件：

```bash
K=work_dirs/manisoft_koopman_history_h10_abs_seed42_20260809/koopman_history/best_validation.pt
S=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
W=data/processed/manisoft_waypoint_bank_v1_merged
```

```bash
python scripts/collect_manisoft_bc_kmpc_expert.py \
  --koopman-checkpoint "$K" --scenario "$S" --waypoint-root "$W" \
  --output data/processed/manisoft_bc_kmpc/expert.npz \
  --episodes 10 --episode-steps 300 --horizon 10 \
  --rollout-noise-std 0.0002 --device cuda
```

小幅 rollout noise 让确定性复位下的专家数据覆盖相邻状态；保存的监督标签
仍是专家在实际 history 上重新求得的动作，而不是加噪后的执行动作。

DAgger（BC 闭环偏离专家分布时，用当前 BC 驱动仿真、由 OSQP 专家重新标注）：

```bash
python scripts/collect_manisoft_bc_kmpc_expert.py \
  --koopman-checkpoint "$K" --scenario "$S" --waypoint-root "$W" \
  --base-dataset data/processed/manisoft_bc_kmpc/expert.npz \
  --rollout-checkpoint runs/manisoft_bc_kmpc/bc/best_validation.pt \
  --output data/processed/manisoft_bc_kmpc/expert_dagger.npz \
  --episodes 3 --episode-steps 300 --rollout-noise-std 0.0001 \
  --device cuda
```

### 5.5 BC 训练

```bash
python scripts/train_manisoft_bc_kmpc_bc.py \
  --koopman-checkpoint "$K" \
  --dataset data/processed/manisoft_bc_kmpc/expert.npz \
  --output runs/manisoft_bc_kmpc/bc \
  --epochs 150 --batch-size 256 --horizon 10 \
  --device cuda
```

除当前动作外，BC 还按参考仓库监督后续 receding-horizon expert actions
（默认 `--sequence-weight 0.25`）；future target 不跨越 episode 或 active
waypoint 边界。实际运行（`runs/manisoft_bc_kmpc_three_waypoint/bc`，
10 episode 专家集）150 epochs 完成，`best_validation_mse ≈ 1.76×10⁻⁷`。

### 5.6 PPO 精调

有限时域 MPC 均值对代价参数敏感，默认用较小 actor 学习率，并用 target-KL
阻止单次 rollout 上的过度更新：

```bash
python scripts/train_manisoft_bc_kmpc_ppo.py \
  --koopman-checkpoint "$K" \
  --bc-checkpoint runs/manisoft_bc_kmpc/bc/best_validation.pt \
  --scenario "$S" --waypoint-root "$W" \
  --output runs/manisoft_bc_kmpc/ppo/seed_42 \
  --horizon 10 --num-envs 1 \
  --actor-learning-rate 0.0001 --target-kl 0.02 --device cuda
```

实际运行（`runs/manisoft_bc_kmpc_three_waypoint/ppo`，seed 42）：
`actor_learning_rate=3×10⁻⁷`、30 updates、61,440 timesteps，最佳完成回合
平均回报 152.4；末次 rollout 的 `completed_success_rate=0`、
`waypoints_completed_mean=0.286`——属于早期小规模实验，更成熟的 PPO 对比
与调参见后续章节。

重点监控指标：`action_saturation_rate`（可行分布下应接近零）、
`distance_minimum`、`completed_success_rate`、`waypoints_completed_mean`、
`approx_kl`、`ppo_early_stopped`（approx_kl 超 target 时本轮提前结束
minibatch 更新，而不是继续破坏 BC 初始化）。

### 5.7 确定性评估与一键脚本

```bash
python scripts/evaluate_manisoft_bc_kmpc.py \
  --checkpoint runs/manisoft_bc_kmpc/ppo/seed_42/last.pt \
  --scenario "$S" --waypoint-root "$W" \
  --output runs/manisoft_bc_kmpc/evaluation/seed_42 \
  --episodes 10 --episode-steps 300 --device cuda
```

结果写入 `<output>/summary.json`（`success_rate`、
`waypoints_completed_mean`、`action_saturation_rate` 等）。

也可以一条命令顺序执行专家采集 + BC + PPO：

```bash
AC_MPC_PYTHON=/root/miniconda3/envs/manisoft/bin/python \
  scripts/run_manisoft_bc_kmpc.sh "$K" "$S" "$W" \
  data/processed/manisoft_bc_kmpc_three_waypoint/expert.npz \
  runs/manisoft_bc_kmpc_three_waypoint/bc \
  runs/manisoft_bc_kmpc_three_waypoint/ppo/seed_42 cuda
```

以下参数必须在专家数据、BC 和 PPO 之间保持一致，否则 checkpoint 拒绝加载：

- `horizon`（默认 10）
- `solver_iterations`（默认 20）
- `absolute_action_limit`（默认 0.30）
- waypoint-bank manifest 的 SHA256

`--waypoint-root` 读取 `manifest.json` 及其中列出的 NPZ；加载器校验
manifest、每个参考文件与 scenario 的 SHA256，并保证同一 episode 的环境目标
与 MPC reference state/action 使用同一个 `waypoint_triplet_index`。

### 5.8 数据集与 checkpoint 现状

专家数据集（`data/processed/`，均为 three-waypoint schema）：

| 目录 | episodes | 说明 |
| --- | ---: | --- |
| `manisoft_bc_kmpc_three_waypoint` | 10 | 初版（v1），BC/PPO 的原始训练集 |
| `manisoft_bc_kmpc_three_waypoint_v3/v4` | 10 | 版本迭代 |
| `manisoft_bc_kmpc_three_waypoint_v5` | 90 | 扩充 |
| `manisoft_bc_kmpc_three_waypoint_v6` | 180 | 含 `part0/1/2` 分片与合并 `expert.npz` |
| `manisoft_bc_kmpc_three_waypoint_v7` | 142 | 已划分 `train.npz`（128）+ `val.npz`（14），`split.json` |

另有早期随机路点尝试（`manisoft_bc_kmpc_random_waypoints_v1–v3` 及其
dagger 版本），closed-loop 成功率 0，已弃用。

训练与评估产物：

- `runs/manisoft_bc_kmpc_three_waypoint/`：`bc/best_validation.pt`、
  `ppo/best_completed_return.pt`、`ppo/last.pt`；
- `runs/manisoft_bc_kmpc_history_h10_seed42/`：BC → STE → DAgger 迭代链
  （`bc`、`bc_v2_ste`、`bc_v3_dagger`、`bc_v4_dagger`）及
  `diagnostics_20260810/` 确定性诊断：最终距离 1.099 m → 0.179 m →
  0.021 m → 0.009 m，说明 DAgger 迭代有效；
- `runs/manisoft_bc_kmpc_random_waypoints_v2/v3`：早期尝试，已弃用。

专家轨迹的成功率参考：`v5` 收集日志中专家本身约 1/3 episode 达到
3/3 waypoints（waypoint 任务对专家 MPC 也有难度），收集时以
`rollout_noise_std` 覆盖相邻状态。
