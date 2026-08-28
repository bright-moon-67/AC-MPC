# AC-MPC + ManiSoft 项目说明

本文档用于 AC-MPC + ManiSoft 工作，覆盖环境搭建、Koopman
数据收集与训练、单点跟踪、三 waypoint PPO-KMPC、离线数据收集、KMPC-IQL，
以及薄墙绕行的平滑教师与统一 residual SAC 跟踪任务。
文档同时区分当前推荐主线、仍可复现的旧实验和正在验证的研究分支。除特别
标注外，命令均从 `AC-MPC/manisoft_port/` 执行；当前服务器上已配好的环境和
数据位置见 2.1。

> **独立目录版说明**：本目录是从已验证的 `manisoft-port` 完整迁入的隔离
> 项目。外层 AC-MPC 保持目标仓库最新 `main` 原样，ManiSoft 扩展的 Python
> 包、配置、脚本、测试和文档全部位于 `manisoft_port/`，因此不会覆盖外层
> 同名模块。首次部署必须按 2.2 节执行布局初始化脚本。

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
5. 在带虚拟薄墙与地面约束的强弯曲场景中，从直立状态绕墙、保持中段拱高，
   再使远端回到 yz 面附近；当前还包含一个不施加物理力的末端视觉推块演示。

两个仓库的职责边界为：

```text
ManiSoft/acmpc-integration
  仿真后端、软体动力学、肌肉力矩、45D 状态抽取、原始数据采集
                  │
                  ▼
AC-MPC/manisoft_port（目标仓库内的隔离项目）
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

先定义工作区路径。除下文 1.6 节说明的 v15e checkpoint 内嵌 Koopman
绝对路径外，后文出现 `/root/autodl-tmp` 时均可用相同方式替换：

```bash
export WORKSPACE=/path/to/workspace
export ACMPC_REPO="$WORKSPACE/AC-MPC"
export ACMPC_ROOT="$ACMPC_REPO/manisoft_port"
export MANISOFT_ROOT="$ACMPC_REPO/ManiSoft"
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
V15E=runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt

"$AC_MPC_PYTHON" scripts/evaluate_manisoft_ppo_comparison.py \
  --checkpoint "$V15E" \
  --scenario "$MANISOFT_ROOT/configs/demo_elastica_fast.yaml" \
  --waypoint-root data/processed/manisoft_waypoint_bank_v2_zmixed_merged \
  --output runs/handoff_smoke/v15e_eval_10ep \
  --episodes 10 --episode-steps 300 --device cuda --seed 42
```

评估器使用确定性 KMPC mean，并保存逐回合轨迹和 `summary.json`。若故意在与
checkpoint 记录不同的 waypoint bank 上测试泛化，必须额外传入
`--allow-other-waypoint-bank`；否则 SHA256 不一致会被拒绝。

### 1.6 v15e 最小交接 artifact

同事已经从 GitHub 获取含 `manisoft_port/` 的 AC-MPC 和 ManiSoft
`acmpc-integration` 配套分支后，若只要求直接运行上面的 v15e 确定性评估，
不需要复制服务器上的整个 `data/`、`runs/` 或 `work_dirs/`。在代码之外至少还需
交付以下三项，并保持相对外层 AC-MPC 仓库根目录的目录结构：

| 必需 artifact | 大小 | 内容与作用 |
| --- | ---: | --- |
| `runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt` | 约 2.7 MB | 当前推荐 v15e PPO-KMPC 行为策略 |
| `work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt` | 约 2.9 MB | v15e 加载时必需的 H=10、谱半径上限 0.95 history Koopman 模型 |
| `data/processed/manisoft_waypoint_bank_v2_zmixed_merged/` | 约 106 MB | 与 checkpoint 匹配的完整 v2 waypoint bank；含 `manifest.json` 和 904 个 triplet，共 2,713 个文件 |

三项合计约 111 MB。用于交接核验的 SHA256 为：

```text
0b82cba4f5c1bf423de3395c15ef2998482ca4db660151127795e531e830c6c6  v15e last.pt
7e2bc0cffe23e095d50a5914716c83545617202ca0d80e29763a3ab2240ec779  history Koopman best_validation.pt
d6e44fe6027a55753ee731c0a7e2bf3e1803090e8b066c051f0de9821e6281cd  v2 waypoint bank manifest.json
```

这三项均被 `.gitignore` 排除，不会随代码 PR 或普通 `git clone`
自动获取。它们已按上述目录结构发布为公开 GitHub Release asset：

- Release：[v15e-artifacts-20260819](https://github.com/bright-moon-67/AC-MPC/releases/tag/v15e-artifacts-20260819)
- 文件：`acmpc-v15e-minimal-artifacts-20260819.tar.gz`
- 大小：87,310,950 bytes（约 83.3 MiB）
- 整包 SHA256：`2cbe57731490c8cbc106f93e1bbd48f190cf79a2f23cdf88403fd21eb0593296`
- 代码基线：AC-MPC `manisoft-port` commit
  `8dba696c5caf79bfd6c90aa41801e16b6508cdf7`

由于下方说明的 v15e 绝对路径限制，当前开箱即用方式是把 AC-MPC 放到
`/root/autodl-tmp/AC-MPC`，并递归初始化其中固定的 ManiSoft submodule，再执行：

```bash
export WORKSPACE=/root/autodl-tmp
export ARTIFACT_ARCHIVE=/tmp/acmpc-v15e-minimal-artifacts-20260819.tar.gz

curl --fail --location \
  --output "$ARTIFACT_ARCHIVE" \
  https://github.com/bright-moon-67/AC-MPC/releases/download/v15e-artifacts-20260819/acmpc-v15e-minimal-artifacts-20260819.tar.gz

echo "2cbe57731490c8cbc106f93e1bbd48f190cf79a2f23cdf88403fd21eb0593296  $ARTIFACT_ARCHIVE" \
  | sha256sum --check

tar -xzf "$ARTIFACT_ARCHIVE" -C "$WORKSPACE"

test -f "$WORKSPACE/AC-MPC/runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt"
test -f "$WORKSPACE/AC-MPC/work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt"
test -f "$WORKSPACE/AC-MPC/data/processed/manisoft_waypoint_bank_v2_zmixed_merged/manifest.json"
```

压缩包只会在已 clone 的外层 `AC-MPC/` 中补充上述三项被忽略的 artifact，
不包含或覆盖代码文件。2.2 节初始化脚本会在 `manisoft_port/` 内建立指向
外层 `data/`、`runs/` 和 `work_dirs/` 的兼容链接。校验成功后才继续执行
1.5 节的 v15e 评估。

这里存在一个当前 checkpoint 格式的可移植性限制：v15e `last.pt` 的 payload
保存了训练机器上的 Koopman 绝对路径：

```text
/root/autodl-tmp/AC-MPC/work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt
```

`load_manisoft_ppo_checkpoint()` 会直接读取这个绝对路径并校验文件 SHA256，当前
evaluator 没有提供覆盖 Koopman 路径的命令行参数。因此，在 loader 尚未增加
路径重映射功能前，开箱即用的交接环境必须把 AC-MPC 放在
`/root/autodl-tmp/AC-MPC`；若安装在其它路径，需要先迁移 checkpoint 元数据或
修改 loader，不能只设置 `ACMPC_ROOT`。`--waypoint-root` 可以显式覆盖 waypoint
bank 路径，不受这一限制。

## 2. 环境配置

### 2.1 基本要求

- Linux（已在 Ubuntu 上验证）；
- Conda/Miniconda 和 Git；
- Python 3.11（`AC-MPC/manisoft_port/pyproject.toml` 要求
  `>=3.10,<3.13`，已在 3.11 上验证）；
- 训练推荐使用支持 CUDA 的 NVIDIA GPU，数据生成和小规模测试可使用 CPU。

隔离项目依赖本项目修改后的 ManiSoft 仿真环境。该依赖作为固定 submodule
放在 AC-MPC 内部；不要用原生 ManiSoft 仓库替代它。

```text
workspace/
├── AC-MPC/                         # yuej0422-dev 最新主项目
│   ├── ManiSoft/                   # 固定到 0096f23 的递归 submodule
│   ├── manisoft_port/              # 本项目的全部受控代码
│   ├── data/                       # artifact 解压后生成
│   ├── runs/                       # artifact 解压后生成
│   └── work_dirs/                  # artifact 解压后生成
```

当前服务器上已完成上述搭建，接手时无需重复：

- 仓库：`/root/autodl-tmp/AC-MPC/manisoft_port`（隔离项目）与
  `/root/autodl-tmp/AC-MPC/ManiSoft`（固定 submodule）；
- Python 环境：名为 `manisoft`（`/root/miniconda3/envs/manisoft`，Python 3.11）；
- GPU：NVIDIA RTX 4090 24 GB（CUDA 12.6），适合训练；数据生成只依赖 CPU。

### 2.2 获取代码

```bash
mkdir -p /root/autodl-tmp
cd /root/autodl-tmp

git clone --recurse-submodules https://github.com/yuej0422-dev/AC-MPC.git

bash AC-MPC/manisoft_port/scripts/bootstrap_isolated_layout.sh
```

若克隆 AC-MPC 时未加 `--recurse-submodules`，需补充执行：

```bash
git -C AC-MPC submodule update --init --recursive
```

PR 合并前审阅者可改为克隆 `bright-moon-67/AC-MPC` 的本 PR 分支：

```bash
git clone --branch manisoft-wall-handoff --recurse-submodules \
  https://github.com/bright-moon-67/AC-MPC.git
```

PR 合并后使用目标仓库默认分支即可。ManiSoft 必须保持在 Git 记录的固定提交；
初始化脚本会核验版本并建立 artifact 兼容链接，不改写外层 AC-MPC 的源文件。

本文最后核对时的兼容基线为：

- AC-MPC 墙体任务源归档 `manisoft-port`：
  `ce8f1a8286e0ea86c20e8c21a3f4d6a5094415f3`；
- 本 PR 固定的 ManiSoft `acmpc-integration`：
  `0096f2358d2605b9d382480a7abd30e5c2292495`；
- ManiSoft 固定的 PyElastica submodule：
  `4084bdaf0438c85b60d4127c287d24d14e80be11`。

后续提交可能使分支 HEAD 前移，因此实验记录仍应同时保存两个仓库的
`git rev-parse HEAD` 和配置文件 SHA256，不要只记录分支名。

### 2.3 创建环境并安装依赖

两个仓库共用一个 Python 环境，环境名可自行指定（下文以 `acmpc-manisoft`
为例，当前服务器上名为 `manisoft`）。先安装 ManiSoft 及其固定版本依赖，
再安装 AC-MPC 的控制、测试和绘图依赖：

```bash
conda create -n acmpc-manisoft python=3.11 -y
conda activate acmpc-manisoft
python -m pip install --upgrade pip

python -m pip install -e /root/autodl-tmp/AC-MPC/ManiSoft
python -m pip install --no-deps \
  -e /root/autodl-tmp/AC-MPC/ManiSoft/third_party/pyelastica \
  -e /root/autodl-tmp/AC-MPC/ManiSoft/third_party/liegroups

python -m pip install -e \
  '/root/autodl-tmp/AC-MPC/manisoft_port[test,mpc,plots,tracking]'
```

固定的 PyElastica commit `4084bdaf...` **已经包含**本项目需要的两处兼容
修改，正常克隆后不要再执行 `git apply`。否则会因为补丁已经存在而报错。

ManiSoft 仍保留两份内容完全相同的历史补丁，供审计和旧 commit 恢复使用：

- `patches/pyelastica_local.patch`：权威归档副本；
- `third_party/pyelastica.patch`：兼容旧脚本和旧文档路径的镜像副本。

两份文件不是两个不同补丁，必须保持逐字节一致。补丁内容是将 face normals
转为 `float64`，以及跳过 ManiSoft 自定义 rod/mesh contact 不兼容的上游
contact-order 检查。只有在检出早于 `4084bdaf...`、尚未包含修复的历史
PyElastica commit 时才可能需要手动应用，而且只能任选一份应用一次。

正常安装后，下列 `--reverse --check` 应成功，表示固定 submodule 已包含补丁；
它只做检查，不修改文件：

```bash
cmp /root/autodl-tmp/AC-MPC/ManiSoft/patches/pyelastica_local.patch \
  /root/autodl-tmp/AC-MPC/ManiSoft/third_party/pyelastica.patch
git -C /root/autodl-tmp/AC-MPC/ManiSoft/third_party/pyelastica \
  apply --reverse --check \
  ../../patches/pyelastica_local.patch
```

### 2.4 可选：下载完整仿真资源

当前 v15e 主线使用 `configs/demo_elastica_fast.yaml`：该配置
`renderer: null`、`objects: []` 且无 gripper，headless 数据采集、训练和
评估不会读取 `ManiSoft/assets/`。如果只需要部署 v15e，可跳过本节，
不需要下载约 3.1 GB 资源。

只有运行 ManiSoft 官方可视化 demo、VLM benchmark、gripper 或物体/纹理
任务时，才需要把 `assets/` 下载并解压到 ManiSoft 仓库根目录：

```bash
cd AC-MPC/ManiSoft
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

#### 必做：headless 主线验证

下列检查不需要 `assets/`、Blender、桌面环境或 checkpoint。命令从
`/root/autodl-tmp` 执行：

```bash
python -c "import manisoft, antmaze_ac; print('imports: OK')"
python -m pytest AC-MPC/manisoft_port/tests -q
(cd AC-MPC/ManiSoft && python -m pytest -q)

export MANISOFT_ROOT="$PWD/AC-MPC/ManiSoft"
(cd AC-MPC/manisoft_port && python - <<'PY'
import os
from pathlib import Path

import numpy as np

from antmaze_ac.envs.manisoft_tracking_env import ManiSoftTipTrackingEnv

scenario = Path(os.environ["MANISOFT_ROOT"]) / "configs/demo_elastica_fast.yaml"
env = ManiSoftTipTrackingEnv(scenario, episode_steps=2)
observation, _ = env.reset(seed=0)
observation, reward, _, _, _ = env.step(np.zeros(18, dtype=np.float32))
assert observation.shape == (45,)
assert np.isfinite(observation).all() and np.isfinite(reward)
print("headless ManiSoft step: OK")
PY
)
```

本文最后核对时，具备本地教师 artifact 的完整环境应为 AC-MPC `206 passed`、
ManiSoft `4 passed`；干净 Git clone 尚未下载第 12.9 节教师 artifact 时，应为
AC-MPC `196 passed, 10 skipped`、ManiSoft `4 passed`。最后一行应输出
`headless ManiSoft step: OK`。这一步真正创建
Elastica 环境、重置软体臂并执行一个 50 Hz 控制步，比只测试 import 更能
发现子模块、补丁或数值依赖错误。

#### 可选：完整渲染 demo 验证

`ManiSoft/scripts/demo.py` 不是 v15e 的安装冒烟；它使用 Blender、
MuJoCo renderer、gripper、纹理和完整 `assets/`。只有需要这些功能时才
安装系统渲染依赖并运行：

```bash
sudo apt-get update
sudo apt-get install -y autoconf automake build-essential cmake \
  libboost-all-dev libpng-dev libjpeg-dev libtiff-dev libopenexr-dev \
  libsdl1.2-dev libsm6 libxext6 freeglut3-dev libxrender1 \
  libxkbcommon-x11-0
conda install ffmpeg -y

(cd AC-MPC/ManiSoft && python scripts/demo.py)
```

该 demo 还要求已完成 2.4 节的 `assets/` 下载。输出位于
`ManiSoft/work_dirs/demo/`，首次运行可能耗时数分钟。在纯 SSH/headless
服务器上，不应把该可选渲染 demo 的失败与 v15e headless 安装失败
混为一谈；优先以上一节的三项检查为准。

服务器长时间训练时，可显式指定解释器；
`AC-MPC/manisoft_port/scripts/run_*_detached.sh` 等后台脚本均以
`AC_MPC_PYTHON`（缺省 `python`）启动训练：

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
cd /root/autodl-tmp/AC-MPC/ManiSoft
python ../manisoft_port/scripts/smoke_manisoft_lqr_track_5mm.py
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
  --scenario /root/autodl-tmp/AC-MPC/ManiSoft/configs/demo_elastica_fast.yaml \
  --reference /root/autodl-tmp/AC-MPC/ManiSoft/work_dirs/random_reference_45d/reference.npz \
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

**各方案最佳结果**（完整命令与复验步骤见
`docs/manisoft_koopman_best_validation_commands.md`）：

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
调参时以 `docs/manisoft_koopman_best_validation_commands.md` 中的现成命令为准。

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
S=/root/autodl-tmp/AC-MPC/ManiSoft/configs/demo_elastica_fast.yaml
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
`waypoints_completed_mean=0.286`——属于早期小规模实验。当前推荐的从头训练
PPO-KMPC、v15e 配置及对照结果见第 6 章。

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

## 6. 当前在线主线：v15 structured PPO-KMPC

### 6.1 v15 与早期 BC-KMPC 的关系

第 5 章从 fixed-cost OSQP 专家出发，经过 BC、DAgger 和 PPO 精调，适合研究
“专家监督能否把 cost-map 带入可用区域”。后续实验发现，更强的结构化归纳
偏置可以让 KMPC cost-map 从零直接用 PPO 训练，因此
`scripts/train_manisoft_ppo_comparison.py` 的 v15 路线：

- 不加载 BC checkpoint；
- 冻结 history Koopman encoder 和 `A/B/C`；
- 随机初始化低维 structured cost-map 和独立 critic；
- 在真实 ManiSoft 三 waypoint 环境中训练 PPO；
- 同时训练同输入的 MLP actor 作为非 KMPC 对照。

因此新使用者要复现当前主要策略时，不需要先跑第 5 章全部 BC/DAgger 管线；
只要具备 history Koopman checkpoint 和 waypoint bank，即可直接训练或评估
第 6 章策略。第 5 章仍是专家数据和历史对照的权威说明。

### 6.2 v15e 的模型结构

v15e 使用 H=10、谱半径上限 0.95 的 history Koopman。当前状态经冻结 encoder
得到 lift，cost-map 输入为当前 lift 加三 waypoint/stage context，仅输出五个
有物理意义的正权重倍率：

```text
tip-x weight
tip-y weight
tip-z weight
shared 18D absolute-action weight
final-stage tip multiplier
```

五个网络输出经 `exp(log(8) * tanh(raw))` 映射到 `[1/8,8]`。非末端 42 个
物理状态维度只保留很小的固定基准权重；线性项不自由学习，而是由显式参考
构造：

```text
x_ref = current non-tip state + active waypoint tip xyz
u_ref = 0
p_x = -Q x_ref
p_u = -R u_ref
```

这使零初始化 cost-map 已经对应一个有效的末端参考跟踪器。与自由输出每个
horizon cost term 的 full actor 相比，v15 structured actor 只有 12,165 个
可训练 actor 参数，降低了 PPO 从零学习的难度。完整数学比较见
`docs/kmpc_v15e_vs_upstream_cost_and_solver.md`。

动力学预测 horizon 为 10。QP 在 normalized-delta 空间求解 180 个变量：

```text
d[0:10] ∈ [-1,1]^(10×18)
u[k] = u[-1] + max_delta * sum(d[0:k])
u[k] ∈ [-0.30,0.30]
```

训练/部署固定展开 80 次 FISTA，另用 320 次求解计算诊断误差。每个控制周期
只执行第一步。零增量初值在物理上等价于“未来保持上一拍绝对动作”，对需要
非零维持激励的柔性系统比每次从零绝对动作开始更连续。

### 6.3 v15 六组对照与推荐模型

`scripts/launch_manisoft_ppo_compare_v15_zmixed_24h.sh` 在相同 Koopman、904
triplet z-mixed waypoint bank、seed 42 和 PPO 设置下启动六组实验：

| 名称 | actor | 主要差异 | 定位 |
| --- | --- | --- | --- |
| v15a | KMPC | range 8、lr 3e-5、std 0.15、delta 0.015 | v14 最强分支基线 |
| v15b | KMPC | actor lr 降至 2.5e-5 | 减少 KL 尖峰 |
| v15c | KMPC | std 降至 0.13 | 更少探索噪声 |
| v15d | KMPC | range 7、lr 3.5e-5 | range/lr 插值 |
| **v15e** | **KMPC** | **range 8、lr 3e-5、std 0.18、delta 0.0125** | **当前主线** |
| v15f | MLP | 256×256、直接输出绝对动作 | 非 KMPC 对照 |

其中 range 8 指 structured multiplier 范围 `[1/8,8]`。a/b/e 被继续训练，
现有记录约为 9.3–9.4M timesteps；e 的最后 PPO rollout 为 12/16 成功、平均
完成 2.625 个 waypoint。a 的 `last.pt` 在独立 v4 test20 上确定性评估为
60% 成功。e 随后成为 v4/v5 大规模离线数据的唯一行为 checkpoint，因此所有
IQL provenance 校验都以 v15e `last.pt` 的 SHA256 为准。

当前建议：

- 部署、继续采集数据或作为 IQL behavior baseline：使用 **v15e `last.pt`**；
- 复核已有独立 test20 结果：使用 v15a `last.pt`；
- 研究 checkpoint 选择：同时评估某一配置的 `best.pt` 和 `last.pt`；
- 不要依据单个 rollout 中偶然出现的 `1.0` success rate 宣称模型最佳，因为
  当轮可能只完成了 1–6 个 episode。

### 6.4 单独复现 v15e 训练

下面命令等价于 launcher 中的 v15e 单项。`K`、`S`、`W` 分别是 history
Koopman、场景和训练 waypoint bank：

```bash
K=work_dirs/manisoft_koopman_history_h10_abs_rho095_seed42_20260811/koopman_history/best_validation.pt
S="$MANISOFT_ROOT/configs/demo_elastica_fast.yaml"
W=data/processed/manisoft_waypoint_bank_v2_zmixed_merged

"$AC_MPC_PYTHON" -u scripts/train_manisoft_ppo_comparison.py \
  --actor ppo_kmpc \
  --koopman-checkpoint "$K" --scenario "$S" --waypoint-root "$W" \
  --output runs/manisoft_ppo_compare_v15_reproduction/v15e_seed42 \
  --episode-steps 300 --absolute-action-limit 0.30 \
  --progress-reward-scale 1.0 --horizon 10 \
  --kmpc-cost-parameterization structured --kmpc-hidden-dims 128 \
  --structured-log-scale 2.0794415416798357 \
  --solver-iterations 80 --solver-diagnostic-iterations 320 \
  --normalized-delta-curvature 0 --max-delta 0.0125 \
  --total-timesteps 100000000 --rollout-steps 4096 --num-envs 16 \
  --parallel-env-processes --minibatch-size 512 --update-epochs 4 \
  --learning-rate 1e-4 --actor-learning-rate 3e-5 \
  --std-learning-rate 1e-6 --freeze-log-std --no-anneal-learning-rate \
  --initial-action-std 0.18 --minimum-action-std 0.001 \
  --maximum-action-std 0.20 --gamma 0.99 --gae-lambda 0.95 \
  --clip-range 0.2 --clip-value-loss --value-coefficient 0.5 \
  --entropy-coefficient 1e-4 --max-grad-norm 0.5 \
  --target-kl 0.02 --kl-soft-stop-multiplier 1.5 \
  --kl-hard-rollback-multiplier 3.0 --normalize-advantages-globally \
  --checkpoint-interval-updates 50 --max-wall-time-hours 24 \
  --device cuda --seed 42
```

16 个环境分别在 spawn 子进程中运行 CPU 仿真，GPU 对 batch observation 做
统一 policy inference。服务器实测这种布局比单进程顺序仿真快；CPU 核数不足
时减少 `--num-envs`，并确保 `rollout-steps` 能被 `num-envs` 整除。

继续训练必须传入与原运行完全相同的 runtime、PPO signature、Koopman hash、
waypoint-bank hash 和 seed，仅允许增加墙钟预算：

```bash
# 在上面的完整命令末尾替换/增加：
--resume runs/manisoft_ppo_compare_v15_reproduction/v15e_seed42/last.pt \
--max-wall-time-hours 48
```

不能只写一个缩短版 resume 命令；训练器会拒绝与 checkpoint 不一致的参数。

### 6.5 `best.pt`、`last.pt` 与训练日志

每个 PPO 运行目录包含：

```text
run_config.json       完整参数、hash、动作语义和可训练参数量
history.jsonl         每个 PPO update 的训练/rollout 指标
training_status.json  最近一次 update 快照
best.pt               按 rollout score 保存的 checkpoint
last.pt               最后一个安全 checkpoint，可用于 resume
```

PPO 的 `best.pt` 按以下字典序 score 选择：

```text
(completed_success_rate,
 waypoints_completed_mean,
 completed_episode_return_mean)
```

由于一个 rollout 中完整 episode 数会变化，这个 score 适合训练期留档，但不是
无偏模型选择。正式选择必须调用第 9 章的确定性 evaluator。v15e 离线数据使用
`last.pt`，因此若要完全复现数据 provenance，不能擅自替换成 `best.pt`。

训练时重点监控：

- `completed_success_rate`、`waypoints_completed_mean`、`distance_minimum`；
- `action_bound_rate` 和 `action_clip_saturation_rate`，正常应接近零；
- `projected_gradient_residual_mean` 和 80/320 次首动作差异；
- `approx_kl`、`ppo_early_stopped`、`ppo_kl_hard_rollbacks`；
- `applied_delta_action_abs_max`，v15e 不应超过 0.0125。

### 6.6 v16 source ablation

v16 不是新推荐模型，而是相对 v15e 的三个单因素消融：

- Q2：显式 `-Q*x_ref` 改为自由学习的隐式线性项；
- Q3：normalized-delta QP 改为直接 absolute-action box QP；
- Q8：关闭最后 stage 的 learned terminal multiplier。

入口为 `scripts/launch_manisoft_ppo_compare_v16_source_ablation.sh`，设计和公式见
`docs/kmpc_v15e_vs_upstream_cost_and_solver.md`。当前运行记录显示 Q2/Q3 出现
很大的 KL 和零成功，Q8 也未达到 v15e 水平；这些是研究证据，不应作为交付
默认 checkpoint。

## 7. 用 v15e 构建离线 KMPC 数据集

### 7.1 数据语义

`scripts/collect_manisoft_kmpc_offline_dataset.py` 加载 PPO-KMPC checkpoint，
在三 waypoint 环境中保存 transition-complete episode。默认按 checkpoint 的
截断 Normal 分布随机采样动作；加 `--deterministic` 才只执行 KMPC mean。

标准字段为：

```text
observations          687D history/waypoint observation
actions               18D normalized delta，策略实际接收的动作
rewards
next_observations
terminals             完成三个 waypoint
timeouts              达到 episode_steps
episode_ids
```

另外保留：

```text
behavior_action_means
behavior_log_probabilities
value_estimates
requested_absolute_actions
applied_actions
applied_delta_actions
waypoint/stage diagnostics
projected-gradient residuals
```

其中 `actions` 不是绝对肌肉激活。v15e 的物理动作由上一绝对动作和
`max_delta=0.0125` 重建。IQL actor likelihood 和部署均使用同一个
state-dependent truncated Normal 支持集。

### 7.2 单进程采集与中断恢复

```bash
V15E=runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt

"$AC_MPC_PYTHON" scripts/collect_manisoft_kmpc_offline_dataset.py \
  --checkpoint "$V15E" \
  --waypoint-root data/processed/manisoft_waypoint_bank_v4_full_merged \
  --output data/processed/manisoft_kmpc_offline/example_v15e \
  --episodes 1000 --episode-steps 300 --device cuda --seed 42 \
  --allow-other-waypoint-bank
```

v15e checkpoint 记录的是 v2 z-mixed bank。用 v4/v5 bank 采集泛化数据时必须
显式使用 `--allow-other-waypoint-bank`；scenario SHA256 仍必须一致。采集器每
完成一个 episode 就写入独立 NPZ，因此崩溃后可在相同参数下加 `--resume`
继续，episode seed 不受中断位置影响。

大规模采集建议加 `--no-merged-dataset`，等所有分片完成后统一合并，避免每个
worker 都构造大压缩文件。

### 7.3 当前 v4/v5 八进程采集

仓库提供：

```bash
bash scripts/collect_kmpc_v4_1498_parallel.sh
bash scripts/collect_kmpc_v5_7109_parallel.sh
```

两者都用 v15e `last.pt`，各启动 8 个独立 worker。脚本里的 Python 路径是
当前服务器值；迁移机器时先替换为 `$AC_MPC_PYTHON`。不要让 worker 共享
`part_N` 目录。

分片完成后分别合并：

```bash
"$AC_MPC_PYTHON" scripts/merge_kmpc_offline_parts.py \
  --root data/processed/manisoft_kmpc_offline/v4_1498_stochastic \
  --parts 8

"$AC_MPC_PYTHON" scripts/merge_kmpc_offline_parts.py \
  --root data/processed/manisoft_kmpc_offline/v5_7109_stochastic \
  --parts 8
```

再合并两个来源：

```bash
"$AC_MPC_PYTHON" scripts/merge_multiple_kmpc_offline_roots.py \
  --roots \
    data/processed/manisoft_kmpc_offline/v4_1498_stochastic \
    data/processed/manisoft_kmpc_offline/v5_7109_stochastic \
  --parts 8 \
  --output \
    data/processed/manisoft_kmpc_offline/combined_v4_1498_v5_7109/dataset.npz
```

当前合并结果为：

| 来源 | episodes | transitions |
| --- | ---: | ---: |
| v4 full | 1,498 | 339,565 |
| v5 10k 中的有效 7,109 triplets | 7,109 | 1,600,628 |
| **合计** | **8,607** | **1,940,193** |

合并集随机闭环成功率 59.56%，平均回报 10.740，平均完成 2.507 个 waypoint。
这些是 behavior dataset 的组成统计，不是离线训练后 candidate 的评估结果。

### 7.4 provenance 校验

`collection_config.json` 和合并 `summary.json` 记录：

- behavior checkpoint 路径和 SHA256；
- scenario、waypoint bank SHA256；
- `max_delta` 和 action semantics；
- episode/transition 数、随机种子和成功统计。

IQL 初始化时会校验所有来源使用完全相同的 behavior checkpoint hash 和动作
语义。如果重新训练了一个“参数相同”的 v15e，它的 hash 仍会不同，不能在不
更新数据 provenance 的情况下替代原 behavior checkpoint。

## 8. 当前离线主线：structured-v2 KMPC-IQL

### 8.1 方法与部署形式

IQL 是独立离线训练路线，不修改现有 PPO 实现。训练时包含：

- twin action-value networks `Q1/Q2`；
- expectile value network `V`；
- exponentially advantage-weighted behavior cloning；
- frozen target-Q 和 behavior-support penalty 的离线 checkpoint score。

策略部署时仍然是 KMPC：

```text
687D observation
  → frozen history Koopman lift/A/B/C
  → learned structured-v2 cost map
  → finite-horizon normalized-delta QP
  → deterministic KMPC mean
```

Q/V 网络只用于训练，不进入 ManiSoft 部署推理。critic feature 额外包含最新
绝对动作，以消除 normalized-delta 可行域对 `u[t-1]` 的状态混叠：

```text
f_t = [z_t, waypoint/stage context, normalized u[t-1]]
```

### 8.2 structured-v2 与 v15e 的区别

v15e cost-map 输出 5 个权重；IQL 默认 candidate 为 structured-v2，共 11 个：

```text
tip xyz                         3
shape/orientation group         1
linear-velocity group           1
angular-velocity group          1
three muscle activation axes    3
final-tip multiplier            1
positive normalized-delta cost  1
total                          11
```

训练开始时先把 v15e 中兼容的 cost-map 行迁移到 candidate，再用数据中保存的
`behavior_action_means` 蒸馏 10,000 步。这一步解决 5-output v15e 和
11-output structured-v2 不能直接完整加载的问题。随后 Q/V 预热 20,000 步，
预热阶段 actor 冻结，之后才开始 IQL actor 更新。

### 8.3 数据加载与内存要求

当前压缩 `dataset.npz` 约 2.2 GiB，但所需字段展开约 10.2 GiB。第一次加载时
`OfflineTransitionDataset` 在数据旁创建只读 NPY cache 并 memory-map；之后
复用 cache，不再把两份 687D observation 全部读入 RAM。磁盘不足时使用
`--cache-dir /other/disk/cache-name`。

默认 timeout 继续 bootstrap，因为保存的每条 timeout transition 都有有效
`next_observations`。若实验需要严格 episodic horizon，可传
`--treat-timeouts-as-terminal`，但该设置会改变 dataset signature，resume 时
必须保持一致。

### 8.4 推荐训练命令

K=60 candidate：

```bash
DATASET=data/processed/manisoft_kmpc_offline/combined_v4_1498_v5_7109/dataset.npz
BEHAVIOR=runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt

"$AC_MPC_PYTHON" -u scripts/train_manisoft_kmpc_iql.py \
  --dataset "$DATASET" \
  --initial-policy-checkpoint "$BEHAVIOR" \
  --candidate-cost-parameterization structured_v2 \
  --candidate-solver-iterations 60 \
  --distillation-steps 10000 --critic-warmup-steps 20000 \
  --selection-behavior-mse-penalty 10 \
  --gradient-steps 500000 --batch-size 256 \
  --validation-batch-size 1024 --validation-interval 5000 \
  --checkpoint-interval 25000 --log-interval 100 \
  --output runs/manisoft_kmpc_iql_v2/k60_seed42 \
  --device cuda --seed 42
```

仓库 launcher 可分别启动 K=40 和 K=60 的配对实验：

```bash
screen -dmS iql_v2_k40_seed42 scripts/launch_manisoft_kmpc_iql_v2.sh 40
screen -dmS iql_v2_k60_seed42 scripts/launch_manisoft_kmpc_iql_v2.sh 60
```

launcher 会拒绝覆盖已有 `history.jsonl`/`last.pt`，并以 lock 文件防止同一路径
重复启动。迁移机器时需修改其中的仓库根目录和 Python 路径。

主要超参数为：

```text
expectile = 0.9
temperature = 0.1
max advantage weight = 100
discount = 0.99
target tau = 0.01
reward scale = 1
reward bias = 0
```

本数据已经是带正负号的 dense progress reward 和 waypoint bonus，因此默认
`reward_bias=0`；不要机械复制 AntMaze IQL 的 `-1` reward shift。

### 8.5 checkpoint、恢复与当前状态

IQL 运行目录包含：

```text
run_config.json
history.jsonl
last.pt
best_offline.pt
recovery_step_XXXXXXXX.pt
```

checkpoint 保存 candidate policy、Q/Q-target、V、所有 optimizer、RNG 状态、
精确 dataset signature、行为策略 provenance 和超参数。恢复必须复用原运行的
完整 training signature；下面是当前 K=60 launcher 的
恢复形式，只有 `--resume`、`--output` 和可增加的墙钟限制不参与 signature：

```bash
DATASET=data/processed/manisoft_kmpc_offline/combined_v4_1498_v5_7109/dataset.npz

"$AC_MPC_PYTHON" scripts/train_manisoft_kmpc_iql.py \
  --dataset "$DATASET" \
  --resume runs/manisoft_kmpc_iql_v2/k60_seed42/last.pt \
  --candidate-cost-parameterization structured_v2 \
  --candidate-solver-iterations 60 \
  --distillation-steps 10000 --critic-warmup-steps 20000 \
  --selection-behavior-mse-penalty 10 \
  --gradient-steps 500000 --batch-size 256 \
  --validation-batch-size 1024 --validation-interval 5000 \
  --checkpoint-interval 25000 --log-interval 100 \
  --output runs/manisoft_kmpc_iql_v2/k60_seed42 \
  --device cuda --seed 42
```

K=40 恢复时只把两处 `k60` 和 `--candidate-solver-iterations 60` 改为 40。
`--initial-policy-checkpoint` 从 resume payload 读取，不需要重复传入；其余参数
若与原 run 不一致会被明确拒绝。

`best_offline.pt` 使用预热结束时冻结的 target twin-Q 评分：candidate 相对
behavior 的 Q 提升减去 behavior-mean MSE penalty。它是同一次运行中的稳定
离线 proxy，不是环境成功率。最终仍必须闭环评估。

截至 2026-08-19，现有 K=40/K=60 运行分别到约 324k/251.9k IQL updates，均
未完成预定 500k，也尚未形成与 v15e 相同测试协议下的正式闭环结论。因此：

- v15e 仍是当前默认可用策略；
- IQL checkpoint 标记为 candidate/实验中；
- 不应因 `best_offline.pt` 分数更高就直接替换 v15e。

### 8.6 IQL 闭环评估

PPO 和 IQL 共用 evaluator，脚本会根据 checkpoint 的 `method` 自动选择 loader：

```bash
"$AC_MPC_PYTHON" scripts/evaluate_manisoft_ppo_comparison.py \
  --checkpoint runs/manisoft_kmpc_iql_v2/k60_seed42/best_offline.pt \
  --scenario "$MANISOFT_ROOT/configs/demo_elastica_fast.yaml" \
  --waypoint-root data/processed/manisoft_waypoint_bank_v2_zmixed_merged \
  --output runs/manisoft_kmpc_iql_v2/k60_seed42/evaluation_100ep \
  --episodes 100 --episode-steps 300 --device cuda --seed 42
```

对 K=40、K=60、v15e 和 v15a 必须使用完全相同的 waypoint schedule 才能比较。

## 9. 统一评估与最终模型选择

### 9.1 推荐协议

模型选择分三层：

1. **代码冒烟**：导入、单元测试、1–3 episodes，排除接口错误；
2. **开发评估**：固定 20–100 triplet 独立 bank，快速比较候选；
3. **正式评估**：未参与训练/采集的固定 bank，至少 100 episodes，多 seed，
   同一 schedule 比较全部候选。

正式报告至少包含：

- `success_rate`：三个 waypoint 全部完成；
- `waypoints_completed_mean`；
- `return_mean`、`episode_steps_mean`；
- `final_distance_mean`；
- `action_bound_rate`、绝对动作和 delta 统计；
- inference mean/p95 和 solver residual；
- checkpoint、scenario、waypoint manifest SHA256。

不要只报告训练 `history.jsonl` 最后一行，也不要把 behavior dataset 的成功率
当成确定性 policy evaluation。

### 9.2 公平比较命令模板

```bash
S="$MANISOFT_ROOT/configs/demo_elastica_fast.yaml"
W=data/processed/manisoft_waypoint_bank_v4_test20

for ITEM in \
  "v15a:runs/manisoft_ppo_compare_v15_zmixed_24h/a_kmpc_r8_lr3_std15_md015/last.pt" \
  "v15e:runs/manisoft_ppo_compare_v15_zmixed_24h/e_kmpc_r8_lr3_std18_md0125/last.pt" \
  "iql_k40:runs/manisoft_kmpc_iql_v2/k40_seed42/best_offline.pt" \
  "iql_k60:runs/manisoft_kmpc_iql_v2/k60_seed42/best_offline.pt"
do
  NAME=${ITEM%%:*}
  CKPT=${ITEM#*:}
  "$AC_MPC_PYTHON" scripts/evaluate_manisoft_ppo_comparison.py \
    --checkpoint "$CKPT" --scenario "$S" --waypoint-root "$W" \
    --allow-other-waypoint-bank \
    --output "runs/formal_compare/$NAME" \
    --episodes 100 --episode-steps 300 --device cuda --seed 4242
done
```

上述 shell 循环是协议模板；只有四个 checkpoint 都存在并与当前代码兼容时
才能直接执行。若正式 bank 与 checkpoint 记录的 bank 相同，可去掉
`--allow-other-waypoint-bank`。

### 9.3 结果文件

每个 evaluator 输出目录包含：

```text
summary.json         汇总指标、runtime 和所有 episode 摘要
trajectory_XXXX.npz  observation/action/reward/distance/stage 轨迹
```

排查失败时优先查看每个 waypoint 的 minimum distance：如果前两点都通过、
第三点失败，通常是远距离/长时域问题；如果第一个点都无法进入 5 mm，先检查
state normalization、场景 hash、动作语义和 checkpoint 是否匹配。

## 10. 代码导航

### 10.1 ManiSoft 侧

| 文件 | 作用 |
| --- | --- |
| `manisoft/backend/elastica.py` | 纯 Elastica 连续体动力学后端 |
| `manisoft/backend/manisoft_sim.py` | Elastica + MuJoCo 组合后端 |
| `manisoft/muscle.py` | 6×3 激活到杆单元力矩的 SplineMuscle |
| `manisoft/utils/math.py` | 45D Koopman 状态布局与旋转表示 |
| `scripts/collect_koopman_data.py` | coverage/rate/targeted-rate 原始轨迹采集 |
| `configs/demo_elastica_fast.yaml` | 当前主场景，0.2 ms 物理步长、50 Hz 控制 |
| `patches/pyelastica_local.patch` | 本项目必需的 PyElastica 兼容补丁 |

ManiSoft 自己也包含一套早期 `manisoft/control/`、`manisoft/koopman/` 和
`manisoft/envs/koopman_tracking.py`。当前 v15/IQL 的权威策略实现位于 AC-MPC；
不要混用两边同名的 tracking wrapper 或 checkpoint loader。

### 10.2 AC-MPC 侧

| 文件 | 作用 |
| --- | --- |
| `antmaze_ac/koopman/history_model.py` | history Koopman encoder 与线性动力学 |
| `antmaze_ac/envs/manisoft_tracking_env.py` | 45D 单点/三 waypoint ManiSoft 环境 |
| `antmaze_ac/envs/history_context_wrapper.py` | H=10、687D observation 和 delta action |
| `antmaze_ac/envs/process_vector_env.py` | 多进程 ManiSoft vector env |
| `antmaze_ac/rl/koopman_mpc_actor.py` | full/structured/structured-v2 KMPC 与 FISTA |
| `antmaze_ac/rl/manisoft_ppo_policies.py` | PPO-KMPC/MLP policy 构造和加载 |
| `antmaze_ac/rl/iql.py` | IQL Q/V、actor update 与 checkpoint loader |
| `antmaze_ac/data/offline_transition_dataset.py` | 大 NPZ cache/memmap loader |
| `scripts/train_manisoft_ppo_comparison.py` | v15/v16 from-scratch PPO 入口 |
| `scripts/collect_manisoft_kmpc_offline_dataset.py` | PPO-KMPC 离线 transition 采集 |
| `scripts/train_manisoft_kmpc_iql.py` | KMPC-IQL 训练入口 |
| `scripts/evaluate_manisoft_ppo_comparison.py` | PPO/IQL 共用确定性评估 |

配置、数据、代码和 checkpoint 都保存自己的语义字段和 hash。遇到加载拒绝时
应先阅读报错并核对 provenance，不建议临时删除校验。

## 11. 常见问题与交接检查

### 11.1 常见问题

**`import manisoft` 失败**

确认两个仓库安装在同一个 Conda 环境，并使用同一个解释器运行脚本。仅在
shell 中激活环境但后台脚本硬编码另一个 Python，是最常见原因。

**`pip check` 报 PyElastica/Numba/PyVista 或 `bpy` 冲突**

当前固定的 PyElastica 0.3.2 包元数据声明 `numba<0.58` 和
`pyvista<0.40`，但 ManiSoft 的已验证环境固定为 `numba==0.61.0` 和
`pyvista==0.45.2`；部分 Linux/headless 环境还会将 `bpy==4.4.0` 报为
platform warning。这是子模块上游 metadata 与 ManiSoft 集成版本的已知
差异，也是安装子模块时必须使用 `--no-deps` 的原因。

不要为了消除 `pip check` 文本而擅自降级 Numba/PyVista，否则可能破坏
ManiSoft 已验证的数值与渲染代码。对 v15e headless 主线，交付验收以
2.5 节的 import、两仓测试和单步 Elastica 冒烟全部通过为准。如果
`pip install` 本身失败，或 headless 冒烟未通过，则不属于可忽略的
metadata 告警。

**PyElastica 补丁无法应用**

先运行：

```bash
git -C "$MANISOFT_ROOT/third_party/pyelastica" apply --reverse --check \
  "$MANISOFT_ROOT/patches/pyelastica_local.patch"
```

若 reverse check 成功，说明补丁已经应用，不要再次 apply。应用补丁后顶层
`git status` 显示 `m third_party/pyelastica` 是预期现象。

**场景或 waypoint hash mismatch**

waypoint bank 是在特定 scenario 上认证的，不能绕过 scenario mismatch。
只有在有意测试另一份、但同 scenario 的 waypoint bank 时才使用
`--allow-other-waypoint-bank`。

**PPO resume 被拒绝**

必须复用完整 runtime/training signature。检查 `run_config.json`，不要只恢复
checkpoint 路径。尤其是 `max_delta`、solver iterations、structured range、
rollout 数、学习率、seed 和 waypoint manifest 都必须一致。

**IQL 首次加载像是卡住或占用大量磁盘**

第一次正在解压 NPZ 字段并建立 memmap cache。检查 cache 目录是否持续增长；
后续运行会直接复用。不要让两个首次构建进程同时写同一个 cache。

**KMPC 很慢**

确认 policy inference 在 GPU batch 上执行、环境使用
`--parallel-env-processes`，并限制每个 worker 的 BLAS 线程数。K=60 IQL
candidate 比 K=40 慢，v15 的 320 次 solver 只应用于诊断，不是执行动作的解。

**动作突然饱和或仿真发散**

先确认没有把 normalized delta 直接当成 absolute action；检查
`applied_delta_action_abs_max`、`action_bound_rate`、scenario hash 和 state
normalizer。v15e 的单维物理 delta 上限是 0.0125，绝对动作上限是 0.30。

### 11.2 PR/交接前检查清单

- [ ] AC-MPC 与 ManiSoft 的兼容分支或 commit 已在文档固定；
- [ ] ManiSoft 固定 submodule 已内含兼容修复，正常安装不再重复 `git apply`；
- [ ] AC-MPC 完整 artifact 环境 `206 passed`，或干净 clone
      `196 passed, 10 skipped`；ManiSoft `4 passed` 和 headless 单步冒烟通过；
- [ ] v15e 所需 Koopman、scenario、waypoint bank 的角色和目标路径明确；
- [ ] v15e 最小 artifact 压缩包已发布到 GitHub Release，本文已填
      Release URL、整包 SHA256 和解压命令；
- [ ] v15e `last.pt` 与离线数据中的 behavior checkpoint SHA256 一致；
- [ ] 数据集 episode/transition 数与 `summary.json` 一致；
- [ ] 至少完成一次 v15e 10-episode smoke；
- [ ] 正式结果明确区分 train rollout、behavior dataset 和 deterministic eval；
- [ ] IQL 若未完成 500k/闭环评估，继续标记为实验中；
- [ ] 所有服务器硬编码路径都已替换或在交接时说明。

当前代码测试基线为 206 tests，其中 10 项需要 `.gitignore` 排除的本地教师
artifact，并会在干净 clone 中明确跳过。涉及策略结构、动作语义、dataset
loader 或 checkpoint 格式的修改，必须至少重新运行全套测试，并对 v15e 做
短闭环 smoke。

## 12. 薄墙绕行、平滑教师与统一 SAC 归档（2026-08-28）

### 12.1 当前结论与适用范围

> [!IMPORTANT]
> **本章薄墙策略使用的是专门调整过的强弯曲软体臂，不是第 3–9 章旧
> Koopman 数据对应的 `demo_elastica_fast.yaml` 软体臂。** 当前正式物理参数为
> 半径 `45 mm`、杨氏模量 `2 MPa`、阻尼 `7`；旧 fast 场景分别为
> `50 mm`、`10 MPa`、`1`。同时，本任务将驱动力矩缩放提高到 `45`、绝对
> 激活上限提高到 `±0.60`。旧 Koopman 数据、旧 Koopman 模型和当前教师/SAC
> checkpoint 不得跨场景直接混用。完整差异和兼容边界见第 12.3 节。

这一分支已经得到一条从直立初态完整执行的、可重复回放的软体臂轨迹：软体臂
先向墙侧面偏转，越过墙的有限 x 向边缘，使 45% 的远端节点进入墙后区域；随后
在墙附近保持超过 0.30 m 的中段拱高，并使末端以低速回到 yz 面负 x 一侧约
2 cm。统一 residual SAC 的 12k checkpoint 在确定性直立回放中成功跟踪该教师，
而且没有发生虚拟墙或地面碰撞。

当前最后一个推块视频在上述真实 ManiSoft 软体臂动力学回放之外增加了一个
**接触校验后的视觉平移**：末端表面碰到落地立方体后，立方体沿负 x 方向显示
移动 5 cm。这个部分没有接触力、摩擦、质量或刚体动力学，不能作为动力学推物
成功率或推力能力的证据。

本节是独立任务归档，不替换第 6–9 章的 v15e/Koopman 三 waypoint 主线。
截至本节归档时：

- AC-MPC 代码位于 `manisoft-port` commit
  `89dbf0596fb368630845ca6994fb070c606923ec`；
- 本任务源码、配置和测试已进入该 commit；
- `data/experiments/` 中的教师、模型、轨迹和视频受 `.gitignore` 排除，普通
  `git clone` 不会获得，但第 12.9 节给出了已发布 Release 的下载、校验和解压
  命令；
- 实验 artifact 的 `run_config.json` 记录的生成时 Git HEAD 为 `b132742`，因为
  artifact 在源文件正式提交前生成；文件 SHA256 是核验 artifact 的权威依据。

### 12.2 坐标、墙体和终止定义

坐标原点是软体臂固定基座中心，初态中心线沿 `+z` 直立，基座与虚拟地面
`z=0` 同高。任务配置仍保留历史目标点 `[0.0, 0.65, 0.05]`，但当前最终任务
不要求到达这个 y/z 点；统一教师跟踪的关键空间终止量是末端到 yz 面的距离
`|tip_x|`、墙后远端比例、中段拱高和末端速度。

最终采用的基础几何为：

| 项目 | 当前值 | 解释 |
| --- | ---: | --- |
| 基座 | `[0.00, 0.00, 0.00]` m | 固定端；初态竖直向上 |
| 任务文件中的墙 | `x∈[-0.05,+0.05]` m | 墙的正 x 边界严格保持在 `+5 cm` |
| 墙的 y 范围 | `[0.27,0.29]` m | 2 cm 厚，位于基座前方 |
| 墙的 z 范围 | `[0.00,1.10]` m | 对当前 1 m 软体臂等效为竖直无限高 |
| 软体臂碰撞半径 | `0.045` m | 碰撞按中心线线段胶囊体计算 |
| 墙安全边距 | `0.010` m | 墙距离判定总 padding 为 55 mm |
| 地面 | `z=0` | 除安装端豁免外，臂体不得进入地下 |
| 地面数值容差 | `0.0005` m | 只吸收离散积分的亚毫米数值误差 |

墙不是 ManiSoft 中施加接触力的实体障碍，而是每个控制步计算的虚拟安全约束：

```text
wall_clearance = min(segment_to_wall_AABB_distance)
                 - arm_radius - wall_safety_margin

ground_clearance = min(non_mount_node_z)
                   - arm_radius - ground_surface_z
```

`wall_clearance < 0` 立即标记 `virtual_wall_collision`；
`ground_clearance < -0.0005` 标记 `ground_violation`。因此视频中的墙是相同几何
的可视化，安全性来自胶囊体/AABB 计算，不是 MuJoCo 接触反作用力。

当前统一 SAC 的终端条件必须同时满足：教师已走完、末端误差不超过 10 mm、
全臂节点 RMSE 不超过 25 mm、`|tip_x|≤25 mm`、远端越墙比例至少 40%、末端
速度不超过 `0.10 m/s`，并且墙附近拱高至少 `0.30 m`。配置文件中的
`target=[0,0.65,0.05]` 只继续用于尺度和旧 route 字段，不应误写成当前策略
最终到达的目标坐标。

### 12.3 当前软体臂动力学与旧 Koopman 场景不同（不可混用）

当前场景使用
`configs/manisoft_strong_bend_e2mpa_r45mm_damping7.yaml`：

#### 软体臂物理参数

| 参数 | 当前薄墙任务 | 早期 `demo_elastica_fast.yaml` |
| --- | ---: | ---: |
| 长度 / 单元数 | `1.0 m / 20` | 相同 |
| 半径 | `45 mm` | `50 mm` |
| 杨氏模量 | `2 MPa` | `10 MPa` |
| 阻尼常数 | `7` | `1` |
| 密度 | `1000 kg/m³` | 相同 |
| 物理步长 / 控制频率 | `0.0002 s / 50 Hz` | 相同 |

#### 驱动、动作与采集限制

以下不是杆体材料参数，但会直接改变可达动作和采集分布，因此也属于模型与
策略的兼容边界：

| 参数 | 当前薄墙任务 | 早期 Koopman/控制主线 |
| --- | ---: | ---: |
| `muscle_torque_scale` | `45` | `30` |
| 绝对激活上限 | `±0.60` | `±0.30` |
| SAC 单步物理动作变化 | `±0.003` | 不适用；旧 DAgger 默认 `±0.002` |
| 自由采集的末端速度终止 | 关闭：`maximum_tip_speed: null` | 旧数据按各自采集配置执行 |

圆截面杆的近似抗弯刚度满足 `EI∝E r⁴`，因此当前弹性抗弯刚度约为旧 fast
场景的 13.1%。同时驱动力矩缩放和动作范围都提高，当前动力学明显更软、驱动
权限更大，并用更高阻尼控制摆动。这不是只改变奖励或障碍物几何，而是改变了
被控动力系统本身。因此旧 824-episode Koopman 数据和由其辨识的线性动力学
不能直接当作当前环境模型；若要重新引入 AC-MPC，应当在当前
45 mm/2 MPa/damping-7 场景重新采集与辨识。

曾做过只把半径从 45 mm 改为 50 mm 的短暂消融：几何静态余量看似足够，但
`r⁴` 项使抗弯刚度增加约 52%，旧教师动作回放在 15.2 s 开始碰墙，终点也明显
漂移。该 50 mm 配置和测试记录随后按要求撤销；**当前正式配置仍是 45 mm**。

当前三份权威配置及 SHA256 为：

```text
920d615c5511996b2292e6a9b3feb694681142151b64a408302146a4f7b094ca  configs/manisoft_strong_bend_e2mpa_r45mm_damping7.yaml
99092f6916c4bfd53e34bded880e7eee9edd2d07205847a225bf186103ab803e  configs/manisoft_wall_route_collection_strong_bend_e2mpa_r45mm_t45_a060_wall_y027_x010.yaml
d689e75d829188fd161bfa392df3de42c1286d3c273851df3c3895208592c458  configs/manisoft_teacher_tracking_sac_silky_negx2cm_speed010.yaml
```

### 12.4 从低效自由采集到分阶段教师轨迹

早期方案是在无障碍自由运动数据中筛选“侧绕—远端越墙—前端回弯”的完整
episode。实践表明完整成功 episode 极稀疏，尤其是同时要求远端越墙比例、墙体
净空、地面净空和返回 yz 面时，纯随机采集效率很低。因此实现演进为：

1. 用受限平滑激励收集可安全接近和侧绕墙体的候选 episode；
2. 从 5%–30% 越墙状态建立左右镜像 snapshot bank；
3. 对 snapshot 主动制动，得到约 `0.066–0.125 m/s` 的低速起点；
4. 先用阶段 SAC 学到 40% 连续远端越墙，再从墙后 snapshot 优化返回 yz 面；
5. 将可行分段动作重放、平滑、重定时，并加入中段拱高与末端主动制动；
6. 把最终单条完整 episode 作为统一 residual SAC 的教师，而不是部署多个阶段
   policy 的切换器。

阶段式研究中曾得到以下诊断结果，但它们不是最终选择：

- 40% 越墙阶段的早期 checkpoint 在 12 个确定性 episode 中成功 2 个；
- 一个错误的联合 reward 能让末端接近 yz 面至约 2.1 mm，但同时丢失越墙比例，
  说明“回到 yz 面”和“保持远端越墙”必须联合约束；
- 40% 越墙并返回 yz 面 20 cm 范围的分层策略可在两个认证 snapshot 上成功，
  但这仍是 snapshot 起步的阶段策略；
- 更激进的 50%/约 14 cm yz 面距离策略出现 `0.25–1.17 mm` 的墙体穿透，未被
  选为安全策略。

这些失败结果说明困难不只是墙在 x 方向有多宽，还包括杆体连续性、近端必须
保持基座约束、远端弯折和中段拱高之间的耦合，以及动作变化率造成的响应滞后。

### 12.5 最终平滑教师是一条完整可用 episode

当前教师位于：

```text
data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/
  silky_negx2cm_speed010_teacher_v1/teacher_episode.npz
```

它从直立初态开始，包含 871 个 50 Hz 控制 transition，总时长 17.42 s；每一步
保存 45D 物理状态、21 个节点的位置/速度、20 个单元 director/omega、内部杆
状态、18D 动作和阶段编号。阶段语义为：

```text
0  直立接近并向有限墙边侧绕
1  平滑动作过渡
2  增加远端越墙比例
3  稳定墙后臂段
4  远端回到 yz 面附近
5  低速主动制动和终端接近
```

独立回放的关键指标为：

| 指标 | 教师结果 |
| --- | ---: |
| 控制步数 / 时长 | `871 / 17.42 s` |
| 最大单步动作变化 | `0.0021848` |
| 最小墙体净空 | `16.297 mm` |
| 最小地面净空 | `4.758 mm` |
| 最终远端越墙比例 | `45%` |
| 最终末端坐标 | `[-0.019823, 0.593944, 0.183425] m` |
| 最终末端到 yz 面距离 | `19.823 mm` |
| 最终末端速度 | `0.09208 m/s` |
| 最终墙附近拱高 | `0.38353 m` |
| 搜索分支最小拱高 | `0.37427 m` |

这里的“平滑”指动作已经连续重定时并取消中段长时间 hold，同时末端主动制动；
并不表示任意动力学扰动下的全局最优或完全无振荡。最大速度可以在中间阶段高于
终端速度阈值，`0.10 m/s` 只约束最终状态；当前没有速度超限提前终止。

教师文件大小约 6.9 MB，SHA256 为：

```text
b39b006a828e6876eb875993f82a933250d2c5f3210817e9ec7c56cc0ed56216
```

### 12.6 统一 residual SAC 的结构与最终结果

最终策略不是直接从目标点生成全新动作，而是在完整教师动作上学习闭环残差：

```text
requested_u = teacher_u[t] + 0.01 * clip(policy_output, -1, 1)
applied_u   = previous_u + clip(requested_u - previous_u, -0.003, 0.003)
applied_u   = clip(applied_u, -0.60, 0.60)
```

统一策略在所有阶段复用同一个网络。139D observation 为：

```text
当前 45D 物理状态                         45
上一步 18D 物理动作                       18
当前教师参考误差                           45
前视末端误差                                3
当前教师动作                               18
轨迹进度及剩余进度                          2
路径侧、越墙、净空、速度等安全特征            8
总计                                      139
```

训练使用两个环境、12k SAC timesteps，actor 网络为 `[512,512,256]`，先对 871
个教师 observation 做 100 epoch 行为克隆，并把确定性残差均值精确置零。前 4k
步冻结 actor 学习，随后用 `actor_anchor_coef=150` 约束其不偏离教师。随机训练
reset 可以从教师任意 snapshot 开始，同时保留 20% 直立起步概率；正式评估始终
从直立初态执行完整 871 步。

当前选择的是：

```text
data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/
  unified_silky_negx2cm_speed010_sac_12k_v1/checkpoints/
    unified_teacher_sac_12000_steps.zip
    unified_teacher_sac_vecnormalize_12000_steps.pkl
```

确定性直立回放结果为：

| 指标 | 12k SAC 结果 |
| --- | ---: |
| 终止状态 | `teacher_terminal_success` |
| 控制步数 | `871` |
| 最终远端越墙比例 | `45%` |
| 最终末端坐标 | `[-0.019817, 0.593933, 0.183530] m` |
| 最终末端速度 | `0.09203 m/s` |
| 轨迹中最大末端速度 | `0.22276 m/s` |
| 最小墙体净空 | `16.382 mm` |
| 最小地面净空 | `4.758 mm` |
| 最大节点跟踪 RMSE | `0.0898 mm` |
| 最大末端跟踪误差 | `0.1769 mm` |
| 最低受约束拱高 | `0.35636 m` |
| 最终拱高 | `0.38354 m` |

模型、normalizer 和标准回放视频的 SHA256 为：

```text
f4596cf3acfb183ec38ef6fbd074148e80343d6697379e2e99bdb87317f24819  unified_teacher_sac_12000_steps.zip
c2e965962276fec8d354529ad2153b348666086c65881cd0b4152543c60e78a0  unified_teacher_sac_vecnormalize_12000_steps.pkl
712e1858c07d84c140f09baf810576096e043080d4e6eb5175bddc0af768dc55  selected_12k_upright_replay.mp4
```

由于策略 observation 和动作执行显式依赖 `teacher_episode.npz`，这个 checkpoint
是**教师条件的闭环跟踪器**，还不是只给墙与目标就能自行规划新路线的通用策略。

### 12.7 最终写实视觉推块场景

最终归档视频为：

```text
data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1/
  unified_silky_negx2cm_speed010_sac_12k_v1/
    fixedwall_p05_grounded_cube_anchor_corner_realistic.mp4
```

该视频使用 1× 实时回放、写实材质和地面，不再放置桌子或小物块支撑平台。
可视墙保持正 x 边界 `+0.05 m`，并按要求只向负 x 方向额外延长 0.20 m，因此
最终可视墙范围为：

```text
x ∈ [-0.25,+0.05] m
y ∈ [ 0.27, 0.29] m
z ∈ [ 0.00, 1.10] m
```

小方块是边长 0.19 m 的正方体，直接落地；初始中心为
`[-0.11,0.67,0.095] m`，最终中心为 `[-0.16,0.67,0.095] m`。因此其
`(+x,-y,+z)` 角从 `[-0.015,0.575,0.190] m` 沿负 x 平移到
`[-0.065,0.575,0.190] m`，y、z 和方块尺寸保持不变。

视觉任务的校验量为：

| 指标 | 结果 |
| --- | ---: |
| 末端表面首次接触 | 第 `839` 步，`16.78 s` |
| 方块负 x 位移 | `0.050 m` |
| 接触后最大表面分离 | `1.812 mm`，小于 2 mm 容差 |
| 最小表面分离 | `-3.823 mm`（视觉几何重叠） |
| 扩展墙最小净空 | `16.382 mm` |
| 方块地面净空 | `0` |
| 平台 | 无 |

视频大小约 0.89 MB，SHA256 为：

```text
a291e8cc3a1e5e625641fdf93402845ae0b17cca00bd8878ffc518b0baf8fcc6
```

视觉平移的实现根据接触后的末端负 x 行程重新缩放方块 x 坐标；MuJoCo 中方块
是 `mocap` body 且禁用碰撞。因此“接触”经过末端球面/AABB 距离验证，但“移动
5 cm”是展示层约束，不由软体臂力矩、接触冲量或摩擦积分产生。

### 12.8 复现实验、评估和视频

先定义公共路径：

```bash
export ACMPC_ROOT=/root/autodl-tmp/AC-MPC/manisoft_port
export AC_MPC_PYTHON=/root/miniconda3/envs/manisoft/bin/python
export EXP="$ACMPC_ROOT/data/experiments/manisoft_strong_bend_e2mpa_r45mm_t45_a060_v1"
export SCENARIO="$ACMPC_ROOT/configs/manisoft_strong_bend_e2mpa_r45mm_damping7.yaml"
export TASK="$ACMPC_ROOT/configs/manisoft_wall_route_collection_strong_bend_e2mpa_r45mm_t45_a060_wall_y027_x010.yaml"
export TRAIN_CFG="$ACMPC_ROOT/configs/manisoft_teacher_tracking_sac_silky_negx2cm_speed010.yaml"
export TEACHER="$EXP/silky_negx2cm_speed010_teacher_v1/teacher_episode.npz"
export RUN="$EXP/unified_silky_negx2cm_speed010_sac_12k_v1"
cd "$ACMPC_ROOT"
```

从已有教师重新训练统一 SAC：

```bash
"$AC_MPC_PYTHON" scripts/train_manisoft_teacher_tracking_sac.py \
  --scenario "$SCENARIO" \
  --task-config "$TASK" \
  --teacher-episode "$TEACHER" \
  --config "$TRAIN_CFG" \
  --output "$EXP/unified_silky_negx2cm_speed010_sac_reproduction" \
  --device cuda
```

确定性直立评估：

```bash
"$AC_MPC_PYTHON" scripts/evaluate_manisoft_teacher_tracking_sac.py \
  --scenario "$SCENARIO" \
  --task-config "$TASK" \
  --teacher-episode "$TEACHER" \
  --config "$TRAIN_CFG" \
  --model "$RUN/checkpoints/unified_teacher_sac_12000_steps.zip" \
  --vecnormalize "$RUN/checkpoints/unified_teacher_sac_vecnormalize_12000_steps.pkl" \
  --output "$RUN/reproduction_evaluation.json" \
  --device cpu
```

渲染标准策略轨迹：

```bash
"$AC_MPC_PYTHON" scripts/render_manisoft_teacher_tracking_sac.py \
  --scenario "$SCENARIO" \
  --task-config "$TASK" \
  --teacher-episode "$TEACHER" \
  --config "$TRAIN_CFG" \
  --model "$RUN/checkpoints/unified_teacher_sac_12000_steps.zip" \
  --vecnormalize "$RUN/checkpoints/unified_teacher_sac_vecnormalize_12000_steps.pkl" \
  --output "$RUN/reproduction_upright_replay.mp4" \
  --playback-speed 1.0 --device cpu
```

重新生成无平台、落地方块、5 cm 视觉推块视频：

```bash
"$AC_MPC_PYTHON" scripts/render_manisoft_terminal_visual_push.py \
  --scenario "$SCENARIO" \
  --task-config "$TASK" \
  --teacher-episode "$TEACHER" \
  --config "$TRAIN_CFG" \
  --model "$RUN/checkpoints/unified_teacher_sac_12000_steps.zip" \
  --vecnormalize "$RUN/checkpoints/unified_teacher_sac_vecnormalize_12000_steps.pkl" \
  --output "$RUN/reproduction_grounded_cube_push.mp4" \
  --playback-speed 1.0 \
  --wall-negative-x-extension 0.20 \
  --platform-y-center 0.67 \
  --cube-size 0.19 --cube-initial-x -0.11 \
  --maximum-push-distance 0.05 --required-push-distance 0.05 \
  --visual-push-target-distance 0.05 \
  --no-platform --realistic-scene --clean-render --device cpu
```

输出脚本拒绝覆盖已有文件；复现时必须使用新的输出名，或在明确确认后移动旧
文件。完整教师的生成经历多轮搜索、重定时和制动，不建议只凭最终摘要从零重做；
交接时应直接保存教师 NPZ 并先核对 SHA256。

### 12.9 代码导航、验证和 artifact 交接

| 文件 | 本任务中的作用 |
| --- | --- |
| `antmaze_ac/data/wall_route_episodes.py` | 胶囊体墙距、地面净空、越墙比例和 episode schema |
| `antmaze_ac/data/wall_crossing_snapshot_bank.py` | snapshot bank 校验和加载 |
| `antmaze_ac/envs/manisoft_wall_crossing_sac_env.py` | 阶段越墙/返回 SAC 环境与安全语义 |
| `antmaze_ac/envs/manisoft_teacher_tracking_sac_env.py` | 139D 统一教师 residual SAC 环境 |
| `scripts/collect_manisoft_wall_route_candidates.py` | 初始候选 episode 收集 |
| `scripts/build_manisoft_wall_crossing_snapshot_bank.py` | 从候选轨迹建立越墙 snapshot |
| `scripts/stabilize_manisoft_wall_crossing_snapshot_bank.py` | snapshot 主动制动 |
| `scripts/search_manisoft_arched_return.py` | 中段拱高、末端回面与低速搜索 |
| `scripts/generate_manisoft_smooth_wall_teacher.py` | 平滑教师构建与认证 |
| `scripts/retime_manisoft_teacher_hold.py` | 删除长 hold、重定时与动作平滑 |
| `scripts/train_manisoft_teacher_tracking_sac.py` | BC 初始化及统一 SAC 训练 |
| `scripts/evaluate_manisoft_teacher_tracking_sac.py` | 直立确定性验证 |
| `scripts/render_manisoft_teacher_tracking_sac.py` | 标准策略视频 |
| `scripts/render_manisoft_terminal_visual_push.py` | 写实墙、落地方块与视觉推块 |

本次代码归档前运行了以下针对性测试：

```bash
"$AC_MPC_PYTHON" -m pytest -q \
  tests/test_wall_route_episodes.py \
  tests/test_manisoft_wall_crossing_sac.py \
  tests/test_manisoft_teacher_tracking_sac.py
```

结果为 `18 passed`。这只覆盖本任务新增模块，不替代前文要求的全仓回归测试。

最小 artifact 交接至少需要：

1. `teacher_episode.npz`；
2. `unified_teacher_sac_12000_steps.zip`；
3. `unified_teacher_sac_vecnormalize_12000_steps.pkl`；
4. 三份 SHA 固定的 YAML 配置；
5. 对应 `run_config.json` 和 `selected_12k_upright_replay.json`；
6. 如需展示，再附标准回放或最终写实视频。

这些 artifact 总量远小于本地 23 GB 的完整实验目录。它们已经打包为以下公开
Release；Release 仓库使用交接分支的 fork，是因为当前账号对上游
`yuej0422-dev/AC-MPC` 只有读取权限：

- Release：[`wall-bypass-sac-artifacts-20260828`](https://github.com/bright-moon-67/AC-MPC/releases/tag/wall-bypass-sac-artifacts-20260828)
- asset：`acmpc-manisoft-wall-bypass-minimal-20260828.tar.gz`
- 大小：`27,772,548` bytes
- 整包 SHA256：
  `431e94f59f3758febea1187f8ebc65f9600ff5041cab6c25c90b3925118a5803`

从上游仓库克隆代码并建立隔离布局：

```bash
cd /root/autodl-tmp
git clone --recurse-submodules https://github.com/yuej0422-dev/AC-MPC.git
bash AC-MPC/manisoft_port/scripts/bootstrap_isolated_layout.sh
```

在 PR 合并前做评审或复现时，应改为直接克隆包含本章代码的交接分支：

```bash
cd /root/autodl-tmp
git clone --branch manisoft-wall-handoff --recurse-submodules \
  https://github.com/bright-moon-67/AC-MPC.git AC-MPC
bash AC-MPC/manisoft_port/scripts/bootstrap_isolated_layout.sh
```

下载、校验并在 AC-MPC 仓库根目录解压 artifact：

```bash
export WORKSPACE=/root/autodl-tmp
export ACMPC_REPO="$WORKSPACE/AC-MPC"
export WALL_RELEASE_TAG=wall-bypass-sac-artifacts-20260828
export WALL_RELEASE_ASSET=acmpc-manisoft-wall-bypass-minimal-20260828.tar.gz
export WALL_ARTIFACT="$WORKSPACE/$WALL_RELEASE_ASSET"

curl --fail --location \
  --output "$WALL_ARTIFACT" \
  "https://github.com/bright-moon-67/AC-MPC/releases/download/$WALL_RELEASE_TAG/$WALL_RELEASE_ASSET"

echo "431e94f59f3758febea1187f8ebc65f9600ff5041cab6c25c90b3925118a5803  $WALL_ARTIFACT" \
  | sha256sum --check
tar -xzf "$WALL_ARTIFACT" -C "$ACMPC_REPO"

cd "$ACMPC_REPO"
sha256sum --check WALL_BYPASS_SHA256SUMS.txt
```

压缩包保持 `data/experiments/...` 的仓库相对路径，并内含教师 NPZ、12k SAC
checkpoint、对应 VecNormalize、运行/评测 JSON、标准回放视频和最终写实展示视频。
`bootstrap_isolated_layout.sh` 建立 `manisoft_port/data -> ../data` 后，第 12.8 节的
`$TEACHER`、`$RUN` 路径可以直接使用。配置 YAML 和评测/渲染脚本属于 Git
版本控制，不在 artifact 包内重复存放。

> [!WARNING]
> Release 只能与本章的强弯曲配置一起使用：半径 `45 mm`、杨氏模量 `2 MPa`、
> 阻尼 `7`、驱动力矩缩放 `45`、绝对激活上限 `±0.60`。不要与旧
> `demo_elastica_fast.yaml` 的 Koopman 模型或数据混用。

### 12.10 已知限制与下一步

- **不是通用路径规划器。** 当前 SAC 需要固定教师的逐步参考、教师动作和轨迹
  进度；墙或目标位置变化后，需要新教师或新的条件化策略。
- **墙体是虚拟约束。** 当前验证证明轨迹几何不穿墙，但没有模拟墙接触后的力学
  响应；这对安全轨迹足够，对接触容错或擦墙控制不够。
- **推块不是动力学任务。** 当前 5 cm 位移只证明末端表面可以沿现有轨迹进入
  适合展示的接触区域，不能证明真实方块在给定摩擦和质量下会移动 5 cm。
- **泛化尚未评估。** 目前关键结论来自一个认证教师和一次固定 seed 的确定性
  直立回放，没有墙位置、材料参数、初态扰动或多 seed 鲁棒性统计。
- **终端 y/z 不是目标约束。** 当前只要求末端回到 yz 面附近且位于墙后；若恢复
  “按下指定按钮”的目标，应加入目标表面法向、接触区域、按压行程/保持时间。
- **旧 Koopman 不匹配。** 若要回到 AC-MPC 预测控制主线，应先在当前强弯曲
  参数下重新采集系统辨识数据，而不是直接复用 `demo_elastica_fast.yaml` 模型。

当前教师和 12k checkpoint 已通过第 12.9 节的 Release 固定。最合理的后续顺序
是：先在小范围初态/墙位扰动上做确定性与随机评估；然后再决定是训练墙/目标条件化
策略，还是加入真实 MuJoCo/软体杆接触与方块动力学，把视觉推块升级为物理推块。
