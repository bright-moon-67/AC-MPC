# AC-MPC + ManiSoft 项目说明

## 2. 环境配置

### 2.1 基本要求

- Linux（已在 Ubuntu 上验证）；
- Conda/Miniconda 和 Git；
- Python 3.11；
- 训练推荐使用支持 CUDA 的 NVIDIA GPU，数据生成和小规模测试可使用 CPU。

AC-MPC 依赖本项目修改后的 ManiSoft 仿真环境。两个仓库应放在同一父目录下，
不要用原生 ManiSoft 仓库替代 `acmpc-integration` 分支。

```text
workspace/
├── AC-MPC/
└── ManiSoft/
```

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

### 2.3 创建环境并安装依赖

两个仓库共用一个 Python 环境。先安装 ManiSoft 及其固定版本依赖，再安装
AC-MPC 的控制、测试和绘图依赖：

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

补丁只需应用一次。如命令提示补丁已应用，不要重复执行。

### 2.4 下载仿真资源

`assets/` 体积较大，未纳入 Git。下载并解压到 ManiSoft 仓库根目录：

```bash
cd ManiSoft
hf download JobsWei/ManiSoft --local-dir ./ \
  --repo-type dataset --include 'assets.tar'
tar -xf assets.tar
cd ..
```

完成后应存在 `ManiSoft/assets/`。`work_dirs/`、训练数据和模型权重也不在 Git 中，
按后续章节生成或下载。

### 2.5 安装验证

```bash
python -c "import manisoft, antmaze_ac; print('imports: OK')"
python -m pytest AC-MPC/tests -q
(cd ManiSoft && python scripts/demo.py)
```

仿真 demo 的输出位于 `ManiSoft/work_dirs/`。服务器长时间训练时，可显式指定解释器：

```bash
export AC_MPC_PYTHON="$(command -v python)"
```

## 3. 数据集收集

### 3.1 收集流程

Koopman 数据的基本收集流程如下：

1. 在 ManiSoft 中按指定激励生成 18 维绝对动作；
2. 以 50 Hz 推进仿真，记录 `(state, action, next_state)`；
3. 过滤触地、末端速度过大或位移越界的轨迹；
4. 将每个 episode 保存为独立 NPZ，同时写入 `metadata.json`；
5. 在 AC-MPC 中合并 episode，按 episode 划分 train/validation/test，
   并仅使用训练集计算归一化参数。

当前采集器输出 45 维物理状态：在软体机械臂的三个代表截面上，分别记录
位置、旋转 6D 表示、线速度和角速度。动作为 `6×3=18` 维肌肉激活量，
范围为 `[-0.30, 0.30]`。

### 3.2 激励类型

`ManiSoft/scripts/collect_koopman_data.py` 提供三种主要激励：

- `coverage`：最小加加速度转移、保持段和局部多正弦激励，用于覆盖常规状态与动作范围；
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
`--excitation` 和输出目录。多进程收集时，每个 worker 必须使用不同的
`--seed` 和 `--output-dir`，不得共享输出目录。

原始 episode 收集完成后，在 AC-MPC 根目录中进行合并和划分：

```bash
python scripts/build_manisoft_sequences.py \
  --config configs/manisoft_coll.yaml \
  --input-root ../ManiSoft/work_dirs/koopman_45d_seed42 \
  --expected-episodes 100 \
  --output data/processed/manisoft_45d_seed42
```

若数据由多个 worker 生成，对每个目录重复传入一次 `--input-root`。

转换后的默认数据格式为：

```text
state          = [45D physical_state_t, 18D previous_action]  # 63D
action         = current_action - previous_action             # 18D 增量动作
next_state     = [45D physical_state_t+1, 18D current_action] # 63D
current_action = 18D 绝对动作
```

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
`AC-MPC/data/processed/`。专家轨迹和三 waypoint 测试数据将在后续章节单独说明。

### 3.5 数据保存与复现

`ManiSoft/work_dirs/` 和 `AC-MPC/data/processed/` 均被 Git 忽略，不会随仓库自动下载。
对外交付时应单独归档核心数据，并保留：

- 数据集下载地址和 SHA256；
- `metadata.json` 中的仿真配置哈希、随机种子、激励类型和安全阈值；
- episode 数、transition 数及 train/validation/test 划分；
- 用于训练的精确数据版本，避免混用基础数据与追加数据。
