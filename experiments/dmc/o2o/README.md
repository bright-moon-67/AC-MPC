# DMC Cartpole offline-to-online AC-KMPC

本目录实现第一个受控的 offline-to-online 验证：在当前 DMC
`cartpole_swingup` 上，用 ExORL Proto 探索数据比较 MLP actor、AC-KMPC
structured actor 和 MPVE。实验单位是 **100,000 个在线真实环境步**；每个 episode
固定 1,000 步，因此 return 范围是 `[0, 1000]`。

> 口径边界：这里是受 REDQ、RLPD 和 Cal-QL 启发的 **style/controlled
> integration**，不是这些作者代码的逐行移植，也不是其论文 benchmark 的精确复现。
> 方法名用于标识本仓库中的受控消融，不应写成“官方 REDQ/RLPD/Cal-QL 结果”，也不应
> 直接与论文表格中的异环境数字比较。

## 固定协议

五组方法共享同一份离线数据、冻结 Koopman、训练 seed、SAC/REDQ Q-learning
核心和评估网格。所有 actor 和 critic 都接收同一个 frozen lifted state；当前
Cartpole 是 5 维标准化 observation 加 10 维 learned lift，共 15 维。critic 是
10-head、每个 head 两层 256-unit LayerNorm MLP，target 随机取 2 个 head 的最小值。

| 运行名 | 离线预训练 | 在线 replay | actor | MPVE |
| --- | --- | --- | --- | --- |
| `REDQ-Online` | 无 | 仅 online | tanh-Gaussian MLP | 无 |
| `RLPD-MLP` | 无 | ExORL/online 50:50 | tanh-Gaussian MLP | 无 |
| `Cal-RLPD-MLP` | Cal-QL-style 500k updates | ExORL/online 50:50 | tanh-Gaussian MLP | 无 |
| `Cal-RLPD-AC-KMPC` | Cal-QL-style 500k updates | ExORL/online 50:50 | differentiable AC-KMPC | 无 |
| `Cal-RLPD-AC-KMPC-MPVE` | 从配对 AC-KMPC 的 `offline.pt` 精确 fork | ExORL/online 50:50 | differentiable AC-KMPC | 有 |

其中：

- `RLPD-MLP` 按 RLPD 思路从头在线学习，不先做离线梯度更新；它和
  `REDQ-Online` 都先收集 5,000 个随机环境步。RLPD 的每一个 critic update
  minibatch 都严格是 128 个 offline + 128 个 online transition，而不是只保证整个
  UTD fused batch 的全局比例。
- `Cal-RLPD-*` 的离线阶段使用 finite-horizon episode MC return 作为 calibration
  lower bound 的 Cal-QL-style CQL penalty；这是本项目的受控实现，不等同于 Cal-QL
  官方代码和原论文所有细节。
- 在线阶段每个真实环境步执行 `UTD=20` 个 critic update，并只用最后一个 minibatch
  做一次 actor/temperature update。`REDQ-Online` 只抽 online replay，其余方法严格
  50:50 混合 ExORL 与 online replay。
- AC-KMPC 用冻结 Koopman 构造可微 QP，规划 horizon `H=20`，即 Cartpole
  `0.20 s`。controller 输出逐时刻对角二次项和线性项，QP 第一项动作作为
  tanh-Gaussian policy 的均值。
- MPVE 的 **总 TD horizon 是 10**：`1` 个 replay 中的真实 transition 加 `9` 个
  Koopman model transition，即 `0.10 s`，不是“真实一步再额外预测十步”。MPVE 只在
  online phase 使用，并且每个真实环境步恰好一次；同一环境步的前 19 个 critic
  update 仍是普通真实 transition TD。MPVE 是加权为 1 的辅助 critic loss，不替换
  普通 Bellman loss。
- Cartpole model rollout reward 使用从预测 physical observation 和 action 解析计算的
  DMC Swingup exact reward oracle，不使用 learned reward model。Koopman export 里即使
  带有 reward-head 相关产物，也不属于这项主实验的 MPVE reward 路径。

默认训练预算是 500,000 个离线 gradient update（仅 Cal 方法）和 100,000 个在线真实
environment step；online `UTD=20`、batch size `256`、discount `0.99`。完整默认值及
配置指纹规则见 [`config.py`](config.py)。collector 默认用 5 个 CPU environment worker
同步采样；这里的 `online_steps` 仍按**单条 transition**计数，而不是按 vector step
计数，因此并行执行不会偷偷放大 100k 样本预算。MPVE 必须从配对的
`Cal-RLPD-AC-KMPC/offline.pt` fork actor、critic、target critic、temperature、optimizer
和 RNG state；这样两条 structured online 分支在看到第一个真实 transition 前完全一致。

## ExORL Proto1M 数据身份

数据来自 ExORL 官方公开的 `cartpole/proto.zip`。官方压缩包包含 10,000 个完整
episode，也就是 10M transitions；本协议的 Proto1M 是按文件名排序后的前 1,000 个
完整 episode（`episode_000000_1000.npz` 到 `episode_000999_1000.npz`），而不是从
10M transition 中另做随机抽样。

| 产物 | SHA256 |
| --- | --- |
| 官方 `proto.zip` | `0384fa7c899777150335b0d602c6b3d945cdc856adbd81341d1eb8848854a042` |
| 1,000 个 source episode 的有序身份 | `1d61021bdc8d2b0d7a208ed5843ba699c247c4c84141bc0f9b4b28ff1c3f0c23` |
| canonical `transitions.npz` | `496282a6c028975f1ffe7944d8a2b6b8eac71ff2283a1af1c12cbea910fa9128` |

ExORL 每个 1,000-step episode 实际保存长度为 1,001。索引 0 是 reset dummy record；
有效 transition `i=1..1000` 必须对齐为
`observation[i-1], action[i], reward[i], discount[i], observation[i]`。转换器恰好跳过
1,000 个 dummy record，得到 1,000,000 个 transition。当前数据中 environment
discount 全部为 `1.0`；time-limit 边界仍保留 bootstrap discount，MC return 则在文件
episode 边界停止递推。记录 reward 与 Cartpole exact observation oracle 的最大绝对
误差是 `1.1920928955078125e-07`。

需要明确记录 version drift：这些文件由历史 ExORL/DMC 软件栈生成，ExORL 环境文件
没有固定当时 `dm_control` 的精确 git revision；本仓库当前在线环境是
`dm_control==1.0.44`、`mujoco==3.11.0`，control timestep `0.01 s`。因此离线
`next_observation` 以公开文件本身为真值，不用当前 simulator 重放覆盖；在线训练和评估
使用当前 DMC。推荐在结果中把协议标为
`ExORL-public-Proto1M/current-DMC-online-v1`，不要声称复现 ExORL 论文的原始 simulator
或其报告分数。

## 1. 数据转换

先用 ExORL 官方 [`download.sh`](https://github.com/denisyarats/exorl/blob/main/download.sh)
获得 `cartpole/proto.zip`，保留原始压缩包以便校验身份。以下命令只抽取协议要求的前
1,000 个 episode，并生成带 manifest 的 canonical transition 数据：

```bash
cd /root/autodl-tmp/AC-MPC

ARCHIVE=runs/o2o/data/exorl/cartpole/proto.zip
SOURCE=runs/o2o/data/exorl/cartpole/proto_1m
DATASET="$SOURCE/transitions.npz"

sha256sum "$ARCHIVE"
unzip -tq "$ARCHIVE"
mkdir -p "$SOURCE"
unzip -q "$ARCHIVE" 'buffer/episode_000???_1000.npz' -d "$SOURCE"

python -m experiments.dmc.o2o.dataset \
  --source-dir "$SOURCE/buffer" \
  --output "$DATASET" \
  --max-transitions 1000000 \
  --gamma 0.99

sha256sum "$DATASET"
```

转换器会检查 episode 连续性、dummy、shape、有限值、动作范围、discount 和 exact
reward parity。审计身份及转换参数写入
`runs/o2o/data/exorl/cartpole/proto_1m/transitions.manifest.json`。如果重新下载后任一
SHA 不匹配，不应与本协议结果合并。

## 2. 准备并训练冻结 Koopman

先把 canonical transition 数据确定性整理成现有 Koopman trainer 的 episode 格式：

```bash
python -m experiments.dmc.o2o.prepare_koopman \
  --dataset runs/o2o/data/exorl/cartpole/proto_1m/transitions.npz \
  --output-dir runs/o2o/data/koopman/CartpoleSwingup/proto_1m
```

`early/mid/late` 只是 episode id 的确定性分区（330/330/340），**不是** Proto policy 的
早中晚训练阶段。trainer 再按每个分区的 episode id modulo 10 做 8:1:1，总计
800 train、100 validation、100 test episode。`manifest.json` 同时绑定 canonical NPZ
SHA 和各分区 SHA。

训练 10 维 learned lift、`K=50`（0.50 s）的共享 Koopman：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m experiments.playground.train_koopman \
  --task CartpoleSwingup \
  --data-dir runs/o2o/data/koopman/CartpoleSwingup/proto_1m \
  --output-dir runs/o2o/koopman/CartpoleSwingup/proto_1m_lift10 \
  --lift-dim 10 \
  --k-step 50 \
  --batch-size 2048 \
  --max-windows 500000 \
  --validation-windows 10000 \
  --epochs 500 \
  --patience 40 \
  --learning-rate 3e-4 \
  --spectral-radius-limit 0.95 \
  --stability-reference-dt 0.04 \
  --seed 20260821
```

正式五组实验必须共享同一个通过验证的
`runs/o2o/koopman/CartpoleSwingup/proto_1m_lift10/best.npz`，训练期间保持冻结。不要把
此前由 PPO replay 训练的旧 Koopman 与 Proto1M 运行混入同一 aggregate。

该模型已在 Proto1M 严格 test split（100 个 episode、95,100 个不跨边界的
H=50 window）上与旧 PPO3M/lift10 模型做过同尺度评估。新模型的 K=50
weighted rollout NMSE 为 `0.03723`（旧模型 `0.12823`），exact-reward
rollout RMSE 为 `0.05499`（旧模型 `0.10762`）。完整 K=1/5/10/20/50
报告位于
`runs/o2o/evaluation/koopman/cartpole_proto1m_new_vs_old.json`，SHA256 为
`ad97eafbad70bcba34e262eabff90930c4dc87b23e38031366b0ca5215ca7b4c`。
可用以下 CPU-only 命令复算：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONPATH=. \
python -m experiments.dmc.o2o.evaluate_koopman \
  --data-dir runs/o2o/data/koopman/CartpoleSwingup/proto_1m \
  --model Proto1M_lift10=runs/o2o/koopman/CartpoleSwingup/proto_1m_lift10/best.npz \
  --model PPO3M_lift10=runs/playground/koopman/CartpoleSwingup/seed_20260812/full_3m_jax/best.npz \
  --batch-size 8192 \
  --output runs/o2o/evaluation/koopman/cartpole_proto1m_new_vs_old.json
```

## 3. 可恢复的五方法矩阵 runner（推荐）

正式多 seed 实验优先使用矩阵 runner。它默认三个 training seed、串行执行；通过
`--max-parallel` 可以让前四种初始方法并发，MPVE 的依赖就绪后也进入同一个并发池。
MPVE 只会在**同一 seed** 的
`Cal-RLPD-AC-KMPC/offline.pt` 已经原子写出后启动，因此不必等待该 AC-KMPC 分支完成
online phase。任一子进程失败后 runner 不再派发新任务，但不会杀死已启动任务。

先做 dry-run；它不导入 trainer、learner 或 DMC，也不启动任何训练/评估子进程：

```bash
python -m experiments.dmc.o2o.runner \
  --dataset runs/o2o/data/exorl/cartpole/proto_1m/transitions.npz \
  --koopman runs/o2o/koopman/CartpoleSwingup/proto_1m_lift10/best.npz \
  --root runs/o2o/matrix/cartpole_proto1m \
  --seeds 20260821,20260822,20260823 \
  --device cuda \
  --offline-updates 500000 \
  --online-steps 100000 \
  --online-utd 20 \
  --num-envs 5 \
  --env-workers 5 \
  --max-parallel 4 \
  --dry-run
```

审核 `matrix_manifest.json` 后，去掉 `--dry-run` 启动。SSH 会断开的机器应保护 **runner
本身**；子命令各有独立日志，动态 PID/argv/return code/timestamp 位于原子的
`matrix_status.json`：

```bash
ROOT=runs/o2o/matrix/cartpole_proto1m
mkdir -p "$ROOT"
nohup python -m experiments.dmc.o2o.runner \
  --dataset runs/o2o/data/exorl/cartpole/proto_1m/transitions.npz \
  --koopman runs/o2o/koopman/CartpoleSwingup/proto_1m_lift10/best.npz \
  --root "$ROOT" \
  --seeds 20260821,20260822,20260823 \
  --device cuda \
  --max-parallel 4 \
  >"$ROOT/runner.log" 2>&1 &
```

重复同一命令会严格检查 config/dataset/Koopman 身份：完成的 run 跳过，未完成且有
`latest.pt` 的 run 交给 `train.py` 精确恢复。五组全部完成后，runner 顺序评估各自
`latest.pt`，然后自动运行严格 aggregate 和 PNG/PDF plot。启动时的 git commit、分支、
dirty 摘要、训练源码逐文件 SHA、Python/Torch/CUDA 和子进程线程环境都会写入 manifest；
运行期间训练核心源码发生变化时，后续任务会 fail-fast。`CUDA_VISIBLE_DEVICES` 原样继承，
而 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 默认固定为 1。
同一个 root 由进程级排他锁保护；活跃 child 会继承该锁，所以 runner 即使恰好在启动
child 后异常退出，第二个 runner 也不能重复派发该任务。MPVE 在恢复、评估和正式聚合时
还会重新核对同 seed AC-KMPC `offline.pt` 的绝对路径与 SHA256。

`--max-parallel=1` 是保守默认值。提高并发数前应按模型显存实测；runner 不会修改或终止
机器上已有的 GPU 进程。

## 4. 手工训练五组方法

下面是一套单 seed 的完整受控运行。重复实验时只改 `SEED`，并让五组方法共享它；
每个目录存在 `latest.pt` 时命令会从完整 learner/replay/RNG checkpoint 恢复。

```bash
cd /root/autodl-tmp/AC-MPC

SEED=20260821
DATASET=runs/o2o/data/exorl/cartpole/proto_1m/transitions.npz
KOOPMAN=runs/o2o/koopman/CartpoleSwingup/proto_1m_lift10/best.npz
RUN_ROOT="runs/o2o/train/CartpoleSwingup/proto_1m/seed_${SEED}"

for METHOD in REDQ-Online RLPD-MLP Cal-RLPD-MLP Cal-RLPD-AC-KMPC; do
  python -m experiments.dmc.o2o.train \
    --method "$METHOD" \
    --dataset "$DATASET" \
    --koopman "$KOOPMAN" \
    --output-dir "$RUN_ROOT/$METHOD" \
    --seed "$SEED" \
    --device cuda \
    --offline-updates 500000 \
    --online-steps 100000 \
    --online-utd 20 \
    --cql-weight 0.01 \
    --eval-episodes 10
done
```

`Cal-RLPD-AC-KMPC` 完成自己的 500k 离线阶段时会保存专用 `offline.pt`。MPVE 从该文件
精确分叉；不要给它另行做一次随机初始化的离线预训练：

```bash
python -m experiments.dmc.o2o.train \
  --method Cal-RLPD-AC-KMPC-MPVE \
  --dataset "$DATASET" \
  --koopman "$KOOPMAN" \
  --output-dir "$RUN_ROOT/Cal-RLPD-AC-KMPC-MPVE" \
  --seed "$SEED" \
  --device cuda \
  --offline-updates 500000 \
  --online-steps 100000 \
  --online-utd 20 \
  --cql-weight 0.01 \
  --eval-episodes 10 \
  --initialize-from-offline "$RUN_ROOT/Cal-RLPD-AC-KMPC/offline.pt"
```

运行中会写 `run.json`、`metrics.jsonl`、`latest.pt` 和 `best.pt`。配置、dataset SHA、
Koopman SHA 和当前 DMC protocol 都写入 run/checkpoint；恢复时任何身份不一致都会拒绝
继续。5 个环境只在同步 autoreset 边界保存可恢复的 `latest.pt`，以免把未保存的 simulator
隐状态伪装成精确恢复；重启时还会把 checkpoint 之后可能残留的 metric 行原子截断。
`--smoke` 只用于链路测试，不得纳入正式结果。

## 5. 独立评估、聚合和绘图

训练内评估使用固定 reset seeds 的 10 个 deterministic episode：online step 0、1,000、
之后每 5,000 步直到 100,000。训练结束后，再对每个 `latest.pt` 做一次身份校验和固定
10-episode 独立评估：

```bash
for METHOD in REDQ-Online RLPD-MLP Cal-RLPD-MLP Cal-RLPD-AC-KMPC Cal-RLPD-AC-KMPC-MPVE; do
  python -m experiments.dmc.o2o.evaluate \
    --run-dir "$RUN_ROOT/$METHOD" \
    --checkpoint latest \
    --device cpu
done
```

跨 seed 严格聚合并绘图：

```bash
python -m experiments.dmc.o2o.aggregate \
  --root runs/o2o/train/CartpoleSwingup/proto_1m \
  --output runs/o2o/results/cartpole_proto1m.json

python -m experiments.dmc.o2o.plot \
  --aggregate runs/o2o/results/cartpole_proto1m.json \
  --output-prefix runs/o2o/results/cartpole_proto1m
```

严格聚合要求每条曲线到达 100k、五组使用相同 dataset/Koopman/DMC/shared config 和评估
grid。`--allow-incomplete` 仅可做运行中诊断，不能生成正式结论。主指标包括固定种子
evaluation return@100k、从 step 0 开始的 trapezoidal AUC、normalized AUC 和 cumulative
regret；统计推断轴是 training seed，输出 mean、sample standard deviation、SEM 和
Student-t 95% CI。单 seed 时标准差、SE 和 CI 均不可估，只能算 development
result；正式比较应按完全相同协议重复多个
training seed。

## 一手资料

- ExORL：[论文](https://arxiv.org/abs/2201.13425)、[官方代码和数据](https://github.com/denisyarats/exorl)、[下载脚本](https://github.com/denisyarats/exorl/blob/main/download.sh)、[episode/dummy/transition 格式](https://github.com/denisyarats/exorl/blob/main/replay_buffer.py)、[DMC wrapper](https://github.com/denisyarats/exorl/blob/main/dmc.py)
- RLPD：[论文](https://proceedings.mlr.press/v202/ball23a.html)、[官方实现](https://github.com/ikostrikov/rlpd)
- Cal-QL：[论文](https://arxiv.org/abs/2303.05479)、[官方实现](https://github.com/nakamotoo/Cal-QL)
- REDQ：[论文](https://arxiv.org/abs/2101.05982)、[作者实现](https://github.com/watchernyu/REDQ)
- DeepMind Control Suite：[论文](https://arxiv.org/abs/1801.00690)、[`dm_control` 官方仓库](https://github.com/google-deepmind/dm_control)
