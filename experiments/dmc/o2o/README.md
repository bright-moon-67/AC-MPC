# DMC Cartpole offline-to-online AC-KMPC

本目录在 DMC `cartpole_swingup` 上比较三条不含 Koopman 的标准 MLP baseline，与两条
包含冻结 Koopman/AC-KMPC 先验的方法。主协议先使用 **50,000 个在线真实环境步**；每个
episode 固定 1,000 步，return 范围为 `[0, 1000]`。

> 口径边界：`Cal-QL-Raw` 和 `RLPD-Raw` 尽量遵循作者公开实现中适用于本任务的算法与
> 超参数；DMC Cartpole 并不是两篇工作的原始 benchmark，因此这里属于
> **official-implementation-derived / official-style port**，不是官方分数或逐行复现。
> `Cal-RLPD-*` 是本项目的组合方法，更不能写成 Cal-QL 或 RLPD 官方算法结果。

## 五方法协议

| 运行名 | 离线阶段 | 在线阶段 | actor/critic 输入 | actor | MPVE |
| --- | --- | --- | --- | --- | --- |
| `Cal-QL-Raw` | Cal-QL | Cal-QL（继续 calibration/CQL） | 标准化 raw observation | MLP | 无 |
| `RLPD-Raw` | 无 | RLPD，offline/online replay 混合 | 标准化 raw observation | MLP | 无 |
| `Cal-RLPD-Raw` | Cal-QL-regularized RLPD | pure RLPD，offline/online replay 混合 | 标准化 raw observation | MLP | 无 |
| `Cal-RLPD-AC-KMPC` | Cal-QL-regularized RLPD | pure RLPD，offline/online replay 混合 | frozen lifted state | differentiable AC-KMPC | 无 |
| `Cal-RLPD-AC-KMPC-MPVE` | 配对 AC-KMPC `offline.pt` 的精确 fork | pure RLPD + MPVE | frozen lifted state | differentiable AC-KMPC | 有 |

前三个 baseline 的 `koopman` 身份必须为 JSON `null`，actor 和 critic 都只接收由离线
数据统计量标准化的 5 维原始 observation；命令行传入 `--koopman` 会被拒绝。后两种方法
共享一个冻结的 Koopman，当前 lifted state 为 5 维标准化 observation 加 10 维 learned
lift。正式 aggregate 会验证这个边界，防止把含 Koopman 的 MLP 错当作标准 baseline。

算法特定的 batch size、Q ensemble、UTD 和优化器参数允许不同，不能为了表面控制变量而
覆盖官方/官方式设置。跨方法只强制相同 dataset、DMC protocol、training seed 集、在线
transition 预算和 deterministic evaluation grid。完整解析参数与出处标签见
[`config.py`](config.py) 中的 method spec。

这里的具体算法边界不能只从 `Cal-*` 名字推断：

- `Cal-QL-Raw` 的 profile 是
  `exorl_cql_backbone_calql_standard_single_tanh_v1`，即适配 DMC 的 ExORL CQL backbone
  加 Cal-QL calibration；离线和在线阶段都启用 calibrated CQL。actor 和两个独立 Q 都
  使用 ExORL 的 `Linear-LayerNorm-Tanh-Linear-ReLU-output` 布局，所有 Linear 使用
  orthogonal weight/zero bias；策略 log-std 截断到 `[-10, 2]`。它刻意保留标准
  single-tanh Gaussian，没有复制 ExORL 先对 mean 做 tanh、再经 SquashedNormal tanh 的
  double-tanh compatibility quirk。其余关键设置是 raw-5、两层 1024-unit、batch 1024、
  每个状态 3 个 CQL proposal、CQL weight
  `0.01`、target `tau=0.01`、三组 learning rate 均为 `1e-4`，online `UTD=1`。target
  使用单个 next-policy action 且不做 entropy backup，actor 使用 `min(Q1,Q2)`，target
  entropy 为 `-1`，temperature 使用 Cal-QL 的 log-alpha objective，两个 Q-head loss 相加。
  特别是 single-action target 不等同于 Cal-QL 官方仓库默认的 max-over-K backup，因此该
  结果只能标为上述 task-matched port，不能简写成官方 Cal-QL 复现。
- `Cal-QL-Raw` 的 calibration lower bound 是文件 episode 内的 finite-horizon discounted
  return-to-go。在线 transition 必须等完整 1,000-step episode 结束、RTG 可确定后才写入
  replay 并补做对应更新；此时 50:50 mixed batch 的 offline/online 每一行都有有效 MC
  target，在线阶段继续 calibrated CQL，而不是切换到 RLPD。
- `RLPD-Raw` 完全不做离线梯度预训练；先收集 5,000 个随机在线 transition，再按 50:50
  混合 offline/online replay。它使用 raw-5、10-Q ensemble、随机 2-Q target subset、
  两层 256-unit 网络、batch 256、`tau=0.005`、三组 learning rate 均为 `3e-4`、online
  `UTD=20`、entropy backup、ensemble-mean actor objective、target entropy `-0.5` 和
  RLPD temperature objective。
- 三个 `Cal-RLPD-*` 的 profile 是
  `calql_regularized_rlpd_offline_then_rlpd_online_v1`：500k 离线更新阶段才加入 K=10、
  weight `0.01` 的 calibrated CQL regularizer；进入在线阶段后 **CQL/calibration loss 为
  零，使用 pure RLPD 更新**。它们在线均使用上述 RLPD 10-Q/UTD20/50:50 replay 语义；
  差别仅是 raw MLP、AC-KMPC structured actor，以及是否增加 MPVE auxiliary loss。离线
  500k 是 gradient-update 数量且每次取一个 batch，不应误读成 `500k × UTD20`。

AC-KMPC 规划 horizon `H=20`（`0.20 s`）。MPVE 总 TD horizon 为 10：一个 replay 中的
真实 transition 加九个 Koopman transition（总计 `0.10 s`）；model rollout reward 由
预测 physical observation 通过 Cartpole exact reward oracle 解析，不使用 learned
reward model。MPVE 分支必须从同 seed AC-KMPC 的 immutable `offline.pt` fork actor、
critic、target critic、temperature、optimizers 和 RNG state，保证在线分叉前一致。

## 在线预算、评估与调整

默认在线预算为 50k 个单条 transition。固定评估点是
`0, 1k, 2.5k, 5k, 10k, ..., 50k`，每点使用同一组 10 个 deterministic reset seeds。
25k 的 return/AUC 是预先注册的中期诊断点，**不是选择性停止某个方法的依据**。

当前唯一允许启动的 development 配置是：training seed `20260821`；root
`runs/o2o/matrix/cartpole_stratified1m_raw_dev1_v2`；上述
Proto-Stratified-1M dataset 和 lift10 Koopman SHA；Cal-QL/Cal-RLPD 离线预算 500k
gradient updates（`RLPD-Raw` 为 0，MPVE 精确 fork 而不重复计算）；五方法在线预算均为
50k transition；train device `cuda`、独立 final evaluation device `cpu`、
`max_parallel=4`。`Cal-QL-Raw` collector 为 1 env/1 worker，其余四种为 5 env/5 worker；
这些差异属于 method-specific recipe，不改变以单条真实 transition 计数的公共预算。三 seed
`20260821,20260822,20260823` 只用于 development 通过后的独立正式复验，不是当前矩阵。

- 允许因数值崩坏或基础设施故障终止异常 run，但不能把它报告成性能早停结果。
- 性能早停只能在相同 online step 对五方法和全部 seed 使用同一预注册准则，并同时停止；
  当前 50k 开发矩阵默认不自动性能早停。
- 若 50k 时学习曲线仍持续上升，只能把五方法和全部 seed 统一扩展到同一个新预算；不得
  只延长较弱或较强的方法。扩展后的正式 aggregate 从 run config 读取新终点，不硬编码
  100k，并重新要求完整公共 evaluation grid。源矩阵完整结束后，以原始 `--online-steps
  50000` 命令加 `--extend-online-steps N` 重新调用 runner；它会先校验五方法×全部 seed
  的 base run，先复制归档原 matrix manifest/status，再统一迁移。扩展调度可幂等重试，
  能识别同一矩阵中 base-complete、target-partial 与 target-complete 的混合中断状态；它
  不会覆盖第一次归档，也不会重复扩展已到目标预算的 run。身份缺失、目标预算不统一或
  单独绕过矩阵扩展都会 fail closed。
- runner 只管理自己的子进程，不会终止或修改机器上其他训练；单实例文件锁、原子状态、
  PID/argv/日志和 resumable checkpoint 用于保护 SSH 断线后的执行。

## ExORL Proto-Stratified-1M 数据身份

数据来自 ExORL 官方公开的 `cartpole/proto.zip`（10,000 个完整 episode、10M
transitions）。当前协议把时间顺序划成十个连续 1,000-episode block，并在每个 block
取局部编号 `0,10,...,990` 的 100 个 episode，共 1,000 episode/1M transitions。它保持
与 ExORL 1M 数据预算相同，同时避免旧版“只取最早 1,000 个 episode”的时间偏差。

| 产物 | SHA256 |
| --- | --- |
| 官方 `proto.zip` | `0384fa7c899777150335b0d602c6b3d945cdc856adbd81341d1eb8848854a042` |
| 1,000 个 source episode index 的有序身份 | `90b453df6379445e1d3c0a2ea9b67eeecd917f8e82b769c29b27025fb33f0343` |
| canonical `transitions.npz` | `681fc1424ee7a19f969e202f4bd2e7d09270202d9646638205dfaa229cdfaaec` |

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
`ExORL-public-Proto-Stratified-1M/current-DMC-online-v1`，不要声称复现 ExORL 论文的原始 simulator
或其报告分数。

## 1. 数据转换

先用 ExORL 官方 [`download.sh`](https://github.com/denisyarats/exorl/blob/main/download.sh)
获得 `cartpole/proto.zip`，保留原始压缩包以便校验身份。转换前先把上述 1,000 个选定
episode 解压到 `SOURCE/buffer`，再生成带 manifest 的 canonical transition 数据：

```bash
cd /root/autodl-tmp/AC-MPC

ARCHIVE=runs/o2o/data/exorl/cartpole/proto.zip
SOURCE=runs/o2o/data/exorl/cartpole/proto_stratified_1m
DATASET="$SOURCE/transitions.npz"

sha256sum "$ARCHIVE"
unzip -tq "$ARCHIVE"

python -m experiments.dmc.o2o.dataset \
  --source-dir "$SOURCE/buffer" \
  --output "$DATASET" \
  --max-transitions 1000000 \
  --gamma 0.99 \
  --temporal-stratified \
  --source-total-episodes 10000 \
  --temporal-deciles 10 \
  --episodes-per-decile 100 \
  --source-archive "$ARCHIVE"

sha256sum "$DATASET"
```

转换器会检查 episode 连续性、dummy、shape、有限值、动作范围、discount 和 exact
reward parity。审计身份及转换参数写入
`runs/o2o/data/exorl/cartpole/proto_stratified_1m/transitions.manifest.json`。如果重新下载后任一
SHA 不匹配，不应与本协议结果合并。

## 2. 准备并训练冻结 Koopman

先把 canonical transition 数据确定性整理成现有 Koopman trainer 的 episode 格式：

```bash
python -m experiments.dmc.o2o.prepare_koopman \
  --dataset runs/o2o/data/exorl/cartpole/proto_stratified_1m/transitions.npz \
  --output-dir runs/o2o/data/koopman/CartpoleSwingup/proto_stratified_1m
```

adapter 保留十个 `decile_00..09` 时间层，每层按 local episode id modulo 10 做
80/10/10，总计 800 train、100 validation、100 test episode；因此 train/validation/test
都覆盖 Proto collection 的十个阶段。`manifest.json` 同时绑定 canonical NPZ SHA、
selection contract 和各层 SHA。

训练 10 维 learned lift、`K=50`（0.50 s）的共享 Koopman：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m experiments.playground.train_koopman \
  --task CartpoleSwingup \
  --data-dir runs/o2o/data/koopman/CartpoleSwingup/proto_stratified_1m \
  --output-dir runs/o2o/koopman/CartpoleSwingup/proto_stratified_1m_lift10 \
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
`runs/o2o/koopman/CartpoleSwingup/proto_stratified_1m_lift10/best.npz`，训练期间保持
冻结。当前 artifact SHA256 为
`7d61b4b13417b70a9b51d55638d4437e05a018e1888af3ab19cbb0e2093e9edc`，best epoch 为
172；adapter manifest SHA256 为
`c881f47de60625efe3b6b09205fa80833981befccaf23caa1711624733368e07`。已经保存的严格
test-split K=1/5/10/20/50 主报告是
`runs/o2o/evaluation/koopman/cartpole_proto_stratified1m_lift10.json`，其 SHA256 为
`dfc3de285e101c869cd144c8de863064ab772bbaea935b677f18661375433241`。该报告覆盖十个
decile 的 100 个 test episode、95,100 个不跨 episode 的 H=50 window；主模型 aggregate
的 one-step NMSE 为 `0.00130942`，step-50 NMSE 为 `0.16853244`，weighted rollout
NMSE 为 `0.05545533`，exact-reward weighted RMSE 为 `0.05593628`。这些是冻结模型的
预测诊断，不是控制 return。旧 first-1M、PPO replay Koopman、旧 lifted-baseline pilot
及其他 diagnostic JSON 都不得替代这份主报告混入协议身份。

可用以下 CPU-only 命令复算到另一个文件；报告包含生成时间，因此只要求协议身份和数值
一致，不要求重跑文件与上述归档报告具有相同的逐字节 SHA：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 PYTHONPATH=. \
python -m experiments.dmc.o2o.evaluate_koopman \
  --data-dir runs/o2o/data/koopman/CartpoleSwingup/proto_stratified_1m \
  --model ProtoStratified1M_lift10=runs/o2o/koopman/CartpoleSwingup/proto_stratified_1m_lift10/best.npz \
  --batch-size 8192 \
  --output runs/o2o/evaluation/koopman/cartpole_proto_stratified1m_lift10.recomputed.json
```

## 3. 可恢复的五方法矩阵 runner（推荐）

当前先运行 `seed=20260821` 的完整五方法 **development matrix**，用于确认算法趋势和
50k 预算是否合适；它不是多 seed 统计结论。三 seed
`20260821,20260822,20260823` 仅在 development 通过后作为正式复验。矩阵 runner 通过
`--max-parallel` 可以让前四种初始方法并发，MPVE 的依赖就绪后也进入同一个并发池。
MPVE 只会在**同一 seed** 的
`Cal-RLPD-AC-KMPC/offline.pt` 已经原子写出后启动，因此不必等待该 AC-KMPC 分支完成
online phase。任一子进程失败后 runner 不再派发新任务，但不会杀死已启动任务。

先做 dry-run；它不导入 trainer、learner 或 DMC，也不启动任何训练/评估子进程：

```bash
python -m experiments.dmc.o2o.runner \
  --dataset runs/o2o/data/exorl/cartpole/proto_stratified_1m/transitions.npz \
  --koopman runs/o2o/koopman/CartpoleSwingup/proto_stratified_1m_lift10/best.npz \
  --root runs/o2o/matrix/cartpole_stratified1m_raw_dev1_v2 \
  --seeds 20260821 \
  --device cuda \
  --offline-updates 500000 \
  --online-steps 50000 \
  --max-parallel 4 \
  --dry-run
```

审核 `matrix_manifest.json` 后，去掉 `--dry-run` 启动。SSH 会断开的机器应保护 **runner
本身**；子命令各有独立日志，动态 PID/argv/return code/timestamp 位于原子的
`matrix_status.json`：

```bash
ROOT=runs/o2o/matrix/cartpole_stratified1m_raw_dev1_v2
mkdir -p "$ROOT"
nohup python -m experiments.dmc.o2o.runner \
  --dataset runs/o2o/data/exorl/cartpole/proto_stratified_1m/transitions.npz \
  --koopman runs/o2o/koopman/CartpoleSwingup/proto_stratified_1m_lift10/best.npz \
  --root "$ROOT" \
  --seeds 20260821 \
  --device cuda \
  --online-steps 50000 \
  --max-parallel 4 \
  >"$ROOT/runner.log" 2>&1 &
```

旧目录 `runs/o2o/matrix/cartpole_proto1m` 是所有方法都使用 lifted state 的 pilot，只保留
审计，不允许被新 aggregate 自动发现。此前的
`runs/o2o/matrix/cartpole_stratified1m_raw_v1` 只是三 seed dry-run 审计，不是当前启动
目标。重复同一命令会严格检查 config/dataset/Koopman
身份：完成的 run 跳过，未完成且有
`latest.pt` 的 run 交给 `train.py` 精确恢复。五组全部完成后，runner 顺序评估各自
`latest.pt`，然后自动运行严格 aggregate 和 PNG/PDF plot。启动时的 git commit、分支、
dirty 摘要、训练源码逐文件 SHA、Python/Torch/CUDA 和子进程线程环境都会写入 manifest；
运行期间训练核心源码发生变化时，后续任务会 fail-fast。`CUDA_VISIBLE_DEVICES` 原样继承，
而 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 默认固定为 1。
同一个 root 由进程级排他锁保护；活跃 child 会继承该锁，所以 runner 即使恰好在启动
child 后异常退出，第二个 runner 也不能重复派发该任务。MPVE 在恢复、评估和正式聚合时
还会重新核对同 seed AC-KMPC `offline.pt` 的绝对路径与 SHA256。fresh run 会先持久化
step-0 `latest.pt` 再执行初始评估，因此即使在评估中断，也不会留下“已有目录但没有可恢复
checkpoint”的半成品。

`--max-parallel=1` 是保守默认值。提高并发数前应按模型显存实测；runner 不会修改或终止
机器上已有的 GPU 进程。

## 4. 手工训练五组方法

下面是一套单 seed 的完整受控运行。重复实验时只改 `SEED`，并让五组方法共享它；
每个目录存在 `latest.pt` 时命令会从完整 learner/replay/RNG checkpoint 恢复。

```bash
cd /root/autodl-tmp/AC-MPC

SEED=20260821
DATASET=runs/o2o/data/exorl/cartpole/proto_stratified_1m/transitions.npz
KOOPMAN=runs/o2o/koopman/CartpoleSwingup/proto_stratified_1m_lift10/best.npz
RUN_ROOT="runs/o2o/matrix/cartpole_stratified1m_raw_dev1_v2/seed_${SEED}"

for METHOD in Cal-QL-Raw RLPD-Raw Cal-RLPD-Raw; do
  python -m experiments.dmc.o2o.train \
    --method "$METHOD" \
    --dataset "$DATASET" \
    --output-dir "$RUN_ROOT/$METHOD" \
    --seed "$SEED" \
    --device cuda \
    --offline-updates 500000 \
    --online-steps 50000 \
    --cql-weight 0.01 \
    --eval-episodes 10
done

python -m experiments.dmc.o2o.train \
  --method Cal-RLPD-AC-KMPC \
  --dataset "$DATASET" \
  --koopman "$KOOPMAN" \
  --output-dir "$RUN_ROOT/Cal-RLPD-AC-KMPC" \
  --seed "$SEED" \
  --device cuda \
  --offline-updates 500000 \
  --online-steps 50000 \
  --eval-episodes 10
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
  --online-steps 50000 \
  --cql-weight 0.01 \
  --eval-episodes 10 \
  --initialize-from-offline "$RUN_ROOT/Cal-RLPD-AC-KMPC/offline.pt"
```

运行中会写 `run.json`、`metrics.jsonl`、`latest.pt` 和 `best.pt`。配置、dataset SHA、
Koopman SHA 和当前 DMC protocol 都写入 run/checkpoint；恢复时任何身份不一致都会拒绝
继续。collector 按 method spec 使用 1 或 5 个环境，并只在同步 autoreset 边界保存可恢复的 `latest.pt`，以免把未保存的 simulator
隐状态伪装成精确恢复；重启时还会把 checkpoint 之后可能残留的 metric 行原子截断。
`--smoke` 只用于链路测试，不得纳入正式结果。

## 5. 独立评估、聚合和绘图

训练内评估使用固定 reset seeds 的 10 个 deterministic episode：online step 0、1,000、
2,500、5,000，之后每 5,000 步直到配置预算。训练结束后，再对每个 `latest.pt` 做一次身份校验和固定
10-episode 独立评估：

```bash
for METHOD in Cal-QL-Raw RLPD-Raw Cal-RLPD-Raw Cal-RLPD-AC-KMPC Cal-RLPD-AC-KMPC-MPVE; do
  python -m experiments.dmc.o2o.evaluate \
    --run-dir "$RUN_ROOT/$METHOD" \
    --checkpoint latest \
    --device cpu
done
```

runner 的真实结果命名是每个 run 下的 `evaluation_latest_10.json`，以及矩阵 root 下的
`results/cartpole_o2o.json`、`results/cartpole_o2o.png` 和
`results/cartpole_o2o.pdf`。手工重建同一结果时也应显式传入该矩阵的 manifest：

```bash
ROOT=runs/o2o/matrix/cartpole_stratified1m_raw_dev1_v2

python -m experiments.dmc.o2o.aggregate \
  --matrix-manifest "$ROOT/matrix_manifest.json" \
  --root "$ROOT" \
  --output "$ROOT/results/cartpole_o2o.json"

python -m experiments.dmc.o2o.plot \
  --aggregate "$ROOT/results/cartpole_o2o.json" \
  --output-prefix "$ROOT/results/cartpole_o2o"
```

严格聚合要求每条曲线到达其共同配置预算，五组使用相同 dataset/DMC/seed set/evaluation
grid，三个 raw baseline 的 Koopman 为 null，两个 structured 方法共享同一个 Koopman。
算法特定超参数可以跨方法不同；同一方法跨 training seed 则必须除 `seed` 外完全一致。
aggregate 还会用 `matrix_manifest.json` 核对五方法的精确 run directory、resolved config、
dataset/Koopman、预算和 training/result/runner 源码逐文件 SHA，并把 source identity 写回
结果。未提供 `--matrix-manifest` 时仍可生成诊断 JSON，但会明确写
`source_verified=false, formal_complete=false`，不得作为正式 complete aggregate。
`--allow-incomplete` 同样只可做运行中诊断。主指标包括固定种子 evaluation
return@budget、25k 中期 AUC、全预算从 step 0 开始的 trapezoidal AUC、normalized AUC
和 cumulative regret；统计推断轴是 training seed，输出 mean、sample standard
deviation、SEM 和
Student-t 95% CI。当前单 seed development 的标准差、SE 和 CI 均不可估，不能当作正式
统计结论；通过后再在独立 root 用完全相同协议运行三个 training seed。

## 一手资料

- ExORL：[论文](https://arxiv.org/abs/2201.13425)、[官方代码和数据](https://github.com/denisyarats/exorl)、[下载脚本](https://github.com/denisyarats/exorl/blob/main/download.sh)、[episode/dummy/transition 格式](https://github.com/denisyarats/exorl/blob/main/replay_buffer.py)、[DMC wrapper](https://github.com/denisyarats/exorl/blob/main/dmc.py)
- RLPD：[论文](https://proceedings.mlr.press/v202/ball23a.html)、[官方实现](https://github.com/ikostrikov/rlpd)
- Cal-QL：[论文](https://arxiv.org/abs/2303.05479)、[官方实现](https://github.com/nakamotoo/Cal-QL)
- DeepMind Control Suite：[论文](https://arxiv.org/abs/1801.00690)、[`dm_control` 官方仓库](https://github.com/google-deepmind/dm_control)
