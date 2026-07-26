# PPO 训练速度与迁移建议

## 两个不同的 batch

每个 PPO update 包含两个阶段：

1. rollout：与环境交互并收集 2048 条 transition；
2. PPO update：将这 2048 条数据打乱，按 minibatch 重算策略并反向传播 10
   epochs。

单环境 rollout 时，每一步只有一个 observation。由于
`gain_update_interval=1`，每一步都执行
`actor(x) -> Q,p -> DARE -> K,d -> delta_u`，因此原实现需要 2048 次串行的
batch-1 DARE。这里的 batch-1 与 PPO 的 `minibatch_size=64` 不是同一个概念。

当前实现支持 `num_envs=16`：一次对 16 个独立环境的当前状态批量求 DARE，
共做 128 个控制时刻，仍得到 2048 条 transition。每个环境每步都用自己的新
状态重新求解，不复用旧增益，也没有引入 feed-forward PPO 无法重建的隐藏
状态。

## RTX 4060 Laptop 实测

实测记录保存在
`runs/antmaze_umaze_local_speed/profile.json`。

| 配置 | 实测结果 |
|---|---:|
| 原单环境、minibatch 64 | 11.91 transitions/s（前 7 updates 平均） |
| 16 env rollout | 8.52 s / 2048 transitions |
| minibatch 256 PPO update | 132.44 s / 10 epochs |
| 优化后端到端 | 14.53 transitions/s |
| 实际整体加速 | 1.22x |

同一 rollout 上只测一个 PPO epoch：

| minibatch | samples/s | 峰值显存 |
|---:|---:|---:|
| 64 | 185.7 | 484 MiB |
| 128 | 199.2 | 915 MiB |
| 256 | 201.9 | 1784 MiB |
| 512 | 201.3 | 3512 MiB |

因此本机使用 16 env + minibatch 256。512 没有吞吐收益，显存接近翻倍。不要
机械地按 batch 线性放大学习率；先保持参考 PPO 的 `3e-4` 并观察 KL、clip
fraction、成功率和轨迹。

单 seed 可恢复启动命令：

```bash
scripts/run_actor_single_detached.sh \
  runs/antmaze_umaze_fulla_formal/koopman/best_validation.pt \
  runs/antmaze_umaze_single/actor/seed_0 0 cuda 16 256 1000000
```

## 高性能 GPU

- 优先重新 profile `num_envs=16/32/64` 和
  `minibatch_size=256/512/1024`；更大显存不等于更大的 batch 一定更快。
- DARE 内部使用 float64 保证近单位圆系统的稳定解，因此应优先选择 FP64
  吞吐较好的数据中心 GPU。消费级 RTX 的 FP64 比例很低。
- 多 seed 时优先“一张 GPU 一个 seed”，用作业数组并行；不要在同一张 GPU
  上启动五个互相争抢的小进程。
- 单 seed 多 GPU 需要把可微 DARE minibatch 和梯度聚合显式分布式化，当前
  未实现；在单卡 batch 吞吐饱和前不值得增加这一复杂度。
- 若 GPU DARE 已很快而 MuJoCo stepping 成为新瓶颈，再考虑进程式异步环境。
  先测再改，因为多进程 MuJoCo 会增加初始化、IPC 和可复现性成本。

## 轨迹诊断

`evaluate_actor.py --plot-paths N` 会保存：

- U-Maze 全局 XY 路径图；
- 每条轨迹的局部放大窗；
- 原始路径 `.npz`；
- 起点、终点、最近目标距离、XY 路径长度和目标进度比例。

正式 100-episode 评估默认为每个 seed 保存前 10 条路径。
