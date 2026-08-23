# ManiSoft 物理可行轨迹 SAC

这是后续绕障推物任务的低层、无障碍轨迹控制器。当前阶段没有夹爪、
障碍物、目标方块或接触反力；策略只负责把 18 维肌肉激活变成可行的末端
轨迹，高层规划器以后再给出绕障参考路径。

## “参考路径”不是“专家轨迹”

- 参考路径只有期望的末端三维点和期望速度，用于构造 observation 与 reward，
  不包含应该施加的肌肉动作。
- 专家轨迹通常包含 `(state, expert_action)`，可用于行为克隆、回放池预填充
  或模仿学习。
- 本流程是纯在线 SAC。训练脚本不读取参考动作、不预填充 replay buffer、
  不执行行为克隆；运行记录明确写入
  `learning_mode=pure_online_sac` 和
  `reference_actions_used_for_learning=false`。

入口库保存了生成物理参考路径时的仿真动作，但训练器看不到这些动作。它们
只用于离线认证路径以及把仿真 reset 到课程所需的物理状态。SAC 在 reset
以后执行的每一个动作仍由 SAC 自己探索并根据 reward 学习。

## 场景尺寸与可行性

当前软体臂长 1 m、物理半径 5 cm、20 个 Cosserat 单元、密度
1000 kg/m3、Young 模量 10 MPa，固定基座、开启重力，控制频率 50 Hz。
自然末端约为 `(0, 0, 1.0)`。

桌面已根据真实全臂扫掠重新布置：

- 台面高 `z=0.36 m`；
- 范围 `x=[-0.55,0.55] m, y=[0.42,0.86] m`；
- SAC 操作框为
  `x=[-0.30,0.30], y=[0.50,0.80], z=[0.445,0.54] m`；
- 碰撞判断使用整根软臂的胶囊体，不只检查末端。

六条从自然直立状态出发的入口路径均已在重力仿真中独立重放认证：

| 名称 | 终点 xyz (m) | 整臂-桌面最小安全余量 |
|---|---:|---:|
| right_entry | `(0.269, 0.691, 0.497)` | 38.9 mm |
| right_center_entry | `(0.138, 0.726, 0.506)` | 37.6 mm |
| left_center_entry | `(-0.131, 0.729, 0.511)` | 40.0 mm |
| left_entry | `(-0.259, 0.692, 0.516)` | 47.5 mm |
| low_right_entry | `(0.207, 0.717, 0.481)` | 30.5 mm |
| low_left_entry | `(-0.276, 0.693, 0.481)` | 33.7 mm |

这里的余量已经扣除了 50 mm 杆半径和 5 mm 额外安全边界。相邻 20 ms
参考动作的最大变化约 `0.00188`，末尾 2 s 的末端漂移小于 0.31 mm。

围绕六个入口做了 90 次局部旋转/缩放探测，80 次落在指定操作框内且全部
保持稳定和桌面净空。10 次未计入是因为 `|x|` 略超 0.30 m，不是碰桌或
动力学发散。可靠单段水平位移约为 5.1--5.5 cm；低位入口向下可达约
2.4 cm，因此可覆盖小方块侧面的接触高度。

## 纯 SAC 课程

1. `entry_tail`：reset 到入口路径的 72%--92%，只学习最后一小段。
2. `entry_mid`：学习更长的弯曲下降。为避免长失败回合在 transition replay
   中淹没短成功回合，训练时75%的episode从55%--85%的已掌握锚点开始，
   25%从30%--55%的新前段开始；显式评估起点不受该采样器影响。
3. `entry`：15%的 episode 从自然直立状态完成整条下降路径，85%从
   30%--65%的认证快照开始。固定网格表明所选 `entry_mid` 模型已经在
   低/中速完整入口达到 12/12，当前重点是高速终端稳定，同时保护已学能力。
4. `table_local`：reset 到稳定入口终点，学习 2--5.5 cm 的近水平局部运动。
5. `recovery`：从入口 45%--90% 开始，完成剩余下降并连接局部运动。
6. `entry_local`：从自然状态完成“下降 + 桌面局部运动”。
7. `mixed`：混合 entry/table_local/entry_local，减少阶段间遗忘。

这种从路径后半段逐渐扩展到完整路径的 reset curriculum 只改变初始状态和
目标难度，不提供动作标签，仍是 SAC。

PyElastica 的移动状态不只包含位置、速度、director 和角速度。Position-
Verlet 还会读取加速度、伸长率、曲率、内外力等缓存；仅恢复前四项会使某些
中段快照第一步数值发散。入口库 schema v3 以 float64 保存并恢复所有杆状态
缓存和仿真时钟。12 个中段/尾段逐步验证的下一帧误差均为 0，随机动作诊断
也不再发生即时发散。真正的数值发散仍会作为 `dynamics_violation` 大惩罚
终止 transition 返回 SAC，而不会杀掉全部并行环境。

早期入口课程可使用绝对 18 维激活；当前桌面 Stage 3 使用
`table_cartesian_delta`，策略输出全局桌面坐标中的 2 维 x/y 命令，再经六种
入口姿态各自的离线标定映射为 18 维肌肉激活。环境仍限制实际激活每帧变化
不超过 `0.008`。Observation 为 70 维：45 维物理状态、18 维上一实际动作、
当前目标误差 3 维、lookahead 误差 3 维和期望速度 1 维。

## 当前 Stage 3 最佳控制方式（2026-08-23）

当前选择的 SAC 检查点为：

```text
runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/
  turn45_weak_a100/checkpoints/
  sac_table_waypoint_polyline_4879664_steps.zip
```

部署时将 SAC 的 2 维动作与标定笛卡尔控制先验融合：先验权重 `0.40`、
比例反馈增益 `20`、前馈系数 `1.0`。先验只利用当前目标、当前末端位置和
入口起点，不读取未来专家动作；SAC 仍保留 60% 控制权来补偿软体动力学。

两批互不重叠、各 192 条三段折线路径的确定性评估结果合计为：

| 控制方式 | 成功数 | 成功率 | 相对纯 SAC 的成对得/失 | 安全违规 |
|---|---:|---:|---:|---:|
| 纯 SAC | 144/384 | 37.50% | - | 0 |
| SAC + 40% 笛卡尔先验 | 154/384 | 40.10% | 16/6 | 0 |

成功回合的平均 RMSE 也由约 `0.876 cm` 降到约 `0.847 cm`。精确 McNemar
检验 `p=0.0525`，因此这是可复现的工程改进，但尚不能宣称已达到最终任务
要求。将先验提高到 60% 会继续降低成功回合 RMSE，却在第二批未见轨迹上
降低总成功率；在先验闭环中继续 SAC 30k--60k 也出现能力重排或遗忘，当前
不采用这些模型。

## 后续推物场景

`push_around_obstacle_kinematic.yaml` 已同步到相同桌面坐标：障碍盒为
`10 x 15 x 5 cm`，目标方块为 `7 x 7 x 8 cm`，目标中心在
`(0,0.77,0.40) m`，终点在 `(0,0.815,0.40) m`，只需推 4.5 cm。
左右绕行门位于 `x=+-0.16 m`，两侧路径连同 24 mm 末端半径均不与障碍物
相交。SAC 低层训练场景的 `objects` 仍为空，这是有意的：先学无障碍轨迹
跟踪，再由高层路径与任务奖励处理绕障和推物。

## 命令

```bash
cd /root/autodl-tmp/AC-MPC
conda activate manisoft

# 生成入口库并检查局部可达性
bash scripts/run_manisoft_waypoint_sac_pipeline.sh generate
bash scripts/run_manisoft_waypoint_sac_pipeline.sh analyze_local

# 测试
pytest -q tests/test_manisoft_waypoint_sac.py tests/test_manisoft_kinematic_push.py
python scripts/smoke_test_manisoft_waypoint_sac.py \
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
  --config configs/manisoft_waypoint_sac_physical.yaml \
  --output runs/manisoft_waypoint_sac_physical_smoke/environment_report_v3.json

# 当前最佳模型的确定性评估（40% 笛卡尔先验）
python scripts/evaluate_manisoft_waypoint_sac.py \
  --model runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100/checkpoints/sac_table_waypoint_polyline_4879664_steps.zip \
  --vec-normalize runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100/checkpoints/sac_table_waypoint_polyline_vecnormalize_4879664_steps.pkl \
  --run-config runs/manisoft_waypoint_sac_stage3_gate_pilot_20260823/turn45_weak_a100/run_config.json \
  --config configs/manisoft_waypoint_sac_table_multipoint_stage3.yaml \
  --scenario /root/autodl-tmp/ManiSoft/configs/sac_waypoint_tracking_table.yaml \
  --output runs/manual_prior40_eval.npz --episodes 48 \
  --families waypoint_polyline --curriculum table_waypoint_polyline \
  --waypoint-maximum-extent 0.035 \
  --waypoint-segment-count-range 3,3 \
  --waypoint-maximum-turn-degrees 60 \
  --cartesian-prior-weight 0.40 \
  --cartesian-prior-proportional-gain 20 \
  --cartesian-prior-feedforward-scale 1

# 纯 SAC 正式课程；每阶段验收后再进入下一阶段
bash scripts/run_manisoft_waypoint_sac_pipeline.sh entry_tail
bash scripts/run_manisoft_waypoint_sac_pipeline.sh entry_mid
bash scripts/run_manisoft_waypoint_sac_pipeline.sh entry
bash scripts/run_manisoft_waypoint_sac_pipeline.sh table_local
bash scripts/run_manisoft_waypoint_sac_pipeline.sh recovery
bash scripts/run_manisoft_waypoint_sac_pipeline.sh entry_local
bash scripts/run_manisoft_waypoint_sac_pipeline.sh mixed
bash scripts/run_manisoft_waypoint_sac_pipeline.sh evaluate
```

建议晋级门槛：独立种子成功率至少 80%，末端 RMSE 小于 2 cm、p95 小于
3 cm、终点 1 cm 内连续 8 帧，`table_violation=0` 且
`dynamics_violation=0`。必须分别报告每个课程阶段，不能只看 mixed 总均值。
