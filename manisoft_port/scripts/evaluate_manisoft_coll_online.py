from pathlib import Path
import gc
import pickle
import warnings

import numpy as np
import torch

from manisoft.envs.vlm_env import VLMEnvironmentForExecutorControl
from manisoft.utils import koopman_section_state, load_yaml
from antmaze_ac.koopman.checkpoint import load_checkpoint

warnings.filterwarnings("ignore", message="Gimbal lock detected.*")

EPISODES = 10
HORIZON = 20

manisoft_root = Path.cwd()
acmpc_root = manisoft_root.parent / "AC-MPC"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = (
    acmpc_root
    / "runs/manisoft_coll_full_seed42/koopman/best_validation.pt"
)

model, payload = load_checkpoint(checkpoint, map_location=device)
model = model.to(device).eval()

stats = payload["normalizers"]["state"]
mean = np.asarray(stats["mean"], dtype=np.float32)
std = np.asarray(stats["std"], dtype=np.float32)


def physical_state(env):
    _, softrobot_state = env.get_state(False)
    return koopman_section_state(softrobot_state)


def label(index):
    section = index // 15
    component = index % 15
    point = (4, 8, 11)[section]
    if component < 3:
        name = f"position[{component}]"
    elif component < 9:
        name = f"rotation_6d[{component - 3}]"
    elif component < 12:
        name = f"linear_velocity[{component - 9}]"
    else:
        name = f"angular_velocity[{component - 12}]"
    return f"section_{point}.{name}"


pairs = []

for trajectory_path in manisoft_root.glob(
    "work_dirs/**/COLL/can/trajectories/*/trajectory.pkl"
):
    scenario = (
        trajectory_path.parents[2]
        / "scenarios"
        / trajectory_path.parent.name
        / "config.yaml"
    )

    if scenario.exists():
        pairs.append((scenario, trajectory_path))

pairs.sort(key=lambda pair: str(pair[1]))

if not pairs:
    raise RuntimeError("没有找到匹配的COLL/can场景和轨迹")

EPISODES = min(10, len(pairs))
print(f"找到有效场景—轨迹配对：{len(pairs)}，本次测试：{EPISODES}")

results = []
h1_dimension_errors = []

for episode, (scenario, trajectory_path) in enumerate(pairs[:EPISODES]):
    config = load_yaml(scenario)
    config.pop("renderer", None)
    config["environment"]["model_path"] = (
        manisoft_root / "work_dirs/rl_models/model_1.zip"
    )

    env = VLMEnvironmentForExecutorControl.from_dict(config)

    with trajectory_path.open("rb") as file:
        trajectory = pickle.load(file)

    actions = np.asarray(
        trajectory["softrobot_actions"],
        dtype=np.float32,
    ).reshape(-1, 18)[:HORIZON]

    deltas = np.diff(
        np.concatenate(
            [np.zeros((1, 18), dtype=np.float32), actions],
            axis=0,
        ),
        axis=0,
    )

    initial_state = np.concatenate(
        [physical_state(env), np.zeros(18, dtype=np.float32)]
    )
    initial_normalized = (initial_state - mean) / std

    with torch.inference_mode():
        predicted, _ = model.rollout(
            torch.from_numpy(initial_normalized)
            .unsqueeze(0).to(device),
            torch.from_numpy(deltas)
            .unsqueeze(0).to(device),
        )

    predicted = predicted.squeeze(0).cpu().numpy()

    real_states = []
    for action in actions:
        env.muscle_step(action.reshape(6, 3))
        real_states.append(
            np.concatenate([physical_state(env), action])
        )

    real_states = np.asarray(real_states, dtype=np.float32)
    real_normalized = (real_states - mean) / std

    # 静止物理状态基线，但正确更新动作状态
    naive_states = np.repeat(
        initial_state[None, :], HORIZON, axis=0
    )
    naive_states[:, 45:] = actions
    naive_normalized = (naive_states - mean) / std

    koopman_h1 = np.mean(
        (predicted[:1, :45] - real_normalized[:1, :45]) ** 2
    )
    baseline_h1 = np.mean(
        (naive_normalized[:1, :45]
         - real_normalized[:1, :45]) ** 2
    )
    koopman_h20 = np.mean(
        (predicted[:, :45] - real_normalized[:, :45]) ** 2
    )
    baseline_h20 = np.mean(
        (naive_normalized[:, :45]
         - real_normalized[:, :45]) ** 2
    )
    action_h20 = np.mean(
        (
            predicted[:, 45:] * std[45:] + mean[45:]
            - real_states[:, 45:]
        ) ** 2
    )

    h1_dimension_errors.append(
        (predicted[0, :45] - real_normalized[0, :45]) ** 2
    )
    results.append(
        [koopman_h1, baseline_h1, koopman_h20, baseline_h20, action_h20]
    )

    print(
        f"episode={episode:02d} | "
        f"H1 {koopman_h1:.5f}/{baseline_h1:.5f} | "
        f"H20 {koopman_h20:.5f}/{baseline_h20:.5f} | "
        f"action {action_h20:.7f}"
    )

    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

results = np.asarray(results)
mean_result = results.mean(axis=0)

h1_wins = int(np.sum(results[:, 0] < results[:, 1]))
h20_wins = int(np.sum(results[:, 2] < results[:, 3]))

h20_improvement = 100 * (
    1 - mean_result[2] / mean_result[3]
)

print("\n===== 10个episode汇总 =====")
print(f"H1平均:  Koopman={mean_result[0]:.6f}, baseline={mean_result[1]:.6f}")
print(f"H20平均: Koopman={mean_result[2]:.6f}, baseline={mean_result[3]:.6f}")
print(f"H1胜出:  {h1_wins}/{EPISODES}")
print(f"H20胜出: {h20_wins}/{EPISODES}")
print(f"H20平均改善: {h20_improvement:.2f}%")
print(f"动作状态平均MSE: {mean_result[4]:.8f}")

dimension_errors = np.mean(h1_dimension_errors, axis=0)
top_indices = np.argsort(dimension_errors)[-5:][::-1]

print("\nH1误差最大的5个物理分量：")
for index in top_indices:
    print(
        f"{index:2d} {label(int(index)):28s} "
        f"MSE={dimension_errors[index]:.6f}"
    )
