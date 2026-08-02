from pathlib import Path
import gc
import warnings

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from manisoft.envs.vlm_env import VLMEnvironmentForExecutorControl
from manisoft.utils import load_yaml
from antmaze_ac.koopman.checkpoint import load_checkpoint

warnings.filterwarnings("ignore", message="Gimbal lock detected.*")

STEPS = 300
MAX_DELTA = 0.02
MAX_ABSOLUTE_ACTION = 0.30
TARGET_OFFSET_X = -0.005

manisoft_root = Path.cwd()
acmpc_root = manisoft_root.parent / "AC-MPC"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env():
    config = load_yaml(
        manisoft_root
        / "work_dirs/data_gen/COLL/can/scenarios/0/config.yaml"
    )
    config.pop("renderer", None)
    config["environment"]["model_path"] = (
        manisoft_root / "work_dirs/rl_models/model_1.zip"
    )
    return VLMEnvironmentForExecutorControl.from_dict(config)


def physical_state(env):
    _, softrobot_state = env.get_state(False)
    state52 = env._patch_state(
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        softrobot_state,
    )
    return np.asarray(state52[:41], dtype=np.float32)


def tip_position(state41):
    return state41[[10, 21, 32]]


# 加载Koopman模型
checkpoint = (
    acmpc_root
    / "runs/manisoft_coll_full_seed42/koopman/best_validation.pt"
)
model, payload = load_checkpoint(checkpoint, map_location=device)
model = model.to(device).eval()

stats = payload["normalizers"]["state"]
mean = np.asarray(stats["mean"], dtype=np.float32)
std = np.asarray(stats["std"], dtype=np.float32)

A = model.A.detach().cpu().double().numpy()
B = model.B.detach().cpu().double().numpy()
C = model.C.detach().cpu().double().numpy()

# 59维归一化状态代价
q_x = np.full(59, 0.001, dtype=np.float64)

# 杆末端位置
q_x[[10, 21, 32]] = 20.0

# 末端速度大小
q_x[33] = 0.1

# 速度方向在低速时不连续，暂时不控制
q_x[34:37] = 0.0

# 末端姿态四元数
q_x[37:41] = 2.0

# 上一绝对动作
q_x[41:59] = 0.01

Qx = np.diag(q_x)
Qz = C.T @ Qx @ C + 1e-4 * np.eye(91)
R = 1000.0 * np.eye(18)

P = solve_discrete_are(A, B, Qz, R)
K = np.linalg.solve(
    R + B.T @ P @ B,
    B.T @ P @ A,
)

closed_loop_radius = np.max(
    np.abs(np.linalg.eigvals(A - B @ K))
)

print("K shape:", K.shape)
print("closed-loop spectral radius:", closed_loop_radius)


def run_controlled():
    env = make_env()

    initial_physical = physical_state(env)
    target_tip = tip_position(initial_physical).copy()
    target_tip[0] += TARGET_OFFSET_X

    previous_action = np.zeros(18, dtype=np.float32)
    reference_state = np.concatenate(
        [initial_physical, previous_action]
    )

    # 第10维是软体杆末端的X坐标
    reference_state[10] += TARGET_OFFSET_X

    reference_normalized = (reference_state - mean) / std

    with torch.inference_mode():
        reference_lifted = model.lift(
            torch.from_numpy(reference_normalized)
            .unsqueeze(0).to(device)
        ).squeeze(0).cpu().numpy()

    drifts = []
    maximum_delta = 0.0
    maximum_action = 0.0

    for step in range(STEPS):
        current_physical = physical_state(env)
        current_state = np.concatenate(
            [current_physical, previous_action]
        )
        current_normalized = (current_state - mean) / std

        with torch.inference_mode():
            current_lifted = model.lift(
                torch.from_numpy(current_normalized)
                .unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()

        delta = -K @ (current_lifted - reference_lifted)
        delta = np.clip(delta, -MAX_DELTA, MAX_DELTA)

        absolute_action = np.clip(
            previous_action + delta,
            -MAX_ABSOLUTE_ACTION,
            MAX_ABSOLUTE_ACTION,
        ).astype(np.float32)

        applied_delta = absolute_action - previous_action

        if not np.isfinite(absolute_action).all():
            raise RuntimeError("LQR产生NaN或Inf动作")

        env.muscle_step(absolute_action.reshape(6, 3))
        previous_action = absolute_action

        next_physical = physical_state(env)
        drift = np.linalg.norm(
            tip_position(next_physical) - target_tip
        )
        drifts.append(drift)

        maximum_delta = max(
            maximum_delta,
            float(np.max(np.abs(applied_delta))),
        )
        maximum_action = max(
            maximum_action,
            float(np.max(np.abs(absolute_action))),
        )

        if (step + 1) % 10 == 0:
            print(
                f"controlled step={step + 1:02d}, "
                f"tip_drift={drift:.6f}, "
                f"max_action={np.max(np.abs(absolute_action)):.4f}"
            )

    return np.asarray(drifts), maximum_delta, maximum_action


def run_zero_action_baseline():
    env = make_env()
    target_tip = tip_position(physical_state(env)).copy()
    target_tip[0] += TARGET_OFFSET_X
    zero_action = np.zeros((6, 3), dtype=np.float32)
    drifts = []

    for _ in range(STEPS):
        env.muscle_step(zero_action)
        drift = np.linalg.norm(
            tip_position(physical_state(env)) - target_tip
        )
        drifts.append(drift)

    return np.asarray(drifts)


controlled, max_delta, max_action = run_controlled()

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

baseline = run_zero_action_baseline()

print("\n===== LQR X方向-5mm目标跟踪测试 =====")
print("LQR平均目标误差:", float(controlled.mean()))
print("基线平均目标误差:", float(baseline.mean()))
print("LQR最终目标误差:", float(controlled[-1]))
print("基线最终目标误差:", float(baseline[-1]))
print("LQR最大目标误差:", float(controlled.max()))
print("实际最大增量动作:", max_delta)
print("实际最大绝对动作:", max_action)
print("LQR最小目标误差:", float(controlled.min()))
print("LQR最小误差步数:", int(controlled.argmin() + 1))
print("LQR最后50步平均误差:", float(controlled[-50:].mean()))
print("基线最后50步平均误差:", float(baseline[-50:].mean()))

if controlled.mean() < baseline.mean():
    improvement = 100 * (1 - controlled.mean() / baseline.mean())
    print(f"结论：LQR目标跟踪优于零动作基线 {improvement:.2f}%")
else:
    print("结论：LQR目标跟踪未优于零动作基线，需要继续调整")

print("X方向-5mm目标跟踪测试完成")
