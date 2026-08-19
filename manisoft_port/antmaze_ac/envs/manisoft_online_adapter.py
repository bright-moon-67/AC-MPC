from __future__ import annotations

import numpy as np


class ManiSoftOnlineStateAdapter:
    """ManiSoft 52D状态 → AC-MPC 59D状态，并维护增量动作时序。"""

    RAW_STATE_DIM = 52
    PHYSICAL_STATE_DIM = 41
    TARGET_STATE_DIM = 11
    ACTION_DIM = 18
    AC_MPC_STATE_DIM = 59

    def __init__(self, action_low: float = -1.0, action_high: float = 1.0):
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.previous_action = np.zeros(self.ACTION_DIM, dtype=np.float32)

    @staticmethod
    def _check_finite(name: str, value: np.ndarray) -> None:
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    def _get_raw_state(self, env) -> np.ndarray:
        raw_state = np.asarray(
            env.get_patched_state(),
            dtype=np.float32,
        ).reshape(-1)

        if raw_state.shape != (self.RAW_STATE_DIM,):
            raise ValueError(
                f"Expected ManiSoft state (52,), got {raw_state.shape}"
            )

        self._check_finite("raw_state", raw_state)
        return raw_state

    def make_state(self, raw_state: np.ndarray) -> np.ndarray:
        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)

        if raw_state.shape != (self.RAW_STATE_DIM,):
            raise ValueError(
                f"Expected raw state (52,), got {raw_state.shape}"
            )

        physical_state = raw_state[:self.PHYSICAL_STATE_DIM]

        state = np.concatenate(
            (physical_state, self.previous_action),
        ).astype(np.float32, copy=False)

        if state.shape != (self.AC_MPC_STATE_DIM,):
            raise RuntimeError(f"Wrong AC-MPC state shape: {state.shape}")

        self._check_finite("acmpc_state", state)
        return state

    def reset(self, env) -> np.ndarray:
        """每个episode开始时，上一动作必须归零。"""
        self.previous_action.fill(0.0)
        return self.make_state(self._get_raw_state(env))

    def observe(self, env) -> np.ndarray:
        return self.make_state(self._get_raw_state(env))

    def target_state(self, env) -> np.ndarray:
        """返回未进入Koopman模型的11维目标信息。"""
        raw_state = self._get_raw_state(env)
        return raw_state[self.PHYSICAL_STATE_DIM:].copy()

    def step(self, env, delta_action: np.ndarray):
        """执行一次增量动作，并返回新的59维状态。"""
        state = self.observe(env)

        requested_delta = np.asarray(
            delta_action,
            dtype=np.float32,
        ).reshape(-1)

        if requested_delta.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"Expected delta action (18,), got {requested_delta.shape}"
            )

        self._check_finite("delta_action", requested_delta)

        previous_action = self.previous_action.copy()

        # u_t = clip(u_{t-1} + delta_u_t, -1, 1)
        current_action = np.clip(
            previous_action + requested_delta,
            self.action_low,
            self.action_high,
        ).astype(np.float32)

        applied_delta = current_action - previous_action

        # ManiSoft内部：18维 → 6×3控制点 → 20×3力矩
        torque = env.action_to_torque(current_action.astype(np.float64))
        env.step(torque)

        # 必须在仿真成功执行后更新
        self.previous_action[:] = current_action
        next_state = self.observe(env)

        if not np.allclose(
            next_state[-self.ACTION_DIM:],
            current_action,
            atol=1e-6,
        ):
            raise RuntimeError("next_state action block is inconsistent")

        info = {
            "state": state,
            "requested_delta_action": requested_delta,
            "applied_delta_action": applied_delta,
            "absolute_action": current_action.copy(),
        }
        return next_state, info


if __name__ == "__main__":
    class DummyEnv:
        def __init__(self):
            self.raw_state = np.zeros(52, dtype=np.float64)

        def get_patched_state(self):
            return self.raw_state

        def action_to_torque(self, action):
            assert action.shape == (18,)
            return np.zeros((20, 3), dtype=np.float64)

        def step(self, torque):
            assert torque.shape == (20, 3)
            self.raw_state[0] += 0.01

    env = DummyEnv()
    adapter = ManiSoftOnlineStateAdapter()

    state = adapter.reset(env)
    next_state, info = adapter.step(
        env,
        np.full(18, 0.1, dtype=np.float32),
    )

    assert state.shape == (59,)
    assert next_state.shape == (59,)
    assert np.allclose(state[-18:], 0.0)
    assert np.allclose(next_state[-18:], 0.1)

    print("状态适配器自检通过")
    print("state:", state.shape, state.dtype)
    print("delta_action:", info["applied_delta_action"].shape)
    print("absolute_action:", info["absolute_action"].shape)
    print("next_state:", next_state.shape, next_state.dtype)
