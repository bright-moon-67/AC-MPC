"""Vector runners for state-based DMC experiments.

The important distinction in this module is between the observation reached by
an action and the observation used by the policy on the following call.  At an
episode boundary those are different:

``transition_observation``
    The final observation reached by the action.  Value bootstrapping for a DMC
    time-limit transition must use this observation because DMC returns
    ``discount=1`` at a timeout.

``observation``
    The next policy input.  For a finished environment this is the observation
    produced by the automatic reset.

Keeping both values prevents the common error of bootstrapping a timeout from
the first observation of the next episode. ``SyncDMCVectorEnv`` remains the
small reference implementation; ``ProcessDMCVectorEnv`` shards the same
contract across CPU workers without changing PPO or experiment identity.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


EnvFactory = Callable[..., Any]


@dataclass(frozen=True)
class VectorStep:
    """One batched environment transition, including autoreset boundaries."""

    observation: np.ndarray
    transition_observation: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    reset_boundary: np.ndarray
    reset_seed: np.ndarray
    applied_action: np.ndarray
    info: tuple[dict[str, Any], ...]

    @property
    def done(self) -> np.ndarray:
        """Compatibility alias for an episode/reset boundary."""

        return self.reset_boundary


def _json_value(value: Any) -> Any:
    """Convert environment metadata into stable JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class SyncDMCVectorEnv:
    """Run independent DMC adapters sequentially with deterministic autoreset."""

    def __init__(
        self,
        task_name: str,
        num_envs: int,
        seed: int,
        *,
        control_timestep: float | None = None,
        time_limit: float | None = None,
        env_factory: EnvFactory | None = None,
        _seed_stride: int | None = None,
        _index_offset: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if control_timestep is not None and control_timestep <= 0:
            raise ValueError("control_timestep must be positive")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if env_factory is None:
            from experiments.dmc.tasks.adapter import make_dmc_adapter

            env_factory = make_dmc_adapter

        self.task_name = str(task_name)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.requested_control_timestep = control_timestep
        self.requested_time_limit = time_limit
        self._seed_stride = int(_seed_stride or num_envs)
        self._index_offset = int(_index_offset)
        if self._seed_stride < self.num_envs or self._index_offset < 0:
            raise ValueError("Invalid vector seed partition")
        if self._index_offset + self.num_envs > self._seed_stride:
            raise ValueError("Vector seed partition exceeds its global stride")
        self._envs: list[Any] = []
        self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._current_reset_seeds = np.zeros(self.num_envs, dtype=np.int64)
        self._observations: np.ndarray | None = None
        self._closed = False

        try:
            for index in range(self.num_envs):
                self._envs.append(
                    env_factory(
                        self.task_name,
                        seed=self.seed + self._index_offset + index,
                        control_timestep=control_timestep,
                        time_limit=time_limit,
                    )
                )
            self._initialize_contract()
        except BaseException:
            self.close()
            raise

    def _initialize_contract(self) -> None:
        first = self._envs[0]
        self.obs_dim = int(first.obs_dim)
        self.action_dim = int(first.action_dim)
        self.action_low = np.asarray(first.action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(first.action_high, dtype=np.float32).reshape(-1)
        if self.action_low.shape != (self.action_dim,) or self.action_high.shape != (
            self.action_dim,
        ):
            raise ValueError("Environment action bounds have the wrong shape")
        if np.any(self.action_low >= self.action_high):
            raise ValueError("Every action lower bound must be below its upper bound")

        self.protocol = self._protocol_metadata(first)
        for env in self._envs[1:]:
            if (
                int(env.obs_dim) != self.obs_dim
                or int(env.action_dim) != self.action_dim
            ):
                raise ValueError("Vector environments have inconsistent dimensions")
            np.testing.assert_allclose(
                np.asarray(env.action_low), self.action_low, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                np.asarray(env.action_high), self.action_high, rtol=0.0, atol=0.0
            )
            if self._protocol_metadata(env) != self.protocol:
                raise ValueError("Vector environments have inconsistent protocols")

    def _protocol_metadata(self, env: Any) -> dict[str, Any]:
        """Return the adapter's exact, seed-independent runtime protocol.

        Checkpoint/dataset fingerprints intentionally use this mapping without
        aliases or runner request fields.  ``n_substeps`` remains the physics
        integration count; callers that need an action-repeat concept default
        it to one rather than adding it to the environment protocol.
        """

        protocol_method = getattr(env, "protocol_metadata", None)
        if callable(protocol_method):
            metadata = dict(protocol_method())
        else:
            metadata = dict(env.metadata())
            metadata.pop("seed", None)
        return _json_value(metadata)

    def _reset_seed(self, index: int) -> int:
        # Seeds are unique across env indices and episode counts, while the
        # complete sequence remains reproducible from the training seed.
        return (
            self.seed
            + self._index_offset
            + index
            + int(self._episode_counts[index]) * self._seed_stride
        )

    def reset(self) -> np.ndarray:
        """Reset every member and return a copied ``[num_envs, obs_dim]`` batch."""

        if self._closed:
            raise RuntimeError("Cannot reset a closed vector environment")
        observations = []
        for index, env in enumerate(self._envs):
            reset_seed = self._reset_seed(index)
            value = np.asarray(
                env.reset(seed=reset_seed), dtype=np.float32
            ).reshape(-1)
            if value.shape != (self.obs_dim,):
                raise ValueError(
                    f"Environment {index} reset returned shape {value.shape}, "
                    f"expected {(self.obs_dim,)}"
                )
            observations.append(value)
            self._current_reset_seeds[index] = reset_seed
        self._observations = np.stack(observations)
        return self._observations.copy()

    @staticmethod
    def _transition_flags(
        done: bool, info: dict[str, Any]
    ) -> tuple[float, bool, bool]:
        """Recover DMC discount and terminal/timeout flags from adapter info.

        New adapters expose all three fields.  The fallback keeps the runner
        compatible with the original DMC adapter, whose tasks only finish by
        time limit: a LAST step is therefore a truncation with discount one.
        """

        explicit_discount = info.get("discount")
        explicit_terminated = info.get("terminated")
        explicit_truncated = info.get("truncated")

        if explicit_discount is None:
            if explicit_terminated is not None:
                discount = 0.0 if bool(explicit_terminated) else 1.0
            else:
                discount = 1.0
        else:
            discount = float(explicit_discount)
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError(f"Invalid environment discount {discount!r}")

        if explicit_terminated is None and explicit_truncated is None:
            terminated = bool(done and discount == 0.0)
            truncated = bool(done and not terminated)
        else:
            terminated = bool(explicit_terminated or False)
            truncated = bool(explicit_truncated or False)
            if terminated and truncated:
                raise ValueError(
                    "A transition cannot be both terminated and truncated"
                )
            if bool(done) != bool(terminated or truncated):
                raise ValueError(
                    "Adapter done must equal terminated or truncated when "
                    "explicit boundary flags are provided"
                )
        return discount, terminated, truncated

    def step(self, latent_action: np.ndarray) -> VectorStep:
        """Step every env, clip to applied actions, and autoreset finished envs."""

        if self._closed:
            raise RuntimeError("Cannot step a closed vector environment")
        if self._observations is None:
            raise RuntimeError("reset() must be called before step()")
        latent = np.asarray(latent_action, dtype=np.float32)
        if latent.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.action_dim)}, "
                f"got {latent.shape}"
            )
        if not np.isfinite(latent).all():
            raise FloatingPointError("Latent action contains NaN or Inf")
        requested_applied = np.clip(latent, self.action_low, self.action_high).astype(
            np.float32
        )

        policy_observations: list[np.ndarray] = []
        transition_observations: list[np.ndarray] = []
        rewards: list[float] = []
        discounts: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        boundaries: list[bool] = []
        reset_seeds: list[int] = []
        applied_actions: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []

        for index, (env, action) in enumerate(zip(self._envs, requested_applied)):
            # Capture the seed of the episode that produced this transition;
            # an autoreset below advances it for the *next* transition.
            transition_reset_seed = int(self._current_reset_seeds[index])
            next_observation, reward, done, raw_info = env.step(action)
            info = dict(raw_info or {})
            transition = np.asarray(next_observation, dtype=np.float32).reshape(-1)
            if transition.shape != (self.obs_dim,):
                raise ValueError(
                    f"Environment {index} step returned shape {transition.shape}, "
                    f"expected {(self.obs_dim,)}"
                )
            discount, terminated, truncated = self._transition_flags(bool(done), info)
            boundary = bool(done or terminated or truncated)
            applied = np.asarray(
                info.get("applied_action", action), dtype=np.float32
            ).reshape(-1)
            if applied.shape != (self.action_dim,):
                raise ValueError("Adapter returned an invalid applied_action shape")
            if not np.isfinite(applied).all():
                raise FloatingPointError("Applied action contains NaN or Inf")
            if np.any(applied < self.action_low - 1e-6) or np.any(
                applied > self.action_high + 1e-6
            ):
                raise ValueError("Adapter reported an action outside its action spec")

            policy_next = transition
            if boundary:
                self._episode_counts[index] += 1
                autoreset_seed = self._reset_seed(index)
                policy_next = np.asarray(
                    env.reset(seed=autoreset_seed), dtype=np.float32
                ).reshape(-1)
                if policy_next.shape != (self.obs_dim,):
                    raise ValueError("Autoreset returned an invalid observation shape")
                info["autoreset_seed"] = autoreset_seed
                info["autoreset_observation"] = policy_next.copy()
                self._current_reset_seeds[index] = autoreset_seed
            info["transition_observation"] = transition.copy()
            info["discount"] = discount
            info["terminated"] = terminated
            info["truncated"] = truncated
            info["applied_action"] = applied.copy()

            policy_observations.append(policy_next)
            transition_observations.append(transition)
            rewards.append(float(reward))
            discounts.append(discount)
            terminateds.append(terminated)
            truncateds.append(truncated)
            boundaries.append(boundary)
            reset_seeds.append(transition_reset_seed)
            applied_actions.append(applied)
            infos.append(info)

        self._observations = np.stack(policy_observations).astype(np.float32)
        return VectorStep(
            observation=self._observations.copy(),
            transition_observation=np.stack(transition_observations).astype(
                np.float32
            ),
            reward=np.asarray(rewards, dtype=np.float32),
            discount=np.asarray(discounts, dtype=np.float32),
            terminated=np.asarray(terminateds, dtype=np.bool_),
            truncated=np.asarray(truncateds, dtype=np.bool_),
            reset_boundary=np.asarray(boundaries, dtype=np.bool_),
            reset_seed=np.asarray(reset_seeds, dtype=np.int64),
            applied_action=np.stack(applied_actions).astype(np.float32),
            info=tuple(infos),
        )

    def close(self) -> None:
        if self._closed:
            return
        for env in self._envs:
            try:
                env.close()
            except Exception:
                # Closing is best effort, particularly while unwinding a
                # partially constructed vector environment.
                pass
        self._closed = True

    def __enter__(self) -> "SyncDMCVectorEnv":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _worker_main(
    connection: Any,
    task_name: str,
    local_envs: int,
    seed: int,
    global_envs: int,
    index_offset: int,
    control_timestep: float | None,
    time_limit: float | None,
) -> None:
    """Own one environment shard and serve reset/step requests."""

    # MuJoCo stepping is scalar here. Prevent numerical libraries imported in
    # a worker from silently creating another thread pool per process.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    env: SyncDMCVectorEnv | None = None
    try:
        env = SyncDMCVectorEnv(
            task_name,
            local_envs,
            seed,
            control_timestep=control_timestep,
            time_limit=time_limit,
            _seed_stride=global_envs,
            _index_offset=index_offset,
        )
        connection.send(
            (
                "ready",
                {
                    "obs_dim": env.obs_dim,
                    "action_dim": env.action_dim,
                    "action_low": env.action_low,
                    "action_high": env.action_high,
                    "protocol": env.protocol,
                },
            )
        )
        while True:
            try:
                command, payload = connection.recv()
            except EOFError:
                break
            if command == "reset":
                connection.send(("result", env.reset()))
            elif command == "step":
                connection.send(("result", env.step(payload)))
            elif command == "close":
                connection.send(("closed", None))
                break
            else:
                raise ValueError(f"Unknown vector worker command {command!r}")
    except BaseException as exc:
        try:
            connection.send(
                (
                    "error",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class ProcessDMCVectorEnv:
    """Run DMC shards concurrently in spawn-based CPU worker processes."""

    def __init__(
        self,
        task_name: str,
        num_envs: int,
        seed: int,
        *,
        workers: int,
        control_timestep: float | None = None,
        time_limit: float | None = None,
        start_method: str = "spawn",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if workers < 1 or workers > num_envs:
            raise ValueError("workers must lie in [1, num_envs]")
        self.task_name = str(task_name)
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.workers = int(workers)
        self._closed = False
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []
        self._slices: list[slice] = []

        counts = [num_envs // workers] * workers
        for index in range(num_envs % workers):
            counts[index] += 1
        context = mp.get_context(start_method)
        offset = 0
        try:
            for count in counts:
                parent, child = context.Pipe(duplex=True)
                process = context.Process(
                    target=_worker_main,
                    args=(
                        child,
                        task_name,
                        count,
                        seed,
                        num_envs,
                        offset,
                        control_timestep,
                        time_limit,
                    ),
                    daemon=False,
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
                self._slices.append(slice(offset, offset + count))
                offset += count
            contracts = [
                self._receive(index, expected="ready", timeout=180.0)
                for index in range(workers)
            ]
            first = contracts[0]
            for contract in contracts[1:]:
                if contract["obs_dim"] != first["obs_dim"] or contract[
                    "action_dim"
                ] != first["action_dim"]:
                    raise ValueError("Vector workers have inconsistent dimensions")
                np.testing.assert_array_equal(contract["action_low"], first["action_low"])
                np.testing.assert_array_equal(contract["action_high"], first["action_high"])
                if contract["protocol"] != first["protocol"]:
                    raise ValueError("Vector workers have inconsistent protocols")
            self.obs_dim = int(first["obs_dim"])
            self.action_dim = int(first["action_dim"])
            self.action_low = np.asarray(first["action_low"], dtype=np.float32)
            self.action_high = np.asarray(first["action_high"], dtype=np.float32)
            self.protocol = dict(first["protocol"])
        except BaseException:
            self.close()
            raise

    def _receive(
        self, index: int, *, expected: str, timeout: float | None = None
    ) -> Any:
        if timeout is not None and not self._connections[index].poll(timeout):
            raise TimeoutError(
                f"DMC worker {index} did not return {expected!r} within "
                f"{timeout:g}s"
            )
        try:
            kind, payload = self._connections[index].recv()
        except EOFError as exc:
            raise RuntimeError(f"DMC worker {index} exited unexpectedly") from exc
        if kind == "error":
            raise RuntimeError(
                f"DMC worker {index} failed with {payload['type']}: "
                f"{payload['message']}\n{payload['traceback']}"
            )
        if kind != expected:
            raise RuntimeError(
                f"DMC worker {index} returned {kind!r}, expected {expected!r}"
            )
        return payload

    def reset(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Cannot reset a closed vector environment")
        for connection in self._connections:
            connection.send(("reset", None))
        observations = [
            np.asarray(self._receive(index, expected="result"), dtype=np.float32)
            for index in range(self.workers)
        ]
        result = np.concatenate(observations, axis=0)
        if result.shape != (self.num_envs, self.obs_dim):
            raise RuntimeError("Parallel reset assembled the wrong shape")
        return result

    def step(self, latent_action: np.ndarray) -> VectorStep:
        if self._closed:
            raise RuntimeError("Cannot step a closed vector environment")
        actions = np.asarray(latent_action, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.action_dim)}, "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise FloatingPointError("Latent action contains NaN or Inf")
        for connection, indices in zip(
            self._connections, self._slices, strict=True
        ):
            connection.send(("step", actions[indices]))
        steps = [
            self._receive(index, expected="result")
            for index in range(self.workers)
        ]
        return VectorStep(
            observation=np.concatenate([step.observation for step in steps]),
            transition_observation=np.concatenate(
                [step.transition_observation for step in steps]
            ),
            reward=np.concatenate([step.reward for step in steps]),
            discount=np.concatenate([step.discount for step in steps]),
            terminated=np.concatenate([step.terminated for step in steps]),
            truncated=np.concatenate([step.truncated for step in steps]),
            reset_boundary=np.concatenate(
                [step.reset_boundary for step in steps]
            ),
            reset_seed=np.concatenate([step.reset_seed for step in steps]),
            applied_action=np.concatenate(
                [step.applied_action for step in steps]
            ),
            info=tuple(item for step in steps for item in step.info),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(
            self._connections, self._processes, strict=True
        ):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for index, (connection, process) in enumerate(
            zip(self._connections, self._processes, strict=True)
        ):
            if process.is_alive():
                try:
                    self._receive(index, expected="closed", timeout=5.0)
                except (RuntimeError, OSError, TimeoutError):
                    pass
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            connection.close()

    def __enter__(self) -> "ProcessDMCVectorEnv":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def default_env_workers(num_envs: int) -> int:
    """Resolve execution-only worker count without changing experiment identity."""

    raw = os.environ.get("DMC_ENV_WORKERS")
    if raw is not None:
        try:
            workers = int(raw)
        except ValueError as exc:
            raise ValueError("DMC_ENV_WORKERS must be an integer") from exc
    else:
        workers = min(16, os.cpu_count() or 1, num_envs)
    if workers < 1 or workers > num_envs:
        raise ValueError("DMC environment workers must lie in [1, num_envs]")
    return workers


def make_dmc_vector_env(
    task_name: str,
    num_envs: int,
    seed: int,
    *,
    workers: int | None = None,
    control_timestep: float | None = None,
    time_limit: float | None = None,
    env_factory: EnvFactory | None = None,
) -> SyncDMCVectorEnv | ProcessDMCVectorEnv:
    """Build the reference or multi-process runner behind one stable contract."""

    # Synthetic test factories are frequently closures and intentionally stay
    # in-process. Real DMC runs use spawn workers unless explicitly set to one.
    resolved_workers = 1 if env_factory is not None else (
        default_env_workers(num_envs) if workers is None else int(workers)
    )
    if resolved_workers == 1:
        return SyncDMCVectorEnv(
            task_name,
            num_envs,
            seed,
            control_timestep=control_timestep,
            time_limit=time_limit,
            env_factory=env_factory,
        )
    if env_factory is not None:
        raise ValueError("Custom env_factory only supports one in-process worker")
    return ProcessDMCVectorEnv(
        task_name,
        num_envs,
        seed,
        workers=resolved_workers,
        control_timestep=control_timestep,
        time_limit=time_limit,
    )
