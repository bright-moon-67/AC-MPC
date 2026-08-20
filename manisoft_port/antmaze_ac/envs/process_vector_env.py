from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import Connection
import traceback
from typing import Callable, Sequence

import cloudpickle
import numpy as np


def _worker(connection: Connection, serialized_factory: bytes) -> None:
    """Own one environment and execute commands in an isolated process."""

    environment = None
    try:
        factory = cloudpickle.loads(serialized_factory)
        environment = factory()
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(("result", environment.reset(**payload)))
            elif command == "step":
                connection.send(("result", environment.step(payload)))
            elif command == "close":
                connection.send(("result", None))
                break
            else:
                raise ValueError(f"Unsupported environment command {command!r}")
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if environment is not None:
            environment.close()
        connection.close()


class ProcessVectorEnv:
    """Minimal synchronous process pool for expensive independent environments.

    Policy inference stays in the parent process and remains GPU-batched.  A
    command is first sent to every worker and only then received, so all
    environment physics steps execute concurrently while result ordering stays
    deterministic.
    """

    def __init__(
        self,
        factories: Sequence[Callable[[], object]],
        *,
        start_method: str = "spawn",
    ) -> None:
        if len(factories) < 2:
            raise ValueError("ProcessVectorEnv requires at least two environments")
        context = mp.get_context(start_method)
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        for index, factory in enumerate(factories):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(child, cloudpickle.dumps(factory)),
                name=f"process-vector-env-{index}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        for connection in self._connections:
            status, payload = connection.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"Environment worker failed to start:\n{payload}")

        self.current_observations: list[np.ndarray] | None = None
        self.episode_returns = np.zeros(len(self), dtype=np.float64)
        self.episode_lengths = np.zeros(len(self), dtype=np.int64)

    def __len__(self) -> int:
        return len(self._connections)

    @staticmethod
    def _receive(connection: Connection):
        status, payload = connection.recv()
        if status == "error":
            raise RuntimeError(f"Environment worker failed:\n{payload}")
        if status != "result":
            raise RuntimeError(f"Unexpected environment worker status {status!r}")
        return payload

    def reset(self, seeds: Sequence[int | None]) -> list[tuple[np.ndarray, dict]]:
        if len(seeds) != len(self):
            raise ValueError("Seed count must match environment count")
        for connection, seed in zip(self._connections, seeds):
            connection.send(("reset", {"seed": seed}))
        results = [self._receive(connection) for connection in self._connections]
        self.current_observations = [
            np.asarray(observation, dtype=np.float32)
            for observation, _ in results
        ]
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        return results

    def step(self, actions: np.ndarray) -> list[tuple]:
        actions = np.asarray(actions)
        if actions.shape[0] != len(self):
            raise ValueError("Action batch must match environment count")
        for connection, action in zip(self._connections, actions):
            connection.send(("step", np.asarray(action).copy()))
        return [self._receive(connection) for connection in self._connections]

    def reset_indices(
        self,
        indices: Sequence[int],
    ) -> dict[int, tuple[np.ndarray, dict]]:
        unique_indices = tuple(dict.fromkeys(map(int, indices)))
        for index in unique_indices:
            if not 0 <= index < len(self):
                raise IndexError("Environment reset index is out of range")
            self._connections[index].send(("reset", {}))
        return {
            index: self._receive(self._connections[index])
            for index in unique_indices
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    self._receive(connection)
                except (BrokenPipeError, EOFError, OSError, RuntimeError):
                    pass
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            connection.close()

    def __enter__(self) -> ProcessVectorEnv:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
