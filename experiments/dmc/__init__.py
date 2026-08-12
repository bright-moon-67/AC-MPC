"""DMC benchmark sub-project for AC-MPC.

Task registry + env adapter + data collection + Koopman training + evaluation
for the DeepMind Control Suite ladder (cartpole_swingup -> reacher_hard ->
hopper_hop -> walker_run -> humanoid_run[pure_state]).
"""

from .tasks.adapter import DMCAdapter, make_dmc_adapter
from .tasks.registry import (
    ALL_TASK_ORDER,
    DMC_CUSTOM_PROTOCOL,
    DMC_NATIVE_PROTOCOL,
    LADDER_ORDER,
    OPTIONAL_COMPARISON_TASKS,
    TASK_SPECS,
    TaskSpec,
    get_task_spec,
    measure_task_spec,
    verify_task_spec,
)

__all__ = [
    "DMCAdapter",
    "make_dmc_adapter",
    "DMC_NATIVE_PROTOCOL",
    "DMC_CUSTOM_PROTOCOL",
    "LADDER_ORDER",
    "OPTIONAL_COMPARISON_TASKS",
    "ALL_TASK_ORDER",
    "TASK_SPECS",
    "TaskSpec",
    "get_task_spec",
    "measure_task_spec",
    "verify_task_spec",
]
