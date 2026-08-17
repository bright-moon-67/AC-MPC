"""DMC task registry and canonical environment adapter."""

from .adapter import DMCAdapter, make_dmc_adapter
from .registry import (
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
