"""GPU-native MuJoCo Playground experiments.

This package intentionally keeps the execution contract small.  The DMC
experiment remains the source of the algorithm definitions, while this
package owns only the Playground task mapping, GPU audit, training config,
and resumable runners.
"""

from experiments.playground.tasks import TASKS, PlaygroundTask

__all__ = ["TASKS", "PlaygroundTask"]
