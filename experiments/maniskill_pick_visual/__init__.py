"""Visual PickCube experiments for controlled Koopman dynamics.

The environment/collector depend on ManiSkill, while :mod:`dataset` only
depends on the project's data extras.  Imports are intentionally kept lazy so
that importing ``experiments`` does not make ManiSkill a core dependency.
"""

__all__ = [
    "collect_visual_pickcube",
    "dataset",
    "run_ablation",
    "visual_pick_cube",
]
