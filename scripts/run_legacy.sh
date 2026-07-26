#!/usr/bin/env bash
set -euo pipefail

# Reproducible launcher for the dedicated Python 3.10 / Gym 0.23 / D4RL 1.1
# environment. Additional command-line arguments are executed inside that env.
legacy_env="${AC_MPC_LEGACY_ENV:-antmaze_legacy}"
conda_bin="${AC_MPC_CONDA:-${CONDA_EXE:-}}"
if [[ -z "${conda_bin}" ]]; then
    conda_bin="$(command -v conda || true)"
fi
if [[ -z "${conda_bin}" ]]; then
    echo "Conda was not found; set AC_MPC_CONDA to its executable path." >&2
    exit 1
fi

legacy_prefix="${AC_MPC_LEGACY_PREFIX:-}"
if [[ -z "${legacy_prefix}" ]]; then
    legacy_prefix="$(
        "${conda_bin}" run -n "${legacy_env}" \
            python -c 'import sys; print(sys.prefix)'
    )"
fi

mujoco_path="${MUJOCO_PY_MUJOCO_PATH:-${HOME}/.mujoco/mujoco210}"
nvidia_lib="${AC_MPC_NVIDIA_LIB:-/usr/lib/nvidia}"
runtime_libs="${mujoco_path}/bin:${legacy_prefix}/lib"
if [[ -d "${nvidia_lib}" ]]; then
    runtime_libs="${runtime_libs}:${nvidia_lib}"
fi

export MUJOCO_PY_MUJOCO_PATH="${mujoco_path}"
export LD_LIBRARY_PATH="${runtime_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export D4RL_SUPPRESS_IMPORT_ERROR=1

exec "${conda_bin}" run --no-capture-output -n "${legacy_env}" "$@"
