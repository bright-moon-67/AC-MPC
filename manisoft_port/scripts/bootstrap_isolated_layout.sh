#!/usr/bin/env bash
set -euo pipefail

bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repository_root=$(dirname -- "$bundle_root")
manisoft_root="$repository_root/ManiSoft"

if ! git -C "$manisoft_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Missing pinned ManiSoft submodule: $manisoft_root" >&2
  echo "Run: git -C $repository_root submodule update --init --recursive" >&2
  exit 1
fi

expected_manisoft_commit=0096f2358d2605b9d382480a7abd30e5c2292495
actual_manisoft_commit=$(git -C "$manisoft_root" rev-parse HEAD)
if [[ "$actual_manisoft_commit" != "$expected_manisoft_commit" ]]; then
  echo "Wrong ManiSoft commit: $actual_manisoft_commit" >&2
  echo "Expected: $expected_manisoft_commit" >&2
  exit 1
fi

for artifact_dir in data runs work_dirs; do
  outer_path="$repository_root/$artifact_dir"
  inner_path="$bundle_root/$artifact_dir"
  mkdir -p -- "$outer_path"
  if [[ -e "$inner_path" && ! -L "$inner_path" ]]; then
    echo "Refusing to replace existing path: $inner_path" >&2
    exit 1
  fi
  if [[ ! -L "$inner_path" ]]; then
    ln -s -- "../$artifact_dir" "$inner_path"
  fi
done

if [[ "$repository_root" != "/root/autodl-tmp/AC-MPC" ]]; then
  echo "WARNING: v15e embeds /root/autodl-tmp/AC-MPC/work_dirs/..." >&2
  echo "Place this checkout at /root/autodl-tmp/AC-MPC for direct v15e use." >&2
fi

echo "isolated ManiSoft port layout: OK"
echo "project:  $bundle_root"
echo "ManiSoft: $manisoft_root"
echo "artifacts: $repository_root/{data,runs,work_dirs}"
