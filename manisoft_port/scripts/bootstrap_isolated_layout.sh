#!/usr/bin/env bash
set -euo pipefail

bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repository_root=$(dirname -- "$bundle_root")
workspace_root=$(dirname -- "$repository_root")
manisoft_root="$workspace_root/ManiSoft"

if [[ ! -d "$manisoft_root/.git" ]]; then
  echo "Missing paired ManiSoft checkout: $manisoft_root" >&2
  echo "Clone bright-moon-67/ManiSoft:acmpc-integration there first." >&2
  exit 1
fi

expected_manisoft_commit=4e02bb87962604c6ab6abf06f3f273a1c49c1270
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

# The historical commands use ../ManiSoft when run inside manisoft_port/.
compat_manisoft="$repository_root/ManiSoft"
if [[ -e "$compat_manisoft" && ! -L "$compat_manisoft" ]]; then
  echo "Refusing to replace existing path: $compat_manisoft" >&2
  exit 1
fi
if [[ ! -L "$compat_manisoft" ]]; then
  ln -s -- ../ManiSoft "$compat_manisoft"
fi

# Keep the runtime-only compatibility link out of the outer repository status.
exclude_file=$(git -C "$repository_root" rev-parse --git-path info/exclude)
if ! grep -qxF '/ManiSoft' "$exclude_file"; then
  printf '%s\n' '/ManiSoft' >> "$exclude_file"
fi

if [[ "$repository_root" != "/root/autodl-tmp/AC-MPC" ]]; then
  echo "WARNING: v15e embeds /root/autodl-tmp/AC-MPC/work_dirs/..." >&2
  echo "Place this checkout at /root/autodl-tmp/AC-MPC for direct v15e use." >&2
fi

echo "isolated ManiSoft port layout: OK"
echo "project:  $bundle_root"
echo "ManiSoft: $manisoft_root"
echo "artifacts: $repository_root/{data,runs,work_dirs}"
