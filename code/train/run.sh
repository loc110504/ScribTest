#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
output_root="${SCRIBBLE_PCE_OUTPUT_ROOT:-$repo_dir/checkpoints/ScribbleBench_pCE}"

for dataset_name in ACDC MSCMR WORD; do
  python "$script_dir/train_pce_3d.py" \
    --dataset "$dataset_name" \
    "$@" \
    --output_dir "$output_root/$dataset_name"
done
