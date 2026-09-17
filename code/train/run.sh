#!/usr/bin/env bash
# pCE only, all 3 datasets: ACDC/MSCMR via the 2D slice pipeline
# (train_pce_2d.py), WORD via the full-3D VNet pipeline (train_pce_3d.py).
#
# Extra args in "$@" are forwarded as-is to every dataset's script, so flags
# that differ in shape between the two pipelines (e.g. --patch_size takes
# two values for 2D, three for 3D) must be passed per-dataset instead of
# through this wrapper.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
output_root="${SCRIBBLE_PCE_OUTPUT_ROOT:-$repo_dir/checkpoints/ScribbleBench_pCE}"

for dataset_name in ACDC MSCMR WORD; do
  case "$dataset_name" in
    WORD) script="$script_dir/train_pce_3d.py" ;;
    *) script="$script_dir/train_pce_2d.py" ;;
  esac
  python "$script" \
    --dataset "$dataset_name" \
    "$@" \
    --output_dir "$output_root/$dataset_name"
done
