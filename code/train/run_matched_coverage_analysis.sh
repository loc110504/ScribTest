#!/usr/bin/env bash
# "Matched-coverage analysis" (paper_icassp2027/main.tex, Sec. 3.3): compares
# Confidence / R_i / EPS pseudo-label selectors on ONE frozen ScribCal (full)
# teacher, at the SAME accepted-pixel budget. Does not train anything --
# reuses the "scribcal_full" checkpoint already produced by
# run_table2_component_ablation.sh (train_voxtrust3d_2d.py --ablation full).
#
# See code/test/test_voxtrust3d_matched_coverage_2d.py's module docstring for
# why this needs exactly one frozen teacher (not three separately trained
# models) and how the matched budget is constructed.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                 space-separated subset, default "ACDC"
#   SCRIBBLE_DEVICE                   e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_ROOT_PATH                ScribbleBench root override (--root_path)
#   SCRIBBLE_TABLE2_CHECKPOINT_ROOT   default "<repo>/checkpoints"; must match
#                                      the root run_table2_component_ablation.sh
#                                      trained the "scribcal_full" arm into
#   SCRIBBLE_MATCHED_COVERAGE_RESULTS_ROOT   default "<repo>/results"
#   SCRIBBLE_EXTRA_EVAL_ARGS           extra args appended to the evaluator call
#
# Example:
#   SCRIBBLE_DATASETS="ACDC MSCMR" SCRIBBLE_DEVICE=cuda \
#   bash code/train/run_matched_coverage_analysis.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC})
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_eval_args="${SCRIBBLE_EXTRA_EVAL_ARGS:-}"

checkpoint_root="${SCRIBBLE_TABLE2_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_MATCHED_COVERAGE_RESULTS_ROOT:-$repo_dir/results}"
namespace="ScribbleBench_VoxTrust3D_table2_ablation"

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: the matched-coverage analysis is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  ckpt="$checkpoint_root/$namespace/$dataset/scribcal_full/best.pth"
  if [ ! -f "$ckpt" ]; then
    echo "=== dataset=$dataset: no ScribCal (full) checkpoint at $ckpt -- skipping ===" >&2
    continue
  fi

  echo "=== dataset=$dataset: matched-coverage selector comparison ===" >&2
  python "$test_dir/test_voxtrust3d_matched_coverage_2d.py" \
    --checkpoint "$ckpt" \
    --output_dir "$results_root/ScribbleBench_VoxTrust3D_matched_coverage/$dataset" \
    $device_flag $root_path_flag $extra_eval_args
done
