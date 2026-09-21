#!/usr/bin/env bash
# Train + test NeSy-Scrib on ACDC and MSCMR, appending each result to one
# summary CSV. Mirrors run_baselines.sh's structure and env-var conventions;
# ACDC/MSCMR-only (2D) -- its anatomical rule bank (RV/MYO/LV connectivity,
# MYO's ring cavity, LV-MYO/RV-MYO adjacency) is specific to cardiac
# short-axis anatomy and has no analog for WORD's 7 abdominal organs (see
# train_nesyscrib_2d.py's module docstring). The checkpoint is a plain
# UNet2D (Mean-Teacher pair), evaluated with the shared test_pce_2d.py,
# exactly like pCE/CycleMix/EFFDNet/SDT-Net/ModelMix/Bayes-WSS/VoxTrust-3D.
#
# Environment variable overrides (all optional, same names as
# run_baselines.sh so scripts can share a shell environment):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_BATCH_SIZE               default 8; forwarded as --batch_size to every train_*.py call
#   SCRIBBLE_AMP_FLAG                default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                  e.g. "cpu"; forwarded as --device to every command
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_BASELINES_CHECKPOINT_ROOT default "<repo>/checkpoints"
#   SCRIBBLE_BASELINES_RESULTS_ROOT    default "<repo>/results"
#   SCRIBBLE_BASELINES_CSV             default "<results_root>/baselines_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_*.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--max_iterations 4 --early_interval 2 --late_interval 2 --num_workers 0" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_nesyscrib.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC MSCMR})
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_BASELINES_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_BASELINES_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_BASELINES_CSV:-$results_root/baselines_summary.csv}"

append_row() {
  # args: method dataset stage checkpoint metrics_json
  python "$test_dir/append_metrics_csv.py" \
    --method "$1" --dataset "$2" --stage "$3" --checkpoint "$4" --metrics_json "$5" --csv "$csv_path"
}

for dataset in "${datasets[@]}"; do
  if [ "$dataset" = "WORD" ]; then
    echo "Skipping WORD: NeSy-Scrib is ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  ckpt_dir="$checkpoint_root/ScribbleBench_NeSyScrib/$dataset"
  results_dir="$results_root/ScribbleBench_NeSyScrib/$dataset"
  echo "=== [NeSy-Scrib] dataset=$dataset: train ==="
  python "$script_dir/train_nesyscrib_2d.py" --dataset "$dataset" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag $extra_train_args
  echo "=== [NeSy-Scrib] dataset=$dataset: test ==="
  python "$test_dir/test_pce_2d.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args
  append_row NeSyScrib "$dataset" "" "$ckpt_dir/best.pth" "$results_dir/metrics.json"
done

echo "Done. Summary CSV: $csv_path"
