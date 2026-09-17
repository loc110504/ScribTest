#!/usr/bin/env bash
# Train + test VoxTrust-3D on ACDC and MSCMR (WORD to follow separately once
# these two are validated), appending each result to the shared baselines
# summary CSV.
#
# ACDC/MSCMR route through the 2D slice pipeline (train_voxtrust3d_2d.py);
# WORD, if added to SCRIBBLE_DATASETS, routes through the full-3D VNet
# pipeline (train_voxtrust3d_3d.py). Either way VoxTrust-3D's checkpointed
# model is the EMA teacher, saved as a plain UNet2D/VNet3D in the same
# schema_version=1 layout the other baselines use (see the training script's
# module docstring), so it is fed through the exact same evaluator as
# pCE/CycleMix/SDT-Net (test_pce_2d.py / test_pce_3d.py): images go through
# the network and the predicted volumes are compared against the dense test
# masks to compute Dice/HD95/ASSD.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC MSCMR"
#   SCRIBBLE_BATCH_SIZE               default 4; forwarded as --batch_size to the train script
#   SCRIBBLE_AMP_FLAG                default "--amp"; set to "" to disable AMP
#   SCRIBBLE_DEVICE                  e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS             default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH               ScribbleBench root override (--root_path)
#   SCRIBBLE_VOXTRUST3D_CHECKPOINT_ROOT default "<repo>/checkpoints"
#   SCRIBBLE_VOXTRUST3D_RESULTS_ROOT    default "<repo>/results"
#   SCRIBBLE_VOXTRUST3D_CSV             default "<results_root>/baselines_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS         extra args appended to every train_voxtrust3d_3d.py call
#   SCRIBBLE_EXTRA_TEST_ARGS          extra args appended to every test_pce_3d.py call
#
# Example - quick end-to-end smoke run on CPU, ACDC only:
#   SCRIBBLE_DATASETS=ACDC SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 SCRIBBLE_NUM_WORKERS=0 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--max_iterations 8 --warmup_frac 0.25 --rampup_frac 0.25 --early_interval 4 --late_interval 4 --late_phase_start 4" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_voxtrust3d.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC MSCMR})
batch_size="${SCRIBBLE_BATCH_SIZE:-4}"
amp_flag="${SCRIBBLE_AMP_FLAG-"--amp"}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_VOXTRUST3D_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_VOXTRUST3D_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_VOXTRUST3D_CSV:-$results_root/baselines_summary.csv}"

dim_for_dataset() {
  case "$1" in
    WORD) echo 3d ;;
    *) echo 2d ;;
  esac
}

for dataset in "${datasets[@]}"; do
  ckpt_dir="$checkpoint_root/ScribbleBench_VoxTrust3D/$dataset"
  results_dir="$results_root/ScribbleBench_VoxTrust3D/$dataset"
  dim="$(dim_for_dataset "$dataset")"

  echo "=== [VoxTrust3D] dataset=$dataset ($dim): train ==="
  python "$script_dir/train_voxtrust3d_${dim}.py" --dataset "$dataset" \
    --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args

  echo "=== [VoxTrust3D] dataset=$dataset ($dim): test ==="
  python "$test_dir/test_pce_${dim}.py" \
    --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args

  python "$test_dir/append_metrics_csv.py" \
    --method VoxTrust3D --dataset "$dataset" --stage "" \
    --checkpoint "$ckpt_dir/best.pth" --metrics_json "$results_dir/metrics.json" --csv "$csv_path"
done

echo "Done. Summary CSV: $csv_path"
