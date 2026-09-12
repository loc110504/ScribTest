#!/usr/bin/env bash
# Train + test pCE, CycleMix, SDT-Net and DMSPS (stage 1 and stage 2) on
# ACDC, MSCMR and WORD, and append every result to one summary CSV.
#
# For each dataset:
#   1. train_pce_3d.py                -> test_pce_3d.py       (pCE baseline)
#   2. train_cyclemix_3d.py           -> test_pce_3d.py       (CycleMix is a
#                                         plain UNet3D checkpoint)
#   3. train_sdtnet_3d.py             -> test_pce_3d.py       (SDT-Net's
#                                         deployed student is also a plain
#                                         UNet3D checkpoint)
#   4. train_dmsps_3d.py --stage 1     -> test_dmsps_3d.py     (stage-1 DB-Net)
#   5. train_dmsps_3d.py --stage 2     -> test_dmsps_3d.py     (re-initialized
#      (--init_checkpoint = stage-1 best.pth)                   from stage 1)
#
# All scripts default to `--foreground_crop_prob 0` (uniform random crop),
# matching the official CycleMix/DMSPS/SDT-Net training recipes -- not the
# 100%-foreground-centered crop this repo used to default to -- so all
# methods are compared under the same, paper-faithful crop policy.
#
# Every train+test cycle appends one row to the results CSV (default:
# results/baselines_summary.csv) via append_metrics_csv.py. The CSV is
# append-only across script runs (a rerun adds new rows rather than
# overwriting), so remove stale rows yourself if you re-run a stage you
# already recorded.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC MSCMR WORD"
#   SCRIBBLE_BATCH_SIZE               default 4; forwarded as --batch_size to every train_*.py call
#   SCRIBBLE_AMP_FLAG                default "--amp"; set to "" to disable AMP
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
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--max_iterations 4 --eval_every 2 --save_every 2 --num_workers 0" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_baselines.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC MSCMR WORD})
batch_size="${SCRIBBLE_BATCH_SIZE:-4}"
amp_flag="${SCRIBBLE_AMP_FLAG-"--amp"}"
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

train_and_test() {
  # args: method_label dataset stage train_checkpoint_dir train_results_dir extra_train_flags...
  local method_label="$1" dataset="$2" stage="$3" ckpt_dir="$4" results_dir="$5"
  shift 5

  echo "=== [$method_label] dataset=$dataset stage=${stage:-none}: train ==="
  case "$method_label" in
    pCE)
      python "$script_dir/train_pce_3d.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    CycleMix)
      python "$script_dir/train_cyclemix_3d.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    SDTNet)
      python "$script_dir/train_sdtnet_3d.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    DMSPS)
      python "$script_dir/train_dmsps_3d.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    *)
      echo "Unknown method_label: $method_label" >&2
      exit 1
      ;;
  esac

  echo "=== [$method_label] dataset=$dataset stage=${stage:-none}: test ==="
  case "$method_label" in
    pCE|CycleMix|SDTNet)
      python "$test_dir/test_pce_3d.py" \
        --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args
      ;;
    DMSPS)
      python "$test_dir/test_dmsps_3d.py" \
        --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args
      ;;
  esac

  append_row "$method_label" "$dataset" "$stage" "$ckpt_dir/best.pth" "$results_dir/metrics.json"
}

for dataset in "${datasets[@]}"; do
  train_and_test pCE "$dataset" "" \
    "$checkpoint_root/ScribbleBench_pCE/$dataset" \
    "$results_root/ScribbleBench_pCE/$dataset"

  train_and_test CycleMix "$dataset" "" \
    "$checkpoint_root/ScribbleBench_CycleMix/$dataset" \
    "$results_root/ScribbleBench_CycleMix/$dataset"

  train_and_test SDTNet "$dataset" "" \
    "$checkpoint_root/ScribbleBench_SDTNet/$dataset" \
    "$results_root/ScribbleBench_SDTNet/$dataset"

  dmsps_stage1_ckpt_dir="$checkpoint_root/ScribbleBench_DMSPS/$dataset/stage1"
  train_and_test DMSPS "$dataset" stage1 \
    "$dmsps_stage1_ckpt_dir" \
    "$results_root/ScribbleBench_DMSPS/$dataset/stage1" \
    --stage 1

  train_and_test DMSPS "$dataset" stage2 \
    "$checkpoint_root/ScribbleBench_DMSPS/$dataset/stage2" \
    "$results_root/ScribbleBench_DMSPS/$dataset/stage2" \
    --stage 2 --init_checkpoint "$dmsps_stage1_ckpt_dir/best.pth"
done

echo "Done. Summary CSV: $csv_path"
