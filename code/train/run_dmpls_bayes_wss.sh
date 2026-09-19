#!/usr/bin/env bash
# Train + test DMPLS (Luo et al., MICCAI 2022) and Bayes-WSS (Zheng et al.,
# MICCAI 2024) on ACDC and MSCMR, appending each result to one summary CSV.
# Mirrors run_baselines.sh's structure and env-var conventions, but is
# 2D-only (ACDC/MSCMR) and scoped to just these two methods -- WORD has no
# DMPLS/Bayes-WSS pipeline in this benchmark (see the module docstrings of
# train_dmpls_2d.py / train_bayes_wss_2d.py).
#
# DMPLS checkpoints a dual-decoder UNetCCT2D DB-Net and is evaluated with
# test_dmpls_2d.py. Bayes-WSS deploys a plain UNet2D (the CVAE used to
# generate pseudo-labels is training-only, never checkpointed) and is
# evaluated with the shared test_pce_2d.py, exactly like pCE/CycleMix/
# EFFDNet/SDT-Net/ModelMix.
#
# Environment variable overrides (all optional, same names as
# run_baselines.sh so both scripts can share a shell environment):
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
#   bash code/train/run_dmpls_bayes_wss.sh

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
    echo "Skipping WORD: DMPLS/Bayes-WSS are ACDC/MSCMR-only (2D) in this benchmark" >&2
    continue
  fi

  dmpls_ckpt_dir="$checkpoint_root/ScribbleBench_DMPLS/$dataset"
  dmpls_results_dir="$results_root/ScribbleBench_DMPLS/$dataset"
  echo "=== [DMPLS] dataset=$dataset: train ==="
  python "$script_dir/train_dmpls_2d.py" --dataset "$dataset" \
    --output_dir "$dmpls_ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag $extra_train_args
  echo "=== [DMPLS] dataset=$dataset: test ==="
  python "$test_dir/test_dmpls_2d.py" \
    --checkpoint "$dmpls_ckpt_dir/best.pth" --output_dir "$dmpls_results_dir" \
    $amp_flag $device_flag $root_path_flag $extra_test_args
  append_row DMPLS "$dataset" "" "$dmpls_ckpt_dir/best.pth" "$dmpls_results_dir/metrics.json"

  bayes_wss_ckpt_dir="$checkpoint_root/ScribbleBench_BayesWSS/$dataset"
  bayes_wss_results_dir="$results_root/ScribbleBench_BayesWSS/$dataset"
  echo "=== [Bayes-WSS] dataset=$dataset: train ==="
  python "$script_dir/train_bayes_wss_2d.py" --dataset "$dataset" \
    --output_dir "$bayes_wss_ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag $extra_train_args
  echo "=== [Bayes-WSS] dataset=$dataset: test ==="
  python "$test_dir/test_pce_2d.py" \
    --checkpoint "$bayes_wss_ckpt_dir/best.pth" --output_dir "$bayes_wss_results_dir" \
    $amp_flag $device_flag $root_path_flag $extra_test_args
  append_row BayesWSS "$dataset" "" "$bayes_wss_ckpt_dir/best.pth" "$bayes_wss_results_dir/metrics.json"
done

echo "Done. Summary CSV: $csv_path"
