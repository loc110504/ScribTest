#!/usr/bin/env bash
# Train + test pCE, CycleMix, SDT-Net, VoxTrust-3D, EFFDNet and DMSPS (stage
# 1 and stage 2) on ACDC, MSCMR and WORD, plus ModelMix jointly on ACDC+MSCMR,
# and append every result to one summary CSV.
#
# ACDC/MSCMR train as independent 2D slices (anisotropic spacing, the
# standard protocol in the scribble-supervision literature -- WSL4MIS,
# DMSPS, CycleMix, ScribFormer); WORD trains as full 3D volumes with a VNet
# backbone. Both pipelines share this same orchestration script and append to
# the same results CSV, since their evaluators score with the same
# dimension-agnostic metrics_3d.py.
#
# For each dataset:
#   ACDC/MSCMR (2D)                       WORD (3D, VNet)
#   1. train_pce_2d.py       -> test_pce_2d.py     1. train_pce_3d.py       -> test_pce_3d.py
#   2. train_cyclemix_2d.py  -> test_pce_2d.py     2. train_cyclemix_3d.py  -> test_pce_3d.py
#   3. train_sdtnet_2d.py    -> test_pce_2d.py     3. train_sdtnet_3d.py    -> test_pce_3d.py
#   4. train_voxtrust3d_2d.py-> test_pce_2d.py     4. train_voxtrust3d_3d.py-> test_pce_3d.py
#   5. train_effdnet_2d.py   -> test_pce_2d.py     5. train_effdnet_3d.py   -> test_pce_3d.py
#   6. train_dmsps_2d.py     -> test_dmsps_2d.py   6. train_dmsps_3d.py     -> test_dmsps_3d.py
#      --stage 1/2 (2)                                --stage 1/2 (2)
# pCE/CycleMix/SDT-Net/VoxTrust-3D/EFFDNet all checkpoint a plain UNet2D (or
# VNet3D on WORD) -- VoxTrust-3D's deployed model is its EMA teacher (Sec. 5:
# "only one EMA network is required at inference"), EFFDNet's is its student
# -- so all five share the same evaluator; DMSPS's dual-decoder DB-Net
# (UNetCCT2D / VNetCCT3D) has its own.
#
# ModelMix (Zhang & Patel, MICCAI 2024) is run separately, once, after the
# per-dataset loop below: it always jointly trains a *pair* of tasks (one
# encoder+decoder each, periodically cross-mixing one encoder layer), and
# ACDC+MSCMR are the only two datasets here sharing a compatible label space
# (matching the paper's own primary experiment). It has no WORD counterpart
# and is skipped unless both ACDC and MSCMR are in `datasets` below.
#
# All scripts default to `--foreground_crop_prob 0` on WORD (uniform random
# crop, matching the official CycleMix/DMSPS/SDT-Net training recipes) and to
# ACDC/MSCMR's shared 2D augmentation policy (RandomGenerator2D), so all
# methods are compared under the same, paper-faithful protocol per dataset.
#
# Every train+test cycle appends one row to the results CSV (default:
# results/baselines_summary.csv) via append_metrics_csv.py. The CSV is
# append-only across script runs (a rerun adds new rows rather than
# overwriting), so remove stale rows yourself if you re-run a stage you
# already recorded.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_DATASETS                space-separated subset, default "ACDC MSCMR WORD"
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
#   bash code/train/run_baselines.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

datasets=(${SCRIBBLE_DATASETS:-ACDC MSCMR WORD})
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_BASELINES_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_BASELINES_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_BASELINES_CSV:-$results_root/baselines_summary.csv}"

dim_for_dataset() {
  case "$1" in
    WORD) echo 3d ;;
    *) echo 2d ;;
  esac
}

append_row() {
  # args: method dataset stage checkpoint metrics_json
  python "$test_dir/append_metrics_csv.py" \
    --method "$1" --dataset "$2" --stage "$3" --checkpoint "$4" --metrics_json "$5" --csv "$csv_path"
}

train_and_test() {
  # args: method_label dataset stage train_checkpoint_dir train_results_dir extra_train_flags...
  local method_label="$1" dataset="$2" stage="$3" ckpt_dir="$4" results_dir="$5"
  shift 5
  local dim
  dim="$(dim_for_dataset "$dataset")"

  echo "=== [$method_label] dataset=$dataset ($dim) stage=${stage:-none}: train ==="
  case "$method_label" in
    pCE)
      python "$script_dir/train_pce_${dim}.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    CycleMix)
      python "$script_dir/train_cyclemix_${dim}.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    SDTNet)
      python "$script_dir/train_sdtnet_${dim}.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    VoxTrust3D)
      python "$script_dir/train_voxtrust3d_${dim}.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    EFFDNet)
      python "$script_dir/train_effdnet_${dim}.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    DMSPS)
      python "$script_dir/train_dmsps_${dim}.py" --dataset "$dataset" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    *)
      echo "Unknown method_label: $method_label" >&2
      exit 1
      ;;
  esac

  echo "=== [$method_label] dataset=$dataset ($dim) stage=${stage:-none}: test ==="
  case "$method_label" in
    pCE|CycleMix|SDTNet|VoxTrust3D|EFFDNet)
      python "$test_dir/test_pce_${dim}.py" \
        --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args
      ;;
    DMSPS)
      python "$test_dir/test_dmsps_${dim}.py" \
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

  train_and_test VoxTrust3D "$dataset" "" \
    "$checkpoint_root/ScribbleBench_VoxTrust3D/$dataset" \
    "$results_root/ScribbleBench_VoxTrust3D/$dataset"

  train_and_test EFFDNet "$dataset" "" \
    "$checkpoint_root/ScribbleBench_EFFDNet/$dataset" \
    "$results_root/ScribbleBench_EFFDNet/$dataset"

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

has_dataset() {
  local wanted="$1"
  for dataset in "${datasets[@]}"; do
    [ "$dataset" = "$wanted" ] && return 0
  done
  return 1
}

if has_dataset ACDC && has_dataset MSCMR; then
  echo "=== [ModelMix] ACDC+MSCMR (joint): train ==="
  modelmix_ckpt_dir="$checkpoint_root/ScribbleBench_ModelMix"
  python "$script_dir/train_modelmix_2d.py" \
    --output_dir "$modelmix_ckpt_dir" --batch_size "$batch_size" $amp_flag $device_flag $root_path_flag $extra_train_args

  for dataset in ACDC MSCMR; do
    echo "=== [ModelMix] dataset=$dataset (2d): test ==="
    results_dir="$results_root/ScribbleBench_ModelMix/$dataset"
    python "$test_dir/test_pce_2d.py" \
      --checkpoint "$modelmix_ckpt_dir/$dataset/best.pth" --output_dir "$results_dir" \
      $amp_flag $device_flag $root_path_flag $extra_test_args
    append_row ModelMix "$dataset" "" "$modelmix_ckpt_dir/$dataset/best.pth" "$results_dir/metrics.json"
  done
else
  echo "Skipping ModelMix: requires both ACDC and MSCMR in SCRIBBLE_DATASETS (got: ${datasets[*]})"
fi

echo "Done. Summary CSV: $csv_path"
