#!/usr/bin/env bash
# Train + test every baseline in Table 1 of paper_icassp2027/main.tex on
# MSCMR, ScribbleBench scribbles (dataset/ScribbleBench/MSCMR), one summary
# CSV row per method:
#   pCE (ECCV'18), CycleMix (CVPR'22), DMSPS (MedIA'24, stage1+stage2),
#   ModelMix (MICCAI'24), EFFDNet (MICCAI'25), SDT-Net (ISBI'26),
#   VoxTrust-3D/"DCC (ours)" (--ablation full, the training script's own
#   default -- see train_voxtrust3d_2d.py's module docstring).
#
# This is a narrower, paper-scoped sibling of run_baselines.sh: it excludes
# WORD (ACDC/MSCMR-only paper). See run_icassp2027_baselines_acdc.sh for
# the ACDC twin -- run both to reproduce the whole of Table 1.
#
# All ACDC/MSCMR methods here checkpoint a plain UNet2D and are evaluated by
# test_pce_2d.py; DMSPS's dual-decoder UNetCCT2D uses test_dmsps_2d.py
# instead. VoxTrust-3D's checkpoint carries both an EMA teacher and a
# student (Mean Teacher) -- test_pce_2d.py's default --eval_target student
# evaluates the student (the network gradient descent actually trained,
# matching every other baseline here); pass SCRIBBLE_EXTRA_TEST_ARGS
#="--eval_target teacher" to instead reproduce the paper's Sec. 3.3
# teacher-at-inference protocol.
#
# ModelMix has no --dataset flag: one run always jointly trains BOTH ACDC
# and MSCMR (see utils/modelmix.py's module docstring for why no other pair
# in this benchmark is a substitute), producing
# <checkpoint_root>/ScribbleBench_ModelMix/{ACDC,MSCMR}/best.pth in a single
# invocation. This script trains it if that checkpoint isn't already there
# (so running this script alone, standalone, still gets you a complete
# MSCMR row), then evaluates only the MSCMR half; if you already ran
# run_icassp2027_baselines_acdc.sh first, this script reuses the same
# checkpoint without retraining (train_modelmix_2d.py would refuse a fresh
# run into an --output_dir that already has a best.pth --
# common_3d.guard_fresh_output_dir).
#
# Every train+test cycle appends one row to the results CSV (default:
# results/icassp2027_baselines_summary.csv, shared with the ACDC script so
# Table 1 ends up as one file) via append_metrics_csv.py. The CSV is
# append-only across script runs, so remove stale rows yourself before
# re-running a stage you already recorded.
#
# Environment variable overrides (all optional):
#   SCRIBBLE_SEED                     default 2026; forwarded as --seed to every train_*.py call
#   SCRIBBLE_MAX_ITERATIONS           default 30000; forwarded as --max_iterations
#   SCRIBBLE_BATCH_SIZE               default 8; forwarded as --batch_size
#   SCRIBBLE_AMP_FLAG                 default "" (AMP off); set to "--amp" to enable AMP
#   SCRIBBLE_DEVICE                   e.g. "cuda" or "cpu"; forwarded as --device
#   SCRIBBLE_NUM_WORKERS              default 4; forwarded as --num_workers
#   SCRIBBLE_ROOT_PATH                ScribbleBench root override (--root_path)
#   SCRIBBLE_BASELINES_CHECKPOINT_ROOT  default "<repo>/checkpoints"
#   SCRIBBLE_BASELINES_RESULTS_ROOT     default "<repo>/results"
#   SCRIBBLE_BASELINES_CSV              default "<results_root>/icassp2027_baselines_summary.csv"
#   SCRIBBLE_EXTRA_TRAIN_ARGS          extra args appended to every train_*.py call
#   SCRIBBLE_EXTRA_TEST_ARGS           extra args appended to every test_*.py call
#
# Example - quick end-to-end smoke run on CPU:
#   SCRIBBLE_DEVICE=cpu SCRIBBLE_AMP_FLAG="" SCRIBBLE_BATCH_SIZE=2 \
#   SCRIBBLE_MAX_ITERATIONS=8 \
#   SCRIBBLE_EXTRA_TRAIN_ARGS="--early_interval 4 --late_interval 4 --num_workers 0" \
#   SCRIBBLE_EXTRA_TEST_ARGS="--case_limit 2" \
#   bash code/train/run_icassp2027_baselines_mscmr.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
test_dir="$repo_dir/code/test"

dataset="MSCMR"
seed="${SCRIBBLE_SEED:-2026}"
max_iterations="${SCRIBBLE_MAX_ITERATIONS:-30000}"
batch_size="${SCRIBBLE_BATCH_SIZE:-8}"
amp_flag="${SCRIBBLE_AMP_FLAG-}"
device_flag="${SCRIBBLE_DEVICE:+--device $SCRIBBLE_DEVICE}"
num_workers="${SCRIBBLE_NUM_WORKERS:-4}"
root_path_flag="${SCRIBBLE_ROOT_PATH:+--root_path $SCRIBBLE_ROOT_PATH}"
extra_train_args="${SCRIBBLE_EXTRA_TRAIN_ARGS:-}"
extra_test_args="${SCRIBBLE_EXTRA_TEST_ARGS:-}"

checkpoint_root="${SCRIBBLE_BASELINES_CHECKPOINT_ROOT:-$repo_dir/checkpoints}"
results_root="${SCRIBBLE_BASELINES_RESULTS_ROOT:-$repo_dir/results}"
csv_path="${SCRIBBLE_BASELINES_CSV:-$results_root/icassp2027_baselines_summary.csv}"

append_row() {
  # args: method stage checkpoint metrics_json
  python "$test_dir/append_metrics_csv.py" \
    --method "$1" --dataset "$dataset" --stage "$2" --checkpoint "$3" --metrics_json "$4" --csv "$csv_path"
}

train_and_test() {
  # args: method_label stage ckpt_dir results_dir extra_train_flags...
  local method_label="$1" stage="$2" ckpt_dir="$3" results_dir="$4"
  shift 4

  echo "=== [$method_label] dataset=$dataset stage=${stage:-none} seed=$seed max_iterations=$max_iterations: train ==="
  case "$method_label" in
    pCE)
      python "$script_dir/train_pce_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    CycleMix)
      python "$script_dir/train_cyclemix_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    SDTNet)
      python "$script_dir/train_sdtnet_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    VoxTrust3D)
      python "$script_dir/train_voxtrust3d_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    EFFDNet)
      python "$script_dir/train_effdnet_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    DMSPS)
      python "$script_dir/train_dmsps_2d.py" --dataset "$dataset" --seed "$seed" --max_iterations "$max_iterations" \
        --output_dir "$ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
        $amp_flag $device_flag $root_path_flag "$@" $extra_train_args
      ;;
    *)
      echo "Unknown method_label: $method_label" >&2
      exit 1
      ;;
  esac

  echo "=== [$method_label] dataset=$dataset stage=${stage:-none}: test ==="
  case "$method_label" in
    pCE|CycleMix|SDTNet|VoxTrust3D|EFFDNet)
      python "$test_dir/test_pce_2d.py" \
        --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args
      ;;
    DMSPS)
      python "$test_dir/test_dmsps_2d.py" \
        --checkpoint "$ckpt_dir/best.pth" --output_dir "$results_dir" $amp_flag $device_flag $root_path_flag $extra_test_args
      ;;
  esac

  append_row "$method_label" "$stage" "$ckpt_dir/best.pth" "$results_dir/metrics.json"
}

train_and_test pCE "" \
  "$checkpoint_root/ScribbleBench_pCE/$dataset" \
  "$results_root/ScribbleBench_pCE/$dataset"

train_and_test CycleMix "" \
  "$checkpoint_root/ScribbleBench_CycleMix/$dataset" \
  "$results_root/ScribbleBench_CycleMix/$dataset"

train_and_test SDTNet "" \
  "$checkpoint_root/ScribbleBench_SDTNet/$dataset" \
  "$results_root/ScribbleBench_SDTNet/$dataset"

train_and_test VoxTrust3D "" \
  "$checkpoint_root/ScribbleBench_VoxTrust3D/$dataset" \
  "$results_root/ScribbleBench_VoxTrust3D/$dataset"

train_and_test EFFDNet "" \
  "$checkpoint_root/ScribbleBench_EFFDNet/$dataset" \
  "$results_root/ScribbleBench_EFFDNet/$dataset"

dmsps_stage1_ckpt_dir="$checkpoint_root/ScribbleBench_DMSPS/$dataset/stage1"
train_and_test DMSPS stage1 \
  "$dmsps_stage1_ckpt_dir" \
  "$results_root/ScribbleBench_DMSPS/$dataset/stage1" \
  --stage 1

train_and_test DMSPS stage2 \
  "$checkpoint_root/ScribbleBench_DMSPS/$dataset/stage2" \
  "$results_root/ScribbleBench_DMSPS/$dataset/stage2" \
  --stage 2 --init_checkpoint "$dmsps_stage1_ckpt_dir/best.pth"

modelmix_ckpt_dir="$checkpoint_root/ScribbleBench_ModelMix"
if [ -f "$modelmix_ckpt_dir/ACDC/best.pth" ] && [ -f "$modelmix_ckpt_dir/MSCMR/best.pth" ]; then
  echo "=== [ModelMix] ACDC+MSCMR checkpoint already present at $modelmix_ckpt_dir, skipping training ==="
else
  echo "=== [ModelMix] ACDC+MSCMR (joint, no --dataset flag) seed=$seed max_iterations=$max_iterations: train ==="
  python "$script_dir/train_modelmix_2d.py" --seed "$seed" --max_iterations "$max_iterations" \
    --output_dir "$modelmix_ckpt_dir" --batch_size "$batch_size" --num_workers "$num_workers" \
    $amp_flag $device_flag $root_path_flag $extra_train_args
fi

echo "=== [ModelMix] dataset=$dataset: test ==="
modelmix_results_dir="$results_root/ScribbleBench_ModelMix/$dataset"
python "$test_dir/test_pce_2d.py" \
  --checkpoint "$modelmix_ckpt_dir/$dataset/best.pth" --output_dir "$modelmix_results_dir" \
  $amp_flag $device_flag $root_path_flag $extra_test_args
append_row ModelMix "" "$modelmix_ckpt_dir/$dataset/best.pth" "$modelmix_results_dir/metrics.json"

echo "Done. Summary CSV: $csv_path"
echo "Run run_icassp2027_baselines_acdc.sh too (if not already) to complete Table 1."
